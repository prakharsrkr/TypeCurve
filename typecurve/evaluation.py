import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fpdf import FPDF

from .decline_curve import modified_hyperbolic, validate_inputs
from .data_preparation import denormalize_data_output, filter_by_basin_and_formation


def _is_keras_model(model):
    """Check if *model* is a Keras Model without importing TensorFlow eagerly."""
    try:
        from tensorflow.keras.models import Model
        return isinstance(model, Model)
    except ImportError:
        return False


def predict_with_model(model, numerical_data, categorical_data,
                       numerical_columns=None, categorical_columns=None):
    """Predict with either a Keras model or sklearn pipeline."""
    if _is_keras_model(model):
        return model.predict([numerical_data] + categorical_data)
    else:
        data_combined = np.hstack([numerical_data] + categorical_data)
        combined_df = pd.DataFrame(data_combined,
                                   columns=numerical_columns + categorical_columns)
        return model.predict(combined_df)


def generate_production_rates_for_comp(y_pred_denormalized, y_true_denormalized,
                                       headers, time, resource_type='Oil'):
    """Generate predicted and actual production rates for comparison."""
    predicted_productions = []
    actual_productions = []

    prefix = {'Oil': 'Oil_Params_P50_', 'Gas': 'Gas_Params_P50_',
              'Water': 'Water_Params_P50_'}
    if resource_type not in prefix:
        raise ValueError("resource_type must be 'Oil', 'Gas', or 'Water'.")
    pfx = prefix[resource_type]

    for idx in range(len(y_pred_denormalized)):
        try:
            qi_pred = np.abs(y_pred_denormalized.iloc[idx, headers.index(f'{pfx}InitialProd')])
            di_pred = y_pred_denormalized.iloc[idx, headers.index(f'{pfx}DiCoefficient')]
            b_pred = y_pred_denormalized.iloc[idx, headers.index(f'{pfx}BCoefficient')]

            qi_true = y_true_denormalized.iloc[idx, headers.index(f'{pfx}InitialProd')]
            di_true = y_true_denormalized.iloc[idx, headers.index(f'{pfx}DiCoefficient')]
            b_true = y_true_denormalized.iloc[idx, headers.index(f'{pfx}BCoefficient')]

            IBU, MBU, Dlim = 0, 0, 7

            if not validate_inputs(qi_pred, di_pred, b_pred, Dlim, IBU, MBU):
                predicted_productions.append(np.zeros_like(time))
                actual_productions.append(
                    modified_hyperbolic(time, qi_true, di_true, b_true, Dlim, IBU, MBU)[1])
                continue

            pred_prod = modified_hyperbolic(time, qi_pred, di_pred, b_pred, Dlim, IBU, MBU)[1]
            act_prod = modified_hyperbolic(time, qi_true, di_true, b_true, Dlim, IBU, MBU)[1]

            if len(pred_prod) == len(time) and len(act_prod) == len(time):
                predicted_productions.append(pred_prod)
                actual_productions.append(act_prod)
        except Exception as e:
            print(f"Error generating production rates: {e}")
            predicted_productions.append(np.zeros_like(time))
            actual_productions.append(np.zeros_like(time))

    return predicted_productions, actual_productions


def calculate_errors(predicted_productions, actual_productions):
    """Calculate MSE, MAE, sMAPE for production rate comparisons."""
    errors = {'MSE': [], 'MAE': [], 'sMAPE': []}

    for pred, actual in zip(predicted_productions, actual_productions):
        if len(pred) != len(actual):
            continue
        if np.any(np.isnan(pred)) or np.any(np.isnan(actual)):
            continue

        errors['MSE'].append(mean_squared_error(actual, pred))
        errors['MAE'].append(mean_absolute_error(actual, pred))
        errors['sMAPE'].append(
            np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred)) / 2)) * 100)

    return errors


def calculate_scalar_errors(y_pred_denormalized, y_true_denormalized,
                            headers, resource_type='Oil'):
    """Calculate per-parameter MAE and RMSE."""
    prefix = {'Oil': 'Oil_Params_P50_', 'Gas': 'Gas_Params_P50_',
              'Water': 'Water_Params_P50_'}
    pfx = prefix[resource_type]

    param_names = ['InitialProd', 'DiCoefficient', 'BCoefficient',
                   'BuildupRate', 'MonthsInProd', 'LimDeclineRate']
    scalar_errors = {}
    for p in param_names:
        scalar_errors[f'{p}_MAE'] = []
        scalar_errors[f'{p}_RMSE'] = []

    for idx in range(len(y_pred_denormalized)):
        try:
            for p in param_names:
                if p in ('BuildupRate', 'MonthsInProd', 'LimDeclineRate'):
                    pred_val = 0 if p != 'LimDeclineRate' else 7
                    true_val = pred_val
                else:
                    pred_val = y_pred_denormalized.iloc[idx, headers.index(f'{pfx}{p}')]
                    true_val = y_true_denormalized.iloc[idx, headers.index(f'{pfx}{p}')]

                scalar_errors[f'{p}_MAE'].append(np.abs(pred_val - true_val))
                scalar_errors[f'{p}_RMSE'].append((pred_val - true_val) ** 2)
        except Exception:
            continue

    for key in list(scalar_errors.keys()):
        vals = scalar_errors[key]
        scalar_errors[key] = np.mean(vals) if vals else np.nan
        if key.endswith('_RMSE'):
            scalar_errors[key] = np.sqrt(scalar_errors[key])

    return scalar_errors


def determine_model_type(config_str):
    """Determine model type from config string."""
    if 'regularization' in config_str or 'dense_layer_sizes' in config_str:
        return 'Neural Network'
    return config_str.split(' ')[0]


def identify_best_worst_matches(errors, y_true_denormalized, y_pred_denormalized,
                                error_metric='MSE'):
    """Find best and worst matching wells by error metric."""
    best_idx = np.argmin(errors[error_metric])
    worst_idx = np.argmax(errors[error_metric])

    return (y_true_denormalized.iloc[best_idx], y_pred_denormalized.iloc[best_idx],
            y_true_denormalized.iloc[worst_idx], y_pred_denormalized.iloc[worst_idx],
            best_idx, worst_idx)


def print_performance_metrics(basin, formation, config_str, model_type,
                              mse, mae, smape, scalar_errors):
    """Format performance metrics as a string."""
    return (
        f"Performance for Basin: {basin}\n"
        f"Formation: {formation}\n"
        f"Config: {model_type}\n"
        f"MSE: {mse:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"sMAPE: {smape:.2f}%\n"
        "Scalar Errors:\n" +
        "\n".join(f"{k}: {v:.4f}" for k, v in scalar_errors.items())
    )


def save_plots_to_pdf(plot_data, performance_data, output_path):
    """Save best/worst match plots and metrics to a PDF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for i, (basin, formation, config_str, time_arr,
            best_true, best_pred, worst_true, worst_pred,
            best_idx, worst_idx) in enumerate(plot_data):

        model_type = determine_model_type(config_str)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        metrics = performance_data.get((basin, formation, config_str), {})
        if metrics:
            metrics_text = print_performance_metrics(
                basin, formation, config_str, model_type,
                metrics.get('MSE', 0), metrics.get('MAE', 0),
                metrics.get('sMAPE', 0), metrics)
            pdf.set_text_color(255, 0, 0)
            pdf.set_font("Arial", style='BU', size=12)
            pdf.set_x(10)
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(page_w, 10, f"Basin: {basin} | Formation: {formation} | {model_type}")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=10)
            pdf.set_x(10)
            pdf.multi_cell(page_w, 8, metrics_text)

        min_len = min(len(time_arr), len(best_true), len(best_pred),
                      len(worst_true), len(worst_pred))
        time_arr = time_arr[:min_len]

        for label, true_data, pred_data in [
            ('Best', best_true[:min_len], best_pred[:min_len]),
            ('Worst', worst_true[:min_len], worst_pred[:min_len])
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(time_arr, true_data, 'b-o', label='Actual', markersize=2)
            ax.plot(time_arr, pred_data, 'r--x', label='Predicted', markersize=2)
            ax.set_yscale('log')
            ax.set_ylim(1, 10000000)
            ax.set_title(f'{label} Match - {model_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (Months)')
            ax.set_ylabel('Production Rate (bbl/day)')
            ax.legend()
            ax.grid(True, which='both', linestyle='--')
            plt.tight_layout()

            plot_path = os.path.join(
                os.path.dirname(output_path), f"_temp_{label.lower()}_{i}.png")
            plt.savefig(plot_path)
            plt.close(fig)

            y_pos = pdf.get_y()
            pdf.image(plot_path, x=10, y=y_pos, w=90)
            pdf.set_y(y_pos + 70)  # Move cursor below image
            os.remove(plot_path)

    pdf.output(output_path)
    print(f"PDF saved to {output_path}")


def run_evaluation_loop(models, test_df, numerical_columns, categorical_columns,
                        y_headers, output_scaler, log_transform_columns, time_array,
                        output_pdf_path):
    """Evaluate all models and generate performance report."""
    evaluation_results = {}
    best_performing_models = {}
    plot_data = []
    performance_data = {}

    for (basin, formation, config_str), model in models.items():
        combo_test = filter_by_basin_and_formation(test_df, basin, formation)
        if len(combo_test) == 0:
            continue

        numerical_data = combo_test[numerical_columns].values
        categorical_data = [combo_test[col].astype(int).values.reshape(-1, 1)
                            for col in categorical_columns]
        y_true = combo_test[y_headers].values

        y_pred = predict_with_model(model, numerical_data, categorical_data,
                                    numerical_columns, categorical_columns)
        y_pred_denorm = denormalize_data_output(y_pred, output_scaler, log_transform_columns)
        y_true_denorm = denormalize_data_output(y_true, output_scaler, log_transform_columns)

        if np.any(np.isnan(y_pred_denorm.values)) or np.any(np.isinf(y_pred_denorm.values)):
            print(f"NaN/Inf in predictions for {basin}-{formation} ({config_str})")
            continue

        pred_prods, act_prods = generate_production_rates_for_comp(
            y_pred_denorm, y_true_denorm, y_headers, time_array)

        errors = calculate_errors(pred_prods, act_prods)
        scalar_errors = calculate_scalar_errors(y_pred_denorm, y_true_denorm, y_headers)

        if not errors['MSE']:
            continue

        (best_true, best_pred, worst_true, worst_pred,
         best_idx, worst_idx) = identify_best_worst_matches(
            errors, y_true_denorm, y_pred_denorm)

        best_act_prod = act_prods[best_idx]
        best_pred_prod = pred_prods[best_idx]
        worst_act_prod = act_prods[worst_idx]
        worst_pred_prod = pred_prods[worst_idx]

        if y_true_denorm.shape == y_pred_denorm.shape:
            mse = mean_squared_error(y_true_denorm, y_pred_denorm)
            mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
            # Guard against division by zero in sMAPE
            denom = (np.abs(y_true_denorm.values) + np.abs(y_pred_denorm.values)) / 2
            denom = np.where(denom == 0, 1.0, denom)
            smape = np.mean(np.abs(y_pred_denorm.values - y_true_denorm.values) / denom) * 100
            performance_data[(basin, formation, config_str)] = {
                'MSE': mse, 'MAE': mae, 'sMAPE': smape, **scalar_errors}
            evaluation_results[(basin, formation, config_str)] = {
                'y_pred': y_pred_denorm, 'y_true': y_true_denorm,
                'errors': errors, 'scalar_errors': scalar_errors,
                'MSE': mse, 'MAE': mae, 'sMAPE': smape,
            }
            print(f"{basin}-{formation} ({config_str}): MSE={mse:.4f}, MAE={mae:.4f}, sMAPE={smape:.2f}%")

            if ((basin, formation) not in best_performing_models or
                    mse < best_performing_models[(basin, formation)]['MSE']):
                best_performing_models[(basin, formation)] = {
                    'MSE': mse, 'MAE': mae, 'sMAPE': smape, **scalar_errors,
                    'config_str': config_str, 'model': model}

        plot_data.append((basin, formation, config_str, time_array,
                          best_act_prod, best_pred_prod,
                          worst_act_prod, worst_pred_prod,
                          best_idx, worst_idx))

    # Print best models
    for (basin, formation), info in best_performing_models.items():
        mt = determine_model_type(info['config_str'])
        print(f"Best for {basin}-{formation}: {mt} MSE={info['MSE']:.4f}")

    save_plots_to_pdf(plot_data, performance_data, output_pdf_path)

    return evaluation_results, best_performing_models, plot_data, performance_data
