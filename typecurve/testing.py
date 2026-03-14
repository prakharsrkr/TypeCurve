import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .decline_curve import generate_production_rates_testing, modified_hyperbolic
from .preprocessing import split_parameters, convert_to_absolute_values
from .feature_engineering import calculate_combined_eur_and_cumulative
from .data_preparation import (
    denormalize_data_output, denormalize_and_decode_inputs, filter_by_basin_and_formation
)
from .evaluation import predict_with_model
from .config import OUTPUT_DIR_TYPE_CURVES


def load_test_data(file_path, dtype_dict):
    """Load and prepare the type curve test/scaling dataset."""
    Testing = pd.read_excel(file_path, dtype=dtype_dict)

    # Remove 'AVG_' prefix from headers
    Testing = Testing.rename(columns={col: col.replace('AVG_', '') for col in Testing.columns})
    Testing['FluidPerFoot_bblft'] = Testing['FluidPerFoot'] / 42
    Testing.drop(columns=['FluidPerFoot'], inplace=True)

    # Parse DCA parameters
    param_columns = ['Oil_DCA_Parameters', 'Gas_DCA_Parameters', 'Water_DCA_Parameters']
    Testing, new_columns = split_parameters(Testing, param_columns)

    # Rename DCA columns to match model expectations
    mapping = {}
    for phase in ['Oil', 'Gas', 'Water']:
        for param in ['Method', 'BuildupRate', 'MonthsInProd', 'InitialProd',
                       'DiCoefficient', 'BCoefficient', 'LimDeclineRate']:
            mapping[f'{phase}_DCA_Parameters_{param}'] = f'{phase}_Params_P50_{param}'
    Testing = Testing.rename(columns=mapping)

    # Drop unnecessary columns
    Testing = Testing.drop(columns=[col for col in Testing.columns if col.endswith('_Method')])
    cols_to_drop = ['Unique_TC_ID', 'Status', 'Polygon_Name',
                    'Oil_Params_P50_LimDeclineRate',
                    'Gas_Params_P50_LimDeclineRate',
                    'Water_Params_P50_LimDeclineRate']
    Testing.drop(columns=[c for c in cols_to_drop if c in Testing.columns], inplace=True)

    Testing['BasinTC'] = Testing['BasinTC'].astype(str)
    Testing['FORMATION_CONDENSED'] = Testing['FORMATION_CONDENSED'].astype(str)
    Testing = convert_to_absolute_values(Testing)
    Testing = calculate_combined_eur_and_cumulative(Testing)

    return Testing


def prepare_testing_data(Testing, numerical_columns, categorical_columns, y_headers,
                         input_scaler):
    """Prepare testing data: reorder columns, add baseline placeholders, scale."""
    Testing = Testing[['BasinTC', 'FORMATION_CONDENSED'] +
                      numerical_columns + categorical_columns + y_headers]

    baseline_columns = [f'{param}_baseline' for param in y_headers]
    for col in baseline_columns:
        Testing[col] = np.nan

    Testing[numerical_columns] = input_scaler.transform(Testing[numerical_columns])
    return Testing, baseline_columns


def generate_baseline_predictions(Testing, models, numerical_columns, categorical_columns,
                                  y_headers, baseline_columns, output_scaler,
                                  log_transform_columns):
    """Generate baseline decline parameters for each basin/formation."""
    for (basin, formation, config_str), model in models.items():
        combo_test = filter_by_basin_and_formation(Testing, basin, formation)
        if len(combo_test) == 0:
            continue

        numerical_data = combo_test[numerical_columns].values
        categorical_data = [combo_test[col].astype(int).values.reshape(-1, 1)
                            for col in categorical_columns]

        y_pred = predict_with_model(model, numerical_data, categorical_data,
                                    numerical_columns, categorical_columns)
        y_pred_denorm = denormalize_data_output(y_pred, output_scaler, log_transform_columns)
        y_pred_denorm.index = combo_test.index

        for param, baseline_col in zip(y_headers, baseline_columns):
            Testing.loc[combo_test.index, baseline_col] = y_pred_denorm[param]

    Testing.dropna(subset=baseline_columns, inplace=True)
    return Testing


def clip_values(df, lower=-1e5, upper=1e5):
    """Clip DataFrame values to prevent overflow."""
    return df.clip(lower=lower, upper=upper)


def run_type_curve_scaling(models, Testing, variations, vary_column,
                           numerical_columns, categorical_columns, y_headers,
                           baseline_columns, input_scaler, output_scaler,
                           encoders, log_transform_columns):
    """Run type curve scaling for a single variation parameter."""
    results = {}

    for (basin, formation, config_str), model in models.items():
        results[(basin, formation, config_str)] = {}
        formation_df = Testing[(Testing['BasinTC'] == basin) &
                               (Testing['FORMATION_CONDENSED'] == formation)]

        for vary_value in variations[vary_column]:
            varied_dfs = []

            for index, row in formation_df.iterrows():
                varied_df = pd.DataFrame([row])
                varied_df = denormalize_and_decode_inputs(
                    varied_df, numerical_columns, categorical_columns,
                    input_scaler, encoders)

                baseline_val = row[vary_column]
                varied_df[vary_column] = vary_value

                varied_df[numerical_columns] = input_scaler.transform(
                    varied_df[numerical_columns])

                # Re-encode categorical columns back to integers after decode
                for col in categorical_columns:
                    le = encoders[col]
                    # Use transform for known labels; unknown labels get code 0
                    try:
                        varied_df[col] = le.transform(varied_df[col].astype(str))
                    except ValueError:
                        varied_df[col] = 0

                numerical_data = varied_df[numerical_columns].values
                categorical_data = [varied_df[col].astype(int).values.reshape(-1, 1)
                                    for col in categorical_columns]

                new_preds = predict_with_model(model, numerical_data, categorical_data,
                                               numerical_columns, categorical_columns)
                new_preds_denorm = clip_values(
                    denormalize_data_output(new_preds, output_scaler, log_transform_columns))

                # Calculate scaling factors
                scaling_factors = {}
                for param, baseline_param in zip(y_headers, baseline_columns):
                    baseline_values = row[baseline_param]
                    new_pred_values = new_preds_denorm[param].values

                    if vary_value == baseline_val:
                        scaling_factors[param] = 1
                    else:
                        scaling_factors[param] = new_pred_values / baseline_values

                # Apply scaling
                scaled_df = varied_df.copy()
                for param in y_headers:
                    if '_P50_MonthsInProd' not in param and '_P50_LimDeclineRate' not in param:
                        scaled_df[param] = varied_df[param] * scaling_factors[param]
                        scaled_df[param] = np.clip(scaled_df[param], -1e5, 1e5)

                varied_dfs.append(scaled_df[y_headers].copy())

            if vary_value not in results[(basin, formation, config_str)]:
                results[(basin, formation, config_str)][vary_value] = []
            results[(basin, formation, config_str)][vary_value].append(varied_dfs)

    return results


def generate_scaled_production_rates(results, y_headers, time):
    """Generate production rates for all scaled type curves."""
    for key, variations in results.items():
        for vary_value, varied_dfs_list in variations.items():
            all_productions = []
            for varied_dfs in varied_dfs_list:
                for scaled_y_df in varied_dfs:
                    if not isinstance(scaled_y_df, pd.DataFrame):
                        scaled_y_df = pd.DataFrame(scaled_y_df)
                    productions = generate_production_rates_testing(
                        scaled_y_df, y_headers, time, resource_type='Oil', use_baseline=False)
                    all_productions.append(productions)

            results[key][vary_value] = {
                'scaled_y_dfs': varied_dfs_list,
                'productions': all_productions
            }

    return results


def plot_type_curves(basin, formation, config_str, original_df, results,
                     numerical_columns, input_scaler, y_headers, time,
                     vary_column='ProppantPerFoot'):
    """Plot original and scaled type curves for all variations."""
    fig, ax = plt.subplots(figsize=(14, 8))

    original_denorm = original_df.copy()
    original_denorm[numerical_columns] = input_scaler.inverse_transform(
        original_df[numerical_columns])
    baseline_value = original_denorm[vary_column].iloc[0]

    original_production = generate_production_rates_testing(
        original_df[y_headers], y_headers, time, resource_type='Oil', use_baseline=False)
    ax.plot(time, np.array(original_production).flatten(), 'k--',
            label=f'Original (Baseline={baseline_value:.2f})')

    for vary_value, data in results.items():
        if 'productions' in data:
            for scaled_production in data['productions']:
                flat = np.array(scaled_production).flatten()
                if len(time) != len(flat):
                    continue
                ax.plot(time, flat, label=f'{vary_column} = {vary_value}')

    ax.set_yscale('log')
    ax.set_ylim(100, 1000000)
    ax.set_xlim(0, max(time))
    ax.set_title(f'Type Curves for {basin} - {formation} - {config_str}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Production Rate (bbl/month)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    os.makedirs(OUTPUT_DIR_TYPE_CURVES, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR_TYPE_CURVES,
                             f'{basin}_{formation}_{config_str}_{vary_column}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
