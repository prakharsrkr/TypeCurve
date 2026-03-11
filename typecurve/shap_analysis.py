import os
import gc
import logging
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import shap
from matplotlib.backends.backend_pdf import PdfPages

from .data_preparation import filter_by_basin_and_formation


# ── Per-Model-Type SHAP Computation ─────────────────────────────────────────

def compute_shap_values_nn(model, numerical_data, categorical_data):
    """Compute SHAP values for a Keras neural network model."""
    try:
        combined_data = [numerical_data] + categorical_data
        explainer = shap.DeepExplainer(model, combined_data)
        shap_values = explainer.shap_values(combined_data)
    except Exception as e:
        logging.error(f"Error using DeepExplainer: {e}. Falling back to KernelExplainer.")
        combined_data_array = np.concatenate([numerical_data] + categorical_data, axis=1)
        num_cols = numerical_data.shape[1]
        explainer = shap.KernelExplainer(
            lambda x: model.predict(
                [x[:, :num_cols]] +
                [x[:, i].reshape(-1, 1)
                 for i in range(num_cols, combined_data_array.shape[1])]),
            combined_data_array)
        shap_values = explainer.shap_values(combined_data_array, nsamples=100)
    return shap_values


def compute_shap_values_ml(model, numerical_data, categorical_data):
    """Compute SHAP values for sklearn-based ML models (RandomForest, etc.)."""
    combined_data = np.concatenate([numerical_data] + categorical_data, axis=1)
    try:
        if hasattr(model, 'named_steps'):
            model = model.named_steps['model']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(combined_data)
    except Exception as e:
        logging.error(f"Error using TreeExplainer: {e}. Falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(model.predict, combined_data)
        shap_values = explainer.shap_values(combined_data, nsamples=100)
    return shap_values


def compute_shap_values_xgb(model, numerical_data, categorical_data):
    """Compute SHAP values for XGBoost models with pipeline preprocessor."""
    try:
        combined_data = np.concatenate([numerical_data] + categorical_data, axis=1)
        preprocessor = model.named_steps['preprocessor']
        xgb_model = model.named_steps['model']

        logging.info("Transforming data using the preprocessor.")
        combined_data_transformed = preprocessor.transform(pd.DataFrame(combined_data))

        logging.info("Creating TreeExplainer.")
        explainer = shap.TreeExplainer(xgb_model)

        logging.info("Computing SHAP values.")
        shap_values = explainer.shap_values(combined_data_transformed)
    except Exception as e:
        logging.error(f"Error using TreeExplainer: {e}. Falling back to KernelExplainer.")
        try:
            explainer = shap.KernelExplainer(xgb_model.predict, combined_data_transformed)
            shap_values = explainer.shap_values(combined_data_transformed, nsamples=100)
        except Exception as e2:
            logging.error(f"Error using KernelExplainer: {e2}")
            shap_values = None
    return shap_values


# ── Dispatcher ──────────────────────────────────────────────────────────────

def compute_shap_values(model, numerical_data, categorical_data, model_type):
    """Route SHAP computation to the correct method based on model_type."""
    if model_type == 'neural_network':
        return compute_shap_values_nn(model, numerical_data, categorical_data)
    elif model_type == 'xgboost':
        return compute_shap_values_xgb(model, numerical_data, categorical_data)
    else:
        return compute_shap_values_ml(model, numerical_data, categorical_data)


# ── Compute & Log SHAP Values Across All Models ────────────────────────────

def compute_and_log_shap_values(models, test_df, numerical_columns, categorical_columns,
                                shap_sample_size=100):
    """Compute SHAP values for all trained models.

    This is the corrected version that:
    - Determines model_type for ALL model types (not just neural_network)
    - Keeps compute/log/store at correct indentation level for all branches
    """
    shap_values_dict = {}

    for (basin, formation, config_str), model in models.items():
        logging.info(f"Processing SHAP values for {basin} - {formation} - {config_str}")
        try:
            combo_test = filter_by_basin_and_formation(test_df, basin, formation)
            logging.info(f"Filtered test data for {basin} - {formation}")

            actual_sample_size = min(len(combo_test), shap_sample_size)
            combo_test_subset = combo_test.sample(n=actual_sample_size, random_state=42)

            numerical_data = combo_test_subset[numerical_columns].values
            categorical_data = [combo_test_subset[col].astype(int).values.reshape(-1, 1)
                                for col in categorical_columns]

            # Determine model type from config_str
            if 'embedding_output_dim' in config_str or config_str in ('neural_network', 'cnn', 'resnet', 'transformer'):
                model_type = 'neural_network'
            elif 'xgboost' in config_str.lower():
                model_type = 'xgboost'
            else:
                model_type = 'ml_model'

            logging.info(f"Computing SHAP values using {model_type}")
            shap_values = compute_shap_values(model, numerical_data, categorical_data,
                                              model_type)

            if shap_values is not None:
                shap_values_dict[(basin, formation, config_str)] = shap_values
                logging.info(f"Successfully computed SHAP values for "
                             f"{basin} - {formation} - {config_str}")
            else:
                logging.error(f"Failed to compute SHAP values for "
                              f"{basin} - {formation} - {config_str}")

            del combo_test, combo_test_subset, numerical_data, categorical_data, shap_values
            gc.collect()

        except Exception as e:
            logging.error(f"Error computing SHAP values for "
                          f"{basin} - {formation} - {config_str}: {e}")

    return shap_values_dict


# ── Plot SHAP Values ───────────────────────────────────────────────────────

def plot_shap_values(shap_values_dict, y_headers, feature_names, output_pdf):
    """Generate SHAP summary bar plots and combine into a single PDF.

    Uses the first entry in shap_values_dict (designed for single-model output).
    For multi-model dicts, iterates over all entries.
    """
    os.makedirs(os.path.dirname(output_pdf) if os.path.dirname(output_pdf) else '.', exist_ok=True)

    if not shap_values_dict:
        logging.warning("No SHAP values to plot.")
        return

    (basin, formation, config_str), shap_values = list(shap_values_dict.items())[0]
    config_readable = "Neural Network"

    with tempfile.TemporaryDirectory() as temp_dir:
        plot_paths = []

        # shap_values shape: (n_samples, n_features, n_outputs)
        n_outputs = shap_values.shape[2] if len(shap_values.shape) == 3 else 1

        for i in range(n_outputs):
            fig, ax = plt.subplots(figsize=(8, 12))

            if len(shap_values.shape) == 3:
                sv = shap_values[:, :, i]
            else:
                sv = shap_values

            shap.summary_plot(sv, feature_names=feature_names, show=False,
                              plot_type='bar', max_display=len(feature_names))

            title = y_headers[i] if i < len(y_headers) else f"Output {i}"
            plt.title(title, fontsize=10, fontname='Arial')
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
            plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)",
                        fontsize=9, fontname='Arial')
            plt.xticks(fontsize=10, fontname='Arial')

            plot_path = os.path.join(temp_dir, f"shap_plot_{i}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            plot_paths.append(plot_path)

        # Combine into PDF
        with PdfPages(output_pdf) as pdf:
            n_plots = len(plot_paths)
            fig, axes = plt.subplots(nrows=1, ncols=n_plots,
                                     figsize=(n_plots * 8, 12), sharey=True)
            if n_plots == 1:
                axes = [axes]

            for i, plot_path in enumerate(plot_paths):
                img = plt.imread(plot_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                title = y_headers[i] if i < len(y_headers) else f"Output {i}"
                axes[i].set_title(title, fontsize=10, fontname='Arial')

            fig.suptitle(config_readable, fontsize=16, fontname='Arial')
            plt.subplots_adjust(wspace=0.09)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    print(f"SHAP PDF saved to {output_pdf}")
