#!/usr/bin/env python
"""
TypeCurve - Oil & Gas Production Forecasting System
Main entry point that orchestrates the full pipeline.
"""

import warnings
import pickle
import os

import numpy as np

from typecurve import config
from typecurve.data_loading import load_data, create_derived_columns
from typecurve.preprocessing import run_preprocessing_pipeline
from typecurve.feature_engineering import (
    scale_parameters, add_neighbor_eur_cumulative, assign_basin_tc,
    fill_zero_parameters, check_and_clean_parameters, check_for_zeros,
    drop_columns, calculate_combined_eur_and_cumulative,
    identify_column_types, encode_categorical_columns,
)
from typecurve.decline_curve import remove_spurious_curves
from typecurve.data_preparation import (
    split_data, fit_and_apply_scalers, filter_by_basin_and_formation,
)
from typecurve.training import execute_training
from typecurve.evaluation import run_evaluation_loop
from typecurve.testing import (
    load_test_data, prepare_testing_data, generate_baseline_predictions,
    run_type_curve_scaling, generate_scaled_production_rates, plot_type_curves,
)

# Suppress noisy warnings
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*deprecated.*')
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*not used.*')
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*No visible GPU is found.*')


def main():
    # ── 1. Load Data ────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(config.DEFAULT_FILE_PATH)
    df = create_derived_columns(df)

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    print("Preprocessing...")
    df = run_preprocessing_pipeline(df, config.PARAM_COLUMNS, config.COLUMNS_TO_CHECK)

    # ── 3. Feature Engineering ──────────────────────────────────────────────
    print("Feature engineering...")
    df = scale_parameters(df, 'HORIZONTIAL_WELL_LENGTH', config.APPLY_SCALING)
    df = add_neighbor_eur_cumulative(df)

    df['BasinTC'] = df.apply(
        lambda row: assign_basin_tc(row, config.BASIN_BOUNDS), axis=1)
    df = df[df['BasinTC'] != 'Unknown']

    df = fill_zero_parameters(df)
    zero_columns = check_for_zeros(df)
    if zero_columns:
        print(f"Columns with zeros after replacement: {zero_columns}")
    else:
        print("No columns contain zeros after replacement.")

    mismatches, tru_dist_below_100, df = check_and_clean_parameters(df)
    df = drop_columns(df, config.COLUMNS_TO_DROP, config.PATTERN_DROP_COLUMNS)
    df_orig = df.copy()
    df = calculate_combined_eur_and_cumulative(df)

    # ── 4. Identify Column Types & Encode ───────────────────────────────────
    categorical_columns, y_headers, numerical_columns, feature_columns = identify_column_types(df)
    print(f"Target columns: {y_headers}")
    print(f"Numerical columns: {numerical_columns}")
    print(f"Categorical columns: {categorical_columns}")

    df, encoders = encode_categorical_columns(df, categorical_columns)

    # ── 5. Remove Spurious Curves ───────────────────────────────────────────
    print("Removing spurious curves...")
    time_array = config.make_time_array(years=20)
    headers = df.columns.tolist()
    df = remove_spurious_curves(df, headers, time_array, resource_type='Oil')
    df = remove_spurious_curves(df, headers, time_array, resource_type='Gas')
    df = remove_spurious_curves(df, headers, time_array, resource_type='Water')
    print(f"DataFrame shape after spurious removal: {df.shape}")

    # ── 6. Train/Val/Test Split & Scaling ───────────────────────────────────
    print("Splitting and scaling data...")
    train_df, val_df, test_df = split_data(df)
    log_transform_columns = []

    train_df, val_df, test_df, input_scaler, output_scaler = fit_and_apply_scalers(
        train_df, val_df, test_df, numerical_columns, y_headers, log_transform_columns)

    # ── 7. Train Models ─────────────────────────────────────────────────────
    print("Training models...")
    specific_combinations = df[
        (df['BasinTC'] == config.TRAINING_BASIN) &
        (df['FORMATION_CONDENSED'].isin(config.TRAINING_FORMATIONS))
    ].drop_duplicates(subset=['BasinTC', 'FORMATION_CONDENSED'])

    output_size = len(y_headers)
    task_times = {}

    models = execute_training(
        specific_combinations, train_df, val_df, test_df,
        numerical_columns, categorical_columns, y_headers,
        output_size, df, task_times, config.ML_CONFIGURATIONS, output_scaler)

    print("\nTask times:")
    for task, duration in task_times.items():
        print(f"  {task}: {duration:.2f} seconds")

    # ── 8. Evaluate Models ──────────────────────────────────────────────────
    print("\nEvaluating models...")
    time_array = config.make_time_array(years=20)

    evaluation_results, best_performing_models, plot_data, performance_data = (
        run_evaluation_loop(
            models, test_df, numerical_columns, categorical_columns,
            y_headers, output_scaler, log_transform_columns, time_array,
            config.OUTPUT_PDF_PATH))

    # ── 9. Save Outputs ─────────────────────────────────────────────────────
    print("\nSaving outputs...")
    os.makedirs(os.path.dirname(config.PICKLE_OUTPUT_PATH), exist_ok=True)
    outputs = {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'models': models,
        'input_scaler': input_scaler,
        'evaluation_results': evaluation_results,
        'best_performing_models': best_performing_models,
        'output_scaler': output_scaler,
        'task_times': task_times,
        'log_transform_columns': log_transform_columns,
        'y_headers': y_headers,
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
    }
    with open(config.PICKLE_OUTPUT_PATH, 'wb') as f:
        pickle.dump(outputs, f)
    print(f"Outputs saved to {config.PICKLE_OUTPUT_PATH}")

    # ── 10. Type Curve Scaling (optional) ───────────────────────────────────
    if os.path.exists(config.TEST_FILE_PATH):
        print("\nRunning type curve scaling...")
        dtype_dict = {col: str for col in config.UWI_COLUMNS}
        Testing = load_test_data(config.TEST_FILE_PATH, dtype_dict)
        Testing, baseline_columns = prepare_testing_data(
            Testing, numerical_columns, categorical_columns, y_headers, input_scaler)
        Testing = generate_baseline_predictions(
            Testing, models, numerical_columns, categorical_columns,
            y_headers, baseline_columns, output_scaler, log_transform_columns)

        # ProppantPerFoot variation
        results = run_type_curve_scaling(
            models, Testing, config.VARIATIONS_PROPPANT, 'ProppantPerFoot',
            numerical_columns, categorical_columns, y_headers, baseline_columns,
            input_scaler, output_scaler, encoders, log_transform_columns)

        time_arr = config.make_time_array(years=20)
        results = generate_scaled_production_rates(results, y_headers, time_arr)

        for (basin, formation, config_str), variations in results.items():
            formation_df = filter_by_basin_and_formation(Testing, basin, formation)
            plot_type_curves(basin, formation, config_str, formation_df, variations,
                             numerical_columns, input_scaler, y_headers, time_arr,
                             vary_column='ProppantPerFoot')

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
