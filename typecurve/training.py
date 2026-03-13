import os
import tempfile
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm

from .models import build_model
from .callbacks import PositivePredictionCallback, custom_xgboost_training
from .config import TOTAL_EPOCHS, BATCH_SIZE


def train_and_evaluate_model(combo_train, combo_val, combo_test,
                             numerical_columns, categorical_columns, y_headers,
                             output_size, model_type, df, task_times, output_scaler):
    """Train a single model for a basin/formation combination."""
    start_time = time.time()
    print(f"Starting {model_type} training")

    if model_type in ['neural_network', 'cnn', 'transformer', 'resnet']:
        model = build_model(numerical_columns, categorical_columns, df,
                            output_size, model_type=model_type)

        model.compile(
            optimizer=Adam(learning_rate=0.001, clipvalue=1.0),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )

        positive_pred_callback = PositivePredictionCallback(
            combo_train, numerical_columns, categorical_columns,
            y_headers, output_scaler)

        # Use a unique temp file for ModelCheckpoint to avoid PermissionError
        # when multiple training runs target the same hardcoded path.
        checkpoint_fd, checkpoint_path = tempfile.mkstemp(suffix='.keras')
        os.close(checkpoint_fd)

        try:
            history = model.fit(
                x=([combo_train[numerical_columns].values] +
                   [combo_train[col].astype(int).values.reshape(-1, 1)
                    for col in categorical_columns]),
                y=combo_train[y_headers].values,
                validation_data=(
                    [combo_val[numerical_columns].values] +
                    [combo_val[col].astype(int).values.reshape(-1, 1)
                     for col in categorical_columns],
                    combo_val[y_headers].values
                ),
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3,
                                  restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                      min_lr=1e-6, verbose=1),
                    ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                    save_best_only=True, mode='min', verbose=1),
                ],
                verbose=0
            )
        finally:
            # Clean up the temp checkpoint file
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title(f'Training and Validation Loss vs Epochs for {model_type}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        trained_model = model
    else:
        # Sklearn-based models
        combo_train_val = pd.concat([combo_train, combo_val])
        model = build_model(numerical_columns, categorical_columns, df,
                            output_size, model_type=model_type)

        if model_type == 'xgboost':
            preprocessor = model.named_steps['preprocessor']
            xgb_model = model.named_steps['model']
            custom_xgboost_training(
                xgb_model, preprocessor, combo_train, combo_val,
                numerical_columns, categorical_columns, y_headers)
            # Return the full pipeline (preprocessor + trained model) so
            # evaluation and SHAP can use model.named_steps as expected.
            trained_model = model
        else:
            model.fit(combo_train_val[numerical_columns + categorical_columns],
                      combo_train_val[y_headers].values)
            trained_model = model

    duration = time.time() - start_time
    task_times[model_type] = duration
    print(f"Training completed for {model_type} in {duration:.2f} seconds")
    return trained_model


def execute_training(specific_combinations, train_df, val_df, test_df,
                     numerical_columns, categorical_columns, y_headers,
                     output_size, df, task_times, ml_configurations, output_scaler):
    """Train models for all basin/formation combinations and ML configs."""
    all_models = {}
    for _, combo_data in tqdm(specific_combinations.iterrows(),
                              total=specific_combinations.shape[0]):
        basin = combo_data['BasinTC']
        formation = combo_data['FORMATION_CONDENSED']

        combo_train = train_df[(train_df['BasinTC'] == basin) &
                               (train_df['FORMATION_CONDENSED'] == formation)]
        combo_val = val_df[(val_df['BasinTC'] == basin) &
                           (val_df['FORMATION_CONDENSED'] == formation)]
        combo_test = test_df[(test_df['BasinTC'] == basin) &
                             (test_df['FORMATION_CONDENSED'] == formation)]

        for ml_config in ml_configurations:
            model = train_and_evaluate_model(
                combo_train, combo_val, combo_test,
                numerical_columns, categorical_columns, y_headers,
                output_size, ml_config['model_type'], df, task_times, output_scaler)
            all_models[(basin, formation, ml_config['model_type'])] = model

    return all_models
