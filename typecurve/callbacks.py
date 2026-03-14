import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .decline_curve import modified_hyperbolic


def _get_keras_callback_base():
    """Lazily import and return the Keras Callback base class."""
    from tensorflow.keras.callbacks import Callback
    return Callback


def RealTimePlottingCallback(combo_description):
    """Factory that returns a RealTimePlottingCallback instance.

    TensorFlow is imported lazily so the module can be loaded even when
    TensorFlow is unavailable (e.g. broken DLL on Windows).
    """
    Callback = _get_keras_callback_base()

    class _RealTimePlottingCallback(Callback):
        def __init__(self):
            super().__init__()
            self.combo_description = combo_description
            self.epochs = []
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.epochs.append(len(self.epochs) + 1)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

            plt.figure(figsize=(10, 4))
            plt.plot(self.epochs, self.losses, label='Training Loss', color='blue')
            plt.plot(self.epochs, self.val_losses, label='Validation Loss', color='red')
            plt.title(f'Training and Validation Loss for {self.combo_description}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(range(1, max(self.epochs) + 1, max(1, len(self.epochs) // 10)))
            plt.legend()
            plt.grid(True)
            plt.pause(0.001)
            plt.close()

        def on_train_end(self, logs=None):
            plt.figure(figsize=(10, 4))
            plt.plot(self.epochs, self.losses, label='Training Loss', color='blue')
            plt.plot(self.epochs, self.val_losses, label='Validation Loss', color='red')
            plt.title(f'Final Training and Validation Loss for {self.combo_description}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(range(1, max(self.epochs) + 1, max(1, len(self.epochs) // 10)))
            plt.legend()
            plt.grid(True)
            plt.show()

    return _RealTimePlottingCallback()


def PositivePredictionCallback(combo_train, numerical_columns, categorical_columns,
                               y_headers, output_scaler, patience=10):
    """Factory that returns a PositivePredictionCallback instance.

    TensorFlow is imported lazily so the module can be loaded even when
    TensorFlow is unavailable.
    """
    Callback = _get_keras_callback_base()

    class _PositivePredictionCallback(Callback):
        def __init__(self):
            super().__init__()
            self.combo_train = combo_train
            self.numerical_columns = numerical_columns
            self.categorical_columns = categorical_columns
            self.y_headers = y_headers
            self.output_scaler = output_scaler
            self._patience = patience
            self.wait = 0
            self.best_weights = None
            self.best_loss = np.inf

        def on_epoch_end(self, epoch, logs=None):
            predictions = self.model.predict(
                x=[self.combo_train[self.numerical_columns].values] +
                  [self.combo_train[col].astype(int).values.reshape(-1, 1)
                   for col in self.categorical_columns]
            )
            predictions_denorm = self.output_scaler.inverse_transform(predictions)

            if np.any(predictions_denorm < 0):
                self.wait += 1
                if self.wait >= self._patience:
                    print(f"Epoch {epoch+1}: Negative predictions detected. Restoring best weights.")
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    self.wait = 0
            else:
                self.wait = 0
                if logs is not None and logs['val_loss'] < self.best_loss:
                    self.best_loss = logs['val_loss']
                    self.best_weights = self.model.get_weights()
                    print(f"Epoch {epoch+1}: New best weights saved.")

    return _PositivePredictionCallback()


def custom_xgboost_training(model, preprocessor, combo_train, combo_val,
                            numerical_columns, categorical_columns, y_headers):
    """Custom XGBoost training with production validation."""
    combo_train_val = pd.concat([combo_train, combo_val])

    preprocessor.fit(combo_train_val[numerical_columns + categorical_columns])
    combo_train_transformed = preprocessor.transform(combo_train[numerical_columns + categorical_columns])
    combo_val_transformed = preprocessor.transform(combo_val[numerical_columns + categorical_columns])

    model.fit(combo_train_transformed, combo_train[y_headers].values,
              eval_set=[(combo_val_transformed, combo_val[y_headers].values)],
              verbose=True)

    for round_num in range(10, model.get_booster().best_iteration + 1, 10):
        predictions = model.predict(combo_val_transformed, iteration_range=(0, round_num))
        valid = validate_productions_xgb(predictions, combo_val, y_headers)
        if not valid:
            print(f"Invalid predictions detected at round {round_num}. Stopping training.")
            break

    return model


def validate_productions_xgb(predictions, combo_val, y_headers):
    """Validate XGBoost predictions using modified hyperbolic model."""
    for idx in range(predictions.shape[0]):
        qi = predictions[idx][y_headers.index('Oil_Params_P50_InitialProd')]
        di = predictions[idx][y_headers.index('Oil_Params_P50_DiCoefficient')]
        b = predictions[idx][y_headers.index('Oil_Params_P50_BCoefficient')]
        IBU = 0
        MBU = 0
        Dlim = 7

        if any(x <= 0 for x in [qi, di, b]):
            return False

        production = modified_hyperbolic(np.arange(1, 121), qi, di, b, Dlim, IBU, MBU)[0]
        if np.isnan(production).any() or (production < 0).any():
            return False

    return True


def mape_loss(y_true, y_pred):
    """Mean Absolute Percentage Error loss."""
    from tensorflow.keras import backend as K
    epsilon = K.epsilon()
    y_true = K.clip(y_true, epsilon, None)
    return K.mean(K.abs((y_true - y_pred) / y_true)) * 100
