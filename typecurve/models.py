import numpy as np
import tensorflow as tf
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# Resolve 'relu' string to actual function to prevent TensorFlow Grappler
# optimizer errors ("Node model_N/activation/Relu has an empty op name").
_RELU = tf.keras.activations.relu


def _import_keras():
    """Lazily import TensorFlow/Keras components.

    Raises a clear error message when TensorFlow is not installed or its
    native libraries fail to load (e.g. DLL errors on Windows).
    """
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, Concatenate, Embedding, Flatten,
            Conv1D, MaxPooling1D, Reshape, GlobalAveragePooling1D,
            Add, Activation, BatchNormalization
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.metrics import MeanSquaredError
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for neural-network model types "
            "(neural_network, cnn, resnet, transformer) but could not be "
            f"imported.  Original error:\n{exc}"
        ) from exc
    return {
        'Model': Model,
        'Input': Input, 'Dense': Dense, 'Dropout': Dropout,
        'Concatenate': Concatenate, 'Embedding': Embedding, 'Flatten': Flatten,
        'Conv1D': Conv1D, 'MaxPooling1D': MaxPooling1D, 'Reshape': Reshape,
        'GlobalAveragePooling1D': GlobalAveragePooling1D,
        'Add': Add, 'Activation': Activation, 'BatchNormalization': BatchNormalization,
        'Adam': Adam, 'MeanSquaredError': MeanSquaredError,
    }


def _build_neural_network(numerical_columns, categorical_columns, df, output_size):
    K = _import_keras()
    dense_layer_sizes = [64, 32]
    dropout_rate = 0.2
    embedding_output_dim = 10

    numerical_input = K['Input'](shape=(len(numerical_columns),), name='num_input')

    if categorical_columns:
        categorical_inputs = [K['Input'](shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [K['Embedding'](input_dim=df[col].nunique() + 1,
                                     output_dim=embedding_output_dim,
                                     name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [K['Flatten']()(emb) for emb in embeddings]
        merged = K['Concatenate']()([numerical_input] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = numerical_input

    x = merged
    for size in dense_layer_sizes:
        x = K['Dense'](size, activation=_RELU)(x)
        x = K['Dropout'](dropout_rate)(x)

    output = K['Dense'](output_size, activation='linear')(x)
    model = K['Model'](inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=K['Adam'](learning_rate=0.0001), loss='mse', metrics=[K['MeanSquaredError']()])
    return model


def _build_cnn(numerical_columns, categorical_columns, df, output_size):
    K = _import_keras()
    embedding_output_dim = 10
    conv_filters = 64
    kernel_size = 3
    dropout_rate = 0.1

    numerical_input = K['Input'](shape=(len(numerical_columns),), name='num_input')
    reshaped = K['Reshape']((len(numerical_columns), 1))(numerical_input)

    # Run Conv1D on numerical features first, then pool to 2D
    x = K['Conv1D'](filters=conv_filters, kernel_size=kernel_size, activation=_RELU, padding='same')(reshaped)
    x = K['MaxPooling1D'](pool_size=2)(x)
    x = K['Dropout'](dropout_rate)(x)
    x = K['GlobalAveragePooling1D']()(x)

    if categorical_columns:
        categorical_inputs = [K['Input'](shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [K['Embedding'](input_dim=df[col].nunique() + 1,
                                     output_dim=embedding_output_dim,
                                     name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [K['Flatten']()(emb) for emb in embeddings]
        # Both x and flat_embeddings are now 2D — safe to concatenate
        x = K['Concatenate']()([x] + flat_embeddings)
    else:
        categorical_inputs = []

    x = K['Dense'](32, activation=_RELU)(x)
    x = K['Dropout'](dropout_rate)(x)

    output = K['Dense'](output_size, activation='linear')(x)
    model = K['Model'](inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=K['Adam'](learning_rate=0.0001), loss='mse', metrics=[K['MeanSquaredError']()])
    return model


def _build_resnet(numerical_columns, categorical_columns, df, output_size):
    K = _import_keras()
    embedding_output_dim = 10
    dense_layer_sizes = [64, 64, 32]
    dropout_rate = 0.3

    numerical_input = K['Input'](shape=(len(numerical_columns),), name='num_input')

    if categorical_columns:
        categorical_inputs = [K['Input'](shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [K['Embedding'](input_dim=df[col].nunique() + 1,
                                     output_dim=embedding_output_dim,
                                     name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [K['Flatten']()(emb) for emb in embeddings]
        merged = K['Concatenate']()([numerical_input] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = numerical_input

    x = K['Dense'](dense_layer_sizes[0], activation=_RELU)(merged)
    x = K['BatchNormalization']()(x)
    x = K['Dropout'](dropout_rate)(x)

    shortcut = K['Dense'](dense_layer_sizes[1])(x)
    x = K['Dense'](dense_layer_sizes[1], activation=_RELU)(x)
    x = K['Add']()([x, shortcut])
    x = K['Activation'](_RELU)(x)
    x = K['BatchNormalization']()(x)
    x = K['Dropout'](dropout_rate)(x)

    output = K['Dense'](output_size, activation='linear')(x)
    model = K['Model'](inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=K['Adam'](learning_rate=0.0001), loss='mse', metrics=[K['MeanSquaredError']()])
    return model


def _build_transformer(numerical_columns, categorical_columns, df, output_size):
    K = _import_keras()
    embedding_output_dim = 10
    dropout_rate = 0.3

    numerical_input = K['Input'](shape=(len(numerical_columns),), name='num_input')

    if categorical_columns:
        categorical_inputs = [K['Input'](shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [K['Embedding'](input_dim=df[col].nunique() + 1,
                                     output_dim=embedding_output_dim,
                                     name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [K['Flatten']()(emb) for emb in embeddings]
        merged = K['Concatenate']()([numerical_input] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = numerical_input

    x = K['Dense'](32, activation=_RELU)(merged)
    x = K['BatchNormalization']()(x)
    x = K['Dropout'](dropout_rate)(x)

    output = K['Dense'](output_size, activation='linear')(x)
    model = K['Model'](inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=K['Adam'](learning_rate=0.0001), loss='mse', metrics=[K['MeanSquaredError']()])
    return model


def _build_sklearn_pipeline(model_type, numerical_columns, categorical_columns):
    """Build a scikit-learn pipeline with polynomial features and a regressor."""
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_jobs=-1, n_estimators=300, max_depth=15,
            min_samples_split=5, min_samples_leaf=1, bootstrap=True)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(max_depth=20, min_samples_split=5, min_samples_leaf=2)
    elif model_type == 'xgboost':
        model = XGBRegressor(
            tree_method='hist', n_estimators=300, max_depth=10,
            learning_rate=0.01, subsample=0.9, colsample_bytree=0.9,
            early_stopping_rounds=10)
    elif model_type in ('ridge', 'lasso', 'multioutput'):
        model = MultiOutputRegressor(Lasso(alpha=0.1, max_iter=10000))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    numerical_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, include_bias=False))])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def build_model(numerical_columns, categorical_columns, df, output_size,
                model_type='neural_network'):
    """Build and compile a model of the specified type."""
    builders = {
        'neural_network': _build_neural_network,
        'cnn': _build_cnn,
        'resnet': _build_resnet,
        'transformer': _build_transformer,
    }

    if model_type in builders:
        return builders[model_type](numerical_columns, categorical_columns, df, output_size)
    elif model_type in ('random_forest', 'decision_tree', 'xgboost', 'ridge', 'lasso', 'multioutput'):
        return _build_sklearn_pipeline(model_type, numerical_columns, categorical_columns)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
