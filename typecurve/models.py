import numpy as np
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, Embedding, Flatten,
    Conv1D, MaxPooling1D, Reshape, GlobalAveragePooling1D,
    Add, Activation, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError


def _build_neural_network(numerical_columns, categorical_columns, df, output_size):
    dense_layer_sizes = [64, 32]
    dropout_rate = 0.2
    embedding_output_dim = 10

    numerical_input = Input(shape=(len(numerical_columns),), name='num_input')

    if categorical_columns:
        categorical_inputs = [Input(shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [Embedding(input_dim=df[col].nunique() + 1,
                                output_dim=embedding_output_dim,
                                name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [Flatten()(emb) for emb in embeddings]
        merged = Concatenate()([numerical_input] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = numerical_input

    x = merged
    for size in dense_layer_sizes:
        x = Dense(size, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(output_size, activation='linear')(x)
    model = Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[MeanSquaredError()])
    return model


def _build_cnn(numerical_columns, categorical_columns, df, output_size):
    embedding_output_dim = 10
    conv_filters = 64
    kernel_size = 3
    dropout_rate = 0.1

    numerical_input = Input(shape=(len(numerical_columns),), name='num_input')
    reshaped = Reshape((len(numerical_columns), 1))(numerical_input)

    if categorical_columns:
        categorical_inputs = [Input(shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [Embedding(input_dim=df[col].nunique() + 1,
                                output_dim=embedding_output_dim,
                                name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [Flatten()(emb) for emb in embeddings]
        merged = Concatenate()([reshaped] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = reshaped

    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(merged)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_size, activation='linear')(x)
    model = Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[MeanSquaredError()])
    return model


def _build_resnet(numerical_columns, categorical_columns, df, output_size):
    embedding_output_dim = 10
    dense_layer_sizes = [64, 64, 32]
    dropout_rate = 0.3

    numerical_input = Input(shape=(len(numerical_columns),), name='num_input')

    if categorical_columns:
        categorical_inputs = [Input(shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [Embedding(input_dim=df[col].nunique() + 1,
                                output_dim=embedding_output_dim,
                                name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [Flatten()(emb) for emb in embeddings]
        merged = Concatenate()([numerical_input] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = numerical_input

    x = Dense(dense_layer_sizes[0], activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    shortcut = Dense(dense_layer_sizes[1])(x)
    x = Dense(dense_layer_sizes[1], activation='relu')(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_size, activation='linear')(x)
    model = Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[MeanSquaredError()])
    return model


def _build_transformer(numerical_columns, categorical_columns, df, output_size):
    embedding_output_dim = 10
    dropout_rate = 0.3

    numerical_input = Input(shape=(len(numerical_columns),), name='num_input')

    if categorical_columns:
        categorical_inputs = [Input(shape=(1,), name=f'cat_input_{i}')
                              for i, _ in enumerate(categorical_columns)]
        embeddings = [Embedding(input_dim=df[col].nunique() + 1,
                                output_dim=embedding_output_dim,
                                name=f'emb_{col}')(cat_input)
                      for cat_input, col in zip(categorical_inputs, categorical_columns)]
        flat_embeddings = [Flatten()(emb) for emb in embeddings]
        merged = Concatenate()([numerical_input] + flat_embeddings)
    else:
        categorical_inputs = []
        merged = numerical_input

    x = Dense(32, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(output_size, activation='linear')(x)
    model = Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[MeanSquaredError()])
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
            learning_rate=0.01, subsample=0.9, colsample_bytree=0.9)
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
