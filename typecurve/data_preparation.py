import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import TEST_SIZE, VAL_SPLIT, RANDOM_STATE


def split_data(df, test_size=None, val_split=None, random_state=None):
    """Split data into train, validation, and test sets."""
    if test_size is None:
        test_size = TEST_SIZE
    if val_split is None:
        val_split = VAL_SPLIT
    if random_state is None:
        random_state = RANDOM_STATE

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_df, test_df = train_test_split(test_df, test_size=val_split, random_state=random_state)
    return train_df, val_df, test_df


def fit_and_apply_scalers(train_df, val_df, test_df, numerical_columns, y_headers,
                          log_transform_columns=None):
    """Fit MinMaxScalers on combined data and transform all splits.

    Returns (train_df, val_df, test_df, input_scaler, output_scaler).
    """
    if log_transform_columns is None:
        log_transform_columns = []

    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    combined_df = pd.concat([train_df, val_df, test_df], axis=0)

    # Apply log transformation
    if log_transform_columns:
        combined_df[log_transform_columns] = combined_df[log_transform_columns].apply(
            lambda x: x.clip(lower=0))
        combined_df[log_transform_columns] = np.log1p(combined_df[log_transform_columns])

        train_df[log_transform_columns] = np.log1p(train_df[log_transform_columns])
        val_df[log_transform_columns] = np.log1p(val_df[log_transform_columns])
        test_df[log_transform_columns] = np.log1p(test_df[log_transform_columns])

    input_scaler.fit(combined_df[numerical_columns])
    output_scaler.fit(combined_df[y_headers])

    combined_df[numerical_columns] = input_scaler.transform(combined_df[numerical_columns])
    combined_df[y_headers] = output_scaler.transform(combined_df[y_headers])

    # Split back
    n_train = len(train_df)
    n_val = len(val_df)
    train_df = combined_df.iloc[:n_train]
    val_df = combined_df.iloc[n_train:n_train + n_val]
    test_df = combined_df.iloc[n_train + n_val:]

    return train_df, val_df, test_df, input_scaler, output_scaler


def prepare_data(df, numerical_columns, categorical_columns):
    """Prepare data by converting categorical columns to category codes."""
    df = df.copy()
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    return df


def filter_by_basin_and_formation(df, basin, formation):
    """Filter DataFrame by basin and formation."""
    return df[(df['BasinTC'] == basin) & (df['FORMATION_CONDENSED'] == formation)]


def denormalize_data_input(data, scaler):
    """Inverse-transform numerical input data."""
    return scaler.inverse_transform(data.copy())


def denormalize_data_output(data, scaler, log_transform_columns=None):
    """Inverse-transform output data and revert log transformation."""
    if log_transform_columns is None:
        log_transform_columns = []

    denormalized_data = scaler.inverse_transform(data.copy())
    denormalized_df = pd.DataFrame(denormalized_data, columns=scaler.get_feature_names_out())

    if log_transform_columns:
        denormalized_df[log_transform_columns] = np.expm1(denormalized_df[log_transform_columns])

    return denormalized_df


def decode_categorical(data, encoders, column_names):
    """Decode label-encoded categorical columns back to original values."""
    decoded_data = {}
    for i, col in enumerate(column_names):
        le = encoders[col]
        decoded_data[col] = le.inverse_transform(data[:, i].astype(int))
    return decoded_data


def denormalize_and_decode(df, numerical_columns, categorical_columns, y_headers,
                           input_scaler, output_scaler, encoders, log_transform_columns=None):
    """Full denormalization: numerical inputs, targets, and categorical decoding."""
    if log_transform_columns is None:
        log_transform_columns = []

    df_copy = df.copy()
    df_copy[numerical_columns] = denormalize_data_input(df_copy[numerical_columns], input_scaler)
    df_copy[y_headers] = denormalize_data_output(df_copy[y_headers], output_scaler,
                                                  log_transform_columns)

    decoded = decode_categorical(df_copy[categorical_columns].values, encoders, categorical_columns)
    for col in categorical_columns:
        df_copy[col] = decoded[col]

    return df_copy


def denormalize_and_decode_inputs(input_df, numerical_columns, categorical_columns,
                                  input_scaler, encoders):
    """Denormalize numerical features and decode categorical features."""
    df_copy = input_df.copy()
    df_copy[numerical_columns] = denormalize_data_input(df_copy[numerical_columns], input_scaler)

    decoded = decode_categorical(df_copy[categorical_columns].values, encoders, categorical_columns)
    for col in categorical_columns:
        df_copy[col] = decoded[col]

    return df_copy
