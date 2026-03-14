import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .config import TEST_SIZE, VAL_SPLIT, RANDOM_STATE


def split_data(df, test_size=None, val_split=None, random_state=None,
               group_columns=None):
    """Split data into train, validation, and test sets.

    When *group_columns* is provided (e.g. ``['BasinTC', 'FORMATION_CONDENSED']``),
    uses ``GroupShuffleSplit`` so that every formation has proportional
    representation in each split.  This prevents the random split from
    concentrating a small formation entirely in one partition, which was a
    major source of instability.
    """
    if test_size is None:
        test_size = TEST_SIZE
    if val_split is None:
        val_split = VAL_SPLIT
    if random_state is None:
        random_state = RANDOM_STATE

    if group_columns is not None:
        # Build a group key per row
        groups = df[group_columns].astype(str).agg('_'.join, axis=1)

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx]
        remaining = df.iloc[test_idx]

        groups_rem = groups.iloc[test_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_split,
                                 random_state=random_state)
        val_idx, test_idx2 = next(gss2.split(remaining, groups=groups_rem))
        val_df = remaining.iloc[val_idx]
        test_df = remaining.iloc[test_idx2]
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state)
        val_df, test_df = train_test_split(
            test_df, test_size=val_split, random_state=random_state)

    return train_df, val_df, test_df


def fit_and_apply_scalers(train_df, val_df, test_df, numerical_columns, y_headers,
                          log_transform_columns=None, use_robust=False):
    """Fit scalers on **training data only** and transform all splits.

    Previous version fitted on ``concat([train, val, test])`` — this leaked
    test-set statistics into the scaler, inflating apparent accuracy and
    making feature scaling dependent on the random split.

    Set *use_robust=True* to use ``RobustScaler`` (median/IQR-based),
    which is less sensitive to outliers than ``MinMaxScaler``.

    Returns (train_df, val_df, test_df, input_scaler, output_scaler).
    """
    if log_transform_columns is None:
        log_transform_columns = []

    ScalerCls = RobustScaler if use_robust else MinMaxScaler
    input_scaler = ScalerCls()
    output_scaler = ScalerCls()

    # Apply log transformation (on each split independently)
    if log_transform_columns:
        for split_df in (train_df, val_df, test_df):
            split_df[log_transform_columns] = split_df[log_transform_columns].clip(lower=0)
            split_df[log_transform_columns] = np.log1p(split_df[log_transform_columns])

    # Fit on training data ONLY — no leakage
    input_scaler.fit(train_df[numerical_columns])
    output_scaler.fit(train_df[y_headers])

    # Transform all splits using training-fitted scaler
    for split_df in (train_df, val_df, test_df):
        split_df = split_df.copy()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[numerical_columns] = input_scaler.transform(train_df[numerical_columns])
    train_df[y_headers] = output_scaler.transform(train_df[y_headers])

    val_df[numerical_columns] = input_scaler.transform(val_df[numerical_columns])
    val_df[y_headers] = output_scaler.transform(val_df[y_headers])

    test_df[numerical_columns] = input_scaler.transform(test_df[numerical_columns])
    test_df[y_headers] = output_scaler.transform(test_df[y_headers])

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
