import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats


def assign_basin_tc(row, basin_bounds):
    """Assign basin type curve based on geographic coordinates."""
    if pd.isna(row['BasinTC']) or row['BasinTC'] == 'Unknown' or row['BasinTC'] == 0:
        for basin, bounds in basin_bounds.items():
            if ((bounds['lat_range'][0] <= row['HEELPOINT_LAT'] <= bounds['lat_range'][1] and
                 bounds['lon_range'][0] <= row['HEELPOINT_LON'] <= bounds['lon_range'][1]) or
                (bounds['lat_range'][0] <= row['MIDPOINT_LAT'] <= bounds['lat_range'][1] and
                 bounds['lon_range'][0] <= row['MIDPOINT_LON'] <= bounds['lon_range'][1]) or
                (bounds['lat_range'][0] <= row['TOEPOINT_LAT'] <= bounds['lat_range'][1] and
                 bounds['lon_range'][0] <= row['TOEPOINT_LON'] <= bounds['lon_range'][1])):
                return basin
        return row['BasinTC']
    return row['BasinTC']


def scale_parameters(df, horizontal_well_length_column, apply_scaling=True):
    """Scale production parameters by horizontal well length."""
    scaling_columns = [col for col in df.columns if
                       'InitialProd' in col or
                       'Cumulative oil mbo' in col or
                       'Cumulative gas mmcf' in col or
                       'Cumulative water mbbl' in col or
                       'EUR_30yr_Actual_Oil' in col or
                       'EUR_30yr_Actual_Gas' in col or
                       'EUR_30yr_Actual_Water' in col]

    if apply_scaling:
        for col in scaling_columns:
            df[col] = df.apply(lambda row: row[col] * (
                (1 - 0.03 * (10000 - row[horizontal_well_length_column]) / 2500) *
                10000 / row[horizontal_well_length_column]), axis=1)
        df = df.drop(columns=[horizontal_well_length_column])

    return df


def add_neighbor_eur_cumulative(df):
    """Add neighbor well EUR and cumulative production columns."""
    if 'UWI' not in df.columns:
        raise ValueError("DataFrame missing 'UWI' column.")
    if 'CompletionDate' not in df.columns:
        raise ValueError("DataFrame missing 'CompletionDate' column.")

    eur_oil_columns = [f'EUR_30yr_Actual_Oil_{p}_MBO' for p in ['P20', 'P35', 'P50', 'P65', 'P80']]
    eur_gas_columns = [f'EUR_30yr_Actual_Gas_{p}_MMCF' for p in ['P20', 'P35', 'P50', 'P65', 'P80']]
    cumulative_columns = ['Cumulative oil mbo', 'Cumulative gas mmcf', 'Cumulative water mbbl']

    missing = [col for col in eur_oil_columns + eur_gas_columns + cumulative_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    mapping_cols = eur_oil_columns + eur_gas_columns + cumulative_columns + ['CompletionDate']
    eur_cumulative_map = df.set_index('UWI')[mapping_cols].fillna(0).copy()

    if not eur_cumulative_map.index.is_unique:
        eur_cumulative_map = eur_cumulative_map.groupby(eur_cumulative_map.index).first()

    new_columns = {}
    for prefix in ['NNAZ', 'NNSZ']:
        num_cols = 6 if prefix == 'NNAZ' else 2
        for i in range(1, num_cols + 1):
            uwi_col = f'{prefix}_{i}_UWI'
            if uwi_col in df.columns:
                for eur_col in eur_oil_columns + eur_gas_columns + cumulative_columns:
                    new_col_name = f'{prefix}_{i}_{eur_col}'
                    new_columns[new_col_name] = df[uwi_col].map(eur_cumulative_map[eur_col]).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df


def fill_zero_parameters(df):
    """Replace zero HZDIST values with 5280 and recalculate TRUEDIST."""
    for prefix in ['NNAZ', 'NNSZ']:
        num_cols = 6 if prefix == 'NNAZ' else 2
        for i in range(1, num_cols + 1):
            hzd_col = f'{prefix}_{i}_HZDIST'
            true_col = f'{prefix}_{i}_TRUEDIST'
            vt_col = f'{prefix}_{i}_VTDIST'

            if hzd_col in df.columns:
                df[hzd_col] = df[hzd_col].replace(0, 5280)
            if true_col in df.columns and vt_col in df.columns:
                df[true_col] = np.sqrt(df[hzd_col] ** 2 + df[vt_col] ** 2)
    return df


def check_and_clean_parameters(df):
    """Check TRUEDIST mismatches and remove rows with TRUEDIST < 100."""
    mismatches = {}
    tru_dist_below_100 = {}
    tru_dist_below_100_indices = set()

    for prefix in ['NNAZ', 'NNSZ']:
        num_cols = 6 if prefix == 'NNAZ' else 2
        for i in range(1, num_cols + 1):
            hzd_col = f'{prefix}_{i}_HZDIST'
            true_col = f'{prefix}_{i}_TRUEDIST'
            vt_col = f'{prefix}_{i}_VTDIST'

            if hzd_col in df.columns and true_col in df.columns and vt_col in df.columns:
                calculated = np.sqrt(df[hzd_col] ** 2 + df[vt_col] ** 2)
                mismatched = df[(df[true_col] != 0) & (df[true_col] != calculated)].index
                mismatches[true_col] = mismatched.tolist()

                tru_dist_below_100_indices.update(df[df[true_col] < 100].index)
                tru_dist_below_100[true_col] = df[df[true_col] < 100].index.tolist()

    df_cleaned = df.drop(list(tru_dist_below_100_indices))
    return mismatches, tru_dist_below_100, df_cleaned


def check_for_zeros(df):
    """Check for remaining zeros in HZDIST/TRUEDIST columns."""
    zero_columns = []
    for prefix in ['NNAZ', 'NNSZ']:
        num_cols = 6 if prefix == 'NNAZ' else 2
        for i in range(1, num_cols + 1):
            hzd_col = f'{prefix}_{i}_HZDIST'
            true_col = f'{prefix}_{i}_TRUEDIST'
            if hzd_col in df.columns and (df[hzd_col] == 0).any():
                zero_columns.append(hzd_col)
            if true_col in df.columns and (df[true_col] == 0).any():
                zero_columns.append(true_col)
    return zero_columns


def drop_columns(df, columns_to_drop, pattern_drops=None):
    """Drop specified columns and columns matching name patterns."""
    existing = [c for c in columns_to_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)

    # Drop _Method columns
    method_cols = [col for col in df.columns if '_Method' in col]
    df.drop(columns=method_cols, axis=1, inplace=True)

    # Drop pattern-based columns
    if pattern_drops:
        for pattern in pattern_drops:
            matching = [col for col in df.columns if pattern in col]
            df.drop(columns=matching, axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def calculate_combined_eur_and_cumulative(df):
    """Calculate combined oil+gas EUR and cumulative (gas/20 conversion)."""
    oil_columns_eur = [col for col in df.columns if col.endswith('_EUR_30yr_Actual_Oil_P50_MBO')]
    for oil_col in oil_columns_eur:
        base_name = oil_col.replace('_EUR_30yr_Actual_Oil_P50_MBO', '')
        gas_col = base_name + '_EUR_30yr_Actual_Gas_P50_MMCF'
        if gas_col in df.columns:
            combined_col = base_name + '_EUR_Combined_P50_MBO'
            df[combined_col] = df[oil_col] + df[gas_col] / 20
            df.drop([oil_col, gas_col], axis=1, inplace=True)

    oil_columns_cum = [col for col in df.columns if col.endswith('_Cumulative oil mbo')]
    for oil_col in oil_columns_cum:
        base_name = oil_col.replace('_Cumulative oil mbo', '')
        gas_col = base_name + '_Cumulative gas mmcf'
        if gas_col in df.columns:
            combined_col = base_name + '_Cumulative combined mbo'
            df[combined_col] = df[oil_col] + df[gas_col] / 20
            df.drop([oil_col, gas_col], axis=1, inplace=True)

    return df


def identify_column_types(df):
    """Identify categorical, target, numerical, and feature columns."""
    df['BasinTC'] = df['BasinTC'].astype(str)
    df['FORMATION_CONDENSED'] = df['FORMATION_CONDENSED'].astype(str)

    categorical_columns = [
        col for col in df.select_dtypes(include=['object', 'category']).columns.tolist()
        if col not in ['BasinTC', 'FORMATION_CONDENSED']
    ]
    y_headers = [col for col in df.columns if 'Params' in col]
    numerical_columns = [
        col for col in df.columns
        if col not in categorical_columns + ['BasinTC', 'FORMATION_CONDENSED'] + y_headers
    ]
    feature_columns = numerical_columns + categorical_columns

    return categorical_columns, y_headers, numerical_columns, feature_columns


def encode_categorical_columns(df, categorical_columns):
    """Label-encode categorical columns. Returns (df, encoders_dict)."""
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def remove_outliers(df, numerical_columns, threshold=3):
    """Remove rows with z-score > threshold in any numerical column."""
    if df is None or len(df) == 0:
        return df, 0

    valid_cols = [col for col in numerical_columns if col in df.columns]
    if not valid_cols:
        return df, 0

    z_scores = np.abs(stats.zscore(df[valid_cols]))
    filter_mask = (z_scores < threshold).all(axis=1)
    filtered_df = df[filter_mask]
    rows_removed = len(df) - len(filtered_df)
    return filtered_df, rows_removed
