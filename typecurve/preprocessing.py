import ast
import pandas as pd
import numpy as np


def robust_parse(x):
    """Parse a stringified list into individual values."""
    if pd.isna(x):
        return [None] * 7
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        else:
            return [x] + [None] * 6
    except Exception:
        return [None] * 7


def split_parameters(df, param_columns):
    """Split parameter columns (stringified lists) into individual columns."""
    new_columns = []
    for col in param_columns:
        expanded_cols = [
            f'{col}_Method', f'{col}_BuildupRate', f'{col}_MonthsInProd',
            f'{col}_InitialProd', f'{col}_DiCoefficient', f'{col}_BCoefficient',
            f'{col}_LimDeclineRate'
        ]
        temp_df = pd.DataFrame(df[col].apply(robust_parse).tolist(), columns=expanded_cols)
        for num_col in expanded_cols:
            if num_col.endswith('_Method'):
                temp_df[num_col] = temp_df[num_col].astype(str)
            else:
                temp_df[num_col] = pd.to_numeric(temp_df[num_col], errors='coerce')
        df = pd.concat([df, temp_df], axis=1)
        new_columns.extend(expanded_cols)
    df.drop(columns=param_columns, inplace=True)
    return df, new_columns


def handle_missing_values(df):
    """Fill missing values: 'Unknown' for categorical, 0 for numerical."""
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for column in df.columns:
        if column in categorical_columns:
            df[column] = df[column].fillna('Unknown')
        else:
            df[column] = df[column].fillna(0)
    return df


def drop_na_parameter_columns(df, new_columns):
    """Drop rows where any Params column has NaN."""
    df.dropna(subset=[col for col in new_columns if 'Params' in col], inplace=True)
    return df


def group_and_summarize(df):
    """Group by Typecurve and count unique wells."""
    dfnew = df[['UWI10', 'Typecurve']].drop_duplicates()
    grouped_df = dfnew.groupby('Typecurve', as_index=False).count()
    uwi10_sum = grouped_df['UWI10'].sum()
    return grouped_df, uwi10_sum


def drop_zero_rows(df, columns):
    """Drop rows where specified columns have zero values."""
    for column in columns:
        initial_count = len(df)
        df = df[df[column] != 0]
        print(f"Data dropped after removing zero '{column}': {initial_count - len(df)}")
    return df


def convert_to_absolute_values(df):
    """Convert _HZDIST and _VTDIST columns to absolute values."""
    columns_to_convert = [col for col in df.columns if '_HZDIST' in col or '_VTDIST' in col]
    df[columns_to_convert] = df[columns_to_convert].abs()
    return df


def replace_missing_water_params(df):
    """Replace zero water params with scaled oil params."""
    oil_columns = {
        'InitialProd': 'Oil_Params_P50_InitialProd',
        'DiCoefficient': 'Oil_Params_P50_DiCoefficient',
        'BCoefficient': 'Oil_Params_P50_BCoefficient',
        'BuildupRate': 'Oil_Params_P50_BuildupRate',
        'MonthsInProd': 'Oil_Params_P50_MonthsInProd',
        'LimDeclineRate': 'Oil_Params_P50_LimDeclineRate'
    }
    water_columns = {
        'InitialProd': 'Water_Params_P50_InitialProd',
        'DiCoefficient': 'Water_Params_P50_DiCoefficient',
        'BCoefficient': 'Water_Params_P50_BCoefficient',
        'BuildupRate': 'Water_Params_P50_BuildupRate',
        'MonthsInProd': 'Water_Params_P50_MonthsInProd',
        'LimDeclineRate': 'Water_Params_P50_LimDeclineRate',
        'EUR_30yr_Actual_Water_P50_MBBL': 'EUR_30yr_Actual_Water_P50_MBBL'
    }

    for param, oil_col in oil_columns.items():
        water_col = water_columns.get(param)
        if water_col:
            if param == 'InitialProd':
                df[water_col] = df.apply(
                    lambda row: row[oil_col] * 3 if row[water_col] == 0 else row[water_col], axis=1)
            else:
                df[water_col] = df.apply(
                    lambda row: row[oil_col] if row[water_col] == 0 else row[water_col], axis=1)

    df['EUR_30yr_Actual_Water_P50_MBBL'] = df.apply(
        lambda row: row['EUR_30yr_Actual_Oil_P50_MBO'] * 3
        if row['EUR_30yr_Actual_Water_P50_MBBL'] == 0
        else row['EUR_30yr_Actual_Water_P50_MBBL'], axis=1)

    return df


def replace_zeros_with_P50(df):
    """Replace zero EUR and parameter values with P50 equivalents."""
    phases = ['Oil', 'Gas', 'Water']
    years = ['30yr']

    for phase in phases:
        for year in years:
            if phase == 'Water':
                p50_col = f'EUR_{year}_Actual_{phase}_P50_MBBL'
            elif phase == 'Gas':
                p50_col = f'EUR_{year}_Actual_{phase}_P50_MMCF'
            else:
                p50_col = f'EUR_{year}_Actual_{phase}_P50_MBO'

            for p in ['P20', 'P35', 'P65', 'P80']:
                if phase == 'Water':
                    p_col = f'EUR_{year}_Actual_{phase}_{p}_MBBL'
                elif phase == 'Gas':
                    p_col = f'EUR_{year}_Actual_{phase}_{p}_MMCF'
                else:
                    p_col = f'EUR_{year}_Actual_{phase}_{p}_MBO'

                if p_col in df.columns and p50_col in df.columns:
                    df.loc[df[p_col] == 0, p_col] = df[p50_col]

    params = ['Method', 'BuildupRate', 'MonthsInProd', 'InitialProd',
              'DiCoefficient', 'BCoefficient', 'LimDeclineRate']
    for phase in ['Oil', 'Gas', 'Water']:
        for param in params:
            p50_col = f'{phase}_Params_P50_{param}'
            for p in ['P20', 'P35', 'P65', 'P80']:
                p_col = f'{phase}_Params_{p}_{param}'
                if p_col in df.columns and p50_col in df.columns:
                    df.loc[df[p_col] == 0, p_col] = df[p50_col]


def run_preprocessing_pipeline(df, param_columns, columns_to_check):
    """Run the full preprocessing pipeline."""
    df, new_columns = split_parameters(df, param_columns)
    df = handle_missing_values(df)
    df = drop_na_parameter_columns(df, new_columns)
    grouped_df, uwi10_sum = group_and_summarize(df)
    print(grouped_df)
    print(uwi10_sum)

    df = drop_zero_rows(df, columns_to_check)
    df = convert_to_absolute_values(df)
    df = replace_missing_water_params(df)
    replace_zeros_with_P50(df)

    return df
