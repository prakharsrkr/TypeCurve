import pandas as pd
from .config import UWI_COLUMNS, DEFAULT_FILE_PATH


def load_data(file_path=None, uwi_columns=None):
    """Load well data from Excel with UWI columns as strings."""
    if file_path is None:
        file_path = DEFAULT_FILE_PATH
    if uwi_columns is None:
        uwi_columns = UWI_COLUMNS

    dtype_dict = {col: str for col in uwi_columns}
    df = pd.read_excel(file_path, dtype=dtype_dict)

    # Format UWI columns as strings
    df['UWI'] = df['UWI'].apply(lambda x: '{:.0f}'.format(float(x)) if pd.notnull(x) else '')
    for col in uwi_columns:
        df[col] = df[col].apply(lambda x: '{:.0f}'.format(float(x)) if pd.notnull(x) else '')

    print(df[['UWI'] + uwi_columns].head())
    return df


def create_derived_columns(df):
    """Create derived columns and apply basic filters."""
    df['Cumulative oil mbo'] = df['CumLiquid'] / 1000
    df['Cumulative gas mmcf'] = df['CumGas'] / 1000
    df['Cumulative water mbbl'] = df['CumWater'] / 1000
    df['FluidPerFoot_bblft'] = df['FluidPerFoot'] / 42
    df['FORMATION_CONDENSED'] = df['FORMATION_CONDENSE']

    df = df[(df['FluidPerFoot_bblft'] <= 300) & (df['ProppantPerFoot'] <= 7000)]
    df.drop(columns=['CumLiquid', 'CumGas', 'CumWater', 'FluidPerFoot'], inplace=True)

    return df
