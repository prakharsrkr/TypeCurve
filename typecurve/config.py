import os
import numpy as np

# ── File Paths ──────────────────────────────────────────────────────────────
# Update these to match your local environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILE_PATH = os.path.join(BASE_DIR, 'Prakhar_Testnew2.xlsx')
TEST_FILE_PATH = os.path.join(BASE_DIR, 'TCTest.xlsx')

# Output directories (override as needed)
OUTPUT_DIR_TRAINING = os.path.join(BASE_DIR, 'output', 'FeaturevsPrediction_Training')
OUTPUT_DIR_TEST = os.path.join(BASE_DIR, 'output', 'FeaturevsPrediction_Test')
OUTPUT_PDF_PATH = os.path.join(BASE_DIR, 'output', 'model_performance_predtuning.pdf')
PICKLE_OUTPUT_PATH = os.path.join(BASE_DIR, 'output', 'outputs_NN_only.pkl')

# ── UWI Columns ─────────────────────────────────────────────────────────────
UWI_COLUMNS = [
    'UWI', 'NNAZ_1_UWI', 'NNAZ_2_UWI', 'NNAZ_3_UWI',
    'NNAZ_4_UWI', 'NNAZ_5_UWI', 'NNAZ_6_UWI',
    'NNSZ_1_UWI', 'NNSZ_2_UWI'
]

# ── Parameter Columns to Parse ──────────────────────────────────────────────
PARAM_COLUMNS = [
    'Oil_Params_P20', 'Gas_Params_P20', 'Oil_Params_P35', 'Gas_Params_P35',
    'Oil_Params_P50', 'Gas_Params_P50', 'Oil_Params_P65', 'Gas_Params_P65',
    'Oil_Params_P80', 'Gas_Params_P80', 'Water_Params_P50'
]

# ── Columns to Check for Zeros ──────────────────────────────────────────────
COLUMNS_TO_CHECK = [
    'FluidPerFoot_bblft', 'ProppantPerFoot', 'EUR_30yr_Actual_Gas_P50_MMCF',
    'EUR_30yr_Actual_Oil_P50_MBO', 'HEELPOINT_LAT'
]

# ── Columns to Drop Before Modeling ─────────────────────────────────────────
COLUMNS_TO_DROP = [
    'UWI10', 'CompletionDate', 'UWI', 'WellName',
    'NNAZ_1_UWI', 'NNAZ_2_UWI', 'NNAZ_3_UWI', 'NNAZ_4_UWI', 'NNAZ_5_UWI',
    'NNAZ_6_UWI', 'NNSZ_1_UWI', 'NNSZ_2_UWI',
    'LeaseName', 'WellNumber', 'CurrentOperatorName', 'OriginalOperatorName',
    'DrillingContractorName', 'PermitDate', 'SpudDate', 'FORMATION_CONDENSE',
    'Unique_PDP_ID',
    'EUR_30yr_Actual_Oil_P20_MBO', 'EUR_30yr_Actual_Gas_P20_MMCF',
    'EUR_30yr_Actual_Oil_P35_MBO', 'EUR_30yr_Actual_Gas_P35_MMCF',
    'EUR_30yr_Actual_Oil_P50_MBO', 'EUR_30yr_Actual_Gas_P50_MMCF',
    'EUR_30yr_Actual_Oil_P65_MBO', 'EUR_30yr_Actual_Gas_P65_MMCF',
    'EUR_30yr_Actual_Oil_P80_MBO', 'EUR_30yr_Actual_Gas_P80_MMCF',
    'EUR_30yr_Actual_Water_P50_MBBL',
    'WELL_TORTUOSITY', 'DEPTH_TO_TOP_2Q', 'DEPTH_TO_TOP_3Q', 'DEPTH_TO_TOP_4Q',
    'AZIMUTH', 'DEPTH_ABOVE_ZONE_2Q', 'DEPTH_ABOVE_ZONE_3Q', 'DEPTH_ABOVE_ZONE_4Q',
    'Cumulative oil mbo', 'Cumulative gas mmcf', 'Cumulative water mbbl',
    'AVERAGE_INCLINATION', 'DEPTH_TO_TOP_1Q', 'HEELPOINT_DEPTH', 'TOEPOINT_DEPTH',
    'NNSZ_1_FORMATION', 'NNSZ_2_FORMATION',
    'NNAZ_1_FORMATION', 'NNAZ_2_FORMATION', 'NNAZ_3_FORMATION',
    'NNAZ_4_FORMATION', 'NNAZ_5_FORMATION', 'NNAZ_6_FORMATION',
    # NNAZ 3-6 EUR and cumulative columns
    'NNAZ_3_EUR_30yr_Actual_Oil_P20_MBO', 'NNAZ_3_EUR_30yr_Actual_Oil_P35_MBO',
    'NNAZ_3_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_3_EUR_30yr_Actual_Oil_P65_MBO',
    'NNAZ_3_EUR_30yr_Actual_Oil_P80_MBO', 'NNAZ_3_EUR_30yr_Actual_Gas_P20_MMCF',
    'NNAZ_3_EUR_30yr_Actual_Gas_P35_MMCF', 'NNAZ_3_EUR_30yr_Actual_Gas_P50_MMCF',
    'NNAZ_3_EUR_30yr_Actual_Gas_P65_MMCF', 'NNAZ_3_EUR_30yr_Actual_Gas_P80_MMCF',
    'NNAZ_3_Cumulative oil mbo', 'NNAZ_3_Cumulative gas mmcf', 'NNAZ_3_Cumulative water mbbl',
    'NNAZ_4_EUR_30yr_Actual_Oil_P20_MBO', 'NNAZ_4_EUR_30yr_Actual_Oil_P35_MBO',
    'NNAZ_4_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_4_EUR_30yr_Actual_Oil_P65_MBO',
    'NNAZ_4_EUR_30yr_Actual_Oil_P80_MBO', 'NNAZ_4_EUR_30yr_Actual_Gas_P20_MMCF',
    'NNAZ_4_EUR_30yr_Actual_Gas_P35_MMCF', 'NNAZ_4_EUR_30yr_Actual_Gas_P50_MMCF',
    'NNAZ_4_EUR_30yr_Actual_Gas_P65_MMCF', 'NNAZ_4_EUR_30yr_Actual_Gas_P80_MMCF',
    'NNAZ_4_Cumulative oil mbo', 'NNAZ_4_Cumulative gas mmcf', 'NNAZ_4_Cumulative water mbbl',
    'NNAZ_5_EUR_30yr_Actual_Oil_P20_MBO', 'NNAZ_5_EUR_30yr_Actual_Oil_P35_MBO',
    'NNAZ_5_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_5_EUR_30yr_Actual_Oil_P65_MBO',
    'NNAZ_5_EUR_30yr_Actual_Oil_P80_MBO', 'NNAZ_5_EUR_30yr_Actual_Gas_P20_MMCF',
    'NNAZ_5_EUR_30yr_Actual_Gas_P35_MMCF', 'NNAZ_5_EUR_30yr_Actual_Gas_P50_MMCF',
    'NNAZ_5_EUR_30yr_Actual_Gas_P65_MMCF', 'NNAZ_5_EUR_30yr_Actual_Gas_P80_MMCF',
    'NNAZ_5_Cumulative oil mbo', 'NNAZ_5_Cumulative gas mmcf', 'NNAZ_5_Cumulative water mbbl',
    'NNAZ_6_EUR_30yr_Actual_Oil_P20_MBO', 'NNAZ_6_EUR_30yr_Actual_Oil_P35_MBO',
    'NNAZ_6_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_6_EUR_30yr_Actual_Oil_P65_MBO',
    'NNAZ_6_EUR_30yr_Actual_Oil_P80_MBO', 'NNAZ_6_EUR_30yr_Actual_Gas_P20_MMCF',
    'NNAZ_6_EUR_30yr_Actual_Gas_P35_MMCF', 'NNAZ_6_EUR_30yr_Actual_Gas_P50_MMCF',
    'NNAZ_6_EUR_30yr_Actual_Gas_P65_MMCF', 'NNAZ_6_EUR_30yr_Actual_Gas_P80_MMCF',
    'NNAZ_6_Cumulative oil mbo', 'NNAZ_6_Cumulative gas mmcf', 'NNAZ_6_Cumulative water mbbl',
    'DEPTH_ABOVE_ZONE_1Q',
    # NNAZ 1-2 non-P50 EUR columns
    'NNAZ_1_EUR_30yr_Actual_Oil_P20_MBO', 'NNAZ_1_EUR_30yr_Actual_Oil_P35_MBO',
    'NNAZ_1_EUR_30yr_Actual_Oil_P65_MBO', 'NNAZ_1_EUR_30yr_Actual_Oil_P80_MBO',
    'NNAZ_1_EUR_30yr_Actual_Gas_P20_MMCF', 'NNAZ_1_EUR_30yr_Actual_Gas_P35_MMCF',
    'NNAZ_1_EUR_30yr_Actual_Gas_P65_MMCF', 'NNAZ_1_EUR_30yr_Actual_Gas_P80_MMCF',
    'NNAZ_2_EUR_30yr_Actual_Oil_P20_MBO', 'NNAZ_2_EUR_30yr_Actual_Oil_P35_MBO',
    'NNAZ_2_EUR_30yr_Actual_Oil_P65_MBO', 'NNAZ_2_EUR_30yr_Actual_Oil_P80_MBO',
    'NNAZ_2_EUR_30yr_Actual_Gas_P20_MMCF', 'NNAZ_2_EUR_30yr_Actual_Gas_P35_MMCF',
    'NNAZ_2_EUR_30yr_Actual_Gas_P65_MMCF', 'NNAZ_2_EUR_30yr_Actual_Gas_P80_MMCF',
    # NNSZ 1-2 non-P50 EUR columns
    'NNSZ_1_EUR_30yr_Actual_Oil_P20_MBO', 'NNSZ_1_EUR_30yr_Actual_Oil_P35_MBO',
    'NNSZ_1_EUR_30yr_Actual_Oil_P65_MBO', 'NNSZ_1_EUR_30yr_Actual_Oil_P80_MBO',
    'NNSZ_1_EUR_30yr_Actual_Gas_P20_MMCF', 'NNSZ_1_EUR_30yr_Actual_Gas_P35_MMCF',
    'NNSZ_1_EUR_30yr_Actual_Gas_P65_MMCF', 'NNSZ_1_EUR_30yr_Actual_Gas_P80_MMCF',
    'NNSZ_2_EUR_30yr_Actual_Oil_P20_MBO', 'NNSZ_2_EUR_30yr_Actual_Oil_P35_MBO',
    'NNSZ_2_EUR_30yr_Actual_Oil_P65_MBO', 'NNSZ_2_EUR_30yr_Actual_Oil_P80_MBO',
    'NNSZ_2_EUR_30yr_Actual_Gas_P20_MMCF', 'NNSZ_2_EUR_30yr_Actual_Gas_P35_MMCF',
    'NNSZ_2_EUR_30yr_Actual_Gas_P65_MMCF', 'NNSZ_2_EUR_30yr_Actual_Gas_P80_MMCF',
    # NNAZ 3-6 distance columns
    'NNAZ_3_TRUEDIST', 'NNAZ_3_HZDIST', 'NNAZ_3_VTDIST',
    'NNAZ_4_TRUEDIST', 'NNAZ_4_HZDIST', 'NNAZ_4_VTDIST',
    'NNAZ_5_TRUEDIST', 'NNAZ_5_HZDIST', 'NNAZ_5_VTDIST',
    'NNAZ_6_TRUEDIST', 'NNAZ_6_HZDIST', 'NNAZ_6_VTDIST',
    # P20/P35/P65/P80 param columns (only P50 kept)
    'Oil_Params_P20_BuildupRate', 'Oil_Params_P20_MonthsInProd',
    'Oil_Params_P20_InitialProd', 'Oil_Params_P20_DiCoefficient',
    'Oil_Params_P20_BCoefficient', 'Oil_Params_P20_LimDeclineRate',
    'Gas_Params_P20_BuildupRate', 'Gas_Params_P20_MonthsInProd',
    'Gas_Params_P20_InitialProd', 'Gas_Params_P20_DiCoefficient',
    'Gas_Params_P20_BCoefficient', 'Gas_Params_P20_LimDeclineRate',
    'Oil_Params_P35_BuildupRate', 'Oil_Params_P35_MonthsInProd',
    'Oil_Params_P35_InitialProd', 'Oil_Params_P35_DiCoefficient',
    'Oil_Params_P35_BCoefficient', 'Oil_Params_P35_LimDeclineRate',
    'Gas_Params_P35_BuildupRate', 'Gas_Params_P35_MonthsInProd',
    'Gas_Params_P35_InitialProd', 'Gas_Params_P35_DiCoefficient',
    'Gas_Params_P35_BCoefficient', 'Gas_Params_P35_LimDeclineRate',
    'Oil_Params_P65_BuildupRate', 'Oil_Params_P65_MonthsInProd',
    'Oil_Params_P65_InitialProd', 'Oil_Params_P65_DiCoefficient',
    'Oil_Params_P65_BCoefficient', 'Oil_Params_P65_LimDeclineRate',
    'Gas_Params_P65_BuildupRate', 'Gas_Params_P65_MonthsInProd',
    'Gas_Params_P65_InitialProd', 'Gas_Params_P65_DiCoefficient',
    'Gas_Params_P65_BCoefficient', 'Gas_Params_P65_LimDeclineRate',
    'Oil_Params_P80_BuildupRate', 'Oil_Params_P80_MonthsInProd',
    'Oil_Params_P80_InitialProd', 'Oil_Params_P80_DiCoefficient',
    'Oil_Params_P80_BCoefficient', 'Oil_Params_P80_LimDeclineRate',
    'Gas_Params_P80_BuildupRate', 'Gas_Params_P80_MonthsInProd',
    'Gas_Params_P80_InitialProd', 'Gas_Params_P80_DiCoefficient',
    'Gas_Params_P80_BCoefficient', 'Gas_Params_P80_LimDeclineRate',
    'PERCENT_IN_ZONE',
    'Gas_Params_P50_LimDeclineRate', 'Oil_Params_P50_LimDeclineRate',
    'Water_Params_P50_LimDeclineRate',
    'MIDPOINT_DEPTH',
    # NNSZ_2 / NNAZ_2 P50 and distance columns
    'NNSZ_2_EUR_30yr_Actual_Oil_P50_MBO', 'NNSZ_2_EUR_30yr_Actual_Gas_P50_MMCF',
    'NNSZ_2_Cumulative oil mbo', 'NNSZ_2_Cumulative gas mmcf', 'NNSZ_2_Cumulative water mbbl',
    'NNAZ_2_TRUEDIST', 'NNAZ_2_HZDIST', 'NNAZ_2_VTDIST',
    'NNAZ_2_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_2_EUR_30yr_Actual_Gas_P50_MMCF',
    'NNAZ_2_Cumulative oil mbo', 'NNAZ_2_Cumulative gas mmcf', 'NNAZ_2_Cumulative water mbbl',
    'NNSZ_2_TRUEDIST', 'NNSZ_2_HZDIST', 'NNSZ_2_VTDIST',
    'AVERAGE_DEPTH_BELOW_TOP',
    'NNAZ_1_TRUEDIST', 'NNSZ_1_TRUEDIST',
    'Oil_Params_P50_BuildupRate', 'Oil_Params_P50_MonthsInProd',
    'Gas_Params_P50_MonthsInProd', 'Gas_Params_P50_BuildupRate',
    'Water_Params_P50_BuildupRate', 'Water_Params_P50_MonthsInProd',
]

# Column name patterns to drop
PATTERN_DROP_COLUMNS = [
    '_LAT', '_LON', 'COMPLETION_RELATIONSHIP', 'WELL_TRAJECTORY',
    'PRIMARY_FORMATION', 'Typecurve'
]

# ── Basin Bounds ─────────────────────────────────────────────────────────────
BASIN_BOUNDS = {
    'Midland': {'lat_range': (29, 34), 'lon_range': (-110, -109)}
}

# ── ML Configurations ───────────────────────────────────────────────────────
ML_CONFIGURATIONS = [
    {'model_type': 'neural_network'},
    {'model_type': 'cnn'},
    {'model_type': 'resnet'},
    {'model_type': 'xgboost'},
]

# ── Training Parameters ─────────────────────────────────────────────────────
APPLY_SCALING = True
TEST_SIZE = 0.3
VAL_SPLIT = 0.5  # of the test portion
RANDOM_STATE = 42
TOTAL_EPOCHS = 1000
BATCH_SIZE = 80
DEFAULT_YEARS = 20

# ── Sensitivity Analysis Variations ─────────────────────────────────────────
VARIATIONS_PROPPANT = {
    'ProppantPerFoot': np.arange(500, 4000, 1000),
}

VARIATIONS_SPACING = {
    'NNSZ_1_HZDIST': np.array([440, 660, 880, 1320]),
}

# ── Specific Basin/Formation Combinations for Training ──────────────────────
TRAINING_BASIN = 'Midland'
TRAINING_FORMATIONS = ['LSS', 'WCA', 'WCB', 'JMS']


def make_time_array(years=DEFAULT_YEARS):
    return np.linspace(1, 12 * years, 12 * years)
