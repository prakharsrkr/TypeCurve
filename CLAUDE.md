# CLAUDE.md - TypeCurve Repository Guide

## Project Overview

TypeCurve is an oil & gas production forecasting system that uses machine learning to predict **Estimated Ultimate Recovery (EUR)** for wells across multiple geological basins and formations. The project builds type curves — standardized production profiles — using neural networks and ensemble ML models trained on well completion parameters, production data, and spatial/neighbor-well features.

**Domain**: Petroleum engineering / reservoir analysis
**Primary targets**: EUR_30yr_Actual_Oil_P50_MBO, EUR_30yr_Actual_Gas_P50_MMCF, EUR_30yr_Actual_Water_P50_MBBL

## Repository Structure

```
TypeCurve/
├── Main.ipynb                          # Entry point: imports, data loading, preprocessing pipeline
├── Refined_TypeCurve_Update-*.ipynb    # Cleaned-up model notebook
├── FirstTest_TypeCurve_Update-*.ipynb  # Experimental notebooks (many variants/copies)
├── Untitled.ipynb                      # Scratch/exploratory notebook
├── Prakhar_Testnew2.xlsx               # Primary dataset (~15 MB, ~15,500 wells)
├── Prakhar_Testnew3.xlsx               # Secondary dataset
├── Prakhar_Testnew4.xlsx               # Tertiary dataset
└── Untitled spreadsheet.xlsx           # Small auxiliary data
```

**Note**: There are no standalone `.py` module files. `Main.ipynb` references `Preprocessing.py` and `Map_Plotting.py` via imports, but these modules are **not present** in the repo — their functions are defined inline within the notebooks themselves. This is a known gap.

### Notebook Naming Conventions

- `*-Copy1`, `*-Copy2`, etc. — Iterative experiment snapshots
- `*-checkpoint.ipynb` — Jupyter auto-save checkpoints (should ideally be in `.gitignore`)
- `*-NoCategorical` — Variants that exclude categorical feature encoding
- `*-withPCA` — Variants using PCA dimensionality reduction
- `*-PhaseSeparate` — Variants with separate oil/gas phase modeling
- `*-withEURGrouping` — Variants that group wells by EUR buckets
- `*-WithCorrections` — Bug-fixed versions of earlier experiments
- `*-withHL` — Variants with hidden layer modifications
- `*-withMLUpdates` — Variants with updated ML pipeline

## Key Concepts

### Well Data Features

- **UWI** (Unique Well Identifier): Primary key for each well (14-digit API number)
- **Typecurve / BasinTC**: Basin-formation grouping (e.g., "CoyoteValley_North", "Mercury_600")
- **EUR parameters**: 30-year estimated ultimate recovery at percentiles P20/P35/P50/P65/P80 for oil (MBO), gas (MMCF), water (MBBL)
- **Decline curve params**: BuildupRate, MonthsInProd, InitialProd, DiCoefficient, BCoefficient, LimDeclineRate
- **Completion features**: FluidPerFoot_bblft, ProppantPerFoot, LateralLength, etc.
- **Spatial features**: HEELPOINT_LAT, HEELPOINT_LON, TOEPOINT_LAT, TOEPOINT_LON
- **Neighbor well features**: NNAZ_1-6 (nearest neighbors in all zones — across all formation benches in multibench targets), NNSZ_1-2 (nearest neighbors in the same zone) with their EUR/cumulative production values

### Data Pipeline (Main.ipynb)

1. `load_and_preprocess_data()` — Load Excel, initial cleaning
2. `split_parameters()` — Parse decline curve parameter strings into individual columns
3. `handle_missing_values()` — Imputation strategies
4. `drop_na_parameter_columns()` — Remove rows with missing key params
5. `group_and_summarize()` — Aggregate by type curve group
6. `drop_zero_rows()` — Remove wells with zero values in critical columns
7. `convert_to_absolute_values()` — Ensure positive numeric values
8. `replace_missing_water_params()` — Handle missing water production parameters
9. `replace_zeros_with_P50()` — Fill zero EUR values with P50 estimates
10. `add_neighbor_eur_cumulative()` — Compute neighbor-well aggregate features
11. `drop_specified_columns()` — Remove identifier and redundant columns before modeling

### ML Models

**Neural Network** (primary model):
- Keras Functional API with mixed input architecture
- Numerical features: Conv1D layers (32 → 64 → 128 filters, kernel_size=3) → GlobalAveragePooling1D
- Categorical features: Embedding layers → Flatten
- Merged path: Dense layers [256, 128, 64, 32] with Dropout (0.2–0.3) and L1/L2 regularization
- Optimizer: Adam (lr=0.001), Loss: MSE
- Training: epochs=1000, batch_size=50, EarlyStopping + ReduceLROnPlateau callbacks

**Ensemble/Traditional models** (for comparison):
- XGBoostRegressor (with hyperparameter tuning via GridSearchCV/RandomizedSearchCV)
- RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
- ElasticNet, SVR, LinearRegression, DecisionTreeRegressor

**Model evaluation**: R2 score, RMSE, MAE; SHAP values for interpretability

### Training Strategy

- Models are trained **per Basin-Formation combination** (not a single global model)
- Standard train/test split via `train_test_split`
- Feature scaling: StandardScaler or RobustScaler for numerical features
- Categorical encoding: LabelEncoder or OneHotEncoder for formation/basin columns
- Cross-validation: `cross_val_score` used for model selection

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, geopandas, scipy, openpyxl |
| ML | scikit-learn, tensorflow/keras, xgboost, shap |
| Visualization | matplotlib, seaborn, folium |
| Interactive | ipywidgets, IPython.display |
| Utilities | joblib, tqdm, PIL |
| Web (partial) | Flask |

**Python version**: 3.x (no version pinned)
**No requirements.txt or environment.yml exists** — dependencies must be inferred from imports.

## Development Workflow

### Running the Project

1. Ensure all dependencies are installed (see Technology Stack)
2. Place well data Excel files in the repository root
3. Open `Main.ipynb` to run the preprocessing pipeline
4. Open the desired model notebook (e.g., `Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb`) to train/evaluate models

### Data File Paths

Notebooks reference Windows paths like `C:\Users\Prakhar.Sarkar\OneDrive - ...`. These must be updated to local paths when running on a different machine. The Excel files in the repo root (`Prakhar_Testnew2.xlsx`, etc.) serve as the data sources.

### No Tests or CI/CD

There are no unit tests, integration tests, or CI/CD pipelines configured.

## Conventions for AI Assistants

### Do

- Treat this as a **data science exploration project** — notebooks are the primary artifact
- When modifying ML pipelines, preserve the per-basin training strategy
- Keep decline curve parameter names consistent (P20/P35/P50/P65/P80 for Oil/Gas/Water)
- Use the existing preprocessing function signatures when adding new data cleaning steps
- When adding new features, add them before the `drop_specified_columns` step
- Use scikit-learn-compatible APIs for new models (fit/predict/score interface)

### Don't

- Don't assume `.py` module files exist — functions referenced in `Main.ipynb` imports are not available as standalone modules
- Don't modify the Excel data files directly
- Don't hardcode Windows file paths — use `os.path` or `pathlib` for cross-platform compatibility
- Don't add large binary files (models, datasets) without discussing storage approach
- Don't remove checkpoint notebooks without confirming — they may contain unique experiment results

### Code Style

- Functions use snake_case naming
- DataFrames commonly named `df`, `grouped_df`, `cumulative_df`
- Column names use mixed conventions: CamelCase for well identifiers (e.g., `FluidPerFoot_bblft`), underscore-separated for EUR columns
- Inline comments are used liberally; docstrings are sparse
- Visualization outputs are embedded in notebook cells
