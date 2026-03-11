# CLAUDE.md - TypeCurve Repository Guide

## Project Overview

TypeCurve is an oil & gas production forecasting system that uses machine learning to predict **Estimated Ultimate Recovery (EUR)** for wells across multiple geological basins and formations. The project builds type curves — standardized production profiles — using neural networks and ensemble ML models trained on well completion parameters, production data, and spatial/neighbor-well features.

**Domain**: Petroleum engineering / reservoir analysis
**Primary prediction targets** (decline curve parameters, dynamically selected):
- All columns containing `Params` in the name (e.g., `Oil_Params_P50_InitialProd`, `Gas_Params_P50_DiCoefficient`, etc.)
- These are derived from splitting original parameter strings into: BuildupRate, MonthsInProd, InitialProd, DiCoefficient, BCoefficient, LimDeclineRate

**EUR columns used during preprocessing** (dropped before modeling):
- `EUR_30yr_Actual_Oil_P50_MBO`, `EUR_30yr_Actual_Gas_P50_MMCF`, `EUR_30yr_Actual_Water_P50_MBBL` (and P20/P35/P65/P80 variants)

## Repository Structure

```
TypeCurve/
├── CLAUDE.md                           # This file
├── Main.ipynb                          # Entry point: imports, data loading, preprocessing (8 cells)
├── Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb
│                                       # Primary model notebook (1491 cells, 307 KB)
├── FirstTest_TypeCurve_Update-*.ipynb  # 18 experimental notebooks (see Notebook Inventory below)
├── Untitled.ipynb                      # Scratch notebook: web scraping (BeautifulSoup), EUR calculator
├── Prakhar_Testnew2.xlsx               # Primary dataset (~15 MB, ~15,500 wells, 114 basin-formation groups)
├── Prakhar_Testnew3.xlsx               # Secondary dataset (~4.5 MB)
├── Prakhar_Testnew4.xlsx               # Tertiary dataset (~1.9 MB)
└── Untitled spreadsheet.xlsx           # Small auxiliary data (~5 KB)
```

**Total repository size**: ~249 MB across all files
**No `.gitignore`** — checkpoint files and large data files are all tracked in git
**No `.py` module files** — `Main.ipynb` imports from `Preprocessing` and `Map_Plotting`, but these modules do **not exist** as files. All functions are defined inline within the notebooks.

### Notebook Inventory

**Core notebooks (3)**:
| Notebook | Cells | Size | Purpose |
|----------|-------|------|---------|
| `Main.ipynb` | 8 | 16 KB | Preprocessing entry point |
| `Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb` | 1491 | 307 KB | Primary cleaned-up model notebook |
| `Untitled.ipynb` | 8 | 258 KB | Scratch: web scraping (BeautifulSoup), `cumulative_production_and_EUR()` calculator |

**Experimental notebooks (18 non-checkpoint files)**:
| Variant suffix | What it changes |
|----------------|-----------------|
| `-NeuralNetwork` (base) | Neural network only, no ensemble models |
| `-NeuralNetworkandOthers` (base) | Neural network + XGBoost, RandomForest, GradientBoosting, etc. |
| `-WithCorrections` | Bug fixes applied to the base NeuralNetworkandOthers |
| `-NoCategorical` | Removes OneHotEncoder/LabelEncoder — numerical features only |
| `-withPCA` | Adds PCA dimensionality reduction before training (51 cells) |
| `-PhaseSeparate` | Separate models for oil vs. gas production |
| `-withEURGrouping` | Groups wells by EUR buckets, trains separate models per group (largest: 57 MB, 136 cells) |
| `-withMLUpdates` | Refined hyperparameters and feature engineering (157 cells) |
| `-withHL` | Modified hidden layer architecture (sizes, activations, dropout) |
| `-Copy1`, `-Copy2`, `-Copy3` | Iterative experiment snapshots with accumulated outputs |

**Checkpoint files (12)**: Auto-saved Jupyter backups (`*-checkpoint.ipynb`) — duplicates of active notebooks.

### Notebook Evolution Tree

```
FirstTest_TypeCurve_Update
├── -Copy1, -Copy2                          # Early iterations
├── -NeuralNetwork                          # NN-only base
│   ├── -withPCA                            # + PCA reduction
│   └── -PhaseSeparate                      # + separate oil/gas models
└── -NeuralNetworkandOthers                 # NN + ensemble base
    ├── -Copy1, -Copy2, -Copy3              # Snapshots
    └── -WithCorrections                    # Bug fixes
        └── -NoCategorical                  # Numerical-only features
            ├── -Copy1, -Copy2, -Copy3      # Snapshots
            ├── -withEURGrouping            # EUR-bucket stratification
            └── -withMLUpdates              # Refined pipeline
                ├── -Copy                   # Snapshot
                └── -withHL                 # Hidden layer tuning
```

**Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb** is the cleaned-up consolidation of the above experiments.

## Key Concepts

### Well Data Features

- **UWI** (Unique Well Identifier): Primary key for each well (14-digit API number, e.g., `42383398110000`)
- **UWI10**: 10-digit shortened UWI
- **Typecurve / BasinTC**: Basin-formation grouping (e.g., "CoyoteValley_North", "Mercury_600") — 114 unique groups across 15,500 wells
- **FORMATION_CONDENSED**: Condensed formation name, used alongside BasinTC for per-combination training
- **EUR parameters**: 30-year estimated ultimate recovery at percentiles P20/P35/P50/P65/P80 for oil (MBO), gas (MMCF), water (MBBL)
- **Decline curve params** (parsed from string columns): BuildupRate, MonthsInProd, InitialProd, DiCoefficient, BCoefficient, LimDeclineRate — prefixed by phase and percentile (e.g., `Oil_Params_P50_InitialProd`)
- **Completion features**: FluidPerFoot_bblft, ProppantPerFoot, LateralLength
- **Spatial features**: HEELPOINT_LAT, HEELPOINT_LON, TOEPOINT_LAT, TOEPOINT_LON, HEELPOINT_DEPTH, TOEPOINT_DEPTH
- **Well geometry**: WELL_TORTUOSITY, AVERAGE_INCLINATION, AZIMUTH, DEPTH_TO_TOP_1Q–4Q, DEPTH_ABOVE_ZONE_1Q–4Q
- **Neighbor well features**: NNAZ_1–6 (nearest neighbors in azimuth zone), NNSZ_1–2 (nearest neighbors in spacing zone) — each with UWI, FORMATION, TRUEDIST, HZDIST, VTDIST, EUR values, and cumulative production
- **Metadata columns** (dropped before modeling): WellName, LeaseName, WellNumber, CurrentOperatorName, OriginalOperatorName, DrillingContractorName, PermitDate, SpudDate, CompletionDate

### Data Pipeline (Main.ipynb)

The pipeline is sequential, executed across 8 notebook cells:

1. **Imports** — Standard libraries + `from Preprocessing import (...)` and `from Map_Plotting import create_well_map` (these imports will fail without the missing `.py` files)
2. **`load_and_preprocess_data(file_path, uwi_columns)`** — Load Excel file, initial cleaning; `uwi_columns` includes UWI + 6 NNAZ UWIs + 2 NNSZ UWIs
3. **`split_parameters(df, param_columns)`** — Parse 11 decline curve parameter string columns (Oil/Gas P20–P80, Water P50) into 66 individual numeric columns
4. **`handle_missing_values(df)`** — Imputation strategies
5. **`drop_na_parameter_columns(df, new_columns)`** — Remove rows with missing key params
6. **`group_and_summarize(df)`** — Aggregate by type curve group; returns `grouped_df` (114 groups) and `uwi10_sum` (15,500 wells)
7. **`drop_zero_rows(df, columns_to_check)`** — Remove wells with zero values in: FluidPerFoot_bblft (drops ~1411), ProppantPerFoot (~187), EUR_Gas_P50 (~4), EUR_Oil_P50 (~1), HEELPOINT_LAT (~331)
8. **`convert_to_absolute_values(df)`** → **`replace_missing_water_params(df)`** → **`replace_zeros_with_P50(df)`** → **`add_neighbor_eur_cumulative(df)`** → filter out `BasinTC == 'Unknown'`
9. **`drop_specified_columns(df)`** — Hard-coded list removes ~40+ identifier, metadata, EUR, geometry, and redundant neighbor columns before modeling

### ML Models (Refined Notebook)

**Neural Network** (primary model):
- Keras Functional API with mixed input architecture
- **Numerical path**: Input → Reshape(n_features, 1) → Conv1D(32, kernel=3, padding='same') → Conv1D(64, kernel=3, padding='same') → Conv1D(128, kernel=3, padding='same') → GlobalAveragePooling1D
- **Categorical path**: Per-column Embedding(input_dim=nunique+1, output_dim=20) → Flatten → Concatenate
- **Merged path**: Concatenate(numerical, categorical) → Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Dropout → Dense(output_size, activation='linear')
- Default config: `{'embedding_output_dim': 20, 'dense_layer_sizes': [256, 128, 64, 32], 'dropout_rate': 0.3, 'regularization': l2(0.01), 'activation': 'relu', 'optimizer': 'adam', 'loss_function': 'mse'}`
- Training: epochs=1000, batch_size=50, EarlyStopping + ReduceLROnPlateau + custom RealTimePlottingCallback

**Ensemble/Traditional models** (for comparison):
- XGBoostRegressor (with hyperparameter tuning via GridSearchCV/RandomizedSearchCV, uses ColumnTransformer for preprocessing)
- RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
- ElasticNet, SVR, LinearRegression, DecisionTreeRegressor

**Model evaluation**: R2 score, RMSE (mean_squared_error), MAE (mean_absolute_error); SHAP values for interpretability

### Key Functions in Model Notebooks

| Function | Purpose |
|----------|---------|
| `prepare_data(df, numerical_columns, categorical_columns)` | Splits data into train/val/test by BasinTC + FORMATION_CONDENSED combination, applies StandardScaler |
| `build_model(numerical_columns, categorical_columns, df, output_size, model_type, **kwargs)` | Constructs neural network or selects ensemble model based on `model_type` parameter |
| `train_and_evaluate_model(combo_train, combo_val, combo_test, ...)` | Full train/evaluate loop for one basin-formation combination |
| `plot_model_performance(history, title)` | Training/validation loss curves |
| `plot_ml_performance(y_true, y_pred, title)` | Actual vs. predicted scatter plots |
| `plot_decline_curves(time, actual, predicted, title)` | Overlay decline curves |
| `plot_sensitivity_analysis(sensitivities, feature_names)` | Feature sensitivity bar charts |
| `plot_feature_vs_EUR(combo_test, feature, EUR_values, ...)` | Feature-EUR relationship plots |
| `create_well_map(df, output_html)` | Folium interactive map of well locations |
| `assign_basin_tc(row)` | Assigns BasinTC using lat/lon bounding boxes when missing |
| `cumulative_production_and_EUR(qi, di, b, Dlim, IBU, MBU, years, buildup_method)` | Decline curve EUR calculator (in Untitled.ipynb) |

### Training Strategy

- Models are trained **per unique (BasinTC, FORMATION_CONDENSED) combination** — not a single global model
- `unique_combinations = df[['BasinTC', 'FORMATION_CONDENSED']].drop_duplicates()` defines the training loop
- Training uses `joblib.Parallel` for parallel execution across combinations
- Data split: train/validation/test via `train_test_split`
- Feature scaling: StandardScaler (primary) or RobustScaler for numerical features
- Categorical encoding: LabelEncoder or OneHotEncoder — excluded in `-NoCategorical` variants
- `BasinTC` and `FORMATION_CONDENSED` are explicitly excluded from feature columns (used only for grouping)
- Target columns (`y_headers`): dynamically selected as all columns containing 'Params' in the name

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, geopandas, scipy, openpyxl |
| ML | scikit-learn, tensorflow/keras, xgboost, shap |
| Visualization | matplotlib, seaborn, folium (interactive maps) |
| Interactive | ipywidgets, IPython.display |
| Parallel | joblib (Parallel, delayed), tqdm |
| Utilities | PIL, textwrap, ast, tempfile |
| Web scraping | BeautifulSoup (bs4) — in Untitled.ipynb only |
| Web (partial) | Flask (imported but minimally used) |

**Python version**: 3.x (no version pinned)
**No requirements.txt, environment.yml, setup.py, or Dockerfile exists** — dependencies must be inferred from imports.

## Git History

- **Initial commit** (`eebe7c3`): "aLL fiLLES ADDED" — single bulk upload of all files
- **Branch**: `master` (default)
- **No `.gitignore`** — checkpoint files (~12), large Excel files (~22 MB total), and 57 MB notebooks are all tracked
- No tags, no CI/CD, no branch protection

## Development Workflow

### Running the Project

1. Install all dependencies (see Technology Stack)
2. Place well data Excel files in the repository root
3. Open `Main.ipynb` — **but note**: the `from Preprocessing import (...)` and `from Map_Plotting import create_well_map` imports will fail. Either:
   - Extract functions from the model notebooks into `Preprocessing.py` / `Map_Plotting.py`, or
   - Copy the function definitions from the model notebooks inline into `Main.ipynb`
4. Update the hardcoded Windows file path: `r'C:\Users\Prakhar.Sarkar\OneDrive - ...\Prakhar_Testnew2.xlsx'` → local path
5. Open the desired model notebook (e.g., `Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb`) to train/evaluate

### Data File Paths

Notebooks reference Windows paths like `C:\Users\Prakhar.Sarkar\OneDrive - SRP Management Services\Documents\_For_Prakhar\Prakhar_Testnew2.xlsx`. These must be updated to local paths. The Excel files in the repo root serve as the data sources.

### Known Issues

1. **Missing `.py` modules**: `Preprocessing.py` and `Map_Plotting.py` are imported but don't exist as files
2. **No `.gitignore`**: Checkpoint files, large binaries, and data files are all committed
3. **Hardcoded Windows paths**: Must be manually updated per environment
4. **No tests or CI/CD**: No unit tests, integration tests, or automated pipelines
5. **Large notebook outputs**: Some notebooks contain embedded training outputs up to 57 MB
6. **Filename spaces**: Some files have spaces before suffixes (e.g., `- withMLUpdates`) which can cause path issues

## Conventions for AI Assistants

### Do

- Treat this as a **data science exploration project** — notebooks are the primary artifact
- When modifying ML pipelines, preserve the **per (BasinTC, FORMATION_CONDENSED) combination** training strategy
- Keep decline curve parameter names consistent: `{Phase}_Params_{Percentile}_{Param}` (e.g., `Oil_Params_P50_InitialProd`)
- Use the existing preprocessing function signatures when adding new data cleaning steps
- When adding new features, add them before the `drop_specified_columns` step
- Use scikit-learn-compatible APIs for new models (fit/predict/score interface)
- Target columns are dynamically selected via `[col for col in df.columns if 'Params' in col]`
- Use `Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb` as the canonical reference for model architecture

### Don't

- Don't assume `.py` module files exist — `Preprocessing.py` and `Map_Plotting.py` are not in the repo
- Don't modify the Excel data files directly
- Don't hardcode Windows file paths — use `os.path` or `pathlib` for cross-platform compatibility
- Don't add large binary files (models, datasets) without discussing storage approach
- Don't remove checkpoint notebooks without confirming — they may contain unique experiment results
- Don't change the `build_model()` function signature — it's called from `train_and_evaluate_model()` with specific kwargs
- Don't merge `BasinTC` or `FORMATION_CONDENSED` into feature columns — they are grouping keys only

### Code Style

- Functions use `snake_case` naming
- DataFrames commonly named `df`, `grouped_df`, `cumulative_df`, `combo_train`, `combo_val`, `combo_test`
- Column names use mixed conventions: CamelCase for well identifiers (e.g., `FluidPerFoot_bblft`), underscore-separated for EUR columns (e.g., `EUR_30yr_Actual_Oil_P50_MBO`)
- Inline comments are used liberally; docstrings are sparse
- Visualization outputs are embedded in notebook cells
- Model configurations are passed as dictionaries with keys: `embedding_output_dim`, `dense_layer_sizes`, `dropout_rate`, `regularization`, `activation`, `optimizer`, `loss_function`
