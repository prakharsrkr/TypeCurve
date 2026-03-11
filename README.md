# TypeCurve

Oil & gas production forecasting system that uses machine learning to predict **Estimated Ultimate Recovery (EUR)** for wells across multiple geological basins and formations. The project builds type curves — standardized production profiles — using neural networks and ensemble ML models trained on well completion parameters, production data, and spatial/neighbor-well features.

**Domain**: Petroleum engineering / reservoir analysis

**Primary prediction targets**:
- `EUR_30yr_Actual_Oil_P50_MBO` — 30-year estimated oil recovery
- `EUR_30yr_Actual_Gas_P50_MMCF` — 30-year estimated gas recovery
- `EUR_30yr_Actual_Water_P50_MBBL` — 30-year estimated water recovery

## Getting Started

### Prerequisites

Python 3.x with the following libraries:

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, geopandas, scipy, openpyxl |
| ML | scikit-learn, tensorflow/keras, xgboost, shap |
| Visualization | matplotlib, seaborn, folium |
| Interactive | ipywidgets, IPython.display |
| Utilities | joblib, tqdm, PIL |

### Running the Project

1. Install all dependencies listed above.
2. Place well data Excel files (`Prakhar_Testnew2.xlsx`, etc.) in the repository root.
3. Open `Main.ipynb` to run the preprocessing pipeline.
4. Open the desired model notebook (e.g., `Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb`) to train and evaluate models.

> **Note**: Some notebooks reference Windows file paths. Update these to your local paths as needed.

## Repository Structure

```
TypeCurve/
├── Main.ipynb                          # Entry point: imports, data loading, preprocessing pipeline
├── Refined_TypeCurve_Update-*.ipynb    # Cleaned-up model notebook
├── FirstTest_TypeCurve_Update-*.ipynb  # Experimental notebooks (multiple variants)
├── Untitled.ipynb                      # Scratch/exploratory notebook
├── Prakhar_Testnew2.xlsx               # Primary dataset (~15,500 wells)
├── Prakhar_Testnew3.xlsx               # Secondary dataset
└── Prakhar_Testnew4.xlsx               # Tertiary dataset
```

### Notebook Naming Conventions

| Suffix | Meaning |
|--------|---------|
| `-Copy1`, `-Copy2` | Iterative experiment snapshots |
| `-NoCategorical` | Excludes categorical feature encoding |
| `-withPCA` | Uses PCA dimensionality reduction |
| `-PhaseSeparate` | Separate oil/gas phase modeling |
| `-withEURGrouping` | Groups wells by EUR buckets |
| `-WithCorrections` | Bug-fixed versions of earlier experiments |
| `-withHL` | Hidden layer modifications |
| `-withMLUpdates` | Updated ML pipeline |

## Data Pipeline

The preprocessing pipeline in `Main.ipynb` runs the following steps:

1. **Load & preprocess** — Load Excel data, initial cleaning
2. **Split parameters** — Parse decline curve parameter strings into columns
3. **Handle missing values** — Imputation strategies
4. **Drop NA columns** — Remove rows with missing key parameters
5. **Group & summarize** — Aggregate by type curve group
6. **Drop zero rows** — Remove wells with zero values in critical columns
7. **Convert to absolute values** — Ensure positive numeric values
8. **Replace missing water params** — Handle missing water production parameters
9. **Replace zeros with P50** — Fill zero EUR values with P50 estimates
10. **Add neighbor EUR features** — Compute neighbor-well aggregate features
11. **Drop specified columns** — Remove identifier/redundant columns before modeling

## Models

### Neural Network (Primary)

- Keras Functional API with mixed input architecture
- Numerical features processed through Conv1D layers (32 → 64 → 128 filters)
- Categorical features through embedding layers
- Dense layers [256, 128, 64, 32] with dropout and L1/L2 regularization
- Adam optimizer, MSE loss, with EarlyStopping and ReduceLROnPlateau

### Ensemble / Traditional Models

- XGBoost (with hyperparameter tuning)
- Random Forest, Gradient Boosting, AdaBoost
- ElasticNet, SVR, Linear Regression, Decision Tree

### Training Strategy

- Models are trained **per basin-formation combination** (not a single global model)
- Feature scaling via StandardScaler or RobustScaler
- Categorical encoding via LabelEncoder or OneHotEncoder
- Evaluation metrics: R2 score, RMSE, MAE
- SHAP values for model interpretability

## Key Concepts

- **UWI** — Unique Well Identifier (14-digit API number)
- **Typecurve / BasinTC** — Basin-formation grouping (e.g., "CoyoteValley_North")
- **EUR** — Estimated Ultimate Recovery at percentiles P20/P35/P50/P65/P80
- **Decline curve parameters** — BuildupRate, MonthsInProd, InitialProd, DiCoefficient, BCoefficient, LimDeclineRate
- **Completion features** — FluidPerFoot, ProppantPerFoot, LateralLength, etc.
- **Neighbor well features** — NNAZ (nearest neighbors azimuth zone), NNSZ (nearest neighbors spacing zone)
