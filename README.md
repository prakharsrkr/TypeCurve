# TypeCurve

A machine learning system for predicting how much oil, gas, and water a well will produce over its lifetime.

## What This Project Does

In the oil & gas industry, companies need to estimate a well's **Estimated Ultimate Recovery (EUR)** — the total volume of hydrocarbons a well is expected to produce over ~30 years. These estimates drive investment decisions worth millions of dollars per well.

Traditionally, petroleum engineers build **type curves** — standardized production decline profiles for a given geological area — by hand using decline curve analysis. This project automates and improves that process using machine learning.

Given a well's completion parameters (how it was drilled and fractured), its location, and data from nearby wells, the system predicts:

| Target | Description | Unit |
|--------|-------------|------|
| **EUR Oil P50** | Expected 30-year oil recovery (median estimate) | Thousand barrels (MBO) |
| **EUR Gas P50** | Expected 30-year gas recovery (median estimate) | Million cubic feet (MMCF) |
| **EUR Water P50** | Expected 30-year water production (median estimate) | Thousand barrels (MBBL) |

The "P50" means a 50th-percentile (median) estimate. The system also works with other percentiles (P20, P35, P65, P80) to capture uncertainty ranges.

## How It Works

### 1. Data

The dataset contains ~15,500 wells with features including:

- **Well identifiers**: UWI (Unique Well Identifier — a 14-digit API number) and basin/formation grouping
- **Completion data**: How the well was drilled — lateral length, fluid pumped per foot, proppant (sand) per foot, etc.
- **Decline curve parameters**: Mathematical parameters describing how production decreases over time (initial production rate, decline coefficients, buildup rate)
- **Location**: Heel and toe point coordinates (lat/lon) of each horizontal well
- **Neighbor well performance**: Production data from the 6 nearest wells in the across all zones and 2 nearest in the same zone — because nearby wells strongly influence each other's performance

### 2. Preprocessing Pipeline

Raw well data goes through an 11-step cleaning pipeline in `Main.ipynb`:

1. Load data from Excel files
2. Parse decline curve parameter strings into individual numeric columns
3. Impute missing values
4. Drop rows missing critical parameters
5. Group wells by their type curve area (basin + formation)
6. Remove wells with zero production in key columns
7. Convert any negative values to absolute values
8. Fill in missing water production parameters
9. Replace zero EUR values with P50 estimates from the well's type curve group
10. Compute aggregate features from neighbor wells (how are nearby wells performing?)
11. Drop identifier columns that shouldn't be used as model features

### 3. Models

Models are trained **separately for each basin-formation combination** — because the geology varies dramatically between areas, a single global model would perform poorly.

**Primary model — Neural Network:**
A Keras neural network with a mixed-input architecture. Numerical features pass through 1D convolutional layers (which capture patterns across feature groups), while categorical features (basin, formation) go through embedding layers. These paths merge into dense layers with dropout and regularization to prevent overfitting.

**Comparison models:**
The neural network is benchmarked against traditional ML approaches — XGBoost (with hyperparameter tuning), Random Forest, Gradient Boosting, AdaBoost, ElasticNet, SVR, and others — all using scikit-learn-compatible interfaces.

**Evaluation:** R2 score, RMSE, and MAE measure prediction accuracy. SHAP values provide interpretability — showing which features matter most for each prediction.

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

> **Note**: There is no `requirements.txt` yet — dependencies must be installed manually.

### Running

1. Install all dependencies listed above.
2. Ensure the well data Excel files (`Prakhar_Testnew2.xlsx`, etc.) are in the repository root.
3. Open and run `Main.ipynb` — this executes the full preprocessing pipeline and produces a clean DataFrame ready for modeling.
4. Open a model notebook to train and evaluate. Start with `Refined_TypeCurve_Update-NeuralNetworkandOthers.ipynb` for the cleaned-up version.

> **Note**: Some notebooks contain hardcoded Windows file paths (e.g., `C:\Users\...`). Update these to your local paths before running.

## Repository Structure

```
TypeCurve/
├── Main.ipynb                          # Start here — data loading & preprocessing
├── Refined_TypeCurve_Update-*.ipynb    # Clean model notebook (neural network + ensembles)
├── FirstTest_TypeCurve_Update-*.ipynb  # Experimental variants (see naming conventions below)
├── Untitled.ipynb                      # Scratch/exploratory work
├── Prakhar_Testnew2.xlsx               # Primary well dataset (~15,500 wells)
├── Prakhar_Testnew3.xlsx               # Secondary dataset
└── Prakhar_Testnew4.xlsx               # Tertiary dataset
```

The experimental notebooks use suffixes to indicate what's different about each variant:

| Suffix | What's different |
|--------|------------------|
| `-NoCategorical` | Drops categorical features — tests if basin/formation encoding helps |
| `-withPCA` | Applies PCA to reduce feature dimensions before training |
| `-PhaseSeparate` | Trains separate models for oil vs. gas phases |
| `-withEURGrouping` | Buckets wells by EUR range before training |
| `-WithCorrections` | Bug fixes applied to an earlier version |
| `-withHL` | Experiments with different hidden layer sizes |
| `-withMLUpdates` | Updated ML pipeline (newer preprocessing or model configs) |
| `-Copy1`, `-Copy2` | Snapshot copies of experiments at different stages |

> **Note**: `Main.ipynb` imports from `Preprocessing.py` and `Map_Plotting.py`, but these files are not in the repo — the functions they reference are defined inline within the notebooks.
