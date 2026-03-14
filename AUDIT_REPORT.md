# TypeCurve ML Workflow Audit Report

**Date**: 2026-03-14
**Scope**: End-to-end audit of the ML pipeline for type-curve scaling stability and repeatability
**Files audited**: `main.py`, `typecurve/config.py`, `typecurve/data_preparation.py`, `typecurve/models.py`, `typecurve/training.py`, `typecurve/shap_analysis.py`, `typecurve/feature_engineering.py`, `typecurve/preprocessing.py`, `typecurve/callbacks.py`, `typecurve/testing.py`, `typecurve/evaluation.py`

---

## Executive Summary

**The current workflow cannot be trusted for repeatable type-curve scaling.**

The model may produce reasonable predictions in aggregate, but the learned scaling relationships (how type curves change with lateral length, spacing, completion parameters, etc.) are **not stable across runs**. This is caused by a combination of: (1) uncontrolled randomness at every layer of the pipeline, (2) data leakage that inflates apparent accuracy while masking instability, (3) an overly flexible model architecture for the available per-formation sample sizes, and (4) SHAP analysis that is structurally unreliable given the model's instability. The diagnosis is: **accurate but unstable** — and therefore unsuitable for operational scaling decisions.

---

## Section 1: Pipeline Walkthrough and Issues Found

### 1.1 Data Loading & Preprocessing (`preprocessing.py`, `feature_engineering.py`)

**What it does**: Loads Excel data, parses stringified decline-curve parameter lists into 66 numeric columns, fills missing values with 0 or 'Unknown', drops zero rows, replaces missing water params with scaled oil params, replaces zero percentile values with P50 values.

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| P1 | `handle_missing_values()` fills ALL numeric NaN with 0 | HIGH | Zero-filling production parameters creates artificial data points that distort trends. A well with zero `DiCoefficient` is physically meaningless but gets treated as real data. |
| P2 | `replace_missing_water_params()` uses `oil * 3` as water proxy | MEDIUM | Introduces a deterministic but physically dubious relationship. Creates correlation between oil and water targets that doesn't reflect real reservoir behavior. |
| P3 | `replace_zeros_with_P50()` replaces zeros in P20/P35/P65/P80 with P50 | HIGH | Destroys the natural variance across percentiles. When a percentile is unknown, substituting P50 makes P20=P35=P50=P65=P80 for that well, creating false precision and reducing meaningful variation. |
| P4 | `scale_parameters()` applies a lateral-length normalization formula then **drops** `HORIZONTIAL_WELL_LENGTH` | CRITICAL | The scaling formula `(1 - 0.03*(10000-LL)/2500) * 10000/LL` normalizes InitialProd and EUR to 10,000ft-equivalent. But then lateral length is dropped from features. This means **the model cannot learn lateral-length scaling** because the feature is removed. To produce a lateral-length scaling curve, you'd need to undo this normalization or keep the raw + normalized values. |
| P5 | `fill_zero_parameters()` replaces zero HZDIST with 5280 (1 mile) | MEDIUM | Wells with no detected neighbor get assigned a 1-mile default distance. This is an arbitrary sentinel value that may or may not be physical. The model treats it as real data. |

### 1.2 Feature Engineering (`feature_engineering.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| F1 | `identify_column_types()` auto-detects categorical columns from dtype | MEDIUM | Any column that happens to be `object` type gets treated as categorical. If preprocessing changes column types between runs (e.g., a column that is sometimes all-numeric vs sometimes has strings), the feature set changes. |
| F2 | `encode_categorical_columns()` fits LabelEncoder on the **entire dataset** before train/test split | CRITICAL | This is data leakage. The encoder sees test-set categories during fitting. More importantly, LabelEncoder assigns arbitrary integer codes based on sort order. If data changes, the same category gets different codes → the embedding layer learns different representations → trends change between runs. |
| F3 | `calculate_combined_eur_and_cumulative()` uses `gas/20` conversion | LOW | This is a common BOE conversion but hardcodes a GOR assumption. Not a stability issue per se, but worth noting as a domain assumption. |

### 1.3 Train/Validation/Test Split (`data_preparation.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| S1 | `split_data()` uses plain `train_test_split()` — NOT stratified by formation | CRITICAL | With 114 basin-formation groups, many having <50 wells, a random split can put vastly different numbers of wells from each formation into train vs test. Formation X might have 80% of its wells in training in run 1 and 60% in run 2 → completely different learned behavior for that formation. |
| S2 | Split happens at the global level, then per-formation subsets are extracted | HIGH | After global split, `execute_training()` filters `train_df` by `(basin, formation)`. If a formation has 30 wells total and the split puts 21 in train, that's 21 training samples. But a different random seed might put only 15 in train. With such small N, learned trends are sample-dependent, not formation-dependent. |
| S3 | `RANDOM_STATE = 42` is set in config but **not propagated to TensorFlow or NumPy** | CRITICAL | `split_data` uses the seed for sklearn's `train_test_split`, but TF weight initialization, dropout masks, batch shuffling, and NumPy operations throughout the pipeline are unseeded. |

### 1.4 Scaling / Normalization (`data_preparation.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| N1 | `fit_and_apply_scalers()` fits MinMaxScaler on `combined_df = concat([train, val, test])` | CRITICAL | **Data leakage.** The scaler sees test data during fitting. This means the [0,1] range is determined by the most extreme values across all splits. If a test-set outlier defines the max, training data gets scaled differently than it would without leakage. Between runs with different splits, the scaler parameters change → feature values change → trends change. |
| N2 | MinMaxScaler is used instead of StandardScaler or RobustScaler | MEDIUM | MinMaxScaler is highly sensitive to outliers. A single extreme value shifts the entire scale. For production data with heavy tails (some wells produce 10x the median), this compresses most data into a narrow range. |
| N3 | Output targets (decline curve parameters) are also MinMaxScaled | MEDIUM | Scaling targets is fine in principle, but with leakage (N1), the target scaling is also contaminated. |

### 1.5 Model Architecture (`models.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| M1 | No random seed set for TensorFlow before model building | CRITICAL | `_build_neural_network()`, `_build_cnn()`, `_build_resnet()` all use random weight initialization with no seed control. Every run starts from different random weights → different local minimum → different learned trends. |
| M2 | Embedding `input_dim=df[col].nunique() + 1` uses the **full dataset** `df` | HIGH | The embedding dimension is computed from the full pre-split DataFrame, not from training data. This creates a dependency on the full dataset's cardinality. If a rare category exists only in the test set, it still gets an embedding slot, but that embedding is never trained → garbage values at inference. |
| M3 | Neural network architectures have no regularization beyond Dropout | MEDIUM | No L1/L2 weight regularization, no BatchNorm on the main NN path, no gradient clipping in the architecture. The `training.py` adds `clipvalue=1.0` to Adam, but this is a coarse control. |
| M4 | `_build_sklearn_pipeline()` applies `PolynomialFeatures(degree=2)` | HIGH | With ~20+ numerical features, degree-2 polynomials create ~200+ interaction terms. This makes the feature space very high-dimensional relative to per-formation sample sizes (often <100). Overfitting is almost guaranteed, and feature importance gets diluted across hundreds of polynomial terms. |
| M5 | Four different model types (NN, CNN, ResNet, XGBoost) are trained per formation | MEDIUM | Training 4 models per formation multiplies the instability — each model type has its own randomness sources. For scaling, you want ONE stable model, not four unstable ones. |

### 1.6 Training (`training.py`, `callbacks.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| T1 | `model.compile()` in `train_and_evaluate_model()` **re-compiles** the model after `build_model()` already compiled it | LOW | Not a stability issue directly, but wasteful and confusing. The second compile with `clipvalue=1.0` overrides the first. |
| T2 | `PositivePredictionCallback` restores weights when negative predictions are detected | HIGH | This callback introduces stochastic weight restoration. If negative predictions occur at different epochs in different runs (due to different random initialization), the model follows different optimization trajectories. This is a major source of non-reproducibility. |
| T3 | `EarlyStopping(patience=10)` + `ReduceLROnPlateau(patience=5)` interact unpredictably | MEDIUM | With small training sets (often <100 samples per formation), the validation loss is noisy. Early stopping may trigger at epoch 30 in one run and epoch 80 in another → different models. |
| T4 | `custom_xgboost_training()` fits preprocessor on `concat([train, val])` | MEDIUM | The ColumnTransformer is fitted on train+val data, which is better than train+val+test but still allows validation data to influence preprocessing. |
| T5 | XGBoost `validate_productions_xgb()` can truncate training early | MEDIUM | If validation predictions produce negative qi/di/b at round N, training stops. With different data splits, this truncation point varies → different models. |
| T6 | No `tf.random.set_seed()`, no `np.random.seed()`, no `os.environ['PYTHONHASHSEED']` | CRITICAL | The single biggest cause of irreproducibility. Without these, every neural network run is fundamentally non-deterministic. |

### 1.7 SHAP Analysis (`shap_analysis.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| SH1 | `DeepExplainer` is used for neural networks — known to be unstable | HIGH | SHAP's DeepExplainer uses the DeepLIFT approximation, which can give different attributions depending on the reference/background data. With a model that's already unstable, SHAP compounds the instability. |
| SH2 | SHAP background data is the same as evaluation data | MEDIUM | `compute_shap_values_nn()` passes the same `combined_data` as both background and explanation data. This is methodologically incorrect — background should represent the "baseline" distribution, not the points being explained. |
| SH3 | `shap_sample_size=100` with `random_state=42` provides one fixed sample | LOW | At least this is deterministic for a given split. But it means SHAP results depend on which 100 wells happen to be in the test set for that formation — which varies with the random split (issue S1). |
| SH4 | For XGBoost, SHAP operates on **polynomial-transformed** features | HIGH | After `PolynomialFeatures(degree=2)`, the SHAP values correspond to interaction terms like `Feature1*Feature2`, not the original features. Feature importance is scattered across hundreds of polynomial terms, making interpretation meaningless. |
| SH5 | CNN/ResNet models are routed to `compute_shap_values_nn()` (DeepExplainer) | MEDIUM | Line 130-131: `if config_str in ('neural_network', 'cnn', 'resnet', 'transformer'): model_type = 'neural_network'`. This means CNN and ResNet both use DeepExplainer, which may not handle Conv1D layers correctly. |
| SH6 | `TreeExplainer` for sklearn models strips the pipeline | MEDIUM | Line 43: `model = model.named_steps['model']` — this gives SHAP the raw model without the preprocessor. But the data hasn't been transformed through the preprocessor first (only via `np.concatenate`). Feature values don't match what the model was trained on. |

### 1.8 Type Curve Scaling (`testing.py`)

**Issues found**:

| # | Issue | Severity | Impact on Stability |
|---|-------|----------|---------------------|
| TC1 | Scaling factors are computed as `new_pred / baseline_pred` | HIGH | If the baseline prediction is itself unstable (which it is, per issues above), the scaling factor inherits that instability. A 10% change in baseline prediction can flip a scaling factor from 1.1 to 0.9. |
| TC2 | Denormalize → modify feature → re-normalize → predict cycle | MEDIUM | The round-trip through `denormalize_and_decode_inputs()` then `input_scaler.transform()` should be exact, but floating-point precision and the encode/decode of categoricals can introduce small errors that compound. |
| TC3 | `clip_values(lower=-1e5, upper=1e5)` on predictions | LOW | Clipping is a safety measure but masks underlying model issues. If the model predicts values that need clipping, the scaling factor is unreliable. |

---

## Section 2: Root Causes of Instability (Ranked)

### Rank 1: Uncontrolled Randomness (T6, M1, S3)
**No seeds are set for TensorFlow, NumPy, or Python hash randomization.** Every neural network run starts from different random weights and follows a different optimization path. This alone makes trends non-reproducible.

### Rank 2: Data Leakage in Scaling (N1, F2)
**Scalers and encoders are fitted on the combined dataset (train+val+test).** This creates a false sense of accuracy and means that the scaling parameters themselves change when the split changes. The model's internal representation of features is split-dependent.

### Rank 3: Non-Stratified Random Split (S1, S2)
**The train/test split is global, not stratified by formation.** With many small formation groups, the per-formation train/test composition varies dramatically between runs. A formation with 25 wells might get 18 in training one run and 12 the next — a 50% change in training data.

### Rank 4: Overly Flexible Models for Small N (M4, M5, M3)
**Per-formation sample sizes are often 20-80 wells, but the model has hundreds of parameters.** Neural networks with [64, 32] dense layers + embeddings, and XGBoost with degree-2 polynomial features, are massively overparameterized for these sample sizes. The models fit noise, and the noise patterns change with the random seed.

### Rank 5: Feature Collinearity
**Neighbor features are highly correlated.** NNAZ_1 EUR, NNAZ_1 cumulative, NNSZ_1 EUR, NNSZ_1 cumulative, and various distance metrics are all measuring "neighbor well quality." When correlated features compete for importance, SHAP attribution splits between them arbitrarily. In one run Feature A gets credit; in another, Feature B does. The model's predictions may be similar, but the attributions (and therefore the apparent scaling logic) flip.

### Rank 6: Lateral Length Removed After Normalization (P4)
**The pipeline normalizes production to 10,000ft-equivalent and then drops lateral length.** This means the model literally cannot learn how type curves scale with lateral length — which is one of the primary use cases.

### Rank 7: SHAP Structural Unreliability (SH1-SH6)
**Even if the model were stable, SHAP results would not be.** DeepExplainer approximations, mismatched background data, polynomial feature expansion, and pipeline stripping all compromise SHAP's reliability.

---

## Section 3: Diagnosis Classification

| Dimension | Assessment |
|-----------|------------|
| **Predictive accuracy** | Moderate — aggregate metrics (MSE/MAE) may look acceptable because of data leakage (N1) inflating apparent performance |
| **Trend stability** | Poor — feature importance, SHAP rankings, and directional effects change across runs |
| **Formation-specific behavior** | Inconsistent — some large formations (>100 wells) may show stable trends; small formations (<30 wells) are essentially random |
| **Scaling reliability** | Not trustworthy — scaling factors inherit all upstream instabilities |

**Overall classification: Accurate but unstable.** The model can interpolate within the training distribution but cannot reliably tell you *why* or *how* type curves should scale.

---

## Section 4: Formation-Specific Behavior Assessment

The current approach of training per `(BasinTC, FORMATION_CONDENSED)` combination is conceptually right — different formations should have different scaling behavior. But the implementation creates problems:

1. **Sample sizes are too small** for the model complexity. With 114 unique combinations across ~15,500 wells, the average group has ~136 wells. After filtering zeros and spurious curves, many groups have <50 wells. Split 70/30, that's <35 training samples for a model with 64+32 dense layers.

2. **The global scaler** (fitted on all formations) means that a formation's feature values are scaled relative to the global distribution, not its own. A formation where all wells have ProppantPerFoot=1500-2000 gets its values compressed into a narrow band on the [0,1] scale, losing resolution.

3. **Categorical encoding** of formation name is redundant — if you're already training per-formation, the formation identity is constant within each training set. The embedding learns nothing.

**Recommendation**: The per-formation training strategy is correct in principle. But it needs: (a) per-formation scalers, (b) simpler models for small groups, (c) hierarchical pooling for very small groups, (d) the categorical formation column excluded from features (it's the grouping variable).

---

## Section 5: Can This Model Support Operational Scaling Decisions?

### Lateral Length Scaling
**No.** Lateral length is removed after normalization (P4). The model cannot produce lateral-length scaling curves.

### Spacing / Development Density
**Partially, but unreliably.** NNSZ_1_HZDIST is retained as a feature and represents nearest spacing-zone neighbor distance. But neighbor distance features are collinear, and the model's response to spacing changes is not stable across runs.

### Number of Neighboring Wells
**No.** The number of neighbors is not a feature — only distances and production values for specific neighbor slots (NNAZ_1 through NNAZ_6, NNSZ_1-2) are included. Slots without neighbors get sentinel values (HZDIST=5280), which the model may or may not interpret correctly.

### Completion Design Changes (ProppantPerFoot, FluidPerFoot)
**Partially, but unreliably.** These are direct features, but the scaling factors depend on baseline predictions that are unstable. Additionally, `FluidPerFoot_bblft` has 1,411 wells dropped for zeros — if zero fluid-per-foot is common in certain formations, those formations lose significant data.

### Formation-Specific Interference Patterns
**No.** Parent-child interference requires knowing the sequence of well completions, relative positions, and timing — none of which are modeled explicitly. Neighbor EUR values give indirect signal, but the model has no mechanism to distinguish parent from child wells.

---

## Section 6: Recommended Fixes (Priority Order)

### Priority 1: Reproducibility (addresses Ranks 1, 3)
- Set global seeds for Python, NumPy, TensorFlow
- Use `GroupShuffleSplit` stratified by `(BasinTC, FORMATION_CONDENSED)` for train/test split
- **Files**: `config.py`, `data_preparation.py`, `main.py`

### Priority 2: Eliminate Data Leakage (addresses Rank 2)
- Fit scalers on training data only
- Fit LabelEncoders on training data only
- **Files**: `data_preparation.py`, `feature_engineering.py`, `main.py`

### Priority 3: Simplify Models (addresses Rank 4)
- Remove PolynomialFeatures from sklearn pipeline
- Add monotonic constraints to XGBoost for physically grounded features
- Consider using only XGBoost with monotonic constraints as the primary model for stability
- **Files**: `models.py`

### Priority 4: Reduce Feature Collinearity (addresses Rank 5)
- Keep only NNAZ_1 and NNSZ_1 combined EUR/cumulative (drop redundant neighbor columns)
- The `COLUMNS_TO_DROP` already removes NNAZ 3-6 and NNAZ_2/NNSZ_2. Verify the remaining features are not highly correlated.
- **Files**: `config.py`

### Priority 5: Fix Lateral Length Handling (addresses Rank 6)
- Either keep `HORIZONTIAL_WELL_LENGTH` as a feature (alongside normalized targets) OR do not normalize targets and let the model learn the lateral-length effect
- **Files**: `feature_engineering.py`

### Priority 6: Replace SHAP with Stable Alternatives (addresses Rank 7)
- Use Accumulated Local Effects (ALE) or repeated PDP instead of SHAP for trend analysis
- If SHAP is retained, use KernelExplainer with a fixed k-means background sample
- Run SHAP across multiple seeds and report only features with consistent direction
- **Files**: `shap_analysis.py`

### Priority 7: Add Stability Measurement
- Create a utility that runs the pipeline N times with different seeds and measures:
  - Prediction variance
  - Feature importance rank correlation (Kendall's tau)
  - SHAP sign consistency
  - Scaling factor confidence intervals
- **Files**: new `typecurve/stability.py`

---

## Section 7: Should This Remain a Predictive Model?

**Recommendation: Reframe as a constrained scaling-factor model.**

The current approach tries to predict absolute decline-curve parameters (qi, Di, b) from well features. This is inherently noisy because decline parameters have high natural variance even within a formation. A better framing:

1. **Establish baseline type curves** from actual production data (not ML — use statistical P10/P50/P90 curves per formation)
2. **Use ML only to predict scaling factors** (multipliers relative to the formation baseline) as a function of completion parameters, spacing, and neighbor effects
3. **Apply monotonic constraints** so that e.g., longer laterals always produce higher EUR (within the model's allowed range)
4. **Use XGBoost with `monotone_constraints`** as the primary model — it's stable, fast, interpretable, and supports monotonic constraints natively
5. **Use bootstrap confidence intervals** on scaling factors to quantify uncertainty

This reframing separates "what is the typical behavior" (statistical, not ML) from "how does behavior change with inputs" (ML with constraints). The ML model only needs to learn deviations from baseline, which is a much easier problem.

---

## Section 8: Final Verdict

**The current workflow is NOT trustworthy for repeatable type-curve scaling.**

The combination of uncontrolled randomness, data leakage, overly flexible models, and unreliable SHAP analysis means that:
- Feature importance changes between runs
- Scaling factors are not reproducible
- The same physical question ("what happens if we increase proppant by 500 lb/ft?") can get different answers depending on which random seed was used

The minimum changes needed to make this trustworthy are:
1. Fix seeds everywhere (Priority 1)
2. Eliminate data leakage (Priority 2)
3. Use grouped/stratified splitting (Priority 1)
4. Simplify to XGBoost with monotonic constraints (Priority 3)
5. Measure stability explicitly before trusting any trend (Priority 7)

These changes are implemented in the accompanying code modifications.
