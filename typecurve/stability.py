"""Stability analysis utilities for TypeCurve models.

Run the pipeline multiple times with different random seeds and measure
whether learned trends are reproducible.  This directly answers the
question: "Can this model be trusted for repeatable type-curve scaling?"

Usage::

    from typecurve.stability import run_stability_analysis
    report = run_stability_analysis(df, numerical_columns, categorical_columns,
                                    y_headers, n_runs=5)
    report.print_summary()
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import GroupShuffleSplit

from .config import set_global_seeds, TEST_SIZE, VAL_SPLIT
from .data_preparation import fit_and_apply_scalers
from .models import build_model


class StabilityReport:
    """Container for stability analysis results."""

    def __init__(self):
        self.run_metrics = []          # List of dicts: {seed, mse, mae, ...}
        self.feature_importances = []  # List of arrays (one per run)
        self.feature_names = []
        self.scaling_factors = {}      # {vary_column: [list of factor dicts per run]}
        self.shap_signs = []           # List of sign arrays per run

    def prediction_stability(self):
        """Return coefficient of variation for key metrics across runs."""
        if not self.run_metrics:
            return {}
        metrics_df = pd.DataFrame(self.run_metrics)
        cv = {}
        for col in metrics_df.columns:
            if col == 'seed':
                continue
            vals = metrics_df[col].dropna()
            if len(vals) > 1 and vals.mean() != 0:
                cv[col] = vals.std() / abs(vals.mean())
        return cv

    def feature_importance_rank_correlation(self):
        """Pairwise Kendall's tau between feature importance rankings.

        Returns the mean tau and individual pairwise taus.
        A mean tau > 0.8 indicates stable rankings.
        """
        if len(self.feature_importances) < 2:
            return {'mean_tau': None, 'pairwise': []}

        taus = []
        n = len(self.feature_importances)
        for i in range(n):
            for j in range(i + 1, n):
                rank_i = np.argsort(-np.abs(self.feature_importances[i]))
                rank_j = np.argsort(-np.abs(self.feature_importances[j]))
                tau, _ = kendalltau(rank_i, rank_j)
                taus.append(tau)

        return {'mean_tau': np.mean(taus), 'pairwise': taus}

    def shap_sign_consistency(self):
        """Fraction of features with consistent SHAP sign direction.

        For each feature, check if the mean SHAP value has the same sign
        across all runs.  Returns fraction of features that are consistent.
        """
        if len(self.shap_signs) < 2:
            return None

        signs = np.array(self.shap_signs)  # (n_runs, n_features)
        # For each feature, check if all runs agree on sign
        consistent = np.all(signs == signs[0:1, :], axis=0)
        return np.mean(consistent)

    def print_summary(self):
        """Print a human-readable stability summary."""
        print("=" * 60)
        print("STABILITY ANALYSIS REPORT")
        print("=" * 60)

        n_runs = len(self.run_metrics)
        print(f"\nRuns: {n_runs}")

        cv = self.prediction_stability()
        if cv:
            print("\nPrediction Stability (Coefficient of Variation):")
            for metric, val in cv.items():
                status = "STABLE" if val < 0.05 else "MODERATE" if val < 0.15 else "UNSTABLE"
                print(f"  {metric}: CV = {val:.4f} [{status}]")

        rank_corr = self.feature_importance_rank_correlation()
        if rank_corr['mean_tau'] is not None:
            tau = rank_corr['mean_tau']
            status = "STABLE" if tau > 0.8 else "MODERATE" if tau > 0.6 else "UNSTABLE"
            print(f"\nFeature Importance Rank Correlation (Kendall's tau):")
            print(f"  Mean tau = {tau:.4f} [{status}]")

        sign_cons = self.shap_sign_consistency()
        if sign_cons is not None:
            status = "STABLE" if sign_cons > 0.9 else "MODERATE" if sign_cons > 0.7 else "UNSTABLE"
            print(f"\nSHAP Sign Consistency: {sign_cons:.2%} [{status}]")

        print("\n" + "=" * 60)
        if cv and any(v > 0.15 for v in cv.values()):
            print("VERDICT: Model trends are NOT stable enough for operational use.")
        elif rank_corr.get('mean_tau') is not None and rank_corr['mean_tau'] < 0.6:
            print("VERDICT: Feature importance is NOT stable across runs.")
        else:
            print("VERDICT: Trends appear reasonably stable.")
        print("=" * 60)


def run_stability_analysis(df, numerical_columns, categorical_columns,
                           y_headers, n_runs=5, model_type='xgboost',
                           basin=None, formation=None, seeds=None):
    """Run the model pipeline *n_runs* times with different seeds.

    Parameters
    ----------
    df : DataFrame
        Preprocessed and feature-engineered DataFrame (before split/scale).
    numerical_columns, categorical_columns, y_headers : lists
        Column specifications.
    n_runs : int
        Number of repeated runs.
    model_type : str
        Which model to test ('xgboost', 'random_forest', 'neural_network').
    basin, formation : str or None
        If provided, only train/evaluate on this formation subset.
    seeds : list of int or None
        Explicit seeds to use; defaults to [42, 123, 456, 789, 1024][:n_runs].

    Returns
    -------
    StabilityReport
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999][:n_runs]

    report = StabilityReport()
    report.feature_names = numerical_columns + categorical_columns

    for seed in seeds:
        set_global_seeds(seed)

        # Split with formation-aware grouping
        groups = df[['BasinTC', 'FORMATION_CONDENSED']].astype(str).agg('_'.join, axis=1)
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx].copy()
        test_df_full = df.iloc[test_idx].copy()

        gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=seed)
        groups_rem = groups.iloc[test_idx]
        val_idx, test_idx2 = next(gss2.split(test_df_full, groups=groups_rem))
        val_df = test_df_full.iloc[val_idx].copy()
        test_df = test_df_full.iloc[test_idx2].copy()

        # Scale (training-only fit)
        train_df, val_df, test_df, input_scaler, output_scaler = fit_and_apply_scalers(
            train_df, val_df, test_df, numerical_columns, y_headers)

        # Filter to specific formation if requested
        if basin and formation:
            combo_train = train_df[(train_df['BasinTC'] == basin) &
                                   (train_df['FORMATION_CONDENSED'] == formation)]
            combo_val = val_df[(val_df['BasinTC'] == basin) &
                               (val_df['FORMATION_CONDENSED'] == formation)]
            combo_test = test_df[(test_df['BasinTC'] == basin) &
                                  (test_df['FORMATION_CONDENSED'] == formation)]
        else:
            combo_train = train_df
            combo_val = val_df
            combo_test = test_df

        if len(combo_train) < 10 or len(combo_test) < 5:
            print(f"Seed {seed}: insufficient data (train={len(combo_train)}, "
                  f"test={len(combo_test)}). Skipping.")
            continue

        # Build and train model
        output_size = len(y_headers)
        model = build_model(numerical_columns, categorical_columns, df,
                            output_size, model_type=model_type)

        if model_type == 'xgboost':
            from .callbacks import custom_xgboost_training
            preprocessor = model.named_steps['preprocessor']
            xgb_model = model.named_steps['model']
            custom_xgboost_training(xgb_model, preprocessor, combo_train, combo_val,
                                    numerical_columns, categorical_columns, y_headers)
        elif model_type in ('random_forest', 'decision_tree'):
            combo_train_val = pd.concat([combo_train, combo_val])
            model.fit(combo_train_val[numerical_columns + categorical_columns],
                      combo_train_val[y_headers].values)
        else:
            # Neural network — minimal training for stability check
            import tensorflow as tf
            model.fit(
                x=[combo_train[numerical_columns].values] +
                  [combo_train[col].astype(int).values.reshape(-1, 1)
                   for col in categorical_columns],
                y=combo_train[y_headers].values,
                validation_data=(
                    [combo_val[numerical_columns].values] +
                    [combo_val[col].astype(int).values.reshape(-1, 1)
                     for col in categorical_columns],
                    combo_val[y_headers].values),
                epochs=200, batch_size=50, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True)])

        # Evaluate
        from .evaluation import predict_with_model
        numerical_data = combo_test[numerical_columns].values
        categorical_data = [combo_test[col].astype(int).values.reshape(-1, 1)
                            for col in categorical_columns]
        y_pred = predict_with_model(model, numerical_data, categorical_data,
                                    numerical_columns, categorical_columns)
        y_true = combo_test[y_headers].values

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        report.run_metrics.append({'seed': seed, 'mse': mse, 'mae': mae})

        # Extract feature importance
        try:
            if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'feature_importances_'):
                imp = model.named_steps['model'].feature_importances_
            elif hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            else:
                imp = np.zeros(len(numerical_columns) + len(categorical_columns))
            report.feature_importances.append(imp)
        except Exception:
            pass

        print(f"Seed {seed}: MSE={mse:.6f}, MAE={mae:.6f}")

    return report
