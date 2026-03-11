import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages

from .decline_curve import modified_hyperbolic


def plot_model_performance(history, title):
    """Plot training and validation loss from a Keras history object."""
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ml_performance(y_true, y_pred, title):
    """Scatter plot of true vs predicted values."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{title} - MSE: {mse}, MAE: {mae}, R2: {r2}")

    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()


def plot_decline_curves(time, actual, predicted, title):
    """Plot actual vs predicted decline curves on log scale."""
    plt.figure(figsize=(10, 6))
    plt.plot(time, actual, 'b-o', label='Actual', markersize=2)
    plt.plot(time, predicted, 'r--x', label='Predicted', markersize=2)
    plt.yscale('log')
    plt.ylim(10, 10000000)
    plt.title(title)
    plt.xlabel('Time (Months)')
    plt.ylabel('Production Monthly Volumes (bbls/month)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()


def plot_feature_vs_EUR(combo_test, feature, EUR_values, output_dir,
                        basin, formation, model_type, EUR_type):
    """Plot a feature vs EUR values with trend line and covariance."""
    plt.figure(figsize=(10, 6))

    feature_values = combo_test[feature].values.copy()
    if combo_test[feature].dtype == 'object':
        le = LabelEncoder()
        feature_values = le.fit_transform(feature_values).astype(float)
        feature_label = f'{feature} (encoded)'
    else:
        feature_values = feature_values.astype(float)
        feature_label = feature

    # Filter NaN/Inf and zeros for specific columns
    mask = (~np.isnan(feature_values) & ~np.isnan(EUR_values) &
            ~np.isinf(feature_values) & ~np.isinf(EUR_values))

    filtered_features = feature_values[mask].reshape(-1, 1)
    filtered_EUR = EUR_values[mask]

    if len(filtered_features) > 1:
        model = LinearRegression()
        model.fit(filtered_features, filtered_EUR)
        trendline = model.predict(filtered_features)
        slope = model.coef_[0]
        r2 = model.score(filtered_features, filtered_EUR)

        plt.scatter(filtered_features, filtered_EUR, alpha=0.5, label='Data Points')
        plt.plot(filtered_features, trendline, color='red', linewidth=2, label='Trend line')
        plt.annotate(f'Slope: {slope:.4f}\nR2: {r2:.2f}', xy=(0.05, 0.95),
                     xycoords='axes fraction', fontsize=10,
                     verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.7))
    else:
        plt.text(0.5, 0.5, 'No data points available', ha='center', va='center')

    plt.title(f'{feature_label} vs. EUR\n{basin}-{formation} ({model_type})', fontsize=14)
    plt.xlabel(feature_label, fontsize=12)
    plt.ylabel('EUR', fontsize=12)
    plt.grid(True)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'{feature}_vs_EUR.jpeg')
    plt.savefig(plot_filename, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename


def calculate_covariance(combo_test, EUR_values, feature):
    """Calculate covariance between a feature and EUR values."""
    feature_values = combo_test[feature].values
    if combo_test[feature].dtype == 'object':
        le = LabelEncoder()
        feature_values = le.fit_transform(feature_values)

    # Exclude zeros for certain neighbor columns
    zero_exclusion_cols = [
        'NNAZ_1_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_1_EUR_30yr_Actual_Gas_P50_MMCF',
        'NNAZ_1_Cumulative oil mbo', 'NNAZ_1_Cumulative gas mmcf',
        'NNAZ_1_Cumulative water mbbl',
        'NNAZ_2_EUR_30yr_Actual_Oil_P50_MBO', 'NNAZ_2_EUR_30yr_Actual_Gas_P50_MMCF',
        'NNAZ_2_Cumulative oil mbo', 'NNAZ_2_Cumulative gas mmcf',
        'NNAZ_2_Cumulative water mbbl',
        'NNSZ_1_EUR_30yr_Actual_Oil_P50_MBO', 'NNSZ_1_EUR_30yr_Actual_Gas_P50_MMCF',
        'NNSZ_1_Cumulative oil mbo', 'NNSZ_1_Cumulative gas mmcf',
        'NNSZ_1_Cumulative water mbbl',
        'NNSZ_2_EUR_30yr_Actual_Oil_P50_MBO', 'NNSZ_2_EUR_30yr_Actual_Gas_P50_MMCF',
        'NNSZ_2_Cumulative oil mbo', 'NNSZ_2_Cumulative gas mmcf',
        'NNSZ_2_Cumulative water mbbl',
    ]

    feature_values_float = feature_values.astype(float)
    mask = ~np.isnan(feature_values_float) & ~np.isnan(EUR_values)
    if feature in zero_exclusion_cols:
        mask &= (feature_values_float != 0) & (EUR_values != 0)

    # Exclude placeholder 5280 values for HZDIST columns
    if feature.endswith('_HZDIST'):
        mask &= (feature_values_float != 5280)

    filtered_features = feature_values_float[mask]
    filtered_EUR = EUR_values[mask]

    if len(filtered_features) > 1:
        return np.cov(filtered_features, filtered_EUR)[0, 1]
    return 0.0


def plot_feature_vs_predicted(combo_test, feature, predicted_values, parameter_name,
                              output_dir, actual_values=None):
    """Plot feature vs predicted parameter values with optional actual overlay."""
    plt.figure(figsize=(10, 6))

    if combo_test[feature].dtype == 'object':
        le = LabelEncoder()
        feature_values = le.fit_transform(combo_test[feature].values).astype(float)
    else:
        feature_values = combo_test[feature].values.astype(float)

    min_length = min(len(feature_values), len(predicted_values))
    feature_values = feature_values[:min_length]
    predicted_values = predicted_values[:min_length]

    mask = (~np.isnan(feature_values) & ~np.isnan(predicted_values) &
            ~np.isinf(feature_values) & ~np.isinf(predicted_values))
    feature_values = feature_values[mask]
    predicted_values = predicted_values[mask]

    if len(feature_values) == 0:
        plt.close()
        return

    plt.scatter(feature_values, predicted_values, alpha=0.5, label='Predicted')

    model = LinearRegression()
    model.fit(feature_values.reshape(-1, 1), predicted_values)
    trendline = model.predict(feature_values.reshape(-1, 1))
    plt.plot(feature_values, trendline, color='blue', linewidth=2, label='Predicted Trend')

    if actual_values is not None:
        actual_values = actual_values[:min_length][mask]
        if len(actual_values) > 0:
            plt.scatter(feature_values, actual_values, color='red', alpha=0.5, label='Actual')
            model_act = LinearRegression()
            model_act.fit(feature_values.reshape(-1, 1), actual_values)
            plt.plot(feature_values, model_act.predict(feature_values.reshape(-1, 1)),
                     color='red', linestyle='--', linewidth=2, label='Actual Trend')

    plt.title(f'{feature} vs. Predicted {parameter_name}')
    plt.xlabel(feature)
    plt.ylabel(f'Predicted {parameter_name}')
    plt.grid(True)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'{feature}_vs_{parameter_name}.jpeg')
    plt.savefig(plot_filename, format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename


def plot_learning_curve(model, X, y, cv=5):
    """Plot learning curve for a sklearn model."""
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    plt.figure()
    plt.plot(train_sizes, -train_scores.mean(axis=1), label="Training error")
    plt.plot(train_sizes, -val_scores.mean(axis=1), label="Validation error")
    plt.xlabel("Training set size")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()


def plot_residuals(model, X, y):
    """Plot residuals for a sklearn model."""
    predictions = model.predict(X)
    residuals = y - predictions
    plt.figure()
    plt.scatter(predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def visualize_outlier_removal(df, numerical_columns, pdf_filename='outlier_removal_plots.pdf'):
    """Plot before/after outlier removal histograms and save to PDF."""
    from .feature_engineering import remove_outliers
    df_cleaned, rows_removed = remove_outliers(df, numerical_columns)

    with PdfPages(pdf_filename) as pdf:
        for col in numerical_columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.histplot(df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Before: {col}')
            sns.histplot(df_cleaned[col], kde=True, ax=axes[1])
            axes[1].set_title(f'After: {col}')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    print(f"Total rows removed: {rows_removed}")
    return df_cleaned


def stitch_plots_to_pdf(plot_files, output_pdf_path):
    """Combine multiple plot image files into a single PDF."""
    from fpdf import FPDF
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    pdf = FPDF()
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            pdf.add_page()
            pdf.image(plot_file, x=10, y=10, w=190)
    pdf.output(output_pdf_path)
    print(f"PDF saved to {output_pdf_path}")
