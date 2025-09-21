# regression_diagnostics_mars.py
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# === USER CONFIG ===
input_file = r'C:\Users\agsse\data_analysis\mars_mission_master_dataset_optimized.csv'
output_dir = r'C:\Users\agsse\data_analysis\max_data'
os.makedirs(output_dir, exist_ok=True)
# ===================

print("Loading Mars mission data...")
df = pd.read_csv(input_file)
print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")


def safe_pinv_dot(X):
    """Return pseudo-inverse (stable) for X'X and handle small numerical issues."""
    return np.linalg.pinv(X)


def get_cooks_distance(df):
    results = []

    target_variables = ['temp_avg', 'wind_speed_avg', 'activity_intensity', 'operational_risk_index']

    predictor_sets = {
        'environmental': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability'],
        'comprehensive': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability',
                         'environmental_stress_index', 'operational_tempo_7sol']
    }

    for target in target_variables:
        if target not in df.columns:
            continue

        for pred_set_name, predictors in predictor_sets.items():
            available_predictors = [p for p in predictors if p in df.columns and p != target]

            if len(available_predictors) < 2:
                continue

            model_data = df[[target] + available_predictors].dropna()

            if len(model_data) < 50:
                continue

            try:
                X = model_data[available_predictors].values
                y = model_data[target].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LinearRegression()
                model.fit(X_scaled, y)

                y_pred = model.predict(X_scaled)
                residuals = y - y_pred

                n = len(X_scaled)
                p = X_scaled.shape[1]

                mse = mean_squared_error(y, y_pred)

                # Use pseudo-inverse for numerical stability
                hat_matrix = X_scaled @ safe_pinv_dot(X_scaled.T @ X_scaled) @ X_scaled.T
                leverage = np.clip(np.diag(hat_matrix), 1e-12, 1 - 1e-12)

                standardized_residuals = residuals / np.sqrt(mse * (1 - leverage))

                cooks_d = (standardized_residuals ** 2 / p) * (leverage / (1 - leverage))

                threshold = 4 / n
                high_influence = cooks_d > threshold

                for j, (cook_d, resid, lev, std_resid) in enumerate(zip(cooks_d, residuals, leverage, standardized_residuals)):
                    obs_index = int(model_data.index[j])
                    results.append({
                        'target_variable': target,
                        'predictor_set': pred_set_name,
                        'observation_id': obs_index,
                        'cooks_distance': float(np.round(cook_d, 6)),
                        'residual': float(np.round(resid, 4)),
                        'leverage': float(np.round(lev, 4)),
                        'standardized_residual': float(np.round(std_resid, 4)),
                        'high_influence': bool(cook_d > threshold),
                        'threshold': float(np.round(threshold, 6)),
                        'actual_value': float(np.round(y[j], 4)),
                        'predicted_value': float(np.round(y_pred[j], 4))
                    })

                high_influence_count = int(sum(high_influence))
                print(f"\n{target} ~ {pred_set_name}:")
                print(f"  Observations: {n}")
                print(f"  High influence points (Cook's D > {threshold:.6f}): {high_influence_count}")
                print(f"  Max Cook's distance: {float(np.max(cooks_d)):.6f}")

            except Exception as e:
                print(f"Error calculating Cook's distance for {target} ~ {pred_set_name}: {str(e)}")

    return pd.DataFrame(results)


def get_fitted_values(df):
    results = []

    target_variables = ['temp_avg', 'wind_speed_avg', 'activity_intensity', 'operational_risk_index']

    predictor_sets = {
        'environmental': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability'],
        'operational': ['activity_intensity', 'operational_tempo_7sol'],
        'comprehensive': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability',
                         'environmental_stress_index', 'operational_tempo_7sol']
    }

    for target in target_variables:
        if target not in df.columns:
            continue

        for pred_set_name, predictors in predictor_sets.items():
            available_predictors = [p for p in predictors if p in df.columns and p != target]

            if len(available_predictors) < 2:
                continue

            model_data = df[[target] + available_predictors].dropna()

            if len(model_data) < 30:
                continue

            try:
                X = model_data[available_predictors].values
                y = model_data[target].values
                idx = model_data.index.values

                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    X, y, idx, test_size=0.3, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LinearRegression()
                model.fit(X_train_scaled, y_train)

                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)

                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)

                # Record training data fitted values
                for j, (actual, fitted, obs_idx) in enumerate(zip(y_train, y_pred_train, idx_train)):
                    results.append({
                        'target_variable': target,
                        'predictor_set': pred_set_name,
                        'dataset': 'train',
                        'observation_id': int(obs_idx),
                        'actual_value': float(np.round(actual, 4)),
                        'fitted_value': float(np.round(fitted, 4)),
                        'residual': float(np.round(actual - fitted, 4)),
                        'squared_residual': float(np.round((actual - fitted) ** 2, 6)),
                        'model_r2': float(np.round(train_r2, 4)),
                        'n_predictors': len(available_predictors)
                    })

                # Record test data fitted values
                for j, (actual, fitted, obs_idx) in enumerate(zip(y_test, y_pred_test, idx_test)):
                    results.append({
                        'target_variable': target,
                        'predictor_set': pred_set_name,
                        'dataset': 'test',
                        'observation_id': int(obs_idx),
                        'actual_value': float(np.round(actual, 4)),
                        'fitted_value': float(np.round(fitted, 4)),
                        'residual': float(np.round(actual - fitted, 4)),
                        'squared_residual': float(np.round((actual - fitted) ** 2, 6)),
                        'model_r2': float(np.round(test_r2, 4)),
                        'n_predictors': len(available_predictors)
                    })

                print(f"\n{target} ~ {pred_set_name}:")
                print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                print(f"  Train observations: {len(y_train)}, Test observations: {len(y_test)}")

            except Exception as e:
                print(f"Error calculating fitted values for {target} ~ {pred_set_name}: {str(e)}")

    return pd.DataFrame(results)


def get_residuals(df):
    results = []

    target_variables = ['temp_avg', 'wind_speed_avg', 'activity_intensity', 'operational_risk_index']

    predictor_sets = {
        'environmental': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability'],
        'comprehensive': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability',
                         'environmental_stress_index', 'operational_tempo_7sol']
    }

    for target in target_variables:
        if target not in df.columns:
            continue

        for pred_set_name, predictors in predictor_sets.items():
            available_predictors = [p for p in predictors if p in df.columns and p != target]

            if len(available_predictors) < 2:
                continue

            model_data = df[[target] + available_predictors].dropna()

            if len(model_data) < 30:
                continue

            try:
                X = model_data[available_predictors].values
                y = model_data[target].values
                idx = model_data.index.values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LinearRegression()
                model.fit(X_scaled, y)

                y_pred = model.predict(X_scaled)
                residuals = y - y_pred

                mse = mean_squared_error(y, y_pred)

                hat_matrix = X_scaled @ safe_pinv_dot(X_scaled.T @ X_scaled) @ X_scaled.T
                leverage = np.clip(np.diag(hat_matrix), 1e-12, 1 - 1e-12)

                standardized_residuals = residuals / np.sqrt(mse * (1 - leverage))
                studentized_residuals = standardized_residuals  # simplified; ok for diagnostics

                for j, (resid, std_resid, stud_resid, lev, fitted, actual, obs_idx) in enumerate(
                        zip(residuals, standardized_residuals, studentized_residuals, leverage, y_pred, y, idx)):

                    results.append({
                        'target_variable': target,
                        'predictor_set': pred_set_name,
                        'observation_id': int(obs_idx),
                        'residual': float(np.round(resid, 4)),
                        'standardized_residual': float(np.round(std_resid, 4)),
                        'studentized_residual': float(np.round(stud_resid, 4)),
                        'leverage': float(np.round(lev, 4)),
                        'fitted_value': float(np.round(fitted, 4)),
                        'actual_value': float(np.round(actual, 4)),
                        'abs_residual': float(np.round(abs(resid), 4)),
                        'squared_residual': float(np.round(resid ** 2, 6)),
                        'outlier_candidate': bool(abs(std_resid) > 2.5)
                    })

                outliers = int(sum(np.abs(standardized_residuals) > 2.5))
                print(f"\n{target} ~ {pred_set_name}:")
                print(f"  Observations: {len(residuals)}")
                print(f"  Potential outliers (|std residual| > 2.5): {outliers}")
                print(f"  Mean squared error: {mse:.6f}")

            except Exception as e:
                print(f"Error calculating residuals for {target} ~ {pred_set_name}: {str(e)}")

    return pd.DataFrame(results)


def plot_diagnostic_plots(df):
    plot_data = []

    target_variables = ['temp_avg', 'wind_speed_avg', 'activity_intensity']

    predictors = {
        'temp_avg': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability'],
        'wind_speed_avg': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability'],
        'activity_intensity': ['temp_max', 'temp_min', 'wind_speed_max', 'environmental_stress_index']
    }

    plt.style.use('default')

    for target in target_variables:
        if target not in df.columns:
            continue

        available_predictors = [p for p in predictors[target] if p in df.columns]
        model_data = df[[target] + available_predictors].dropna()

        if len(model_data) < 30:
            continue

        try:
            X = model_data[available_predictors].values
            y = model_data[target].values
            idx = model_data.index.values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            y_pred = model.predict(X_scaled)
            residuals = y - y_pred

            mse = mean_squared_error(y, y_pred)
            hat_matrix = X_scaled @ safe_pinv_dot(X_scaled.T @ X_scaled) @ X_scaled.T
            leverage = np.clip(np.diag(hat_matrix), 1e-12, 1 - 1e-12)
            standardized_residuals = residuals / np.sqrt(mse * (1 - leverage))

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Diagnostic Plots: {target}', fontsize=16, fontweight='bold')

            # Residuals vs fitted
            axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
            axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            axes[0, 0].grid(True, alpha=0.3)

            # QQ plot
            stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            axes[0, 1].grid(True, alpha=0.3)

            # Scale-location
            sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
            axes[1, 0].scatter(y_pred, sqrt_abs_std_residuals, alpha=0.6, s=30)
            axes[1, 0].set_xlabel('Fitted Values')
            axes[1, 0].set_ylabel('√|Standardized Residuals|')
            axes[1, 0].set_title('Scale-Location Plot')
            axes[1, 0].grid(True, alpha=0.3)

            # Residuals vs leverage
            axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.6, s=30)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
            # FIX: use `linestyle` (not `linetype`)
            axes[1, 1].axhline(y=2, color='orange', linestyle=':', linewidth=1, label='±2σ')
            axes[1, 1].axhline(y=-2, color='orange', linestyle=':', linewidth=1)
            axes[1, 1].set_xlabel('Leverage')
            axes[1, 1].set_ylabel('Standardized Residuals')
            axes[1, 1].set_title('Residuals vs Leverage')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'diagnostic_plots_{target}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Diagnostic plots saved: {plot_filename}")

            # QQ theoretical/sample for plot_data
            qq_probs = (np.arange(1, len(standardized_residuals) + 1) - 0.5) / len(standardized_residuals)
            qq_theoretical = stats.norm.ppf(qq_probs)
            qq_sample = np.sort(standardized_residuals)

            for j, (actual, fitted, resid, std_resid, lev, obs_idx) in enumerate(
                    zip(y, y_pred, residuals, standardized_residuals, leverage, idx)):

                plot_data.append({
                    'target_variable': target,
                    'observation_id': int(obs_idx),
                    'actual_value': float(np.round(actual, 4)),
                    'fitted_value': float(np.round(fitted, 4)),
                    'residual': float(np.round(resid, 4)),
                    'standardized_residual': float(np.round(std_resid, 4)),
                    'leverage': float(np.round(lev, 4)),
                    'sqrt_abs_std_residual': float(np.round(np.sqrt(abs(std_resid)), 4)),
                    'qq_theoretical': float(np.round(qq_theoretical[min(j, len(qq_theoretical) - 1)], 4)),
                    'qq_sample': float(np.round(qq_sample[min(j, len(qq_sample) - 1)], 4)),
                    'plot_type': 'diagnostic'
                })

            print(f"\n{target} diagnostic plots:")
            print(f"  Observations: {len(y)}")
            print(f"  R²: {r2_score(y, y_pred):.4f}")
            print(f"  RMSE: {np.sqrt(mse):.4f}")

        except Exception as e:
            print(f"Error creating diagnostic plots for {target}: {str(e)}")

    # Combined residual analysis (robust)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Mars Mission: Residual Analysis Summary', fontsize=16, fontweight='bold')

        all_residuals = []
        all_fitted = []
        all_targets = []

        for target in ['temp_avg', 'wind_speed_avg', 'activity_intensity']:
            target_data = [d for d in plot_data if d['target_variable'] == target]
            if not target_data:
                continue

            residuals = [d['residual'] for d in target_data]
            fitted = [d['fitted_value'] for d in target_data]

            all_residuals.extend(residuals)
            all_fitted.extend(fitted)
            all_targets.extend([target] * len(residuals))

            axes[0, 0].scatter(fitted, residuals, alpha=0.5, label=target, s=20)

        if len(all_residuals) == 0:
            print("No residuals available for combined plot. Skipping combined residual analysis.")
        else:
            axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('All Models: Residuals vs Fitted')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].hist(all_residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of All Residuals')
            axes[0, 1].grid(True, alpha=0.3)

            unique_targets = sorted(set(all_targets))
            box_data = [[r for r, t in zip(all_residuals, all_targets) if t == ut] for ut in unique_targets]
            labels = [str(ut) for ut in unique_targets]

            if len(box_data) == len(labels) and len(box_data) > 0:
                axes[1, 0].boxplot(box_data, labels=labels)
                axes[1, 0].set_ylabel('Residuals')
                axes[1, 0].set_title('Residuals by Target Variable')
                axes[1, 0].grid(True, alpha=0.3)
                plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
            else:
                print("Skipping boxplot (incompatible or empty data). Data lengths:", len(box_data), len(labels))

            axes[1, 1].plot(range(len(all_residuals)), np.abs(all_residuals), 'o-', markersize=2, alpha=0.6)
            axes[1, 1].set_xlabel('Observation Order')
            axes[1, 1].set_ylabel('|Residuals|')
            axes[1, 1].set_title('Residual Magnitude Pattern')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            combined_plot = os.path.join(output_dir, 'combined_residual_analysis.png')
            plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Combined residual analysis plot saved: {combined_plot}")

    except Exception as e:
        print(f"Error creating combined residual plot: {str(e)}")

    return pd.DataFrame(plot_data)


def fit_simple_lm(df):
    results = []

    simple_models = [
        ('temp_avg', 'temp_max'),
        ('temp_avg', 'temp_min'),
        ('wind_speed_avg', 'wind_speed_max'),
        ('activity_intensity', 'operational_tempo_7sol'),
        ('operational_risk_index', 'environmental_stress_index'),
        ('dust_storm_probability', 'temp_avg'),
        ('environmental_stress_index', 'wind_speed_avg')
    ]

    for target, predictor in simple_models:
        if target not in df.columns or predictor not in df.columns:
            continue

        model_data = df[[target, predictor]].dropna()

        if len(model_data) < 20:
            continue

        try:
            X = model_data[[predictor]].values
            y = model_data[target].values
            idx = model_data.index.values

            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, idx, test_size=0.3, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            n = len(X_train)
            adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - 2) if n > 2 else train_r2

            slope = float(model.coef_[0])
            intercept = float(model.intercept_)

            residuals = y_train - y_pred_train
            mse = mean_squared_error(y_train, y_pred_train)

            X_mean = X_train.mean()
            ss_xx = ((X_train.flatten() - X_mean) ** 2).sum() if n > 1 else 0.0

            if ss_xx <= 0:
                se_slope = np.nan
                se_intercept = np.nan
            else:
                se_slope = np.sqrt(mse / ss_xx)
                se_intercept = np.sqrt(mse * (1 / n + X_mean ** 2 / ss_xx))

            t_stat_slope = slope / se_slope if se_slope and not np.isnan(se_slope) else np.nan
            t_stat_intercept = intercept / se_intercept if se_intercept and not np.isnan(se_intercept) else np.nan

            p_value_slope = 2 * (1 - stats.t.cdf(abs(t_stat_slope), n - 2)) if not np.isnan(t_stat_slope) else np.nan
            p_value_intercept = 2 * (1 - stats.t.cdf(abs(t_stat_intercept), n - 2)) if not np.isnan(t_stat_intercept) else np.nan

            f_statistic = train_r2 / ((1 - train_r2) / (n - 2)) if n > 2 and (1 - train_r2) != 0 else np.nan
            f_p_value = 1 - stats.f.cdf(f_statistic, 1, n - 2) if not np.isnan(f_statistic) else np.nan

            # Plot simple regression diagnostics
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Simple Linear Regression: {target} ~ {predictor}', fontsize=14, fontweight='bold')

            axes[0].scatter(X_train, y_train, alpha=0.6, s=30, label='Train')
            axes[0].scatter(X_test, y_test, alpha=0.6, s=30, label='Test')

            x_min, x_max = X.min(), X.max()
            x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            y_range_pred = model.predict(x_range)
            axes[0].plot(x_range, y_range_pred, 'g-', linewidth=2, label=f'y = {intercept:.3f} + {slope:.3f}x')

            axes[0].set_xlabel(predictor)
            axes[0].set_ylabel(target)
            axes[0].set_title('Data and Fitted Line')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(y_pred_train, residuals, alpha=0.6, s=30)
            axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xlabel('Fitted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residuals vs Fitted')
            axes[1].grid(True, alpha=0.3)

            stats.probplot(residuals, dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot of Residuals')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            simple_plot_file = os.path.join(output_dir, f'simple_regression_{target}_{predictor}.png')
            plt.savefig(simple_plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Simple regression plot saved: {simple_plot_file}")

            results.append({
                'target_variable': target,
                'predictor_variable': predictor,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'intercept': float(np.round(intercept, 4)),
                'slope': float(np.round(slope, 4)),
                'se_intercept': float(np.round(se_intercept, 4)) if not np.isnan(se_intercept) else np.nan,
                'se_slope': float(np.round(se_slope, 4)) if not np.isnan(se_slope) else np.nan,
                't_stat_intercept': float(np.round(t_stat_intercept, 4)) if not np.isnan(t_stat_intercept) else np.nan,
                't_stat_slope': float(np.round(t_stat_slope, 4)) if not np.isnan(t_stat_slope) else np.nan,
                'p_value_intercept': float(np.round(p_value_intercept, 6)) if not np.isnan(p_value_intercept) else np.nan,
                'p_value_slope': float(np.round(p_value_slope, 6)) if not np.isnan(p_value_slope) else np.nan,
                'train_r2': float(np.round(train_r2, 4)),
                'test_r2': float(np.round(test_r2, 4)),
                'adjusted_r2': float(np.round(adjusted_r2, 4)) if not np.isnan(adjusted_r2) else np.nan,
                'train_rmse': float(np.round(train_rmse, 4)),
                'test_rmse': float(np.round(test_rmse, 4)),
                'f_statistic': float(np.round(f_statistic, 4)) if not np.isnan(f_statistic) else np.nan,
                'f_p_value': float(np.round(f_p_value, 6)) if not np.isnan(f_p_value) else np.nan,
                'significant_at_0.05': bool(p_value_slope < 0.05) if not np.isnan(p_value_slope) else False,
                'equation': f"y = {intercept:.4f} + {slope:.4f}*x"
            })

            print(f"\n{target} ~ {predictor}:")
            print(f"  Equation: y = {intercept:.4f} + {slope:.4f}*x")
            print(f"  R²: {train_r2:.4f} (train), {test_r2:.4f} (test)")
            print(f"  P-value (slope): {p_value_slope if not np.isnan(p_value_slope) else 'nan'}")
            print(f"  Significant: {'Yes' if (not np.isnan(p_value_slope) and p_value_slope < 0.05) else 'No'}")

        except Exception as e:
            print(f"Error fitting simple linear model {target} ~ {predictor}: {str(e)}")

    return pd.DataFrame(results)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("COOK'S DISTANCE ANALYSIS")
    print("=" * 60)
    cooks_results = get_cooks_distance(df)

    print("\n" + "=" * 60)
    print("FITTED VALUES ANALYSIS")
    print("=" * 60)
    fitted_results = get_fitted_values(df)

    print("\n" + "=" * 60)
    print("RESIDUALS ANALYSIS")
    print("=" * 60)
    residuals_results = get_residuals(df)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC PLOTS DATA")
    print("=" * 60)
    diagnostic_plots_data = plot_diagnostic_plots(df)

    print("\n" + "=" * 60)
    print("SIMPLE LINEAR REGRESSION MODELS")
    print("=" * 60)
    simple_lm_results = fit_simple_lm(df)

    # Cook's distance plots (if any)
    try:
        if not cooks_results.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Cook's Distance Analysis", fontsize=16, fontweight='bold')

            # plot two main targets (if present)
            targets_to_plot = ['temp_avg', 'wind_speed_avg']
            for i, target in enumerate(targets_to_plot):
                target_data = cooks_results[cooks_results['target_variable'] == target]
                if target_data.empty:
                    continue

                row = i // 2
                col = i % 2

                obs_ids = target_data['observation_id'].values
                cook_distances = target_data['cooks_distance'].values
                threshold = float(target_data['threshold'].iloc[0]) if 'threshold' in target_data.columns and not target_data['threshold'].empty else 0.0

                colors = ['red' if cd > threshold else 'blue' for cd in cook_distances]

                axes[row, col].scatter(obs_ids, cook_distances, c=colors, alpha=0.6, s=30)
                if threshold > 0:
                    axes[row, col].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.6f}')
                axes[row, col].set_xlabel('Observation ID')
                axes[row, col].set_ylabel("Cook's Distance")
                axes[row, col].set_title(f"{target}: Cook's Distance")
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

            # Distribution plot
            all_cook_d = cooks_results['cooks_distance'].values if 'cooks_distance' in cooks_results.columns else np.array([])
            if all_cook_d.size > 0:
                axes[1, 0].hist(all_cook_d, bins=50, alpha=0.7, edgecolor='black')
                axes[1, 0].axvline(x=np.mean(all_cook_d), color='red', linestyle='--', label=f'Mean = {np.mean(all_cook_d):.6f}')
                axes[1, 0].set_xlabel("Cook's Distance")
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title("Distribution of Cook's Distances")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Leverage vs standardized residuals plot
            if not cooks_results.empty and {'leverage', 'standardized_residual', 'cooks_distance'}.issubset(cooks_results.columns):
                sc = axes[1, 1].scatter(cooks_results['leverage'], cooks_results['standardized_residual'],
                                        c=cooks_results['cooks_distance'], cmap='viridis', alpha=0.6, s=30)
                axes[1, 1].set_xlabel('Leverage')
                axes[1, 1].set_ylabel('Standardized Residual')
                axes[1, 1].set_title('Leverage vs Standardized Residuals')
                plt.colorbar(sc, ax=axes[1, 1], label="Cook's Distance")
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            cooks_plot_file = os.path.join(output_dir, 'cooks_distance_analysis.png')
            plt.savefig(cooks_plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Cook's distance plots saved: {cooks_plot_file}")

    except Exception as e:
        print(f"Error creating Cook's distance plots: {str(e)}")

    # Fitted values comparison plots
    try:
        if not fitted_results.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Fitted Values Analysis', fontsize=16, fontweight='bold')

            best_models = fitted_results.groupby(['target_variable', 'predictor_set'])['model_r2'].first().nlargest(4)

            for i, (model_id, r2) in enumerate(best_models.items()):
                if i >= 4:
                    break

                target, pred_set = model_id
                model_data = fitted_results[
                    (fitted_results['target_variable'] == target) &
                    (fitted_results['predictor_set'] == pred_set)
                ]

                row = i // 2
                col = i % 2

                train_data = model_data[model_data['dataset'] == 'train']
                test_data = model_data[model_data['dataset'] == 'test']

                if not train_data.empty:
                    axes[row, col].scatter(train_data['fitted_value'], train_data['actual_value'],
                                           alpha=0.6, label='Train', s=20)
                if not test_data.empty:
                    axes[row, col].scatter(test_data['fitted_value'], test_data['actual_value'],
                                           alpha=0.6, label='Test', s=20)

                try:
                    min_val = min(model_data['fitted_value'].min(), model_data['actual_value'].min())
                    max_val = max(model_data['fitted_value'].max(), model_data['actual_value'].max())
                    axes[row, col].plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect Fit')
                except Exception:
                    pass

                axes[row, col].set_xlabel('Fitted Values')
                axes[row, col].set_ylabel('Actual Values')
                axes[row, col].set_title(f'{target} ~ {pred_set} (R² = {r2:.3f})')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

            plt.tight_layout()
            fitted_plot_file = os.path.join(output_dir, 'fitted_values_analysis.png')
            plt.savefig(fitted_plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Fitted values plots saved: {fitted_plot_file}")

    except Exception as e:
        print(f"Error creating fitted values plots: {str(e)}")

    # === SAVE RESULTS & SUMMARY ===
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    if not cooks_results.empty:
        cooks_file = os.path.join(output_dir, 'cooks_distance_results.csv')
        cooks_results.to_csv(cooks_file, index=False)
        print(f"Cook's distance results saved to: {cooks_file}")

    if not fitted_results.empty:
        fitted_file = os.path.join(output_dir, 'fitted_values_results.csv')
        fitted_results.to_csv(fitted_file, index=False)
        print(f"Fitted values results saved to: {fitted_file}")

    if not residuals_results.empty:
        residuals_file = os.path.join(output_dir, 'residuals_analysis_results.csv')
        residuals_results.to_csv(residuals_file, index=False)
        print(f"Residuals analysis results saved to: {residuals_file}")

    if not diagnostic_plots_data.empty:
        plots_file = os.path.join(output_dir, 'diagnostic_plots_data.csv')
        diagnostic_plots_data.to_csv(plots_file, index=False)
        print(f"Diagnostic plots data saved to: {plots_file}")

    if not simple_lm_results.empty:
        simple_file = os.path.join(output_dir, 'simple_linear_regression_results.csv')
        simple_lm_results.to_csv(simple_file, index=False)
        print(f"Simple linear regression results saved to: {simple_file}")

    summary_file = os.path.join(output_dir, 'regression_diagnostics_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MARS MISSION DATA - REGRESSION DIAGNOSTICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET OVERVIEW:\n")
        f.write(f"- Total observations: {len(df)}\n")
        f.write(f"- Total variables: {len(df.columns)}\n\n")

        if not cooks_results.empty:
            high_influence = int(sum(cooks_results['high_influence']))
            f.write("COOK'S DISTANCE ANALYSIS:\n")
            f.write(f"- Total observations analyzed: {len(cooks_results)}\n")
            f.write(f"- High influence points detected: {high_influence}\n")
            f.write(f"- Max Cook's distance: {float(cooks_results['cooks_distance'].max()):.6f}\n\n")

        if not fitted_results.empty:
            best_r2 = float(fitted_results['model_r2'].max())
            f.write("FITTED VALUES ANALYSIS:\n")
            f.write(f"- Model fits analyzed: {len(fitted_results.groupby(['target_variable', 'predictor_set']))}\n")
            f.write(f"- Best R² achieved: {best_r2:.4f}\n")
            f.write(f"- Total fitted values: {len(fitted_results)}\n\n")

        if not residuals_results.empty:
            outliers = int(sum(residuals_results['outlier_candidate']))
            f.write("RESIDUALS ANALYSIS:\n")
            f.write(f"- Total residuals analyzed: {len(residuals_results)}\n")
            f.write(f"- Potential outliers detected: {outliers}\n")
            f.write(f"- Max absolute residual: {float(residuals_results['abs_residual'].max()):.4f}\n\n")

        if not simple_lm_results.empty:
            significant_models = int(sum(simple_lm_results['significant_at_0.05']))
            f.write("SIMPLE LINEAR REGRESSION:\n")
            f.write(f"- Simple models fitted: {len(simple_lm_results)}\n")
            f.write(f"- Significant models (p < 0.05): {significant_models}\n")
            f.write(f"- Best test R²: {float(simple_lm_results['test_r2'].max()):.4f}\n\n")

        f.write("PLOTS CREATED:\n")
        f.write("- Individual diagnostic plots for each target variable\n")
        f.write("- Combined residual analysis summary plot\n")
        f.write("- Cook's distance analysis plots\n")
        f.write("- Fitted values comparison plots\n")
        f.write("- Simple regression plots with diagnostics\n\n")

        f.write("DIAGNOSTIC FILES CREATED:\n")
        f.write("- cooks_distance_results.csv: Influence analysis\n")
        f.write("- fitted_values_results.csv: Model predictions\n")
        f.write("- residuals_analysis_results.csv: Residual diagnostics\n")
        f.write("- diagnostic_plots_data.csv: Plot data for diagnostics\n")
        f.write("- simple_linear_regression_results.csv: Simple model results\n\n")

        f.write("INTERPRETATION NOTES:\n")
        f.write("- Cook's distance > 4/n indicates high influence observations\n")
        f.write("- Standardized residuals > 2.5 suggest potential outliers\n")
        f.write("- Use diagnostic plots to check model assumptions\n")
        f.write("- Compare train vs test R² to assess overfitting\n")

    print(f"Summary report saved to: {summary_file}")

    print("\n" + "=" * 60)
    print("REGRESSION DIAGNOSTICS COMPLETE!")
    print("=" * 60)
    print(f"All results saved to: {output_dir}")
    print("Files created:")
    if not cooks_results.empty:
        print(f"  - cooks_distance_results.csv ({len(cooks_results)} observations)")
    if not fitted_results.empty:
        print(f"  - fitted_values_results.csv ({len(fitted_results)} fitted values)")
    if not residuals_results.empty:
        print(f"  - residuals_analysis_results.csv ({len(residuals_results)} residuals)")
    if not diagnostic_plots_data.empty:
        print(f"  - diagnostic_plots_data.csv ({len(diagnostic_plots_data)} plot points)")
    if not simple_lm_results.empty:
        print(f"  - simple_linear_regression_results.csv ({len(simple_lm_results)} models)")
    print(f"  - regression_diagnostics_summary.txt")
    print("\nPlots created:")
    print(f"  - diagnostic_plots_[target].png for each target variable")
    print(f"  - combined_residual_analysis.png")
    print(f"  - cooks_distance_analysis.png")
    print(f"  - fitted_values_analysis.png")
    print(f"  - simple_regression_[target]_[predictor].png for each simple model")
