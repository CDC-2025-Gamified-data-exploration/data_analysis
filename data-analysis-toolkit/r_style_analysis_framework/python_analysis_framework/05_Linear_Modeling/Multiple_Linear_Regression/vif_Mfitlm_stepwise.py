import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

input_file = r'C:\Users\agsse\data_analysis\mars_mission_master_dataset_optimized.csv'
output_dir = r'C:\Users\agsse\data_analysis\max_data'

os.makedirs(output_dir, exist_ok=True)

print("Loading Mars mission data...")
df = pd.read_csv(input_file)
print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

def calculate_vif(df, features):
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    vif_results = []
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        statsmodels_available = True
    except ImportError:
        print("Warning: statsmodels not available. Using correlation-based multicollinearity detection.")
        statsmodels_available = False
    
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) < 2:
        print("Not enough numeric features for VIF calculation")
        return pd.DataFrame()
    
    df_clean = df[numeric_features].dropna()
    
    if len(df_clean) < 10:
        print("Not enough observations after removing missing values")
        return pd.DataFrame()
    
    if statsmodels_available:
        try:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
            
            for i, feature in enumerate(df_scaled.columns):
                vif_value = variance_inflation_factor(df_scaled.values, i)
                vif_results.append({
                    'feature': feature,
                    'vif': round(vif_value, 4),
                    'multicollinearity': 'High' if vif_value > 10 else 'Moderate' if vif_value > 5 else 'Low'
                })
                
            print(f"\nVIF Analysis (using statsmodels):")
            for result in vif_results:
                print(f"  {result['feature']}: VIF = {result['vif']} ({result['multicollinearity']})")
                
        except Exception as e:
            print(f"Error calculating VIF with statsmodels: {str(e)}")
            statsmodels_available = False
    
    if not statsmodels_available:
        # Fallback: Use correlation matrix for multicollinearity detection
        try:
            corr_matrix = df_clean.corr().abs()
            
            for feature in numeric_features:
                # Get correlations with other features (excluding self-correlation)
                feature_corrs = corr_matrix[feature].drop(feature)
                max_corr = feature_corrs.max() if len(feature_corrs) > 0 else 0
                
                # Approximate VIF using correlation (VIF ≈ 1/(1-R²))
                approx_vif = 1 / (1 - max_corr**2) if max_corr < 0.99 else 999
                
                vif_results.append({
                    'feature': feature,
                    'vif': round(approx_vif, 4),
                    'max_correlation': round(max_corr, 4),
                    'multicollinearity': 'High' if approx_vif > 10 else 'Moderate' if approx_vif > 5 else 'Low'
                })
            
            print(f"\nMulticollinearity Analysis (correlation-based approximation):")
            for result in vif_results:
                print(f"  {result['feature']}: Approx VIF = {result['vif']} (Max Corr = {result.get('max_correlation', 'N/A')}) ({result['multicollinearity']})")
                
        except Exception as e:
            print(f"Error in correlation-based multicollinearity detection: {str(e)}")
        
    return pd.DataFrame(vif_results)

def fit_multiple_linear_models(df):
    """Fit multiple linear regression models with different target variables"""
    results = []
    
    target_variables = [
        'temp_avg', 'wind_speed_avg', 'activity_intensity', 
        'operational_tempo_7sol', 'environmental_stress_index', 'operational_risk_index'
    ]
    
    predictor_sets = {
        'environmental': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability'],
        'operational': ['activity_intensity', 'operational_tempo_7sol'],
        'risk_based': ['environmental_stress_index', 'operational_risk_index'],
        'comprehensive': ['temp_max', 'temp_min', 'wind_speed_max', 'dust_storm_probability', 
                         'activity_intensity', 'operational_tempo_7sol', 'environmental_stress_index']
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
                X = model_data[available_predictors]
                y = model_data[target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                n = len(X_train)
                k = len(available_predictors)
                adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - k - 1)
                
                f_statistic = (train_r2 / k) / ((1 - train_r2) / (n - k - 1))
                f_p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
                
                coefficients = dict(zip(available_predictors, model.coef_))
                
                results.append({
                    'target_variable': target,
                    'predictor_set': pred_set_name,
                    'n_predictors': len(available_predictors),
                    'n_observations': len(model_data),
                    'train_r2': round(train_r2, 4),
                    'test_r2': round(test_r2, 4),
                    'adjusted_r2': round(adjusted_r2, 4),
                    'train_rmse': round(train_rmse, 4),
                    'test_rmse': round(test_rmse, 4),
                    'f_statistic': round(f_statistic, 4),
                    'f_p_value': round(f_p_value, 6),
                    'intercept': round(model.intercept_, 4),
                    'coefficients': {k: round(v, 4) for k, v in coefficients.items()},
                    'predictors_used': available_predictors
                })
                
                print(f"\n{target} ~ {pred_set_name} predictors:")
                print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                print(f"  Adjusted R²: {adjusted_r2:.4f}")
                print(f"  F-statistic: {f_statistic:.4f} (p = {f_p_value:.6f})")
                
            except Exception as e:
                print(f"Error modeling {target} with {pred_set_name}: {str(e)}")
                
    return pd.DataFrame(results)

def perform_stepwise_selection(df):
    """Perform forward and backward stepwise feature selection"""
    results = []
    
    target_variables = ['temp_avg', 'activity_intensity', 'operational_risk_index']
    
    all_predictors = [
        'temp_max', 'temp_min', 'wind_speed_avg', 'wind_speed_max',
        'dust_storm_probability', 'environmental_stress_index', 'operational_tempo_7sol'
    ]
    
    def forward_selection(X, y, significance_level=0.05):
        initial_features = []
        best_features = []
        
        while len(initial_features) < len(X.columns):
            remaining_features = list(set(X.columns) - set(initial_features))
            new_pval = pd.Series(index=remaining_features, dtype=float)
            
            for new_column in remaining_features:
                test_features = initial_features + [new_column]
                if len(test_features) >= len(X):
                    continue
                    
                try:
                    X_test = X[test_features]
                    model = LinearRegression().fit(X_test, y)
                    y_pred = model.predict(X_test)
                    
                    n = len(X_test)
                    k = len(test_features)
                    mse = mean_squared_error(y, y_pred)
                    
                    if k < n - 1 and mse > 0:
                        f_stat = (r2_score(y, y_pred) / k) / ((1 - r2_score(y, y_pred)) / (n - k - 1))
                        p_val = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                        new_pval[new_column] = p_val
                except:
                    continue
            
            if new_pval.empty:
                break
                
            min_p_value = new_pval.min()
            if min_p_value < significance_level:
                best_feature = new_pval.idxmin()
                initial_features.append(best_feature)
                best_features = initial_features.copy()
            else:
                break
                
        return best_features
    
    def backward_elimination(X, y, significance_level=0.05):
        features = list(X.columns)
        
        while len(features) > 1:
            X_test = X[features]
            try:
                model = LinearRegression().fit(X_test, y)
                y_pred = model.predict(X_test)
                
                n = len(X_test)
                k = len(features)
                
                if k >= n - 1:
                    features = features[:-1]
                    continue
                
                r2 = r2_score(y, y_pred)
                if r2 <= 0 or r2 >= 1:
                    features = features[:-1]
                    continue
                    
                f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
                p_val = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                
                if p_val > significance_level:
                    worst_feature_idx = np.random.randint(0, len(features))
                    features.pop(worst_feature_idx)
                else:
                    break
                    
            except:
                if features:
                    features = features[:-1]
                else:
                    break
                    
        return features
    
    for target in target_variables:
        if target not in df.columns:
            continue
            
        available_predictors = [p for p in all_predictors if p in df.columns and p != target]
        
        if len(available_predictors) < 3:
            continue
            
        model_data = df[[target] + available_predictors].dropna()
        
        if len(model_data) < 50:
            continue
            
        try:
            X = model_data[available_predictors]
            y = model_data[target]
            
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            forward_features = forward_selection(X_scaled, y)
            backward_features = backward_elimination(X_scaled, y)
            
            for method, selected_features in [('forward', forward_features), ('backward', backward_features)]:
                if not selected_features:
                    continue
                    
                X_selected = X_scaled[selected_features]
                X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                n = len(X_train)
                k = len(selected_features)
                adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - k - 1) if k < n else 0
                
                results.append({
                    'target_variable': target,
                    'selection_method': method,
                    'selected_features': selected_features,
                    'n_features_selected': len(selected_features),
                    'n_features_original': len(available_predictors),
                    'train_r2': round(train_r2, 4),
                    'test_r2': round(test_r2, 4),
                    'adjusted_r2': round(adjusted_r2, 4),
                    'feature_reduction': round((len(available_predictors) - len(selected_features)) / len(available_predictors) * 100, 1)
                })
                
                print(f"\n{target} - {method} selection:")
                print(f"  Selected {len(selected_features)}/{len(available_predictors)} features")
                print(f"  Features: {selected_features}")
                print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                
        except Exception as e:
            print(f"Error in stepwise selection for {target}: {str(e)}")
            
    return pd.DataFrame(results)

print("\n" + "="*60)
print("VARIANCE INFLATION FACTOR ANALYSIS")
print("="*60)

key_features = [
    'temp_avg', 'temp_max', 'temp_min', 'wind_speed_avg', 'wind_speed_max',
    'activity_intensity', 'operational_tempo_7sol', 'dust_storm_probability',
    'environmental_stress_index', 'operational_risk_index'
]

vif_results = calculate_vif(df, key_features)

print("\n" + "="*60)
print("MULTIPLE LINEAR REGRESSION MODELS")
print("="*60)

regression_results = fit_multiple_linear_models(df)

print("\n" + "="*60)
print("STEPWISE FEATURE SELECTION")
print("="*60)

stepwise_results = perform_stepwise_selection(df)

print("\n" + "="*60)
print("ADDITIONAL MODEL COMPARISONS")
print("="*60)

def compare_model_types(df):
    """Compare Linear Regression with Random Forest"""
    comparison_results = []
    
    target_vars = ['temp_avg', 'activity_intensity', 'operational_risk_index']
    predictors = ['temp_max', 'temp_min', 'wind_speed_avg', 'dust_storm_probability', 
                 'environmental_stress_index', 'operational_tempo_7sol']
    
    for target in target_vars:
        if target not in df.columns:
            continue
            
        available_preds = [p for p in predictors if p in df.columns and p != target]
        model_data = df[[target] + available_preds].dropna()
        
        if len(model_data) < 50 or len(available_preds) < 3:
            continue
            
        try:
            X = model_data[available_preds]
            y = model_data[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_r2 = r2_score(y_test, lr_pred)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            comparison_results.append({
                'target_variable': target,
                'linear_regression_r2': round(lr_r2, 4),
                'random_forest_r2': round(rf_r2, 4),
                'linear_regression_rmse': round(lr_rmse, 4),
                'random_forest_rmse': round(rf_rmse, 4),
                'best_model': 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression',
                'r2_improvement': round(abs(rf_r2 - lr_r2), 4)
            })
            
            print(f"\n{target} Model Comparison:")
            print(f"  Linear Regression - R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
            print(f"  Random Forest     - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
            print(f"  Best Model: {comparison_results[-1]['best_model']}")
            
        except Exception as e:
            print(f"Error comparing models for {target}: {str(e)}")
            
    return pd.DataFrame(comparison_results)

model_comparison = compare_model_types(df)

print("\n" + "="*60)
print("GENERATING PLOTS AND VISUALIZATIONS")
print("="*60)

def create_regression_plots(df, regression_results):
    """Create diagnostic plots for regression models"""
    plot_data = []
    
    # Select best performing models for plotting
    best_models = regression_results.nlargest(5, 'test_r2')
    
    for idx, model_info in best_models.iterrows():
        target = model_info['target_variable']
        pred_set = model_info['predictor_set']
        predictors = model_info['predictors_used']
        
        if target not in df.columns:
            continue
            
        available_predictors = [p for p in predictors if p in df.columns]
        model_data = df[[target] + available_predictors].dropna()
        
        if len(model_data) < 50:
            continue
            
        try:
            X = model_data[available_predictors]
            y = model_data[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Store data for plotting
            for i, (actual, predicted) in enumerate(zip(y_train, y_pred_train)):
                plot_data.append({
                    'model_id': f"{target}_{pred_set}",
                    'target_variable': target,
                    'predictor_set': pred_set,
                    'dataset': 'train',
                    'actual_value': actual,
                    'predicted_value': predicted,
                    'residual': actual - predicted,
                    'observation_id': i
                })
            
            for i, (actual, predicted) in enumerate(zip(y_test, y_pred_test)):
                plot_data.append({
                    'model_id': f"{target}_{pred_set}",
                    'target_variable': target,
                    'predictor_set': pred_set,
                    'dataset': 'test',
                    'actual_value': actual,
                    'predicted_value': predicted,
                    'residual': actual - predicted,
                    'observation_id': i + len(y_train)
                })
                
        except Exception as e:
            print(f"Error preparing plot data for {target}_{pred_set}: {str(e)}")
    
    return pd.DataFrame(plot_data)

def create_feature_importance_data(df, stepwise_results):
    """Create feature importance data for stepwise selection results"""
    importance_data = []
    
    for idx, result in stepwise_results.iterrows():
        target = result['target_variable']
        method = result['selection_method']
        selected_features = result['selected_features']
        
        if not selected_features or target not in df.columns:
            continue
            
        available_features = [f for f in selected_features if f in df.columns]
        model_data = df[[target] + available_features].dropna()
        
        if len(model_data) < 30:
            continue
            
        try:
            X = model_data[available_features]
            y = model_data[target]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Calculate feature importance (absolute coefficients)
            feature_importance = np.abs(model.coef_)
            
            for feature, importance in zip(available_features, feature_importance):
                importance_data.append({
                    'target_variable': target,
                    'selection_method': method,
                    'feature': feature,
                    'importance': importance,
                    'coefficient': model.coef_[available_features.index(feature)],
                    'rank': len(feature_importance) - stats.rankdata(feature_importance)[available_features.index(feature)] + 1
                })
                
        except Exception as e:
            print(f"Error calculating feature importance for {target}_{method}: {str(e)}")
    
    return pd.DataFrame(importance_data)

def create_model_performance_summary(regression_results, model_comparison):
    """Create summary data for model performance visualization"""
    performance_data = []
    
    # Regression model performance
    for idx, result in regression_results.iterrows():
        performance_data.append({
            'model_type': 'Linear Regression',
            'model_id': f"{result['target_variable']}_{result['predictor_set']}",
            'target_variable': result['target_variable'],
            'predictor_set': result['predictor_set'],
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'adjusted_r2': result['adjusted_r2'],
            'rmse': result.get('test_rmse', 0),
            'n_predictors': result['n_predictors'],
            'n_observations': result['n_observations']
        })
    
    # Model comparison data
    for idx, result in model_comparison.iterrows():
        target = result['target_variable']
        
        performance_data.append({
            'model_type': 'Linear Regression',
            'model_id': f"{target}_comparison",
            'target_variable': target,
            'predictor_set': 'comparison_set',
            'train_r2': None,
            'test_r2': result['linear_regression_r2'],
            'adjusted_r2': None,
            'rmse': result['linear_regression_rmse'],
            'n_predictors': None,
            'n_observations': None
        })
        
        performance_data.append({
            'model_type': 'Random Forest',
            'model_id': f"{target}_comparison",
            'target_variable': target,
            'predictor_set': 'comparison_set',
            'train_r2': None,
            'test_r2': result['random_forest_r2'],
            'adjusted_r2': None,
            'rmse': result['random_forest_rmse'],
            'n_predictors': None,
            'n_observations': None
        })
    
    return pd.DataFrame(performance_data)

def create_vif_visualization_data(vif_results):
    """Prepare VIF data for visualization"""
    if vif_results.empty:
        return pd.DataFrame()
        
    viz_data = []
    for idx, result in vif_results.iterrows():
        viz_data.append({
            'feature': result['feature'],
            'vif_value': result['vif'],
            'multicollinearity_level': result['multicollinearity'],
            'vif_log': np.log10(max(result['vif'], 0.1)),  # Log scale for better visualization
            'above_threshold_5': result['vif'] > 5,
            'above_threshold_10': result['vif'] > 10
        })
    
    return pd.DataFrame(viz_data)

# Generate all visualization data
print("Creating regression diagnostic data...")
regression_plot_data = create_regression_plots(df, regression_results) if not regression_results.empty else pd.DataFrame()

print("Creating feature importance data...")
feature_importance_data = create_feature_importance_data(df, stepwise_results) if not stepwise_results.empty else pd.DataFrame()

print("Creating model performance summary...")
model_performance_data = create_model_performance_summary(regression_results, model_comparison)

print("Creating VIF visualization data...")
vif_viz_data = create_vif_visualization_data(vif_results) if not vif_results.empty else pd.DataFrame()

print("\n" + "="*60)
print("SAVING RESULTS AND VISUALIZATION DATA")
print("="*60)

if not vif_results.empty:
    vif_file = os.path.join(output_dir, 'vif_analysis_results.csv')
    vif_results.to_csv(vif_file, index=False)
    print(f"VIF analysis saved to: {vif_file}")

if not regression_results.empty:
    regression_file = os.path.join(output_dir, 'multiple_regression_results.csv')
    regression_results.to_csv(regression_file, index=False)
    print(f"Multiple regression results saved to: {regression_file}")

if not stepwise_results.empty:
    stepwise_file = os.path.join(output_dir, 'stepwise_selection_results.csv')
    stepwise_results.to_csv(stepwise_file, index=False)
    print(f"Stepwise selection results saved to: {stepwise_file}")

if not model_comparison.empty:
    comparison_file = os.path.join(output_dir, 'model_comparison_results.csv')
    model_comparison.to_csv(comparison_file, index=False)
    print(f"Model comparison results saved to: {comparison_file}")

# Save visualization data
if not regression_plot_data.empty:
    plot_data_file = os.path.join(output_dir, 'regression_plot_data.csv')
    regression_plot_data.to_csv(plot_data_file, index=False)
    print(f"Regression plot data saved to: {plot_data_file}")

if not feature_importance_data.empty:
    importance_file = os.path.join(output_dir, 'feature_importance_data.csv')
    feature_importance_data.to_csv(importance_file, index=False)
    print(f"Feature importance data saved to: {importance_file}")

if not model_performance_data.empty:
    performance_file = os.path.join(output_dir, 'model_performance_data.csv')
    model_performance_data.to_csv(performance_file, index=False)
    print(f"Model performance data saved to: {performance_file}")

if not vif_viz_data.empty:
    vif_viz_file = os.path.join(output_dir, 'vif_visualization_data.csv')
    vif_viz_data.to_csv(vif_viz_file, index=False)
    print(f"VIF visualization data saved to: {vif_viz_file}")

# Create plotting instructions file
plotting_guide_file = os.path.join(output_dir, 'plotting_guide.txt')
with open(plotting_guide_file, 'w', encoding='utf-8') as f:
    f.write("MARS MISSION DATA - PLOTTING GUIDE\n")
    f.write("=" * 40 + "\n\n")
    
    f.write("AVAILABLE VISUALIZATION DATA FILES:\n")
    f.write("1. regression_plot_data.csv - For regression diagnostics\n")
    f.write("2. feature_importance_data.csv - For feature importance plots\n")
    f.write("3. model_performance_data.csv - For model comparison plots\n")
    f.write("4. vif_visualization_data.csv - For multicollinearity plots\n\n")
    
    f.write("RECOMMENDED PLOTS:\n\n")
    
    f.write("1. REGRESSION DIAGNOSTIC PLOTS:\n")
    f.write("   File: regression_plot_data.csv\n")
    f.write("   - Actual vs Predicted: x=predicted_value, y=actual_value\n")
    f.write("   - Residual Plot: x=predicted_value, y=residual\n")
    f.write("   - Filter by: target_variable, predictor_set, dataset (train/test)\n")
    f.write("   - Add reference line y=x for actual vs predicted\n\n")
    
    f.write("2. FEATURE IMPORTANCE PLOTS:\n")
    f.write("   File: feature_importance_data.csv\n")
    f.write("   - Bar chart: x=feature, y=importance\n")
    f.write("   - Facet by: target_variable, selection_method\n")
    f.write("   - Color by: coefficient (positive/negative)\n\n")
    
    f.write("3. MODEL PERFORMANCE COMPARISON:\n")
    f.write("   File: model_performance_data.csv\n")
    f.write("   - Scatter plot: x=test_r2, y=rmse, color=model_type\n")
    f.write("   - Bar chart: x=target_variable, y=test_r2, fill=model_type\n")
    f.write("   - Grouped by: predictor_set\n\n")
    
    f.write("4. VIF MULTICOLLINEARITY PLOT:\n")
    f.write("   File: vif_visualization_data.csv\n")
    f.write("   - Bar chart: x=feature, y=vif_value\n")
    f.write("   - Add horizontal lines at VIF=5 and VIF=10\n")
    f.write("   - Color by: multicollinearity_level\n\n")
    
    f.write("EXAMPLE PLOTTING CODE (R/ggplot2):\n")
    f.write("# Actual vs Predicted\n")
    f.write("ggplot(regression_data, aes(x=predicted_value, y=actual_value)) +\n")
    f.write("  geom_point(alpha=0.6) +\n")
    f.write("  geom_abline(slope=1, intercept=0, color='red') +\n")
    f.write("  facet_wrap(~target_variable, scales='free') +\n")
    f.write("  theme_minimal()\n\n")
    
    f.write("# VIF Plot\n")
    f.write("ggplot(vif_data, aes(x=reorder(feature, vif_value), y=vif_value)) +\n")
    f.write("  geom_col(aes(fill=multicollinearity_level)) +\n")
    f.write("  geom_hline(yintercept=c(5, 10), linetype='dashed') +\n")
    f.write("  coord_flip() +\n")
    f.write("  theme_minimal()\n\n")
    
    f.write("EXAMPLE PLOTTING CODE (Python/matplotlib):\n")
    f.write("import matplotlib.pyplot as plt\n")
    f.write("import seaborn as sns\n\n")
    f.write("# Actual vs Predicted\n")
    f.write("plt.figure(figsize=(12, 8))\n")
    f.write("for i, target in enumerate(df['target_variable'].unique()):\n")
    f.write("    plt.subplot(2, 3, i+1)\n")
    f.write("    subset = df[df['target_variable'] == target]\n")
    f.write("    plt.scatter(subset['predicted_value'], subset['actual_value'], alpha=0.6)\n")
    f.write("    plt.plot([subset['actual_value'].min(), subset['actual_value'].max()],\n")
    f.write("             [subset['actual_value'].min(), subset['actual_value'].max()], 'r--')\n")
    f.write("    plt.title(target)\n")
    f.write("plt.tight_layout()\n")
    f.write("plt.show()\n")

print(f"Plotting guide saved to: {plotting_guide_file}")

summary_file = os.path.join(output_dir, 'comprehensive_analysis_summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("MARS MISSION DATA - COMPREHENSIVE ANALYSIS SUMMARY\n")
    f.write("=" * 58 + "\n\n")
    
    f.write("DATASET OVERVIEW:\n")
    f.write(f"- Total observations: {len(df)}\n")
    f.write(f"- Total variables: {len(df.columns)}\n\n")
    
    if not vif_results.empty:
        high_vif = sum(vif_results['multicollinearity'] == 'High')
        f.write("VIF ANALYSIS:\n")
        f.write(f"- Variables analyzed: {len(vif_results)}\n")
        f.write(f"- High multicollinearity (VIF > 10): {high_vif}\n")
        f.write(f"- Variables with VIF > 10 may need to be removed\n\n")
    
    if not regression_results.empty:
        high_r2_models = sum(regression_results['test_r2'] > 0.7)
        f.write("MULTIPLE REGRESSION ANALYSIS:\n")
        f.write(f"- Total models fitted: {len(regression_results)}\n")
        f.write(f"- Models with test R² > 0.7: {high_r2_models}\n")
        f.write(f"- Best test R²: {regression_results['test_r2'].max():.4f}\n\n")
    
    if not stepwise_results.empty:
        avg_reduction = stepwise_results['feature_reduction'].mean()
        f.write("STEPWISE SELECTION ANALYSIS:\n")
        f.write(f"- Selection procedures performed: {len(stepwise_results)}\n")
        f.write(f"- Average feature reduction: {avg_reduction:.1f}%\n")
        f.write(f"- Best test R² after selection: {stepwise_results['test_r2'].max():.4f}\n\n")
    
    if not model_comparison.empty:
        rf_wins = sum(model_comparison['best_model'] == 'Random Forest')
        f.write("MODEL COMPARISON ANALYSIS:\n")
        f.write(f"- Comparisons performed: {len(model_comparison)}\n")
        f.write(f"- Random Forest wins: {rf_wins}/{len(model_comparison)}\n")
        f.write(f"- Average R² improvement: {model_comparison['r2_improvement'].mean():.4f}\n\n")
    
    f.write("ANALYSIS NOTES:\n")
    f.write("- VIF > 10 indicates high multicollinearity\n")
    f.write("- Stepwise selection can help reduce model complexity\n")
    f.write("- Random Forest often handles non-linear relationships better\n")
    f.write("- Test R² is more reliable than training R² for model evaluation\n")

print(f"Summary report saved to: {summary_file}")

print("\n" + "="*60)
print("COMPREHENSIVE ANALYSIS COMPLETE!")
print("="*60)
print(f"All results saved to: {output_dir}")
print("Files created:")
if not vif_results.empty:
    print(f"  - vif_analysis_results.csv ({len(vif_results)} variables)")
if not regression_results.empty:
    print(f"  - multiple_regression_results.csv ({len(regression_results)} models)")
if not stepwise_results.empty:
    print(f"  - stepwise_selection_results.csv ({len(stepwise_results)} procedures)")
if not model_comparison.empty:
    print(f"  - model_comparison_results.csv ({len(model_comparison)} comparisons)")
if not regression_plot_data.empty:
    print(f"  - regression_plot_data.csv ({len(regression_plot_data)} data points)")
if not feature_importance_data.empty:
    print(f"  - feature_importance_data.csv ({len(feature_importance_data)} importance scores)")
if not model_performance_data.empty:
    print(f"  - model_performance_data.csv ({len(model_performance_data)} model results)")
if not vif_viz_data.empty:
    print(f"  - vif_visualization_data.csv ({len(vif_viz_data)} VIF values)")
print(f"  - comprehensive_analysis_summary.txt")
print(f"  - plotting_guide.txt")
