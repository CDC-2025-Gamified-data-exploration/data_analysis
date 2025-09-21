# bartlett, f test, and levene


import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import bartlett, levene, f_oneway
import os
import warnings
warnings.filterwarnings('ignore')

# File paths
input_file = r'C:\Users\agsse\data_analysis\mars_mission_master_dataset_optimized.csv'
output_dir = r'C:\Users\agsse\data_analysis\max_data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the Mars mission data
print("Loading Mars mission data...")
df = pd.read_csv(input_file)
print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 1. BARTLETT'S TEST FOR HOMOGENEITY OF VARIANCES
# =============================================================================
print("\n" + "="*60)
print("BARTLETT'S TEST FOR HOMOGENEITY OF VARIANCES")
print("="*60)

def perform_bartlett_tests(df):
    """Perform Bartlett's test on key variables grouped by categorical variables"""
    results = []
    
    # Key continuous variables to test
    continuous_vars = [
        'temp_avg', 'temp_max', 'temp_min', 'wind_speed_avg', 'wind_speed_max',
        'activity_intensity', 'operational_tempo_7sol', 'dust_storm_probability',
        'environmental_stress_index', 'operational_risk_index'
    ]
    
    # Categorical grouping variables
    categorical_vars = [
        'mars_season', 'mission_phase', 'primary_instrument', 'primary_activity_type'
    ]
    
    for cont_var in continuous_vars:
        if cont_var not in df.columns:
            continue
            
        for cat_var in categorical_vars:
            if cat_var not in df.columns:
                continue
                
            # Filter out missing values
            subset = df[[cont_var, cat_var]].dropna()
            if len(subset) < 10:
                continue
                
            # Group data by categorical variable
            groups = [group[cont_var].values for name, group in subset.groupby(cat_var) if len(group) >= 3]
            
            if len(groups) < 2:
                continue
                
            try:
                # Perform Bartlett's test
                statistic, p_value = bartlett(*groups)
                
                # Get group information
                group_info = subset.groupby(cat_var)[cont_var].agg(['count', 'mean', 'std']).round(4)
                
                results.append({
                    'continuous_variable': cont_var,
                    'grouping_variable': cat_var,
                    'bartlett_statistic': round(statistic, 4),
                    'p_value': round(p_value, 6),
                    'significant_at_0.05': p_value < 0.05,
                    'num_groups': len(groups),
                    'total_observations': len(subset),
                    'interpretation': 'Reject null hypothesis - variances differ' if p_value < 0.05 else 'Fail to reject null hypothesis - variances similar'
                })
                
                print(f"\n{cont_var} grouped by {cat_var}:")
                print(f"  Bartlett statistic: {statistic:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"  Groups: {len(groups)}, Total N: {len(subset)}")
                
            except Exception as e:
                print(f"Error testing {cont_var} by {cat_var}: {str(e)}")
    
    return pd.DataFrame(results)

bartlett_results = perform_bartlett_tests(df)

# =============================================================================
# 2. LEVENE'S TEST FOR HOMOGENEITY OF VARIANCES
# =============================================================================
print("\n" + "="*60)
print("LEVENE'S TEST FOR HOMOGENEITY OF VARIANCES")
print("="*60)

def perform_levene_tests(df):
    """Perform Levene's test (more robust to non-normality than Bartlett's)"""
    results = []
    
    continuous_vars = [
        'temp_avg', 'temp_max', 'temp_min', 'wind_speed_avg', 'wind_speed_max',
        'activity_intensity', 'operational_tempo_7sol', 'dust_storm_probability',
        'environmental_stress_index', 'operational_risk_index'
    ]
    
    categorical_vars = [
        'mars_season', 'mission_phase', 'primary_instrument', 'primary_activity_type'
    ]
    
    for cont_var in continuous_vars:
        if cont_var not in df.columns:
            continue
            
        for cat_var in categorical_vars:
            if cat_var not in df.columns:
                continue
                
            subset = df[[cont_var, cat_var]].dropna()
            if len(subset) < 10:
                continue
                
            groups = [group[cont_var].values for name, group in subset.groupby(cat_var) if len(group) >= 3]
            
            if len(groups) < 2:
                continue
                
            try:
                # Perform Levene's test (using median - more robust)
                statistic, p_value = levene(*groups, center='median')
                
                results.append({
                    'continuous_variable': cont_var,
                    'grouping_variable': cat_var,
                    'levene_statistic': round(statistic, 4),
                    'p_value': round(p_value, 6),
                    'significant_at_0.05': p_value < 0.05,
                    'num_groups': len(groups),
                    'total_observations': len(subset),
                    'interpretation': 'Reject null hypothesis - variances differ' if p_value < 0.05 else 'Fail to reject null hypothesis - variances similar'
                })
                
                print(f"\n{cont_var} grouped by {cat_var}:")
                print(f"  Levene statistic: {statistic:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                
            except Exception as e:
                print(f"Error testing {cont_var} by {cat_var}: {str(e)}")
    
    return pd.DataFrame(results)

levene_results = perform_levene_tests(df)

# =============================================================================
# 3. F-TESTS (ONE-WAY ANOVA)
# =============================================================================
print("\n" + "="*60)
print("F-TESTS (ONE-WAY ANOVA)")
print("="*60)

def perform_f_tests(df):
    """Perform F-tests (one-way ANOVA) to test for mean differences between groups"""
    results = []
    
    continuous_vars = [
        'temp_avg', 'temp_max', 'temp_min', 'wind_speed_avg', 'wind_speed_max',
        'activity_intensity', 'operational_tempo_7sol', 'dust_storm_probability',
        'environmental_stress_index', 'operational_risk_index'
    ]
    
    categorical_vars = [
        'mars_season', 'mission_phase', 'primary_instrument', 'primary_activity_type'
    ]
    
    for cont_var in continuous_vars:
        if cont_var not in df.columns:
            continue
            
        for cat_var in categorical_vars:
            if cat_var not in df.columns:
                continue
                
            subset = df[[cont_var, cat_var]].dropna()
            if len(subset) < 10:
                continue
                
            groups = [group[cont_var].values for name, group in subset.groupby(cat_var) if len(group) >= 3]
            
            if len(groups) < 2:
                continue
                
            try:
                # Perform one-way ANOVA
                f_statistic, p_value = f_oneway(*groups)
                
                # Calculate effect size (eta-squared)
                group_means = [np.mean(group) for group in groups]
                overall_mean = np.mean([val for group in groups for val in group])
                ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups)
                ss_total = sum((val - overall_mean)**2 for group in groups for val in group)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Get descriptive statistics
                group_stats = subset.groupby(cat_var)[cont_var].agg(['count', 'mean', 'std']).round(4)
                
                results.append({
                    'continuous_variable': cont_var,
                    'grouping_variable': cat_var,
                    'f_statistic': round(f_statistic, 4),
                    'p_value': round(p_value, 6),
                    'significant_at_0.05': p_value < 0.05,
                    'eta_squared': round(eta_squared, 4),
                    'effect_size': 'Small' if eta_squared < 0.06 else 'Medium' if eta_squared < 0.14 else 'Large',
                    'num_groups': len(groups),
                    'total_observations': len(subset),
                    'interpretation': 'Reject null hypothesis - group means differ' if p_value < 0.05 else 'Fail to reject null hypothesis - group means similar'
                })
                
                print(f"\n{cont_var} grouped by {cat_var}:")
                print(f"  F-statistic: {f_statistic:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"  Effect size (η²): {eta_squared:.4f}")
                
            except Exception as e:
                print(f"Error testing {cont_var} by {cat_var}: {str(e)}")
    
    return pd.DataFrame(results)

f_test_results = perform_f_tests(df)

# =============================================================================
# SAVE RESULTS TO FILES
# =============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save Bartlett test results
bartlett_file = os.path.join(output_dir, 'bartlett_test_results.csv')
bartlett_results.to_csv(bartlett_file, index=False)
print(f"Bartlett test results saved to: {bartlett_file}")

# Save Levene test results
levene_file = os.path.join(output_dir, 'levene_test_results.csv')
levene_results.to_csv(levene_file, index=False)
print(f"Levene test results saved to: {levene_file}")

# Save F-test results
f_test_file = os.path.join(output_dir, 'f_test_results.csv')
f_test_results.to_csv(f_test_file, index=False)
print(f"F-test results saved to: {f_test_file}")

# Create summary report
summary_file = os.path.join(output_dir, 'statistical_tests_summary.txt')
with open(summary_file, 'w') as f:
    f.write("MARS MISSION DATA - STATISTICAL TESTS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("DATASET OVERVIEW:\n")
    f.write(f"- Total observations: {len(df)}\n")
    f.write(f"- Total variables: {len(df.columns)}\n")
    f.write(f"- Date range: Sol {df['sol'].min()} to Sol {df['sol'].max()}\n\n")
    
    f.write("BARTLETT'S TEST RESULTS:\n")
    f.write(f"- Total tests performed: {len(bartlett_results)}\n")
    f.write(f"- Significant results (p < 0.05): {sum(bartlett_results['significant_at_0.05'])}\n")
    f.write(f"- Percentage significant: {100*sum(bartlett_results['significant_at_0.05'])/len(bartlett_results):.1f}%\n\n")
    
    f.write("LEVENE'S TEST RESULTS:\n")
    f.write(f"- Total tests performed: {len(levene_results)}\n")
    f.write(f"- Significant results (p < 0.05): {sum(levene_results['significant_at_0.05'])}\n")
    f.write(f"- Percentage significant: {100*sum(levene_results['significant_at_0.05'])/len(levene_results):.1f}%\n\n")
    
    f.write("F-TEST (ANOVA) RESULTS:\n")
    f.write(f"- Total tests performed: {len(f_test_results)}\n")
    f.write(f"- Significant results (p < 0.05): {sum(f_test_results['significant_at_0.05'])}\n")
    f.write(f"- Percentage significant: {100*sum(f_test_results['significant_at_0.05'])/len(f_test_results):.1f}%\n\n")
    
    if len(f_test_results) > 0:
        large_effects = sum(f_test_results['effect_size'] == 'Large')
        medium_effects = sum(f_test_results['effect_size'] == 'Medium')
        f.write(f"- Large effect sizes: {large_effects}\n")
        f.write(f"- Medium effect sizes: {medium_effects}\n\n")
    
    f.write("INTERPRETATION NOTES:\n")
    f.write("- Bartlett's test assumes normality; Levene's test is more robust\n")
    f.write("- Significant variance tests suggest using non-parametric alternatives\n")
    f.write("- F-tests assume equal variances; check variance test results first\n")
    f.write("- Effect sizes: Small (η² < 0.06), Medium (0.06-0.14), Large (> 0.14)\n")

print(f"Summary report saved to: {summary_file}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"All results saved to: {output_dir}")
print(f"Files created:")
print(f"  - bartlett_test_results.csv ({len(bartlett_results)} tests)")
print(f"  - levene_test_results.csv ({len(levene_results)} tests)")
print(f"  - f_test_results.csv ({len(f_test_results)} tests)")
print(f"  - statistical_tests_summary.txt")