import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import kruskal, mannwhitneyu, ranksums, wilcoxon
from scipy.stats import bootstrap, trim_mean
import itertools
import os
import warnings
warnings.filterwarnings('ignore')

input_file = r'C:\Users\agsse\data_analysis\mars_mission_master_dataset_optimized.csv'
output_dir = r'C:\Users\agsse\data_analysis\max_data'

os.makedirs(output_dir, exist_ok=True)

print("Loading Mars mission data...")
df = pd.read_csv(input_file)
print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

def epsilon_squared(h_statistic, n, k):
    """Calculate epsilon-squared effect size for Kruskal-Wallis test"""
    return (h_statistic - k + 1) / (n - k)

def eta_squared_kw(groups):
    """Calculate eta-squared effect size for Kruskal-Wallis test"""
    all_data = np.concatenate(groups)
    n = len(all_data)
    ranks = stats.rankdata(all_data)
    
    start_idx = 0
    sum_ranks_squared = 0
    for group in groups:
        end_idx = start_idx + len(group)
        group_ranks = ranks[start_idx:end_idx]
        sum_ranks_squared += (np.sum(group_ranks)**2) / len(group)
        start_idx = end_idx
    
    eta_sq = (12 / (n * (n + 1))) * sum_ranks_squared - 3 * (n + 1)
    return eta_sq / (n - 1)

def perform_kruskal_wallis_tests(df):
    """Perform Kruskal-Wallis tests (non-parametric alternative to one-way ANOVA)"""
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
                h_statistic, p_value = kruskal(*groups)
                
                n = len(subset)
                k = len(groups)
                epsilon_sq = epsilon_squared(h_statistic, n, k)
                eta_sq = eta_squared_kw(groups)
                
                group_medians = [np.median(group) for group in groups]
                group_names = [name for name, group in subset.groupby(cat_var) if len(group) >= 3]
                
                effect_size = 'Small' if epsilon_sq < 0.01 else 'Medium' if epsilon_sq < 0.04 else 'Large'
                
                results.append({
                    'continuous_variable': cont_var,
                    'grouping_variable': cat_var,
                    'kruskal_h_statistic': round(h_statistic, 4),
                    'p_value': round(p_value, 6),
                    'significant_at_0.05': p_value < 0.05,
                    'epsilon_squared': round(epsilon_sq, 4),
                    'eta_squared': round(eta_sq, 4),
                    'effect_size': effect_size,
                    'num_groups': len(groups),
                    'total_observations': n,
                    'group_medians': [round(m, 3) for m in group_medians],
                    'group_names': group_names,
                    'interpretation': 'Reject null hypothesis - group distributions differ' if p_value < 0.05 else 'Fail to reject null hypothesis - group distributions similar'
                })
                
                print(f"\n{cont_var} grouped by {cat_var}:")
                print(f"  Kruskal-Wallis H: {h_statistic:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"  Effect size (ε²): {epsilon_sq:.4f} ({effect_size})")
                
            except Exception as e:
                print(f"Error testing {cont_var} by {cat_var}: {str(e)}")
    
    return pd.DataFrame(results)

def perform_mann_whitney_tests(df):
    """Perform Mann-Whitney U tests for pairwise comparisons"""
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
                
            categories = subset[cat_var].unique()
            if len(categories) < 2:
                continue
                
            for cat1, cat2 in itertools.combinations(categories, 2):
                group1 = subset[subset[cat_var] == cat1][cont_var].values
                group2 = subset[subset[cat_var] == cat2][cont_var].values
                
                if len(group1) < 3 or len(group2) < 3:
                    continue
                    
                try:
                    u_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    
                    n1, n2 = len(group1), len(group2)
                    z_score = (u_statistic - (n1 * n2) / 2) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
                    r_effect_size = abs(z_score) / np.sqrt(n1 + n2)
                    
                    effect_size = 'Small' if r_effect_size < 0.3 else 'Medium' if r_effect_size < 0.5 else 'Large'
                    
                    results.append({
                        'continuous_variable': cont_var,
                        'grouping_variable': cat_var,
                        'group_1': cat1,
                        'group_2': cat2,
                        'u_statistic': round(u_statistic, 4),
                        'p_value': round(p_value, 6),
                        'significant_at_0.05': p_value < 0.05,
                        'z_score': round(z_score, 4),
                        'r_effect_size': round(r_effect_size, 4),
                        'effect_size': effect_size,
                        'n_group_1': n1,
                        'n_group_2': n2,
                        'median_group_1': round(np.median(group1), 3),
                        'median_group_2': round(np.median(group2), 3),
                        'interpretation': 'Reject null hypothesis - distributions differ' if p_value < 0.05 else 'Fail to reject null hypothesis - distributions similar'
                    })
                    
                except Exception as e:
                    print(f"Error in Mann-Whitney test {cont_var} {cat1} vs {cat2}: {str(e)}")
    
    return pd.DataFrame(results)

def perform_robust_statistics(df):
    """Calculate robust descriptive statistics"""
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
            
            for category, group in subset.groupby(cat_var):
                if len(group) < 5:
                    continue
                    
                data = group[cont_var].values
                
                try:
                    results.append({
                        'continuous_variable': cont_var,
                        'grouping_variable': cat_var,
                        'category': category,
                        'n': len(data),
                        'mean': round(np.mean(data), 4),
                        'trimmed_mean_10pct': round(trim_mean(data, 0.1), 4),
                        'median': round(np.median(data), 4),
                        'std': round(np.std(data, ddof=1), 4),
                        'mad': round(stats.median_abs_deviation(data), 4),
                        'iqr': round(stats.iqr(data), 4),
                        'q25': round(np.percentile(data, 25), 4),
                        'q75': round(np.percentile(data, 75), 4),
                        'min': round(np.min(data), 4),
                        'max': round(np.max(data), 4)
                    })
                    
                except Exception as e:
                    print(f"Error calculating robust stats for {cont_var} {category}: {str(e)}")
    
    return pd.DataFrame(results)

def perform_bootstrap_comparisons(df, n_bootstrap=1000):
    """Perform bootstrap tests for mean differences between groups"""
    results = []
    
    continuous_vars = ['temp_avg', 'wind_speed_avg', 'activity_intensity']
    categorical_vars = ['mars_season', 'mission_phase']
    
    rng = np.random.default_rng(42)
    
    for cont_var in continuous_vars:
        if cont_var not in df.columns:
            continue
            
        for cat_var in categorical_vars:
            if cat_var not in df.columns:
                continue
                
            subset = df[[cont_var, cat_var]].dropna()
            categories = subset[cat_var].unique()
            
            if len(categories) < 2:
                continue
                
            for cat1, cat2 in itertools.combinations(categories, 2):
                group1 = subset[subset[cat_var] == cat1][cont_var].values
                group2 = subset[subset[cat_var] == cat2][cont_var].values
                
                if len(group1) < 10 or len(group2) < 10:
                    continue
                    
                try:
                    def mean_diff(x, y, axis=-1):
                        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
                    
                    res = bootstrap((group1, group2), mean_diff, 
                                  n_resamples=n_bootstrap, 
                                  confidence_level=0.95,
                                  random_state=rng)
                    
                    observed_diff = np.mean(group1) - np.mean(group2)
                    ci_low, ci_high = res.confidence_interval
                    
                    significant = not (ci_low <= 0 <= ci_high)
                    
                    results.append({
                        'continuous_variable': cont_var,
                        'grouping_variable': cat_var,
                        'group_1': cat1,
                        'group_2': cat2,
                        'observed_mean_diff': round(observed_diff, 4),
                        'bootstrap_ci_low': round(ci_low, 4),
                        'bootstrap_ci_high': round(ci_high, 4),
                        'significant_at_0.05': significant,
                        'n_bootstrap': n_bootstrap,
                        'interpretation': 'Significant difference (CI excludes 0)' if significant else 'No significant difference (CI includes 0)'
                    })
                    
                except Exception as e:
                    print(f"Error in bootstrap test {cont_var} {cat1} vs {cat2}: {str(e)}")
    
    return pd.DataFrame(results)

print("\n" + "="*60)
print("KRUSKAL-WALLIS TESTS (NON-PARAMETRIC ANOVA)")
print("="*60)
kruskal_results = perform_kruskal_wallis_tests(df)

print("\n" + "="*60)
print("MANN-WHITNEY U TESTS (PAIRWISE COMPARISONS)")
print("="*60)
mannwhitney_results = perform_mann_whitney_tests(df)

print("\n" + "="*60)
print("ROBUST DESCRIPTIVE STATISTICS")
print("="*60)
robust_stats = perform_robust_statistics(df)

print("\n" + "="*60)
print("BOOTSTRAP TESTS (SELECTED COMPARISONS)")
print("="*60)
bootstrap_results = perform_bootstrap_comparisons(df)

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

kruskal_file = os.path.join(output_dir, 'kruskal_wallis_results.csv')
kruskal_results.to_csv(kruskal_file, index=False)
print(f"Kruskal-Wallis results saved to: {kruskal_file}")

mannwhitney_file = os.path.join(output_dir, 'mann_whitney_results.csv')
mannwhitney_results.to_csv(mannwhitney_file, index=False)
print(f"Mann-Whitney U results saved to: {mannwhitney_file}")

robust_file = os.path.join(output_dir, 'robust_statistics.csv')
robust_stats.to_csv(robust_file, index=False)
print(f"Robust statistics saved to: {robust_file}")

bootstrap_file = os.path.join(output_dir, 'bootstrap_results.csv')
bootstrap_results.to_csv(bootstrap_file, index=False)
print(f"Bootstrap results saved to: {bootstrap_file}")

summary_file = os.path.join(output_dir, 'nonparametric_tests_summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("MARS MISSION DATA - NON-PARAMETRIC TESTS SUMMARY\n")
    f.write("=" * 55 + "\n\n")
    
    f.write("DATASET OVERVIEW:\n")
    f.write(f"- Total observations: {len(df)}\n")
    f.write(f"- Total variables: {len(df.columns)}\n\n")
    
    if len(kruskal_results) > 0:
        f.write("KRUSKAL-WALLIS TEST RESULTS:\n")
        f.write(f"- Total tests performed: {len(kruskal_results)}\n")
        f.write(f"- Significant results (p < 0.05): {sum(kruskal_results['significant_at_0.05'])}\n")
        f.write(f"- Percentage significant: {100*sum(kruskal_results['significant_at_0.05'])/len(kruskal_results):.1f}%\n")
        large_effects = sum(kruskal_results['effect_size'] == 'Large')
        medium_effects = sum(kruskal_results['effect_size'] == 'Medium')
        f.write(f"- Large effect sizes: {large_effects}\n")
        f.write(f"- Medium effect sizes: {medium_effects}\n\n")
    
    if len(mannwhitney_results) > 0:
        f.write("MANN-WHITNEY U TEST RESULTS:\n")
        f.write(f"- Total pairwise tests: {len(mannwhitney_results)}\n")
        f.write(f"- Significant results: {sum(mannwhitney_results['significant_at_0.05'])}\n")
        f.write(f"- Percentage significant: {100*sum(mannwhitney_results['significant_at_0.05'])/len(mannwhitney_results):.1f}%\n\n")
    
    if len(bootstrap_results) > 0:
        f.write("BOOTSTRAP TEST RESULTS:\n")
        f.write(f"- Total bootstrap tests: {len(bootstrap_results)}\n")
        f.write(f"- Significant results: {sum(bootstrap_results['significant_at_0.05'])}\n")
        f.write(f"- Bootstrap samples per test: 1000\n\n")
    
    f.write("METHOD ADVANTAGES:\n")
    f.write("- Kruskal-Wallis: No normality assumption, handles unequal variances\n")
    f.write("- Mann-Whitney U: Robust pairwise comparisons\n")
    f.write("- Bootstrap: Distribution-free confidence intervals\n")
    f.write("- Robust statistics: Less sensitive to outliers\n\n")
    
    f.write("INTERPRETATION NOTES:\n")
    f.write("- These tests focus on distribution differences, not just means\n")
    f.write("- Effect sizes: Small (ε² < 0.01), Medium (0.01-0.04), Large (> 0.04)\n")
    f.write("- Medians and IQR are more robust than means and standard deviations\n")
    f.write("- Bootstrap CIs that exclude 0 indicate significant differences\n")

print(f"Summary report saved to: {summary_file}")

print("\n" + "="*60)
print("NON-PARAMETRIC ANALYSIS COMPLETE!")
print("="*60)
print(f"All results saved to: {output_dir}")
print(f"Files created:")
print(f"  - kruskal_wallis_results.csv ({len(kruskal_results)} tests)")
print(f"  - mann_whitney_results.csv ({len(mannwhitney_results)} tests)")
print(f"  - robust_statistics.csv ({len(robust_stats)} statistics)")
print(f"  - bootstrap_results.csv ({len(bootstrap_results)} tests)")
print(f"  - nonparametric_tests_summary.txt")