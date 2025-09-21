import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson, kstest, normaltest
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_summary_statistics(df, output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary_stats = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
            
        try:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            if len(data) >= 8:
                try:
                    jb_stat, jb_p = jarque_bera(data)
                except:
                    jb_stat, jb_p = np.nan, np.nan
            else:
                jb_stat, jb_p = np.nan, np.nan
                
            if len(data) >= 3 and len(data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(data)
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            summary_stats.append({
                'variable': col,
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q1': q1,
                'median': data.median(),
                'q3': q3,
                'max': data.max(),
                'range': data.max() - data.min(),
                'iqr': iqr,
                'variance': data.var(),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,
                'mad': np.median(np.abs(data - data.median())),
                'sem': stats.sem(data),
                'jarque_bera_stat': jb_stat,
                'jarque_bera_p': jb_p,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'missing_count': df[col].isna().sum(),
                'missing_percent': (df[col].isna().sum() / len(df)) * 100,
                'unique_values': data.nunique(),
                'mode': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan
            })
            
        except Exception as e:
            print(f"Error processing {col}: {e}")
            continue
    
    summary_df = pd.DataFrame(summary_stats)
    
    csv_filepath = os.path.join(output_folder, f'summary_statistics_{ts}.csv')
    summary_df.to_csv(csv_filepath, index=False)
    
    print(f"Saved summary statistics to: {csv_filepath}")
    return summary_df, csv_filepath

def plot_histogram(df, variables=None, bins=30, figsize=(15, 12), output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if variables is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        variables = numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols
    
    histogram_data = []
    
    n_vars = len(variables)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if var not in df.columns:
            ax.text(0.5, 0.5, f'{var} not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
            
        data = df[var].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
        
        n, bin_edges, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        
        ax.set_title(f'{var}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for j, (center, count) in enumerate(zip(bin_centers, n)):
            histogram_data.append({
                'variable': var,
                'bin_center': center,
                'bin_left_edge': bin_edges[j],
                'bin_right_edge': bin_edges[j+1],
                'frequency': count,
                'density': count / len(data),
                'bin_width': bin_edges[j+1] - bin_edges[j]
            })
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    plot_filepath = os.path.join(output_folder, f'histograms_{len(variables)}vars_{ts}.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    histogram_df = pd.DataFrame(histogram_data)
    csv_filepath = os.path.join(output_folder, f'histogram_data_{len(variables)}vars_{ts}.csv')
    histogram_df.to_csv(csv_filepath, index=False)
    
    print(f"Saved histogram plots to: {plot_filepath}")
    print(f"Saved histogram data to: {csv_filepath}")
    
    return plot_filepath, csv_filepath

def plot_density(df, variables=None, figsize=(15, 12), output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if variables is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        variables = numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols
    
    density_data = []
    
    n_vars = len(variables)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if var not in df.columns:
            ax.text(0.5, 0.5, f'{var} not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
            
        data = df[var].dropna()
        
        if len(data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
        
        try:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            density_values = kde(x_range)
            
            ax.plot(x_range, density_values, linewidth=2, label='KDE')
            ax.fill_between(x_range, density_values, alpha=0.3)
            
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
            
            ax.set_title(f'{var} - Density Plot')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for j, (x, y) in enumerate(zip(x_range, density_values)):
                density_data.append({
                    'variable': var,
                    'x_value': x,
                    'density': y,
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std()
                })
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    plot_filepath = os.path.join(output_folder, f'density_plots_{len(variables)}vars_{ts}.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    density_df = pd.DataFrame(density_data)
    csv_filepath = os.path.join(output_folder, f'density_data_{len(variables)}vars_{ts}.csv')
    density_df.to_csv(csv_filepath, index=False)
    
    print(f"Saved density plots to: {plot_filepath}")
    print(f"Saved density data to: {csv_filepath}")
    
    return plot_filepath, csv_filepath

def plot_barchart(df, variables=None, figsize=(15, 12), max_categories=20, output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if variables is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude activity variables except primary_instrument and primary_activity_type
        activity_vars_to_exclude = [
            'has_activity', 'activity_count', 'activity_text', 'multiple_activities',
            'activity_word_count', 'instruction_complexity', 'is_multipart_operation',
            'operation_part_number', 'systems_involved', 'uses_ida', 'uses_hp3',
            'uses_seis', 'uses_apss', 'uses_camera', 'uses_wts', 'is_science_activity',
            'is_maintenance_activity', 'is_deployment_activity', 'is_recovery_activity',
            'is_thermal_activity', 'is_communication_activity', 'activity_intensity',
            'operational_status_rate', 'has_operational_problem', 'criticality_rate',
            'is_critical_operation', 'power_status_rate', 'has_power_issue',
            'operational_tempo_3sol', 'operational_tempo_7sol', 'operational_tempo_15sol',
            'sols_since_last_problem', 'sols_to_next_problem', 'sols_since_last_activity',
            'activity_days_30sol', 'cumulative_activity_count', 'ida_cumulative_usage',
            'hp3_cumulative_usage', 'seis_cumulative_usage', 'apss_cumulative_usage',
            'camera_cumulative_usage', 'wts_cumulative_usage', 'ida_usage_intensity',
            'hp3_usage_intensity', 'seis_usage_intensity', 'apss_usage_intensity',
            'camera_usage_intensity', 'wts_usage_intensity', 'mission_operational_cycles',
            'equipment_reliability_score', 'maintenance_activity', 'sols_since_maintenance'
        ]
        
        categorical_cols = [col for col in categorical_cols if col not in activity_vars_to_exclude]
        
        low_cardinality_numeric = []
        for col in numeric_cols:
            if col not in activity_vars_to_exclude and df[col].nunique() <= max_categories:
                low_cardinality_numeric.append(col)
        
        # Explicitly include the two allowed activity variables if they exist
        if 'primary_instrument' in df.columns:
            categorical_cols.append('primary_instrument')
        if 'primary_activity_type' in df.columns:
            categorical_cols.append('primary_activity_type')
        
        variables = categorical_cols + low_cardinality_numeric
        variables = variables[:12] if len(variables) > 12 else variables
    
    barchart_data = []
    
    n_vars = len(variables)
    if n_vars == 0:
        print("No suitable variables found for bar charts")
        return None, None
        
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if var not in df.columns:
            ax.text(0.5, 0.5, f'{var} not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
        
        data = df[var].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
        
        value_counts = data.value_counts()
        
        if len(value_counts) > max_categories:
            value_counts = value_counts.head(max_categories)
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        ax.set_title(f'{var}')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        for j, (category, count) in enumerate(value_counts.items()):
            barchart_data.append({
                'variable': var,
                'category': category,
                'count': count,
                'percentage': (count / len(data)) * 100,
                'cumulative_count': value_counts.iloc[:j+1].sum(),
                'cumulative_percentage': (value_counts.iloc[:j+1].sum() / len(data)) * 100
            })
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    plot_filepath = os.path.join(output_folder, f'bar_charts_{len(variables)}vars_{ts}.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    barchart_data_df = pd.DataFrame(barchart_data)
    csv_filepath = os.path.join(output_folder, f'bar_chart_data_{len(variables)}vars_{ts}.csv')
    barchart_data_df.to_csv(csv_filepath, index=False)
    
    print(f"Saved bar charts to: {plot_filepath}")
    print(f"Saved bar chart data to: {csv_filepath}")
    
    return plot_filepath, csv_filepath

def plot_qq(df, variables=None, figsize=(15, 12), distribution='norm', output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if variables is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        variables = numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols
    
    qq_data = []
    normality_tests = []
    
    n_vars = len(variables)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if var not in df.columns:
            ax.text(0.5, 0.5, f'{var} not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
            
        data = df[var].dropna()
        
        if len(data) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
            continue
        
        try:
            if distribution == 'norm':
                theoretical_quantiles, sample_quantiles = stats.probplot(data, dist='norm')
            else:
                theoretical_quantiles, sample_quantiles = stats.probplot(data, dist=distribution)
            
            ax.scatter(theoretical_quantiles[0], theoretical_quantiles[1], alpha=0.6)
            ax.plot(theoretical_quantiles[0], sample_quantiles[1] + sample_quantiles[0] * theoretical_quantiles[0], 'r-', linewidth=2)
            
            ax.set_title(f'{var} - Q-Q Plot')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.grid(True, alpha=0.3)
            
            r_squared = stats.pearsonr(theoretical_quantiles[0], theoretical_quantiles[1])[0] ** 2
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            for j, (theo, samp) in enumerate(zip(theoretical_quantiles[0], theoretical_quantiles[1])):
                qq_data.append({
                    'variable': var,
                    'theoretical_quantile': theo,
                    'sample_quantile': samp,
                    'distribution': distribution,
                    'r_squared': r_squared
                })
            
            if len(data) >= 8:
                try:
                    jb_stat, jb_p = jarque_bera(data)
                except:
                    jb_stat, jb_p = np.nan, np.nan
            else:
                jb_stat, jb_p = np.nan, np.nan
                
            if len(data) >= 3 and len(data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(data)
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            try:
                ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            except:
                ks_stat, ks_p = np.nan, np.nan
            
            normality_tests.append({
                'variable': var,
                'n_observations': len(data),
                'jarque_bera_stat': jb_stat,
                'jarque_bera_p': jb_p,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'qq_r_squared': r_squared,
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            })
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}')
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    plot_filepath = os.path.join(output_folder, f'qq_plots_{distribution}_{len(variables)}vars_{ts}.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    qq_df = pd.DataFrame(qq_data)
    normality_df = pd.DataFrame(normality_tests)
    
    qq_csv_filepath = os.path.join(output_folder, f'qq_data_{distribution}_{len(variables)}vars_{ts}.csv')
    qq_df.to_csv(qq_csv_filepath, index=False)
    
    normality_csv_filepath = os.path.join(output_folder, f'normality_tests_{len(variables)}vars_{ts}.csv')
    normality_df.to_csv(normality_csv_filepath, index=False)
    
    print(f"Saved Q-Q plots to: {plot_filepath}")
    print(f"Saved Q-Q data to: {qq_csv_filepath}")
    print(f"Saved normality tests to: {normality_csv_filepath}")
    
    return plot_filepath, qq_csv_filepath, normality_csv_filepath

def demo_univariate_analysis():
    print("Loading Mars mission dataset...")
    try:
        df = pd.read_csv('mars_mission_master_dataset_optimized.csv')
        print(f"Successfully loaded dataset with shape: {df.shape}")
        print(f"Columns: {len(df.columns)} variables")
        
        results = analyze_dataframe(df)
        return results
    except FileNotFoundError:
        print("Error: mars_mission_master_dataset_optimized.csv not found in current directory")
        print("Please ensure the file is in the same folder as this script")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def analyze_dataframe(df):
    print("Running complete univariate analysis on Mars mission dataset...")
    print(f"Data shape: {df.shape}")
    print(f"Date range: Sol {df['sol'].min()} to Sol {df['sol'].max()}")
    
    print("\n1. Getting summary statistics...")
    summary_df, summary_path = get_summary_statistics(df)
    
    print("\n2. Creating histograms...")
    hist_plot, hist_data = plot_histogram(df)
    
    print("\n3. Creating density plots...")
    density_plot, density_data = plot_density(df)
    
    print("\n4. Creating bar charts...")
    bar_plot, bar_data = plot_barchart(df)
    
    print("\n5. Creating Q-Q plots...")
    qq_plot, qq_data, normality_data = plot_qq(df)
    
    print("\nAnalysis completed! All files saved to C:\\Users\\agsse\\data_analysis\\max_data")
    
    return {
        'summary_statistics': (summary_df, summary_path),
        'histograms': (hist_plot, hist_data),
        'density_plots': (density_plot, density_data),
        'bar_charts': (bar_plot, bar_data),
        'qq_plots': (qq_plot, qq_data, normality_data)
    }

if __name__ == "__main__":
    demo_univariate_analysis()