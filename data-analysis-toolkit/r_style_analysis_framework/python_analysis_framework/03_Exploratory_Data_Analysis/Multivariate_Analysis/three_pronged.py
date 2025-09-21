import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def plot_correlation_pairs_matrix(df, variables=None, method='pearson', figsize=(20, 16),
                                output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    """
    Create correlation pairs matrix with both visualization and CSV output
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if variables is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            variables = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        
        clean_data = df[variables].dropna()
        
        if len(clean_data) < 10:
            print(f"Insufficient data: only {len(clean_data)} rows available")
            return None, None
        
        pairs_data = []
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)
        
        if n_vars == 1:
            axes = np.array([[axes]])
        elif n_vars == 2:
            axes = axes.reshape(2, 2)
        
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j]
                
                if i == j:
                    ax.hist(clean_data[variables[i]], bins=30, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{variables[i]}')
                    ax.grid(True, alpha=0.3)
                    
                elif i > j:
                    x_data = clean_data[variables[j]]
                    y_data = clean_data[variables[i]]
                    
                    ax.scatter(x_data, y_data, alpha=0.6, s=20)
                    
                    if method == 'pearson':
                        correlation, p_value = stats.pearsonr(x_data, y_data)
                    elif method == 'spearman':
                        correlation, p_value = stats.spearmanr(x_data, y_data)
                    else:
                        correlation, p_value = stats.pearsonr(x_data, y_data)
                    
                    pairs_data.append({
                        'var1': variables[j],
                        'var2': variables[i],
                        'correlation': correlation,
                        'p_value': p_value,
                        'method': method,
                        'n_observations': len(clean_data)
                    })
                    
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                    
                    ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel(variables[j])
                    ax.set_ylabel(variables[i])
                    ax.grid(True, alpha=0.3)
                    
                else:
                    x_data = clean_data[variables[j]]
                    y_data = clean_data[variables[i]]
                    
                    if method == 'pearson':
                        correlation, p_value = stats.pearsonr(x_data, y_data)
                    elif method == 'spearman':
                        correlation, p_value = stats.spearmanr(x_data, y_data)
                    else:
                        correlation, p_value = stats.pearsonr(x_data, y_data)
                    
                    ax.text(0.5, 0.5, f'r = {correlation:.3f}\np = {p_value:.4f}', 
                           transform=ax.transAxes, fontsize=12,
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        
        filepath = os.path.join(output_folder, f'pairs_matrix_{method}_{len(variables)}vars_{ts}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        pairs_df = pd.DataFrame(pairs_data)
        csv_filepath = os.path.join(output_folder, f'pairs_matrix_data_{method}_{len(variables)}vars_{ts}.csv')
        pairs_df.to_csv(csv_filepath, index=False)
        
        print(f"Saved pairs matrix plot to: {filepath}")
        print(f"Saved pairs matrix data to: {csv_filepath}")
        
        return filepath, csv_filepath
        
    except Exception as e:
        print(f"Error in plot_correlation_pairs_matrix: {e}")
        return None, None

def plot_enhanced_pairs_matrix(df, variables=None, color_by=None, figsize=(20, 16),
                              output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    """
    Create enhanced pairs matrix with color coding and CSV output
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if variables is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            variables = numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
        
        if color_by and color_by in df.columns:
            clean_data = df[variables + [color_by]].dropna()
        else:
            clean_data = df[variables].dropna()
        
        if len(clean_data) < 10:
            print(f"Insufficient data: only {len(clean_data)} rows available")
            return None, None
        
        enhanced_pairs_data = []
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)
        
        if n_vars == 1:
            axes = np.array([[axes]])
        elif n_vars == 2:
            axes = axes.reshape(2, 2)
        
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j]
                
                if i == j:
                    if color_by and color_by in clean_data.columns:
                        unique_vals = clean_data[color_by].unique()
                        if len(unique_vals) <= 10:
                            for val in unique_vals:
                                subset = clean_data[clean_data[color_by] == val]
                                ax.hist(subset[variables[i]], bins=20, alpha=0.6, 
                                       label=f'{color_by}={val}', edgecolor='black')
                            ax.legend(fontsize=8)
                        else:
                            scatter = ax.scatter(clean_data[variables[i]], 
                                               np.random.normal(0, 0.1, len(clean_data)),
                                               c=clean_data[color_by], alpha=0.6, cmap='viridis')
                            plt.colorbar(scatter, ax=ax)
                    else:
                        ax.hist(clean_data[variables[i]], bins=30, alpha=0.7, edgecolor='black')
                    
                    ax.set_title(f'{variables[i]}')
                    ax.grid(True, alpha=0.3)
                    
                elif i > j:
                    x_data = clean_data[variables[j]]
                    y_data = clean_data[variables[i]]
                    
                    if color_by and color_by in clean_data.columns:
                        scatter = ax.scatter(x_data, y_data, c=clean_data[color_by], 
                                           alpha=0.6, s=20, cmap='viridis')
                    else:
                        ax.scatter(x_data, y_data, alpha=0.6, s=20)
                    
                    correlation, p_value = stats.pearsonr(x_data, y_data)
                    spearman_corr, spearman_p = stats.spearmanr(x_data, y_data)
                    
                    enhanced_pairs_data.append({
                        'var1': variables[j],
                        'var2': variables[i],
                        'pearson_correlation': correlation,
                        'pearson_p_value': p_value,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'n_observations': len(clean_data),
                        'colored_by': color_by if color_by else 'None'
                    })
                    
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
                    
                    ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel(variables[j])
                    ax.set_ylabel(variables[i])
                    ax.grid(True, alpha=0.3)
                    
                else:
                    x_data = clean_data[variables[j]]
                    y_data = clean_data[variables[i]]
                    
                    correlation, p_value = stats.pearsonr(x_data, y_data)
                    spearman_corr, _ = stats.spearmanr(x_data, y_data)
                    
                    ax.text(0.5, 0.6, f'Pearson r = {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           horizontalalignment='center', verticalalignment='center')
                    ax.text(0.5, 0.4, f'Spearman ρ = {spearman_corr:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           horizontalalignment='center', verticalalignment='center')
                    ax.text(0.5, 0.2, f'p = {p_value:.4f}', 
                           transform=ax.transAxes, fontsize=10,
                           horizontalalignment='center', verticalalignment='center')
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        
        color_suffix = f'_colored_by_{color_by}' if color_by else ''
        filepath = os.path.join(output_folder, f'enhanced_pairs_matrix_{len(variables)}vars{color_suffix}_{ts}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        enhanced_df = pd.DataFrame(enhanced_pairs_data)
        csv_filepath = os.path.join(output_folder, f'enhanced_pairs_data_{len(variables)}vars{color_suffix}_{ts}.csv')
        enhanced_df.to_csv(csv_filepath, index=False)
        
        print(f"Saved enhanced pairs matrix plot to: {filepath}")
        print(f"Saved enhanced pairs matrix data to: {csv_filepath}")
        
        return filepath, csv_filepath
        
    except Exception as e:
        print(f"Error in plot_enhanced_pairs_matrix: {e}")
        return None, None

def plot_correlation_pairs_subset(df, variable_pairs, figsize=(15, 10),
                                output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    """
    Create plots for specific variable pairs with CSV output
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        n_pairs = len(variable_pairs)
        
        if n_pairs > 12:
            variable_pairs = variable_pairs[:12]
            n_pairs = 12
        
        subset_data = []
        
        cols = 4
        rows = (n_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if n_pairs == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (var1, var2) in enumerate(variable_pairs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if var1 not in df.columns or var2 not in df.columns:
                ax.text(0.5, 0.5, f'{var1} or {var2} not found in data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{var1} vs {var2}')
                
                subset_data.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n_observations': 0,
                    'status': 'variables_not_found'
                })
                continue
            
            clean_data = df[[var1, var2]].dropna()
            
            if len(clean_data) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{var1} vs {var2}')
                
                subset_data.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n_observations': len(clean_data),
                    'status': 'insufficient_data'
                })
                continue
            
            x_data = clean_data[var1]
            y_data = clean_data[var2]
            
            ax.scatter(x_data, y_data, alpha=0.6, s=30)
            
            correlation, p_value = stats.pearsonr(x_data, y_data)
            
            subset_data.append({
                'var1': var1,
                'var2': var2,
                'correlation': correlation,
                'p_value': p_value,
                'n_observations': len(clean_data),
                'status': 'valid'
            })
            
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
            
            ax.text(0.05, 0.95, f'r = {correlation:.3f}\np = {p_value:.4f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_title(f'{var1} vs {var2}')
            ax.grid(True, alpha=0.3)
        
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_folder, f'correlation_pairs_subset_{n_pairs}pairs_{ts}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        subset_df = pd.DataFrame(subset_data)
        csv_filepath = os.path.join(output_folder, f'correlation_pairs_subset_data_{n_pairs}pairs_{ts}.csv')
        subset_df.to_csv(csv_filepath, index=False)
        
        print(f"Saved pairs subset plot to: {filepath}")
        print(f"Saved pairs subset data to: {csv_filepath}")
        
        return filepath, csv_filepath
        
    except Exception as e:
        print(f"Error in plot_correlation_pairs_subset: {e}")
        return None, None

def plot_time_series_pairs_matrix(df, variables=None, time_col='sol', window=50, 
                                 figsize=(20, 16), output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    """
    Create time series pairs matrix with CSV output
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if time_col not in df.columns:
            print(f"Time column '{time_col}' not found in dataframe")
            return None, None, None
        
        if variables is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col in numeric_cols:
                numeric_cols.remove(time_col)
            variables = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
        
        clean_data = df[variables + [time_col]].dropna().sort_values(time_col)
        
        if len(clean_data) < window:
            print(f"Insufficient data: need at least {window} rows, have {len(clean_data)}")
            return None, None, None
        
        time_series_data = []
        rolling_corr_data = []
        
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)
        
        if n_vars == 1:
            axes = np.array([[axes]])
        elif n_vars == 2:
            axes = axes.reshape(2, 2)
        
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axes[i, j]
                
                if i == j:
                    ax.plot(clean_data[time_col], clean_data[variables[i]], alpha=0.7, linewidth=1)
                    ax.set_title(f'{variables[i]} over time')
                    ax.set_xlabel(time_col)
                    ax.set_ylabel(variables[i])
                    ax.grid(True, alpha=0.3)
                    
                    time_series_data.extend([{
                        'variable': variables[i],
                        'time': t,
                        'value': v
                    } for t, v in zip(clean_data[time_col], clean_data[variables[i]])])
                    
                elif i > j:
                    rolling_corr = clean_data[variables[j]].rolling(window=window).corr(clean_data[variables[i]])
                    
                    ax.plot(clean_data[time_col], rolling_corr, linewidth=2)
                    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                    ax.axhline(0.5, color='green', linestyle='--', alpha=0.3)
                    ax.axhline(-0.5, color='green', linestyle='--', alpha=0.3)
                    
                    ax.set_title(f'Rolling correlation: {variables[j]} vs {variables[i]}')
                    ax.set_xlabel(time_col)
                    ax.set_ylabel('Correlation')
                    ax.grid(True, alpha=0.3)
                    
                    for t, corr in zip(clean_data[time_col], rolling_corr):
                        if not pd.isna(corr):
                            rolling_corr_data.append({
                                'var1': variables[j],
                                'var2': variables[i],
                                'time': t,
                                'rolling_correlation': corr,
                                'window_size': window
                            })
                    
                else:
                    x_data = clean_data[variables[j]]
                    y_data = clean_data[variables[i]]
                    
                    scatter = ax.scatter(x_data, y_data, c=clean_data[time_col], 
                                       alpha=0.6, s=20, cmap='viridis')
                    
                    correlation, _ = stats.pearsonr(x_data, y_data)
                    
                    ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel(variables[j])
                    ax.set_ylabel(variables[i])
                    ax.grid(True, alpha=0.3)
                    
                    if i == 0 and j == n_vars - 1:
                        plt.colorbar(scatter, ax=ax, label=time_col)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_folder, f'time_series_pairs_matrix_{len(variables)}vars_{window}window_{ts}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        time_series_df = pd.DataFrame(time_series_data)
        rolling_corr_df = pd.DataFrame(rolling_corr_data)
        
        ts_csv_filepath = os.path.join(output_folder, f'time_series_data_{len(variables)}vars_{window}window_{ts}.csv')
        time_series_df.to_csv(ts_csv_filepath, index=False)
        
        rolling_csv_filepath = os.path.join(output_folder, f'rolling_correlation_data_{len(variables)}vars_{window}window_{ts}.csv')
        rolling_corr_df.to_csv(rolling_csv_filepath, index=False)
        
        print(f"Saved time series pairs matrix plot to: {filepath}")
        print(f"Saved time series data to: {ts_csv_filepath}")
        print(f"Saved rolling correlation data to: {rolling_csv_filepath}")
        
        return filepath, ts_csv_filepath, rolling_csv_filepath
        
    except Exception as e:
        print(f"Error in plot_time_series_pairs_matrix: {e}")
        return None, None, None

def plot_seaborn_pairplot(df, variables=None, hue=None, plot_kws=None, diag_kind='hist',
                         output_folder=r'C:\Users\agsse\data_analysis\max_data'):
    """
    Create seaborn pairplot with CSV output
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if variables is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            variables = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
        
        if hue and hue in df.columns:
            plot_data = df[variables + [hue]].dropna()
        else:
            plot_data = df[variables].dropna()
            hue = None
        
        if len(plot_data) < 10:
            print(f"Insufficient data: only {len(plot_data)} rows available")
            return None, None
        
        if plot_kws is None:
            plot_kws = {'alpha': 0.6, 's': 20}
        
        seaborn_data = []
        
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i != j:
                    x_data = plot_data[variables[j]]
                    y_data = plot_data[variables[i]]
                    correlation, p_value = stats.pearsonr(x_data, y_data)
                    
                    seaborn_data.append({
                        'var1': variables[j],
                        'var2': variables[i],
                        'correlation': correlation,
                        'p_value': p_value,
                        'hue_variable': hue if hue else 'None',
                        'n_observations': len(plot_data)
                    })
        
        g = sns.pairplot(plot_data, vars=variables, hue=hue, 
                        plot_kws=plot_kws, diag_kind=diag_kind)
        
        g.fig.suptitle('Pairwise Relationships', y=1.02)
        
        hue_suffix = f'_hue_{hue}' if hue else ''
        filepath = os.path.join(output_folder, f'seaborn_pairplot_{len(variables)}vars{hue_suffix}_{ts}.png')
        g.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        seaborn_df = pd.DataFrame(seaborn_data)
        csv_filepath = os.path.join(output_folder, f'seaborn_pairplot_data_{len(variables)}vars{hue_suffix}_{ts}.csv')
        seaborn_df.to_csv(csv_filepath, index=False)
        
        print(f"Saved seaborn pairplot to: {filepath}")
        print(f"Saved seaborn pairplot data to: {csv_filepath}")
        
        return filepath, csv_filepath
        
    except Exception as e:
        print(f"Error in plot_seaborn_pairplot: {e}")
        return None, None

# Example usage function
def demo_pairs_matrix_plotting():
    """
    Demonstration function showing how to use the pairs matrix plotting functions
    """
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    sample_data = pd.DataFrame({
        'temp_avg': np.random.normal(20, 5, n_samples),
        'wind_speed_avg': np.random.normal(15, 3, n_samples),
        'pressure': np.random.normal(1013, 20, n_samples),
        'humidity': np.random.normal(60, 15, n_samples),
        'sol': range(n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
    })
    
    # Add some correlations
    sample_data['temp_correlated'] = sample_data['temp_avg'] * 0.8 + np.random.normal(0, 2, n_samples)
    sample_data['wind_correlated'] = sample_data['wind_speed_avg'] * -0.6 + np.random.normal(0, 3, n_samples)
    
    print("Running pairs matrix plotting demonstration...")
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test 1: Basic correlation pairs matrix
    print("\n1. Testing basic correlation pairs matrix...")
    plot_correlation_pairs_matrix(sample_data, method='pearson')
    
    # Test 2: Enhanced pairs matrix with color coding
    print("\n2. Testing enhanced pairs matrix...")
    plot_enhanced_pairs_matrix(sample_data, color_by='season')
    
    # Test 3: Specific variable pairs
    print("\n3. Testing specific variable pairs...")
    variable_pairs = [('temp_avg', 'temp_correlated'), ('wind_speed_avg', 'wind_correlated'), 
                     ('temp_avg', 'humidity'), ('pressure', 'wind_speed_avg')]
    plot_correlation_pairs_subset(sample_data, variable_pairs)
    
    # Test 4: Time series pairs matrix
    print("\n4. Testing time series pairs matrix...")
    plot_time_series_pairs_matrix(sample_data, time_col='sol', window=50)
    
    # Test 5: Seaborn pairplot
    print("\n5. Testing seaborn pairplot...")
    plot_seaborn_pairplot(sample_data, hue='season')
    
    print("\nDemo completed! Check the output folder for generated files.")

if __name__ == "__main__":
    # Run the demonstration
    demo_pairs_matrix_plotting()