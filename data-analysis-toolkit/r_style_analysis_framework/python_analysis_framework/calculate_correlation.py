import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def analyze_temp_stress_relationship(input_file, output_file='mars_temp_stress_cleaned.csv'):
    """
    Analyze relationship between temperature range and environmental stress index
    Creates visualization and cleaned dataset
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path for cleaned output CSV file
    """
    
    # Read the data
    print("Loading Mars mission data...")
    df = pd.read_csv(input_file)
    
    # Check required columns exist
    required_cols = ['temp_range', 'environmental_stress_index', 'sol', 'mars_season', 'mission_phase']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create cleaned dataset with relevant columns for analysis
    analysis_cols = [
        'sol',
        'temp_range', 
        'environmental_stress_index',
        'temp_min',
        'temp_max', 
        'temp_avg',
        'mars_season',
        'mission_phase',
        'solar_longitude_deg',
        'dust_season_risk',
        'orbital_thermal_factor',
        'wind_speed_avg',
        'wind_speed_max'
    ]
    
    # Filter to only include columns that exist in the dataset
    available_cols = [col for col in analysis_cols if col in df.columns]
    df_clean = df[available_cols].copy()
    
    # Remove rows where either temp_range or environmental_stress_index is null
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['temp_range', 'environmental_stress_index'])
    final_rows = len(df_clean)
    
    print(f"Data cleaning: {initial_rows} -> {final_rows} rows ({initial_rows - final_rows} removed due to missing values)")
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    # Extract data for analysis
    temp_range = df_clean['temp_range'].values
    stress_index = df_clean['environmental_stress_index'].values
    
    # Calculate correlation and regression
    correlation = np.corrcoef(temp_range, stress_index)[0, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(temp_range, stress_index)
    
    # Create the visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')
    
    # Create color mapping based on mission phase if available
    if 'mission_phase' in df_clean.columns:
        phases = df_clean['mission_phase'].unique()
        colors = ['#ff6b35', '#f77f00', '#fcbf49', '#06ffa5', '#4cc9f0']
        phase_colors = {phase: colors[i % len(colors)] for i, phase in enumerate(phases)}
        
        for phase in phases:
            mask = df_clean['mission_phase'] == phase
            ax.scatter(temp_range[mask], stress_index[mask], 
                      c=phase_colors[phase], label=phase, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    else:
        # Use gradient coloring based on sol if mission_phase not available
        scatter = ax.scatter(temp_range, stress_index, c=df_clean['sol'], 
                           cmap='plasma', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sol (Mars Day)', fontsize=12, color='white')
        cbar.ax.tick_params(colors='white')
    
    # Add line of best fit
    line_x = np.linspace(temp_range.min(), temp_range.max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='#ff4444', linewidth=2, alpha=0.8, label=f'Best Fit Line (r={correlation:.3f})')
    
    # Add confidence interval
    def predict_interval(x, y, new_x, confidence=0.95):
        n = len(x)
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        sxy = np.sum((x - x_mean) * (y - np.mean(y)))
        syy = np.sum((y - np.mean(y)) ** 2)
        s_yx = np.sqrt((syy - sxy**2/sxx) / (n-2))
        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)
        
        margin = t_val * s_yx * np.sqrt(1/n + (new_x - x_mean)**2 / sxx)
        return margin
    
    margin = predict_interval(temp_range, stress_index, line_x)
    ax.fill_between(line_x, line_y - margin, line_y + margin, 
                    color='#ff4444', alpha=0.2, label='95% Confidence Interval')
    
    # Styling
    ax.set_xlabel('Temperature Range (°C)', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('Environmental Stress Index', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Mars Temperature Range vs Environmental Stress Index\nInSight Mission Analysis', 
                fontsize=16, color='white', fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='white', labelsize=11)
    
    # Add statistics text box
    stats_text = f"""Statistical Analysis:
Correlation: r = {correlation:.3f}
R-squared: R² = {r_value**2:.3f}
P-value: p = {p_value:.2e}
Sample size: n = {len(temp_range)}
Equation: y = {slope:.3f}x + {intercept:.3f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
            color='white', fontfamily='monospace')
    
    # Legend
    ax.legend(loc='lower right', fontsize=10, facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'mars_temp_range_vs_stress_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"Visualization saved to: {plot_filename}")
    
    # Display the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Temperature Range Statistics:")
    print(f"  Mean: {np.mean(temp_range):.2f}°C")
    print(f"  Std:  {np.std(temp_range):.2f}°C")
    print(f"  Min:  {np.min(temp_range):.2f}°C")
    print(f"  Max:  {np.max(temp_range):.2f}°C")
    
    print(f"\nEnvironmental Stress Index Statistics:")
    print(f"  Mean: {np.mean(stress_index):.3f}")
    print(f"  Std:  {np.std(stress_index):.3f}")
    print(f"  Min:  {np.min(stress_index):.3f}")
    print(f"  Max:  {np.max(stress_index):.3f}")
    
    print(f"\nRelationship Analysis:")
    print(f"  Correlation coefficient: {correlation:.4f}")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Regression equation: Stress = {slope:.4f} * TempRange + {intercept:.4f}")
    
    # Interpretation
    if abs(correlation) > 0.7:
        strength = "strong"
    elif abs(correlation) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    
    direction = "positive" if correlation > 0 else "negative"
    
    print(f"\nInterpretation:")
    print(f"  There is a {strength} {direction} correlation between temperature range")
    print(f"  and environmental stress index on Mars.")
    
    if p_value < 0.001:
        print(f"  This relationship is highly statistically significant (p < 0.001)")
    elif p_value < 0.05:
        print(f"  This relationship is statistically significant (p < 0.05)")
    else:
        print(f"  This relationship is not statistically significant (p ≥ 0.05)")
    
    return df_clean, correlation, r_value**2, p_value

# Example usage
if __name__ == "__main__":
    input_filename = 'data_analysis\mars_mission_master_dataset_optimized.csv'  
    try:
        cleaned_data, correlation, r_squared, p_value = analyze_temp_stress_relationship(input_filename)
        print(f"\nAnalysis complete! Check output files for results.")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_filename}'")
        print("Please make sure the CSV file exists and update the filename in the script.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check your data format and column names.")