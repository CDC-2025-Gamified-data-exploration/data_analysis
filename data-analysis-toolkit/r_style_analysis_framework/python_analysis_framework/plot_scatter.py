import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

def create_mars_wind_speed_scatterplot(input_file, output_csv='mars_wind_speed_data.csv'):
    """
    Create a scatterplot showing wind speed max vs sols, color-coded by Mars season
    
    Parameters:
    input_file (str): Path to input CSV file
    output_csv (str): Path for cleaned output CSV file
    """
    
    print("🌪️  Mars Wind Speed Analysis")
    print("=" * 40)
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{input_file}'")
        print("Please make sure the CSV file exists and update the filename.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Check for required columns
    required_cols = ['sol', 'wind_speed_avg', 'mars_season']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Additional useful columns if available
    optional_cols = ['temp_avg', 'environmental_stress_index', 'dust_season_risk', 
                    'solar_longitude_deg', 'mission_phase', 'activity_count']
    
    # Select columns for cleaned dataset
    output_cols = required_cols.copy()
    for col in optional_cols:
        if col in df.columns:
            output_cols.append(col)
    
    # Create cleaned dataset
    df_clean = df[output_cols].copy()
    
    # Remove rows where key columns are null
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['sol', 'wind_speed_avg', 'mars_season'])
    final_rows = len(df_clean)
    
    print(f"✓ Data cleaning: {initial_rows} → {final_rows} rows")
    print(f"  ({initial_rows - final_rows} rows removed due to missing values)")
    
    if final_rows == 0:
        print("❌ No valid data remaining after cleaning")
        return
    
    # Save cleaned dataset
    df_clean.to_csv(output_csv, index=False)
    print(f"✓ Cleaned dataset saved: {output_csv}")
    
    # Prepare data for visualization
    sol_data = df_clean['sol'].values
    wind_speed_data = df_clean['wind_speed_avg'].values
    seasons = df_clean['mars_season'].values
    
    # Define Mars season colors (Mars-themed palette)
    season_colors = {
        'Spring': '#06ffa5',    # Fresh green
        'Summer': '#ff6b35',    # Mars orange  
        'Autumn': '#fcbf49',    # Golden yellow
        'Fall': '#fcbf49',      # Alternative for Autumn
        'Winter': '#4cc9f0'     # Cool blue
    }
    
    # Get unique seasons and assign colors
    unique_seasons = sorted(df_clean['mars_season'].unique())
    print(f"✓ Mars seasons found: {unique_seasons}")
    
    # Ensure all seasons have colors
    default_colors = ['#ff6b35', '#06ffa5', '#fcbf49', '#4cc9f0', '#f72585']
    for i, season in enumerate(unique_seasons):
        if season not in season_colors:
            season_colors[season] = default_colors[i % len(default_colors)]
    
    # Create the visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#111111')
    
    # Create scatter plot for each season
    for season in unique_seasons:
        mask = seasons == season
        season_sols = sol_data[mask]
        season_winds = wind_speed_data[mask]
        
        ax.scatter(season_sols, season_winds, 
                  c=season_colors[season], 
                  label=f'{season} ({len(season_sols)} points)',
                  alpha=0.7, 
                  s=60, 
                  edgecolors='white', 
                  linewidth=0.5)
    
    # Calculate and display statistics
    print(f"\n📊 Wind Speed Statistics:")
    print(f"  Mean: {np.mean(wind_speed_data):.2f} m/s")
    print(f"  Max:  {np.max(wind_speed_data):.2f} m/s")
    print(f"  Min:  {np.min(wind_speed_data):.2f} m/s")
    print(f"  Std:  {np.std(wind_speed_data):.2f} m/s")
    
    # Add trend line for overall pattern
    if len(sol_data) > 5:
        # Calculate polynomial fit
        coeffs = np.polyfit(sol_data, wind_speed_data, 2)
        poly_func = np.poly1d(coeffs)
        
        # Generate smooth line points
        sol_range = np.linspace(sol_data.min(), sol_data.max(), 100)
        wind_trend = poly_func(sol_range)
        
        ax.plot(sol_range, wind_trend, 
               color='white', 
               linewidth=2, 
               alpha=0.8, 
               linestyle='--',
               label='Trend Line')
    
    # Add seasonal statistics as text annotations
    season_stats = df_clean.groupby('mars_season')['wind_speed_avg'].agg(['mean', 'max', 'count'])
    stats_text = "Seasonal Wind Speed Stats:\n"
    for season in unique_seasons:
        if season in season_stats.index:
            mean_wind = season_stats.loc[season, 'mean']
            max_wind = season_stats.loc[season, 'max']
            count = season_stats.loc[season, 'count']
            stats_text += f"{season}: μ={mean_wind:.1f}, max={max_wind:.1f} m/s (n={count})\n"
    
    # Add statistics box
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes, 
           fontsize=10,
           verticalalignment='top', 
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='black', 
                    alpha=0.8,
                    edgecolor='white'),
           color='white', 
           fontfamily='monospace')
    
    # Styling and labels
    ax.set_xlabel('Sol (Mars Day)', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Average Wind Speed (m/s)', fontsize=14, fontweight='bold', color='white')
    ax.set_title('Mars Average Wind Speed by Sol\nColor-Coded by Mars Season', 
                fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Grid and aesthetics
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    ax.tick_params(colors='white', labelsize=11)
    
    # Legend
    ax.legend(loc='upper right', 
             fontsize=11, 
             facecolor='black', 
             edgecolor='white',
             framealpha=0.9)
    
    # Add mission context if available
    if 'mission_phase' in df_clean.columns:
        # Add subtle phase boundaries
        phase_changes = df_clean.groupby('mission_phase')['sol'].agg(['min', 'max'])
        y_min, y_max = ax.get_ylim()
        
        for i, (phase, bounds) in enumerate(phase_changes.iterrows()):
            if i > 0:  # Don't draw line at the very beginning
                ax.axvline(bounds['min'], color='white', alpha=0.2, linestyle='-', linewidth=1)
                ax.text(bounds['min'] + 10, y_max * 0.95, phase, 
                       rotation=90, ha='left', va='top', 
                       color='white', alpha=0.6, fontsize=9)
    
    # Add wind speed severity indicators
    wind_percentiles = np.percentile(wind_speed_data, [25, 50, 75, 95])
    severity_colors = ['#06ffa5', '#fcbf49', '#ff6b35', '#f72585']
    severity_labels = ['Low', 'Moderate', 'High', 'Extreme']
    
    for i, (percentile, color, label) in enumerate(zip(wind_percentiles, severity_colors, severity_labels)):
        ax.axhline(percentile, color=color, alpha=0.3, linestyle=':', linewidth=1)
        ax.text(sol_data.max() * 0.98, percentile, 
               f'{label} ({percentile:.1f} m/s)', 
               ha='right', va='bottom', color=color, fontsize=9, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save high-resolution image
    output_image = 'mars_wind_speed_by_sol_scatterplot.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"✓ Visualization saved: {output_image}")
    
    plt.show()
    
    # Additional analysis and insights
    print(f"\n🔍 Analysis Insights:")
    
    # Seasonal analysis
    seasonal_analysis = df_clean.groupby('mars_season')['wind_speed_max'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    print(f"\n📊 Seasonal Wind Speed Analysis:")
    print(seasonal_analysis)
    
    # Find windiest periods
    windiest_sols = df_clean.nlargest(5, 'wind_speed_max')[['sol', 'wind_speed_max', 'mars_season']]
    print(f"\n🌪️  Top 5 Windiest Sols:")
    for _, row in windiest_sols.iterrows():
        print(f"  Sol {int(row['sol'])}: {row['wind_speed_max']:.2f} m/s ({row['mars_season']})")
    
    # Correlation with other factors if available
    if 'temp_avg' in df_clean.columns:
        temp_wind_corr = df_clean['temp_avg'].corr(df_clean['wind_speed_max'])
        print(f"\n🌡️  Temperature-Wind Correlation: {temp_wind_corr:.3f}")
    
    if 'dust_season_risk' in df_clean.columns:
        dust_wind_corr = df_clean['dust_season_risk'].corr(df_clean['wind_speed_max'])
        print(f"🌪️  Dust Risk-Wind Correlation: {dust_wind_corr:.3f}")
    
    # Mission timeline analysis
    sol_span = sol_data.max() - sol_data.min()
    print(f"\n⏱️  Mission Timeline Analysis:")
    print(f"  Total mission span: {sol_span} sols ({sol_span/687:.1f} Mars years)")
    print(f"  Wind data coverage: {final_rows} measurements")
    print(f"  Average measurement frequency: {sol_span/final_rows:.1f} sols between measurements")
    
    return df_clean

# =====================================================
# ADDITIONAL UTILITY FUNCTIONS
# =====================================================

def generate_wind_speed_summary_stats(df_clean):
    """Generate comprehensive summary statistics for the wind speed data"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE WIND SPEED ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    wind_data = df_clean['wind_speed_max']
    print(f"📊 Overall Wind Speed Statistics:")
    print(f"  Count:      {len(wind_data)}")
    print(f"  Mean:       {wind_data.mean():.2f} m/s")
    print(f"  Median:     {wind_data.median():.2f} m/s")
    print(f"  Std Dev:    {wind_data.std():.2f} m/s")
    print(f"  Min:        {wind_data.min():.2f} m/s")
    print(f"  Max:        {wind_data.max():.2f} m/s")
    print(f"  Range:      {wind_data.max() - wind_data.min():.2f} m/s")
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n📈 Wind Speed Percentiles:")
    for p in percentiles:
        value = np.percentile(wind_data, p)
        print(f"  {p:2d}th percentile: {value:.2f} m/s")
    
    # Seasonal breakdown
    if 'mars_season' in df_clean.columns:
        print(f"\n🌍 Seasonal Analysis:")
        seasonal_stats = df_clean.groupby('mars_season')['wind_speed_max'].describe().round(2)
        print(seasonal_stats)
    
    return wind_data.describe()

# =====================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = r"data_analysis\mars_mission_master_dataset_optimized.csv"  # Update this with your actual filename
    OUTPUT_CSV = r"mars_wind_speed_analysis_data.csv"
    
    print("🔴 Mars Wind Speed Analysis Script")
    print("Analyzing maximum wind speeds across Mars mission timeline")
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print("=" * 50)
    
    # Run the analysis
    try:
        cleaned_data = create_mars_wind_speed_scatterplot(INPUT_CSV, OUTPUT_CSV)
        
        if cleaned_data is not None and len(cleaned_data) > 0:
            # Generate additional summary statistics
            summary_stats = generate_wind_speed_summary_stats(cleaned_data)
            
            print("\n✅ Analysis Complete!")
            print("Generated files:")
            print(f"  📊 Data: {OUTPUT_CSV}")
            print(f"  📈 Chart: mars_wind_speed_by_sol_scatterplot.png")
            
        else:
            print("❌ Analysis failed - please check your data file")
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("Please check your file path and data format")

# =====================================================
# ADVANCED ANALYSIS OPTION
# =====================================================

def create_enhanced_wind_analysis(input_file):
    """
    Create additional enhanced wind speed analysis with multiple views
    This is an optional extended analysis
    """
    
    df = pd.read_csv(input_file)
    
    if all(col in df.columns for col in ['sol', 'wind_speed_max', 'mars_season']):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#0a0a0a')
        
        # 1. Main scatterplot (same as above but smaller)
        create_mars_wind_speed_scatterplot(input_file, 'temp_wind_data.csv')
        
        # 2. Seasonal box plots
        df_clean = df.dropna(subset=['wind_speed_max', 'mars_season'])
        seasons = sorted(df_clean['mars_season'].unique())
        season_data = [df_clean[df_clean['mars_season'] == season]['wind_speed_max'].values 
                      for season in seasons]
        
        box_plot = ax2.boxplot(season_data, labels=seasons, patch_artist=True)
        colors = ['#06ffa5', '#ff6b35', '#fcbf49', '#4cc9f0']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Wind Speed Max (m/s)', color='white', fontweight='bold')
        ax2.set_title('Wind Speed Distribution by Season', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram
        ax3.hist(df_clean['wind_speed_max'], bins=20, color='#ff6b35', alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Wind Speed Max (m/s)', color='white', fontweight='bold')
        ax3.set_ylabel('Frequency', color='white', fontweight='bold')
        ax3.set_title('Wind Speed Distribution', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling average
        df_sorted = df_clean.sort_values('sol')
        if len(df_sorted) > 10:
            window_size = max(7, len(df_sorted) // 20)  # Adaptive window size
            df_sorted['wind_rolling'] = df_sorted['wind_speed_max'].rolling(window=window_size, center=True).mean()
            
            ax4.plot(df_sorted['sol'], df_sorted['wind_speed_max'], alpha=0.3, color='gray', linewidth=1)
            ax4.plot(df_sorted['sol'], df_sorted['wind_rolling'], color='#ff6b35', linewidth=3)
            ax4.set_xlabel('Sol', color='white', fontweight='bold')
            ax4.set_ylabel('Wind Speed (m/s)', color='white', fontweight='bold')
            ax4.set_title(f'Wind Speed Trend ({window_size}-Sol Rolling Average)', color='white', fontweight='bold')
            ax4.tick_params(colors='white')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mars_enhanced_wind_analysis.png', dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.show()
        
        print("✓ Enhanced analysis saved: mars_enhanced_wind_analysis.png")

# Uncomment the line below to run enhanced analysis
# create_enhanced_wind_analysis(INPUT_CSV)