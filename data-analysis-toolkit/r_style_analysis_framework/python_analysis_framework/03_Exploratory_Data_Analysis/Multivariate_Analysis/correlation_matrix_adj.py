import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_mars_weather_correlations(csv_file_path):
    """
    Analyzes Mars weather data to create correlation matrices adjusted for Martian years:
    1. Average temperature vs wind speed
    2. Martian time of sol vs average temperature  
    3. Martian time of sol vs wind speed
    4. Martian season vs average temperature
    5. Martian season vs wind speed
    
    Uses Fed-style seasonal adjustment methodology for Martian year conversion.
    """
    
    # Martian year constants
    MARS_YEAR_DAYS = 687  # Earth days in a Martian year
    MARS_SOL_LENGTH = 24.6597  # Hours in a Martian sol (day)
    EARTH_YEAR_DAYS = 365.25
    MARS_SEASONS = 6  # Traditional Martian seasons (some use 4, but 6 provides better granularity)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records from CSV")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Identify temperature and wind speed columns
    temp_columns = [col for col in df.columns if 'TEMP' in col.upper()]
    wind_column = [col for col in df.columns if 'WIND_SPEED' in col.upper()][0]
    datetime_column = 'bin'  # Based on your description
    
    print(f"\nTemperature columns found: {temp_columns}")
    print(f"Wind speed column: {wind_column}")
    print(f"DateTime column: {datetime_column}")
    
    # Convert datetime column to datetime format
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Calculate average temperature across the 3 temperature measurements
    df['avg_temperature'] = df[temp_columns].mean(axis=1)
    
    # MARTIAN YEAR ADJUSTMENTS (Fed-style seasonal adjustment)
    # Convert Earth time to Mars time
    df['earth_days_from_start'] = (df[datetime_column] - df[datetime_column].min()).dt.days
    
    # Calculate Martian year and sol (Martian day)
    df['mars_year'] = (df['earth_days_from_start'] / MARS_YEAR_DAYS).astype(int)
    df['mars_sol_in_year'] = df['earth_days_from_start'] % MARS_YEAR_DAYS
    df['mars_season'] = ((df['mars_sol_in_year'] / MARS_YEAR_DAYS) * MARS_SEASONS).astype(int)
    
    # Martian hour of sol (24.6 hour Martian day cycle)
    df['mars_hour_of_sol'] = ((df['earth_days_from_start'] * 24) % MARS_SOL_LENGTH).astype(int)
    
    # Fed-style seasonal adjustment factors
    # Calculate long-term seasonal patterns for detrending
    seasonal_temp_factors = df.groupby('mars_season')['avg_temperature'].transform('mean')
    seasonal_wind_factors = df.groupby('mars_season')[wind_column].transform('mean')
    
    # Apply seasonal adjustment (similar to Fed X-13ARIMA-SEATS methodology)
    overall_temp_mean = df['avg_temperature'].mean()
    overall_wind_mean = df[wind_column].mean()
    
    df['seasonally_adjusted_temp'] = df['avg_temperature'] - (seasonal_temp_factors - overall_temp_mean)
    df['seasonally_adjusted_wind'] = df[wind_column] - (seasonal_wind_factors - overall_wind_mean)
    
    # Clean data - remove NaN values
    clean_df = df.dropna(subset=['avg_temperature', wind_column])
    print(f"\nAfter removing NaN values: {len(clean_df)} records")
    print(f"Data spans {clean_df['mars_year'].max() - clean_df['mars_year'].min() + 1} Martian years")
    print(f"Martian seasons represented: {sorted(clean_df['mars_season'].unique())}")
    
    # Create correlation matrix for seasonally adjusted variables
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('Mars Weather Data Correlation Analysis (Seasonally Adjusted)', fontsize=16, fontweight='bold')
    
    combined_vars = clean_df[['seasonally_adjusted_temp', 'seasonally_adjusted_wind', 
                             'mars_hour_of_sol', 'mars_season']]
    correlation_combined = combined_vars.corr()
    sns.heatmap(correlation_combined, annot=True, cmap='RdBu_r', center=0, 
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Seasonally Adjusted Variables\nCorrelation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Create detailed time series plots for Martian patterns with tighter y-axis scaling
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Mars Weather Patterns Analysis (Martian Time)', fontsize=16, fontweight='bold')
    
    # Plot 1: Average temperature by Martian hour of sol
    ax1 = axes[0, 0]
    sol_temp = clean_df.groupby('mars_hour_of_sol')['avg_temperature'].mean()
    sol_temp_std = clean_df.groupby('mars_hour_of_sol')['avg_temperature'].std()
    ax1.plot(sol_temp.index, sol_temp.values, 'o-', color='red', linewidth=2, markersize=6)
    ax1.fill_between(sol_temp.index, 
                     sol_temp.values - sol_temp_std.values,
                     sol_temp.values + sol_temp_std.values,
                     alpha=0.3, color='red')
    ax1.set_xlabel('Martian Hour of Sol')
    ax1.set_ylabel('Average Temperature')
    ax1.set_title('Temperature by Martian Hour of Sol\n(24.66 hour Martian day cycle)')
    ax1.grid(True, alpha=0.3)
    # Tighter y-axis scaling
    temp_range = sol_temp.max() - sol_temp.min()
    ax1.set_ylim(sol_temp.min() - temp_range * 0.1, sol_temp.max() + temp_range * 0.1)
    
    # Plot 2: Wind speed by Martian hour of sol
    ax2 = axes[0, 1]
    sol_wind = clean_df.groupby('mars_hour_of_sol')[wind_column].mean()
    sol_wind_std = clean_df.groupby('mars_hour_of_sol')[wind_column].std()
    ax2.plot(sol_wind.index, sol_wind.values, 'o-', color='blue', linewidth=2, markersize=6)
    ax2.fill_between(sol_wind.index,
                     sol_wind.values - sol_wind_std.values,
                     sol_wind.values + sol_wind_std.values,
                     alpha=0.3, color='blue')
    ax2.set_xlabel('Martian Hour of Sol')
    ax2.set_ylabel('Wind Speed')
    ax2.set_title('Wind Speed by Martian Hour of Sol')
    ax2.grid(True, alpha=0.3)
    # Tighter y-axis scaling
    wind_range = sol_wind.max() - sol_wind.min()
    ax2.set_ylim(sol_wind.min() - wind_range * 0.1, sol_wind.max() + wind_range * 0.1)
    
    # Plot 3: Temperature by Martian season
    ax3 = axes[1, 0]
    season_temp = clean_df.groupby('mars_season')['avg_temperature'].mean()
    season_temp_std = clean_df.groupby('mars_season')['avg_temperature'].std()
    ax3.plot(season_temp.index, season_temp.values, 'o-', color='red', linewidth=2, markersize=8)
    ax3.fill_between(season_temp.index,
                     season_temp.values - season_temp_std.values,
                     season_temp.values + season_temp_std.values,
                     alpha=0.3, color='red')
    ax3.set_xlabel('Martian Season (0-5)')
    ax3.set_ylabel('Average Temperature')
    ax3.set_title('Temperature by Martian Season\n(687 Earth day year)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, MARS_SEASONS))
    # Tighter y-axis scaling
    season_temp_range = season_temp.max() - season_temp.min()
    ax3.set_ylim(season_temp.min() - season_temp_range * 0.1, season_temp.max() + season_temp_range * 0.1)
    
    # Plot 4: Wind speed by Martian season
    ax4 = axes[1, 1]
    season_wind = clean_df.groupby('mars_season')[wind_column].mean()
    season_wind_std = clean_df.groupby('mars_season')[wind_column].std()
    ax4.plot(season_wind.index, season_wind.values, 'o-', color='blue', linewidth=2, markersize=8)
    ax4.fill_between(season_wind.index,
                     season_wind.values - season_wind_std.values,
                     season_wind.values + season_wind_std.values,
                     alpha=0.3, color='blue')
    ax4.set_xlabel('Martian Season (0-5)')
    ax4.set_ylabel('Wind Speed')
    ax4.set_title('Wind Speed by Martian Season')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(0, MARS_SEASONS))
    # Tighter y-axis scaling
    season_wind_range = season_wind.max() - season_wind.min()
    ax4.set_ylim(season_wind.min() - season_wind_range * 0.1, season_wind.max() + season_wind_range * 0.1)
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation summary
    print("\n" + "="*80)
    print("MARS WEATHER CORRELATION ANALYSIS SUMMARY (MARTIAN TIME ADJUSTED)")
    print("="*80)
    
    corr_temp_wind = clean_df['avg_temperature'].corr(clean_df[wind_column])
    corr_sol_temp = clean_df['mars_hour_of_sol'].corr(clean_df['avg_temperature'])
    corr_sol_wind = clean_df['mars_hour_of_sol'].corr(clean_df[wind_column])
    corr_season_temp = clean_df['mars_season'].corr(clean_df['avg_temperature'])
    corr_season_wind = clean_df['mars_season'].corr(clean_df[wind_column])
    
    # Seasonally adjusted correlations
    corr_adj_temp_wind = clean_df['seasonally_adjusted_temp'].corr(clean_df['seasonally_adjusted_wind'])
    
    print(f"ORIGINAL CORRELATIONS:")
    print(f"1. Temperature vs Wind Speed:           {corr_temp_wind:.4f}")
    print(f"2. Martian Hour of Sol vs Temperature:  {corr_sol_temp:.4f}")
    print(f"3. Martian Hour of Sol vs Wind Speed:   {corr_sol_wind:.4f}")
    print(f"4. Martian Season vs Temperature:       {corr_season_temp:.4f}")
    print(f"5. Martian Season vs Wind Speed:        {corr_season_wind:.4f}")
    print(f"\nSEASONALLY ADJUSTED CORRELATIONS:")
    print(f"6. Adj. Temperature vs Adj. Wind Speed: {corr_adj_temp_wind:.4f}")
    
    print(f"\nCorrelation strength interpretation:")
    print(f"0.0 - 0.3: Weak correlation")
    print(f"0.3 - 0.7: Moderate correlation")  
    print(f"0.7 - 1.0: Strong correlation")
    
    # Martian time statistics
    print(f"\n" + "="*80)
    print("MARTIAN TIME CONVERSION SUMMARY")
    print("="*80)
    print(f"Earth date range: {clean_df[datetime_column].min()} to {clean_df[datetime_column].max()}")
    print(f"Martian years covered: {clean_df['mars_year'].min()} to {clean_df['mars_year'].max()}")
    print(f"Total Earth days: {clean_df['earth_days_from_start'].max()}")
    print(f"Equivalent Martian years: {clean_df['earth_days_from_start'].max() / MARS_YEAR_DAYS:.2f}")
    print(f"Martian sol length: {MARS_SOL_LENGTH} Earth hours")
    print(f"Martian year length: {MARS_YEAR_DAYS} Earth days")
    
    # Statistical summary
    print(f"\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"Total records: {len(clean_df)}")
    print(f"\nOriginal Temperature Statistics:")
    print(f"  Mean: {clean_df['avg_temperature'].mean():.2f}")
    print(f"  Std:  {clean_df['avg_temperature'].std():.2f}")
    print(f"  Min:  {clean_df['avg_temperature'].min():.2f}")
    print(f"  Max:  {clean_df['avg_temperature'].max():.2f}")
    print(f"\nSeasonally Adjusted Temperature Statistics:")
    print(f"  Mean: {clean_df['seasonally_adjusted_temp'].mean():.2f}")
    print(f"  Std:  {clean_df['seasonally_adjusted_temp'].std():.2f}")
    print(f"\nOriginal Wind Speed Statistics:")
    print(f"  Mean: {clean_df[wind_column].mean():.2f}")
    print(f"  Std:  {clean_df[wind_column].std():.2f}")
    print(f"  Min:  {clean_df[wind_column].min():.2f}")
    print(f"  Max:  {clean_df[wind_column].max():.2f}")
    print(f"\nSeasonally Adjusted Wind Speed Statistics:")
    print(f"  Mean: {clean_df['seasonally_adjusted_wind'].mean():.2f}")
    print(f"  Std:  {clean_df['seasonally_adjusted_wind'].std():.2f}")
    
    # Save processed data with Martian time adjustments
    output_file = csv_file_path.replace('.csv', '_mars_adjusted.csv')
    processed_df = clean_df[['bin', 'avg_temperature', wind_column, 'mars_hour_of_sol', 
                            'mars_season', 'mars_year', 'seasonally_adjusted_temp', 
                            'seasonally_adjusted_wind']].copy()
    processed_df.to_csv(output_file, index=False)
    print(f"\nProcessed Mars-adjusted data saved to: {output_file}")
    
    return clean_df

# Example usage
if __name__ == "__main__":
    # Update this path to your actual CSV file
    csv_file_path = r"data-analysis-toolkit\r_style_analysis_framework\python_analysis_framework\02_Data_Cleaning_and_Preparation\Prepped_data\mars_weather_15min.csv"
    
    analyze_mars_weather_correlations(csv_file_path)
    