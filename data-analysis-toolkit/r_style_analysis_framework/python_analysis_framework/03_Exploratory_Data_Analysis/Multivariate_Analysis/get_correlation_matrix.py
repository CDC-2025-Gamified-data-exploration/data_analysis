import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_weather_correlations(csv_file_path):
    """
    Analyzes weather data to create correlation matrices for:
    1. Average temperature vs wind speed
    2. Time of day vs average temperature
    3. Time of day vs wind speed
    4. Month of year vs average temperature
    5. Month of year vs wind speed
    """
    
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
    
    # Extract time features
    df['hour'] = df[datetime_column].dt.hour
    df['month'] = df[datetime_column].dt.month
    df['day_of_year'] = df[datetime_column].dt.dayofyear
    
    # Clean data - remove NaN values
    clean_df = df.dropna(subset=['avg_temperature', wind_column])
    print(f"\nAfter removing NaN values: {len(clean_df)} records")
    
    # Create figure with subplots for all correlation analyses
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Weather Data Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average Temperature vs Wind Speed Correlation
    ax1 = axes[0, 0]
    correlation_temp_wind = clean_df[['avg_temperature', wind_column]].corr()
    sns.heatmap(correlation_temp_wind, annot=True, cmap='RdBu_r', center=0, 
                ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Temperature vs Wind Speed\nCorrelation Matrix')
    
    # 2. Time of Day vs Average Temperature
    ax2 = axes[0, 1]
    hourly_temp = clean_df.groupby('hour')['avg_temperature'].mean().reset_index()
    correlation_hour_temp = clean_df[['hour', 'avg_temperature']].corr()
    sns.heatmap(correlation_hour_temp, annot=True, cmap='RdBu_r', center=0, 
                ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Hour of Day vs Temperature\nCorrelation Matrix')
    
    # 3. Time of Day vs Wind Speed
    ax3 = axes[0, 2]
    correlation_hour_wind = clean_df[['hour', wind_column]].corr()
    sns.heatmap(correlation_hour_wind, annot=True, cmap='RdBu_r', center=0, 
                ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Hour of Day vs Wind Speed\nCorrelation Matrix')
    
    # 4. Month of Year vs Average Temperature
    ax4 = axes[1, 0]
    correlation_month_temp = clean_df[['month', 'avg_temperature']].corr()
    sns.heatmap(correlation_month_temp, annot=True, cmap='RdBu_r', center=0, 
                ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Month vs Temperature\nCorrelation Matrix')
    
    # 5. Month of Year vs Wind Speed
    ax5 = axes[1, 1]
    correlation_month_wind = clean_df[['month', wind_column]].corr()
    sns.heatmap(correlation_month_wind, annot=True, cmap='RdBu_r', center=0, 
                ax=ax5, cbar_kws={'shrink': 0.8})
    ax5.set_title('Month vs Wind Speed\nCorrelation Matrix')
    
    # 6. Combined correlation matrix
    ax6 = axes[1, 2]
    combined_vars = clean_df[['avg_temperature', wind_column, 'hour', 'month']]
    correlation_combined = combined_vars.corr()
    sns.heatmap(correlation_combined, annot=True, cmap='RdBu_r', center=0, 
                ax=ax6, cbar_kws={'shrink': 0.8})
    ax6.set_title('Combined Variables\nCorrelation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Create detailed time series plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Weather Patterns Analysis', fontsize=16, fontweight='bold')
    
    # Hourly patterns
    hourly_stats = clean_df.groupby('hour').agg({
        'avg_temperature': ['mean', 'std'],
        wind_column: ['mean', 'std']
    }).round(2)
    
    # Plot 1: Average temperature by hour
    ax1 = axes[0, 0]
    hourly_temp = clean_df.groupby('hour')['avg_temperature'].mean()
    hourly_temp_std = clean_df.groupby('hour')['avg_temperature'].std()
    ax1.plot(hourly_temp.index, hourly_temp.values, 'o-', color='red', linewidth=2, markersize=6)
    ax1.fill_between(hourly_temp.index, 
                     hourly_temp.values - hourly_temp_std.values,
                     hourly_temp.values + hourly_temp_std.values,
                     alpha=0.3, color='red')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Temperature')
    ax1.set_title('Temperature by Hour of Day')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # Plot 2: Wind speed by hour
    ax2 = axes[0, 1]
    hourly_wind = clean_df.groupby('hour')[wind_column].mean()
    hourly_wind_std = clean_df.groupby('hour')[wind_column].std()
    ax2.plot(hourly_wind.index, hourly_wind.values, 'o-', color='blue', linewidth=2, markersize=6)
    ax2.fill_between(hourly_wind.index,
                     hourly_wind.values - hourly_wind_std.values,
                     hourly_wind.values + hourly_wind_std.values,
                     alpha=0.3, color='blue')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Wind Speed')
    ax2.set_title('Wind Speed by Hour of Day')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    # Plot 3: Temperature by month
    ax3 = axes[1, 0]
    monthly_temp = clean_df.groupby('month')['avg_temperature'].mean()
    monthly_temp_std = clean_df.groupby('month')['avg_temperature'].std()
    ax3.plot(monthly_temp.index, monthly_temp.values, 'o-', color='red', linewidth=2, markersize=8)
    ax3.fill_between(monthly_temp.index,
                     monthly_temp.values - monthly_temp_std.values,
                     monthly_temp.values + monthly_temp_std.values,
                     alpha=0.3, color='red')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Temperature')
    ax3.set_title('Temperature by Month')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, 13))
    
    # Plot 4: Wind speed by month
    ax4 = axes[1, 1]
    monthly_wind = clean_df.groupby('month')[wind_column].mean()
    monthly_wind_std = clean_df.groupby('month')[wind_column].std()
    ax4.plot(monthly_wind.index, monthly_wind.values, 'o-', color='blue', linewidth=2, markersize=8)
    ax4.fill_between(monthly_wind.index,
                     monthly_wind.values - monthly_wind_std.values,
                     monthly_wind.values + monthly_wind_std.values,
                     alpha=0.3, color='blue')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Wind Speed')
    ax4.set_title('Wind Speed by Month')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 13))
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation summary
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    
    corr_temp_wind = clean_df['avg_temperature'].corr(clean_df[wind_column])
    corr_hour_temp = clean_df['hour'].corr(clean_df['avg_temperature'])
    corr_hour_wind = clean_df['hour'].corr(clean_df[wind_column])
    corr_month_temp = clean_df['month'].corr(clean_df['avg_temperature'])
    corr_month_wind = clean_df['month'].corr(clean_df[wind_column])
    
    print(f"1. Temperature vs Wind Speed:        {corr_temp_wind:.4f}")
    print(f"2. Hour of Day vs Temperature:       {corr_hour_temp:.4f}")
    print(f"3. Hour of Day vs Wind Speed:        {corr_hour_wind:.4f}")
    print(f"4. Month vs Temperature:             {corr_month_temp:.4f}")
    print(f"5. Month vs Wind Speed:              {corr_month_wind:.4f}")
    
    print(f"\nCorrelation strength interpretation:")
    print(f"0.0 - 0.3: Weak correlation")
    print(f"0.3 - 0.7: Moderate correlation")
    print(f"0.7 - 1.0: Strong correlation")
    
    # Statistical summary
    print(f"\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    print(f"Data period: {clean_df[datetime_column].min()} to {clean_df[datetime_column].max()}")
    print(f"Total records: {len(clean_df)}")
    print(f"\nTemperature Statistics:")
    print(f"  Mean: {clean_df['avg_temperature'].mean():.2f}")
    print(f"  Std:  {clean_df['avg_temperature'].std():.2f}")
    print(f"  Min:  {clean_df['avg_temperature'].min():.2f}")
    print(f"  Max:  {clean_df['avg_temperature'].max():.2f}")
    print(f"\nWind Speed Statistics:")
    print(f"  Mean: {clean_df[wind_column].mean():.2f}")
    print(f"  Std:  {clean_df[wind_column].std():.2f}")
    print(f"  Min:  {clean_df[wind_column].min():.2f}")
    print(f"  Max:  {clean_df[wind_column].max():.2f}")
    
    # Save processed data option
    output_file = csv_file_path.replace('.csv', '_processed.csv')
    processed_df = clean_df[['bin', 'avg_temperature', wind_column, 'hour', 'month']].copy()
    processed_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    return clean_df

# Example usage
if __name__ == "__main__":
    
    csv_file_path = r"data-analysis-toolkit\r_style_analysis_framework\python_analysis_framework\02_Data_Cleaning_and_Preparation\Prepped_data\mars_weather_15min.csv"
    analyze_weather_correlations(csv_file_path)
    