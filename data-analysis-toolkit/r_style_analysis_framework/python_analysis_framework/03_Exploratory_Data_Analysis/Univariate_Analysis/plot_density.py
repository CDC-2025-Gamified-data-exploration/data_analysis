import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def create_activity_density_plot(csv_file_path):
    """
    Creates a density plot showing number of activities performed at every hour during the day.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records from {csv_file_path}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Assuming the UTC timestamp is in the last column based on your image
    # Adjust column name if needed
    utc_column = df.columns[-1]  # Last column appears to be UTC timestamp
    
    print(f"Using UTC column: '{utc_column}'")
    print(f"Sample timestamps: {df[utc_column].head()}")
    
    # Extract hour from UTC timestamp
    # Handle different possible timestamp formats
    hours = []
    
    for timestamp in df[utc_column]:
        try:
            # Try parsing as datetime string (e.g., "2018-11-27 18:50")
            if isinstance(timestamp, str):
                # Split by space and take the time part
                if ' ' in timestamp:
                    time_part = timestamp.split(' ')[1]
                    hour = int(time_part.split(':')[0])
                else:
                    # If no space, might be just time
                    hour = int(timestamp.split(':')[0])
            else:
                # If already datetime object
                hour = timestamp.hour
            
            hours.append(hour)
            
        except (ValueError, AttributeError, IndexError) as e:
            print(f"Warning: Could not parse timestamp '{timestamp}': {e}")
            continue
    
    if not hours:
        print("Error: No valid timestamps found")
        return
    
    print(f"Successfully extracted {len(hours)} hour values")
    
    # Create DataFrame with hours
    hour_df = pd.DataFrame({'hour': hours})
    
    # Count activities by hour
    hourly_counts = hour_df['hour'].value_counts().sort_index()
    
    # Ensure all 24 hours are represented (fill missing hours with 0)
    all_hours = pd.Series(0, index=range(24))
    all_hours.update(hourly_counts)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.bar(all_hours.index, all_hours.values, 
                   color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
    
    # Customize the plot
    plt.title('Activity Density by Hour of Day (UTC)', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day (UTC)', fontsize=12)
    plt.ylabel('Number of Activities', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis ticks and labels
    plt.xticks(range(0, 24, 2))  # Show every 2 hours
    plt.xlim(-0.5, 23.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, all_hours.values):
        if value > 0:  # Only show labels for non-zero values
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(all_hours.values)*0.01,
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    # Add statistics text
    total_activities = len(hours)
    peak_hour = all_hours.idxmax()
    peak_count = all_hours.max()
    
    stats_text = f'Total Activities: {total_activities}\nPeak Hour: {peak_hour:02d}:00 UTC ({peak_count} activities)'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Also create a density plot (smooth curve)
    plt.figure(figsize=(12, 8))
    
    # Create a smooth density plot
    hours_array = np.array(hours)
    plt.hist(hours_array, bins=24, range=(0, 24), density=True, alpha=0.7, 
             color='lightcoral', edgecolor='darkred', linewidth=0.5)
    
    # Add KDE curve
    from scipy import stats
    density = stats.gaussian_kde(hours_array)
    x_range = np.linspace(0, 24, 1000)
    plt.plot(x_range, density(x_range), 'navy', linewidth=2, label='Density Curve')
    
    plt.title('Activity Density Distribution by Hour of Day (UTC)', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day (UTC)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.xlim(0, 24)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ACTIVITY DENSITY SUMMARY")
    print("="*50)
    print(f"Total activities analyzed: {total_activities}")
    print(f"Peak activity hour: {peak_hour:02d}:00 UTC ({peak_count} activities)")
    print(f"Quietest hour: {all_hours.idxmin():02d}:00 UTC ({all_hours.min()} activities)")
    print(f"Average activities per hour: {total_activities/24:.1f}")
    
    print("\nHourly breakdown:")
    for hour in range(24):
        count = all_hours[hour]
        bar = "█" * int(count * 30 / max(all_hours.values)) if count > 0 else ""
        print(f"{hour:02d}:00 - {hour:02d}:59 UTC: {count:3d} activities {bar}")

# Example usage
if __name__ == "__main__":
    # Use raw string (r"") or forward slashes to handle file paths properly
    # Option 1: Raw string (recommended for Windows paths)
    csv_file_path = r"data-analysis-toolkit\r_style_analysis_framework\python_analysis_framework\02_Data_Cleaning_and_Preparation\Prepped_data\INS_InSight__Surface_Activity_Overview.csv"
    
    # Option 2: Forward slashes (works on all systems)
    # csv_file_path = "data-analysis-toolkit/r_style_analysis_framework/python_analysis_framework/02_Data_Cleaning_and_Preparation/Prepped_data/INS_InSight__Surface_Activity_Overview.csv"
    
    # Option 3: Double backslashes
    # csv_file_path = "data-analysis-toolkit\\r_style_analysis_framework\\python_analysis_framework\\02_Data_Cleaning_and_Preparation\\Prepped_data\\INS_InSight__Surface_Activity_Overview.csv"
    
    create_activity_density_plot(csv_file_path)
    