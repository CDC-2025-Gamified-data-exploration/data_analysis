import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_mars_mission_data():
    """Generate Mars InSight Mission Master Dataset column names"""
    
    # All column names based on the brief
    columns = [
        # Original Mission Data (12 columns)
        'datetime', 'sol', 'activity', 'solar_longitude_deg',
        'year', 'month', 'day', 'hour', 'day_of_year', 'weekday',
        
        # Mission Enrichment Features (~30 columns)
        'mission_phase', 'mission_age_factor', 'phase_progression',
        'complexity_score', 'activity_word_count', 'instruction_count', 'systems_involved',
        
        # Equipment Usage (Binary indicators)
        'uses_ida', 'uses_hp3', 'uses_seis', 'uses_apss', 'uses_camera', 'uses_wts',
        'primary_instrument',
        
        # Activity Classifications (Binary 0/1)
        'operational_status', 'criticality', 'power_status',
        'is_science', 'is_maintenance', 'is_deployment', 'is_recovery', 'is_thermal', 'is_communication',
        
        # Operational Context
        'operational_tempo', 'requires_movement', 'power_intensive', 'data_intensity',
        
        # Weather Integration Features (~15 columns)
        'weather_avg_temp', 'weather_temp_severity', 'weather_wind_intensity',
        'weather_atmospheric_instability', 'BMY_HORIZONTAL_WIND_SPEED', 'weather_time_diff_hours',
        
        # Seasonal & Temporal Features
        'seasonal_intensity', 'orbital_thermal_factor', 'diurnal_thermal_load',
        'thermal_stress_combined',
        
        # Mission Timeline
        'sols_since_last_science', 'sols_since_last_maintenance', 'sols_since_last_deployment',
        'cumulative_ida_usage', 'cumulative_hp3_usage', 'cumulative_seis_usage',
        'sols_to_next_milestone'
    ]
    
    return columns

def main():
    """Generate CSV with column names"""
    columns = generate_mars_mission_data()
    
    # Create DataFrame with column names
    df = pd.DataFrame({'column_names': columns})
    
    # Save to CSV
    df.to_csv('mars_mission_column_names.csv', index=False)
    print(f"Generated mars_mission_column_names.csv with {len(columns)} columns")
    
    # Display columns
    for i, col in enumerate(columns, 1):
        print(f"{i:2d}. {col}")

if __name__ == "__main__":
    main()