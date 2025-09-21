import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_activities():
    """Generate realistic Mars mission activity descriptions"""
    activities = [
        "Deploy IDA robotic arm for HP3 heat probe positioning and soil analysis preparation",
        "Execute SEIS seismometer calibration sequence with thermal protection adjustment",
        "Perform HP3 heat probe penetration attempt #15 with modified hammer sequence",
        "Conduct routine APSS weather station data collection and atmospheric monitoring",
        "Initiate camera system health check and lens cleaning protocol",
        "Execute emergency power management protocol due to dust accumulation on solar panels"
    ]
    return activities

def generate_mars_mission_head():
    """Generate first 5-10 rows of Mars mission data"""
    np.random.seed(42)
    random.seed(42)
    
    n_rows = 6
    activities = generate_sample_activities()
    
    # Base datetime starting from InSight landing
    start_date = datetime(2018, 11, 26, 14, 30, 0)
    
    data = {
        # Original Mission Data
        'datetime': [start_date + timedelta(days=i*5.2, hours=random.randint(8, 16)) for i in range(n_rows)],
        'sol': [1, 5, 12, 28, 45, 67],
        'activity': activities,
        'solar_longitude_deg': [248.2, 249.8, 252.1, 257.3, 264.7, 273.1],
        'year': [2018, 2018, 2018, 2018, 2018, 2019],
        'month': [11, 12, 12, 12, 1, 1],
        'day': [26, 1, 8, 24, 10, 31],
        'hour': [14, 9, 13, 11, 15, 8],
        'day_of_year': [330, 335, 342, 358, 10, 31],
        'weekday': [0, 5, 5, 0, 3, 3],  # Monday=0
        
        # Mission Enrichment Features
        'mission_phase': ['Early_Deployment', 'Early_Deployment', 'Early_Deployment', 
                         'Primary_Science', 'Primary_Science', 'Primary_Science'],
        'mission_age_factor': [0.1, 0.15, 0.25, 0.4, 0.5, 0.6],
        'phase_progression': [0.05, 0.12, 0.25, 0.08, 0.15, 0.22],
        'complexity_score': [0.85, 0.72, 0.91, 0.45, 0.38, 0.76],
        'activity_word_count': [14, 11, 13, 12, 10, 13],
        'instruction_count': [3, 2, 4, 2, 2, 3],
        'systems_involved': [2, 1, 2, 1, 1, 1],
        
        # Equipment Usage (Binary)
        'uses_ida': [1, 0, 1, 0, 0, 0],
        'uses_hp3': [1, 0, 1, 0, 0, 0],
        'uses_seis': [0, 1, 0, 0, 0, 0],
        'uses_apss': [0, 0, 0, 1, 0, 0],
        'uses_camera': [0, 0, 0, 0, 1, 0],
        'uses_wts': [0, 0, 0, 0, 0, 1],
        'primary_instrument': ['IDA', 'SEIS', 'HP3', 'APSS', 'CAMERA', 'WTS'],
        
        # Activity Classifications  
        'operational_status': [1, 1, 0, 1, 1, 0],  # 0=problem, 1=normal
        'criticality': [0, 1, 0, 1, 1, 0],  # 0=critical, 1=routine
        'power_status': [1, 1, 1, 1, 1, 0],  # 0=power issue, 1=normal
        'is_science': [0, 1, 0, 1, 0, 0],
        'is_maintenance': [0, 0, 0, 0, 1, 1],
        'is_deployment': [1, 0, 0, 0, 0, 0],
        'is_recovery': [0, 0, 1, 0, 0, 1],
        'is_thermal': [0, 1, 0, 0, 0, 0],
        'is_communication': [0, 0, 0, 0, 0, 0],
        
        # Operational Context
        'operational_tempo': [0.2, 0.15, 0.3, 0.25, 0.2, 0.35],
        'requires_movement': [1, 0, 1, 0, 0, 0],
        'power_intensive': [1, 0, 1, 0, 0, 1],
        'data_intensity': [0.4, 0.7, 0.3, 0.8, 0.5, 0.2],
        
        # Weather Integration (with some NaN values)
        'weather_avg_temp': [-63.2, -71.5, np.nan, -58.7, -76.3, np.nan],
        'weather_temp_severity': [1, 2, np.nan, 1, 2, np.nan],
        'weather_wind_intensity': [1, 0, np.nan, 2, 1, np.nan],
        'weather_atmospheric_instability': [1, 2, np.nan, 2, 1, np.nan],
        'BMY_HORIZONTAL_WIND_SPEED': [3.2, 1.1, np.nan, 5.7, 2.8, np.nan],
        'weather_time_diff_hours': [1.5, 0.8, np.nan, 2.1, 1.2, np.nan],
        
        # Seasonal & Temporal
        'seasonal_intensity': [0.7, 0.75, 0.8, 0.9, 0.95, 0.85],
        'orbital_thermal_factor': [0.3, 0.28, 0.25, 0.2, 0.15, 0.1],
        'diurnal_thermal_load': [0.6, 0.4, 0.7, 0.5, 0.8, 0.3],
        'thermal_stress_combined': [1.6, 1.43, 1.75, 1.6, 1.9, 1.25],
        
        # Mission Timeline
        'sols_since_last_science': [np.nan, 0, 1, 0, 17, 22],
        'sols_since_last_maintenance': [np.nan, np.nan, np.nan, np.nan, 0, 22],
        'sols_since_last_deployment': [0, 4, 11, 27, 44, 66],
        'cumulative_ida_usage': [1, 1, 2, 2, 2, 2],
        'cumulative_hp3_usage': [1, 1, 2, 2, 2, 2], 
        'cumulative_seis_usage': [0, 1, 1, 1, 1, 1],
        'sols_to_next_milestone': [94, 90, 83, 67, 50, 28]
    }
    
    return pd.DataFrame(data)

def main():
    """Generate CSV with first few rows of Mars mission data"""
    df_head = generate_mars_mission_head()
    
    # Save to CSV
    df_head.to_csv('mars_mission_head.csv', index=False)
    print(f"Generated mars_mission_head.csv with {len(df_head)} rows and {len(df_head.columns)} columns")
    
    # Display first few rows
    print("\nFirst few rows:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df_head.head(3))
    
    print(f"\nColumns: {list(df_head.columns)}")

if __name__ == "__main__":
    main()