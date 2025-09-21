import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_activities_late():
    """Generate realistic Mars mission activity descriptions for end of mission"""
    activities = [
        "Execute power conservation protocol due to accumulated dust on solar arrays",
        "Perform final HP3 heat probe data analysis and thermal conductivity measurements",
        "Conduct SEIS seismometer long-term stability assessment and calibration check",
        "Initiate end-of-mission data backup sequence and memory optimization",
        "Execute camera system final panoramic survey of landing site surroundings", 
        "Perform WTS wind sensor final calibration before mission conclusion"
    ]
    return activities

def generate_mars_mission_tail():
    """Generate last 5-6 rows of Mars mission data (end of mission)"""
    np.random.seed(42)
    random.seed(42)
    
    n_rows = 6
    activities = generate_sample_activities_late()
    
    # End-of-mission datetime (late 2022)
    end_date = datetime(2022, 12, 18, 10, 15, 0)
    
    # Late mission sols
    late_sols = [1398, 1407, 1418, 1429, 1436, 1442]
    
    data = {
        # Original Mission Data
        'datetime': [end_date - timedelta(days=(1442-sol)*1.027, hours=random.randint(8, 16)) for sol in late_sols],
        'sol': late_sols,
        'activity': activities,
        'solar_longitude_deg': [310.8, 313.2, 316.4, 319.7, 321.5, 323.1],  # Late northern winter
        'year': [2022, 2022, 2022, 2022, 2022, 2022],
        'month': [11, 11, 12, 12, 12, 12],
        'day': [23, 25, 2, 8, 12, 18],
        'hour': [10, 14, 9, 13, 11, 10],
        'day_of_year': [327, 329, 336, 342, 346, 352],
        'weekday': [2, 4, 4, 3, 0, 6],  # Tuesday, Friday, Friday, Thursday, Monday, Sunday
        
        # Mission Enrichment Features - Late mission characteristics
        'mission_phase': ['Long_Term', 'Long_Term', 'Long_Term', 'Long_Term', 'Long_Term', 'Long_Term'],
        'mission_age_factor': [2.85, 2.89, 2.94, 2.97, 2.98, 3.0],  # Maximum aging
        'phase_progression': [0.89, 0.92, 0.96, 0.98, 0.99, 1.0],  # End of phase
        'complexity_score': [0.68, 0.54, 0.71, 0.82, 0.59, 0.63],  # Mixed complexity
        'activity_word_count': [12, 13, 14, 11, 13, 12],
        'instruction_count': [2, 3, 3, 4, 2, 2],
        'systems_involved': [1, 2, 1, 3, 1, 1],
        
        # Equipment Usage (Binary) - End of mission focus
        'uses_ida': [0, 1, 0, 0, 0, 0],
        'uses_hp3': [0, 1, 0, 0, 0, 0],
        'uses_seis': [0, 0, 1, 0, 0, 0],
        'uses_apss': [0, 0, 0, 1, 0, 0],
        'uses_camera': [0, 0, 0, 0, 1, 0],
        'uses_wts': [1, 0, 0, 0, 0, 1],  # Power conservation + final WTS
        'primary_instrument': ['WTS', 'HP3', 'SEIS', 'APSS', 'CAMERA', 'WTS'],
        
        # Activity Classifications - End of mission patterns
        'operational_status': [0, 1, 1, 1, 1, 1],  # One power issue, rest normal
        'criticality': [0, 1, 1, 0, 1, 1],  # Some critical end-of-mission activities
        'power_status': [0, 1, 1, 1, 1, 1],  # Power conservation issue
        'is_science': [0, 1, 1, 0, 1, 0],  # Final science activities
        'is_maintenance': [1, 0, 1, 0, 0, 1],  # End-of-mission maintenance
        'is_deployment': [0, 0, 0, 0, 0, 0],  # No new deployments
        'is_recovery': [1, 0, 0, 0, 0, 0],  # Power recovery
        'is_thermal': [0, 0, 0, 0, 0, 0],
        'is_communication': [0, 0, 0, 1, 0, 0],  # Data backup
        
        # Operational Context - End of mission
        'operational_tempo': [0.15, 0.28, 0.22, 0.35, 0.18, 0.12],  # Slowing down
        'requires_movement': [0, 1, 0, 0, 1, 0],  # Limited movement
        'power_intensive': [1, 1, 0, 1, 1, 0],  # Power constraints
        'data_intensity': [0.3, 0.6, 0.8, 0.9, 0.7, 0.4],  # Final data collection
        
        # Weather Integration (late mission, some missing data)
        'weather_avg_temp': [-81.2, -78.9, np.nan, -74.3, np.nan, -79.8],
        'weather_temp_severity': [2, 2, np.nan, 2, np.nan, 2],
        'weather_wind_intensity': [0, 1, np.nan, 2, np.nan, 1],
        'weather_atmospheric_instability': [2, 2, np.nan, 2, np.nan, 2],
        'BMY_HORIZONTAL_WIND_SPEED': [1.8, 3.4, np.nan, 4.9, np.nan, 2.7],
        'weather_time_diff_hours': [2.8, 1.1, np.nan, 0.9, np.nan, 1.7],
        
        # Seasonal & Temporal - Late northern winter
        'seasonal_intensity': [0.95, 0.97, 0.98, 0.96, 0.94, 0.92],
        'orbital_thermal_factor': [0.08, 0.06, 0.05, 0.04, 0.03, 0.02],
        'diurnal_thermal_load': [0.4, 0.6, 0.3, 0.7, 0.5, 0.4],
        'thermal_stress_combined': [1.47, 1.69, 1.36, 1.73, 1.48, 1.38],
        
        # Mission Timeline - End of mission values
        'sols_since_last_science': [156, 0, 0, 11, 0, 6],
        'sols_since_last_maintenance': [45, 46, 0, 11, 18, 0],
        'sols_since_last_deployment': [1345, 1354, 1365, 1376, 1383, 1389],
        'cumulative_ida_usage': [28, 29, 29, 29, 29, 29],  # Final counts
        'cumulative_hp3_usage': [34, 35, 35, 35, 35, 35],
        'cumulative_seis_usage': [67, 67, 68, 68, 68, 68],
        'sols_to_next_milestone': [0, 0, 0, 0, 0, 0]  # Mission ending
    }
    
    return pd.DataFrame(data)

def main():
    """Generate CSV with last few rows of Mars mission data"""
    df_tail = generate_mars_mission_tail()
    
    # Save to CSV
    df_tail.to_csv('mars_mission_tail.csv', index=False)
    print(f"Generated mars_mission_tail.csv with {len(df_tail)} rows and {len(df_tail.columns)} columns")
    
    # Display last few rows
    print("\nLast few rows (end of mission):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df_tail.tail(3))
    
    print(f"\nEnd-of-mission characteristics:")
    print(f"Final sol: {df_tail['sol'].max()}")
    print(f"Mission phase: {df_tail['mission_phase'].iloc[-1]}")
    print(f"Mission age factor: {df_tail['mission_age_factor'].max()}")
    print(f"Power issues: {(df_tail['power_status'] == 0).sum()} activities")
    print(f"Final activities focus: {', '.join(df_tail['primary_instrument'].tolist())}")

if __name__ == "__main__":
    main()