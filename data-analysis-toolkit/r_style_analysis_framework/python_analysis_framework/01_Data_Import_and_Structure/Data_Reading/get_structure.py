import pandas as pd
import numpy as np

def generate_mars_mission_structure():
    """Generate Mars mission dataset structure information"""
    
    structure_data = {
        'column_name': [
            # Original Mission Data (12 columns)
            'datetime', 'sol', 'activity', 'solar_longitude_deg', 'year', 'month', 
            'day', 'hour', 'day_of_year', 'weekday',
            
            # Mission Enrichment Features
            'mission_phase', 'mission_age_factor', 'phase_progression', 'complexity_score',
            'activity_word_count', 'instruction_count', 'systems_involved',
            
            # Equipment Usage
            'uses_ida', 'uses_hp3', 'uses_seis', 'uses_apss', 'uses_camera', 'uses_wts',
            'primary_instrument',
            
            # Activity Classifications
            'operational_status', 'criticality', 'power_status', 'is_science', 
            'is_maintenance', 'is_deployment', 'is_recovery', 'is_thermal', 'is_communication',
            
            # Operational Context
            'operational_tempo', 'requires_movement', 'power_intensive', 'data_intensity',
            
            # Weather Integration
            'weather_avg_temp', 'weather_temp_severity', 'weather_wind_intensity',
            'weather_atmospheric_instability', 'BMY_HORIZONTAL_WIND_SPEED', 'weather_time_diff_hours',
            
            # Seasonal & Temporal
            'seasonal_intensity', 'orbital_thermal_factor', 'diurnal_thermal_load', 'thermal_stress_combined',
            
            # Mission Timeline
            'sols_since_last_science', 'sols_since_last_maintenance', 'sols_since_last_deployment',
            'cumulative_ida_usage', 'cumulative_hp3_usage', 'cumulative_seis_usage', 'sols_to_next_milestone'
        ],
        
        'data_type': [
            # Original Mission Data
            'datetime64', 'int64', 'object', 'float64', 'int64', 'int64', 
            'int64', 'int64', 'int64', 'int64',
            
            # Mission Enrichment
            'object', 'float64', 'float64', 'float64', 'int64', 'int64', 'int64',
            
            # Equipment Usage
            'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'object',
            
            # Activity Classifications  
            'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64',
            
            # Operational Context
            'float64', 'int64', 'int64', 'float64',
            
            # Weather Integration
            'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
            
            # Seasonal & Temporal
            'float64', 'float64', 'float64', 'float64',
            
            # Mission Timeline
            'float64', 'float64', 'float64', 'int64', 'int64', 'int64', 'int64'
        ],
        
        'category': [
            # Original Mission Data
            'Original', 'Original', 'Original', 'Original', 'Original', 'Original',
            'Original', 'Original', 'Original', 'Original',
            
            # Mission Enrichment
            'Mission_Phase', 'Mission_Phase', 'Mission_Phase', 'Activity_Complexity',
            'Activity_Complexity', 'Activity_Complexity', 'Activity_Complexity',
            
            # Equipment Usage
            'Equipment', 'Equipment', 'Equipment', 'Equipment', 'Equipment', 'Equipment', 'Equipment',
            
            # Activity Classifications
            'Classification', 'Classification', 'Classification', 'Classification',
            'Classification', 'Classification', 'Classification', 'Classification', 'Classification',
            
            # Operational Context
            'Operational', 'Operational', 'Operational', 'Operational',
            
            # Weather Integration
            'Weather', 'Weather', 'Weather', 'Weather', 'Weather', 'Weather',
            
            # Seasonal & Temporal
            'Temporal', 'Temporal', 'Temporal', 'Temporal',
            
            # Mission Timeline
            'Timeline', 'Timeline', 'Timeline', 'Timeline', 'Timeline', 'Timeline', 'Timeline'
        ],
        
        'description': [
            # Original Mission Data
            'Mission activity timestamp', 'Mars day number (1-1442)', 'Detailed activity description',
            'Mars seasonal position (0-360°)', 'Earth year', 'Earth month', 'Earth day', 'Earth hour',
            'Day of year', 'Weekday (0=Monday)',
            
            # Mission Enrichment
            'Mission phase (Early_Deployment, Primary_Science, Extended_1, Extended_2, Long_Term)',
            'Logarithmic equipment aging factor (0-3)', 'Progress within current phase (0-1)',
            'Composite complexity rating (0-1)', 'Words in activity description', 
            'Number of discrete tasks', 'Count of instruments used (0-6)',
            
            # Equipment Usage
            'Uses IDA robotic arm (0/1)', 'Uses HP3 heat probe (0/1)', 'Uses SEIS seismometer (0/1)',
            'Uses APSS weather station (0/1)', 'Uses camera system (0/1)', 'Uses WTS wind sensor (0/1)',
            'Primary instrument (IDA, HP3, SEIS, APSS, CAMERA, WTS)',
            
            # Activity Classifications
            'Operational status (0=problem, 1=normal)', 'Criticality (0=critical, 1=routine)',
            'Power status (0=power issue, 1=normal)', 'Science activity (0/1)', 'Maintenance activity (0/1)',
            'Deployment activity (0/1)', 'Recovery activity (0/1)', 'Thermal activity (0/1)', 'Communication activity (0/1)',
            
            # Operational Context
            'Rolling 7-sol activity intensity', 'Requires mechanical movement (0/1)',
            'High-power operation (0/1)', 'Data collection/transmission intensity',
            
            # Weather Integration
            'Average temperature (°C)', 'Temperature severity scale (0-3)', 'Wind intensity scale (0-3)',
            'Combined atmospheric instability (0-3)', 'Horizontal wind speed (m/s)', 
            'Time difference to weather reading (hours)',
            
            # Seasonal & Temporal
            'Distance from seasonal extremes', 'Mars orbital thermal effects',
            'Daily thermal cycle position', 'Combined seasonal-diurnal thermal stress',
            
            # Mission Timeline
            'Days since last science activity', 'Days since last maintenance', 'Days since last deployment',
            'Cumulative IDA usage count', 'Cumulative HP3 usage count', 'Cumulative SEIS usage count',
            'Days to next mission milestone'
        ],
        
        'missing_data': [
            # Original Mission Data
            'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
            
            # Mission Enrichment  
            'No', 'No', 'No', 'No', 'No', 'No', 'No',
            
            # Equipment Usage
            'No', 'No', 'No', 'No', 'No', 'No', 'No',
            
            # Activity Classifications
            'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No',
            
            # Operational Context
            'No', 'No', 'No', 'No',
            
            # Weather Integration (~30% missing)
            'Yes (~30%)', 'Yes (~30%)', 'Yes (~30%)', 'Yes (~30%)', 'Yes (~30%)', 'Yes (~30%)',
            
            # Seasonal & Temporal
            'No', 'No', 'No', 'No',
            
            # Mission Timeline
            'Yes (early mission)', 'Yes (early mission)', 'No', 'No', 'No', 'No', 'No'
        ]
    }
    
    return pd.DataFrame(structure_data)

def generate_category_summary():
    """Generate summary by category"""
    category_summary = {
        'category': ['Original', 'Mission_Phase', 'Activity_Complexity', 'Equipment', 
                    'Classification', 'Operational', 'Weather', 'Temporal', 'Timeline'],
        'column_count': [10, 3, 4, 7, 9, 4, 6, 4, 7],
        'primary_purpose': [
            'Base mission data and timestamps',
            'Mission lifecycle and aging factors', 
            'Activity difficulty and resource requirements',
            'Instrument usage and co-dependencies',
            'Activity type and operational status',
            'Real-time operational metrics',
            'Environmental conditions and correlations',
            'Mars seasonal and thermal effects',
            'Historical usage and future planning'
        ],
        'key_insights': [
            'Temporal context for all activities',
            'Mission evolution over 4+ years',
            'Resource planning and complexity trends',
            'Equipment reliability and usage patterns',
            'Problem identification and categorization',
            'Operational tempo and efficiency',
            'Environmental impact on operations',
            'Mars seasonal effects on mission',
            'Learning curves and milestone tracking'
        ]
    }
    
    return pd.DataFrame(category_summary)

def main():
    """Generate CSV files with dataset structure information"""
    
    # Generate detailed structure
    structure_df = generate_mars_mission_structure()
    structure_df.to_csv('mars_mission_structure.csv', index=False)
    print(f"Generated mars_mission_structure.csv with {len(structure_df)} columns")
    
    # Generate category summary
    summary_df = generate_category_summary()
    summary_df.to_csv('mars_mission_category_summary.csv', index=False)
    print(f"Generated mars_mission_category_summary.csv")
    
    # Display structure overview
    print(f"\nDataset Structure Overview:")
    print(f"Total columns: {len(structure_df)}")
    print(f"Data types: {structure_df['data_type'].value_counts().to_dict()}")
    print(f"Categories: {structure_df['category'].value_counts().to_dict()}")
    
    print(f"\nColumns with missing data:")
    missing_cols = structure_df[structure_df['missing_data'] != 'No']
    for _, row in missing_cols.iterrows():
        print(f"  {row['column_name']}: {row['missing_data']}")

if __name__ == "__main__":
    main()