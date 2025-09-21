import pandas as pd
import numpy as np
from calculate_correlation import calculate_correlations
from plot_scatter import plot_scatter
import itertools

df = pd.read_csv(r'C:\Users\agsse\data_analysis\mars_mission_master_dataset_optimized.csv')

print("=== COMPREHENSIVE MARS MISSION ANALYSIS ===")
print(f"Dataset shape: {df.shape}")

# Get all numeric variables
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Total numeric variables: {len(numeric_vars)}")

# Get meaningful categorical variables for hue coloring (exclude event-related)
categorical_vars = []
for col in df.columns:
    if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 8:
        if 'event' not in col.lower():  # Exclude event-related variables
            categorical_vars.append(col)

print(f"Categorical variables for hue: {categorical_vars}")

# Define variable groups - limit to 8 variables max per heatmap, exclude event variables
variable_groups = {
    'core_temperature': ['temp_avg', 'temp_range', 'temp_median', 'temp_volatility_7sol', 'temp_min', 'temp_max'],
    'core_wind': ['wind_speed_avg', 'wind_speed_max', 'wind_speed_std', 'wind_gustiness', 'wind_volatility_7sol'],
    'wind_direction': ['wind_direction_avg', 'wind_direction_std', 'wind_resultant_length', 'wind_u_mean', 'wind_v_mean'],
    'activity_metrics': ['activity_count', 'activity_intensity', 'systems_involved', 'instruction_complexity', 'activity_word_count'],
    'mission_timeline': ['sol', 'solar_longitude_deg', 'mission_age_linear', 'phase_progress'],
    'risk_assessment': ['dust_storm_probability', 'environmental_stress_index', 'operational_risk_index'],
    'equipment_tracking': ['equipment_age_factor', 'wear_accumulation', 'mission_operational_cycles', 'equipment_reliability_score'],
    'power_orbital': ['orbital_thermal_factor', 'power_availability_proxy', 'dust_season_risk'],
    'operational_tempo': ['operational_tempo_3sol', 'operational_tempo_7sol', 'operational_tempo_15sol']
}

# Filter to existing variables and limit to 8 per group
MAX_VARS_PER_HEATMAP = 8
for group_name, vars_list in variable_groups.items():
    available_vars = [v for v in vars_list if v in df.columns and 'event' not in v.lower()]
    available_vars = available_vars[:MAX_VARS_PER_HEATMAP]  # Limit to 8 variables
    variable_groups[group_name] = available_vars
    print(f"{group_name}: {len(available_vars)} variables")

# 1. DOMAIN-SPECIFIC CORRELATION HEATMAPS (8 variables max each)
print(f"\n=== GENERATING DOMAIN-SPECIFIC CORRELATION HEATMAPS (MAX {MAX_VARS_PER_HEATMAP} VARS) ===")

for domain, vars_list in variable_groups.items():
    if len(vars_list) >= 2:
        print(f"Creating {domain} correlation heatmap ({len(vars_list)} variables)")
        try:
            calculate_correlations(
                df,
                cols=vars_list,
                method='pearson',
                output_folder=r'C:\Users\agsse\data_analysis\max_data',
                save_heatmap=True
            )
        except Exception as e:
            print(f"Error creating {domain} heatmap: {e}")

# 2. CROSS-DOMAIN CORRELATION HEATMAPS (limited combinations, max 8 vars)
print(f"\n=== GENERATING CROSS-DOMAIN CORRELATION HEATMAPS ===")

cross_domain_combinations = [
    ('core_temperature', 'core_wind'),
    ('core_temperature', 'activity_metrics'),
    ('core_wind', 'activity_metrics'),
    ('mission_timeline', 'core_temperature'),
    ('risk_assessment', 'core_temperature'),
    ('risk_assessment', 'core_wind'),
    ('power_orbital', 'core_temperature')
]

for domain1, domain2 in cross_domain_combinations:
    vars1 = variable_groups.get(domain1, [])[:4]  # Max 4 from each domain
    vars2 = variable_groups.get(domain2, [])[:4]  # Max 4 from each domain
    combined_vars = vars1 + vars2
    
    if len(combined_vars) >= 3 and len(combined_vars) <= MAX_VARS_PER_HEATMAP:
        print(f"Creating {domain1} x {domain2} cross-correlation ({len(combined_vars)} variables)")
        try:
            calculate_correlations(
                df,
                cols=combined_vars,
                method='spearman',
                output_folder=r'C:\Users\agsse\data_analysis\max_data',
                save_heatmap=True
            )
        except Exception as e:
            print(f"Error creating {domain1} x {domain2} heatmap: {e}")

# 3. TOP VARIABLES COMPREHENSIVE MATRIX (limit to most important 15 variables)
print(f"\n=== GENERATING TOP VARIABLES COMPREHENSIVE MATRIX ===")
top_variables = [
    'temp_avg', 'temp_range', 'temp_volatility_7sol',
    'wind_speed_avg', 'wind_speed_std', 'wind_volatility_7sol', 
    'activity_intensity', 'systems_involved',
    'sol', 'solar_longitude_deg', 'mission_age_linear',
    'dust_storm_probability', 'environmental_stress_index', 'operational_risk_index',
    'power_availability_proxy'
]

available_top_vars = [v for v in top_variables if v in df.columns]
if len(available_top_vars) >= 5:
    print(f"Creating comprehensive top variables matrix ({len(available_top_vars)} variables)")
    try:
        calculate_correlations(
            df,
            cols=available_top_vars,
            method='spearman',
            output_folder=r'C:\Users\agsse\data_analysis\max_data',
            save_heatmap=True
        )
    except Exception as e:
        print(f"Error creating comprehensive matrix: {e}")

# 4. HIGH-PRIORITY SCATTER PLOTS
print(f"\n=== GENERATING HIGH-PRIORITY SCATTER PLOTS ===")

# Core relationships to explore
priority_scatter_pairs = [
    ('temp_avg', 'wind_speed_avg'),
    ('temp_avg', 'solar_longitude_deg'),
    ('temp_volatility_7sol', 'wind_volatility_7sol'),
    ('sol', 'temp_avg'),
    ('sol', 'wind_speed_avg'),
    ('sol', 'activity_intensity'),
    ('activity_intensity', 'temp_avg'),
    ('activity_intensity', 'wind_speed_avg'),
    ('dust_storm_probability', 'wind_speed_avg'),
    ('environmental_stress_index', 'temp_volatility_7sol'),
    ('operational_risk_index', 'activity_intensity'),
    ('power_availability_proxy', 'temp_avg'),
    ('mission_age_linear', 'equipment_reliability_score'),
    ('solar_longitude_deg', 'dust_storm_probability')
]

scatter_count = 0
for x_var, y_var in priority_scatter_pairs:
    if x_var in df.columns and y_var in df.columns:
        # Cycle through different hue variables
        hue_var = categorical_vars[scatter_count % len(categorical_vars)] if categorical_vars else None
        
        try:
            plot_scatter(
                df,
                x=x_var,
                y=y_var,
                hue=hue_var,
                add_regression=True,
                method_corr='spearman',
                output_folder=r'C:\Users\agsse\data_analysis\max_data'
            )
            scatter_count += 1
            print(f"Generated scatter plot {scatter_count}: {x_var} vs {y_var}")
        except Exception as e:
            print(f"Error with {x_var} vs {y_var}: {e}")

# 5. TEMPORAL EVOLUTION SCATTER PLOTS
print(f"\n=== GENERATING TEMPORAL EVOLUTION SCATTER PLOTS ===")
if 'sol' in df.columns:
    temporal_targets = [
        'temp_avg', 'temp_volatility_7sol',
        'wind_speed_avg', 'wind_volatility_7sol',
        'activity_intensity', 'systems_involved',
        'dust_storm_probability', 'environmental_stress_index',
        'power_availability_proxy', 'equipment_reliability_score'
    ]
    
    available_temporal = [v for v in temporal_targets if v in df.columns]
    
    for i, target_var in enumerate(available_temporal[:12]):  # Limit to 12
        hue_var = categorical_vars[i % len(categorical_vars)] if categorical_vars else None
        
        try:
            plot_scatter(
                df,
                x='sol',
                y=target_var,
                hue=hue_var,
                add_regression=True,
                method_corr='spearman',
                output_folder=r'C:\Users\agsse\data_analysis\max_data'
            )
            print(f"Temporal evolution plot: sol vs {target_var}")
        except Exception as e:
            print(f"Error with temporal plot {target_var}: {e}")

# 6. SEASONAL RELATIONSHIP SCATTER PLOTS
print(f"\n=== GENERATING SEASONAL SCATTER PLOTS ===")
if 'solar_longitude_deg' in df.columns:
    seasonal_targets = [
        'temp_avg', 'temp_range', 'temp_volatility_7sol',
        'wind_speed_avg', 'wind_speed_max', 'wind_volatility_7sol',
        'dust_storm_probability', 'environmental_stress_index'
    ]
    
    available_seasonal = [v for v in seasonal_targets if v in df.columns]
    
    for target_var in available_seasonal[:10]:  # Limit to 10
        try:
            plot_scatter(
                df,
                x='solar_longitude_deg',
                y=target_var,
                hue='mars_season' if 'mars_season' in categorical_vars else None,
                add_regression=False,  # Non-linear seasonal relationships
                method_corr='spearman',
                output_folder=r'C:\Users\agsse\data_analysis\max_data'
            )
            print(f"Seasonal plot: solar_longitude_deg vs {target_var}")
        except Exception as e:
            print(f"Error with seasonal plot {target_var}: {e}")

# 7. ACTIVITY vs ENVIRONMENT RELATIONSHIPS
print(f"\n=== GENERATING ACTIVITY vs ENVIRONMENT RELATIONSHIPS ===")
activity_vars = ['activity_intensity', 'systems_involved', 'activity_count']
environment_vars = ['temp_avg', 'wind_speed_avg', 'dust_storm_probability', 'environmental_stress_index']

available_activity = [v for v in activity_vars if v in df.columns]
available_environment = [v for v in environment_vars if v in df.columns]

for act_var in available_activity:
    for env_var in available_environment:
        try:
            plot_scatter(
                df,
                x=env_var,
                y=act_var,
                hue='mission_phase' if 'mission_phase' in categorical_vars else None,
                add_regression=True,
                method_corr='spearman',
                output_folder=r'C:\Users\agsse\data_analysis\max_data'
            )
            print(f"Activity-Environment plot: {env_var} vs {act_var}")
        except Exception as e:
            continue

print(f"\n=== COMPREHENSIVE ANALYSIS COMPLETE ===")
print("Generated outputs:")
print(f"- Domain-specific correlation heatmaps (max {MAX_VARS_PER_HEATMAP} variables each)")
print("- Cross-domain correlation matrices")
print("- Top variables comprehensive matrix")
print(f"- Priority scatter plots: {scatter_count}")
print("- Temporal evolution plots: up to 12")
print("- Seasonal relationship plots: up to 10") 
print("- Activity vs environment plots")
print(f"\nAll files saved to: C:\\Users\\agsse\\data_analysis\\max_data\\")
print("Event-related variables excluded from all analyses")