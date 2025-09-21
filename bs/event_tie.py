import pandas as pd
import numpy as np
from datetime import datetime

def categorize_mars_events(input_csv_path, output_csv_path='mars_events_classified.csv'):
    """
    Load Mars InSight mission data, categorize events, and save to new CSV
    """
    
    # Load the data
    try:
        mars_df = pd.read_csv(input_csv_path)
        print(f"✓ Loaded {len(mars_df)} events from {input_csv_path}")
    except FileNotFoundError:
        print(f"✗ File not found: {input_csv_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None
    
    # Create a copy to work with
    classified_df = mars_df.copy()
    
    def classify_activity(activity):
        """Classify a single activity into binary categories"""
        activity_lower = str(activity).lower()
        
        # Category 1: OPERATIONAL vs NON-OPERATIONAL (0/1)
        non_operational_keywords = [
            'safe mode', 'recovery', 'runout', 'fault', 'error', 
            'unsafe', 'safed', 'retirement', 'monitoring only', 
            'characterization', 'temperature monitoring'
        ]
        operational_status = 0 if any(keyword in activity_lower for keyword in non_operational_keywords) else 1
        
        # Category 2: SCIENCE vs ENGINEERING (0/1)
        science_keywords = [
            'imaging', 'data collection', 'measurements', 'observations', 
            'panorama', 'survey', 'tau', 'calibration', 'science', 'eclipse',
            'mosaic', 'stereo', 'dust devil', 'meteor', 'shadow', 'cloud'
        ]
        activity_type = 1 if any(keyword in activity_lower for keyword in science_keywords) else 0
        
        # Category 3: DEPLOYMENT vs ROUTINE_OPS (0/1)
        deployment_keywords = [
            'unstow', 'deploy', 'capture', 'release', 'placement', 'stow',
            'grapple', 'adjustment', 'leveling', 'pinning', 'landing',
            'checkouts', 'cover open', 'cover'
        ]
        operation_phase = 1 if any(keyword in deployment_keywords for keyword in deployment_keywords) else 0
        
        # Category 4: POWER_ISSUES vs NORMAL_OPS (0/1)
        power_keywords = [
            'heater', 'power', 'battery', 'solar array', 'energy', 'thermal',
            'cleaning', 'dust', 'safe mode', 'conjunction', 'survival'
        ]
        power_status = 1 if any(keyword in activity_lower for keyword in power_keywords) else 0
        
        # Category 5: CRITICAL vs ROUTINE (0/1)
        critical_keywords = [
            'landing', 'thermal checkouts', 'seis deployment', 'hp3 deployment',
            'mole', 'hammering', 'recovery', 'safe mode', 'conjunction',
            'wts deploy', 'instrument', 'unstow', 'first'
        ]
        criticality = 1 if any(keyword in activity_lower for keyword in critical_keywords) else 0
        
        return {
            'operational_status': operational_status,
            'activity_type': activity_type,
            'operation_phase': operation_phase, 
            'power_status': power_status,
            'criticality': criticality
        }
    
    print("Classifying activities...")
    
    # Apply classification to each activity
    classifications = []
    for idx, activity in enumerate(classified_df['activity']):
        classification = classify_activity(activity)
        classifications.append(classification)
        
        # Show progress
        if (idx + 1) % 20 == 0 or (idx + 1) == len(classified_df):
            print(f"  Processed {idx + 1}/{len(classified_df)} activities")
    
    # Add classification columns to dataframe
    for key in ['operational_status', 'activity_type', 'operation_phase', 'power_status', 'criticality']:
        classified_df[key] = [c[key] for c in classifications]
    
    # Add derived binary categories based on existing data
    
    # Mission phase (Early vs Late) - split at Sol 400
    if 'sol' in classified_df.columns:
        classified_df['mission_phase'] = (classified_df['sol'] >= 400).astype(int)
    
    # Season (Warm vs Cold) based on solar longitude
    if 'solar_longitude_deg' in classified_df.columns:
        classified_df['season_warm'] = (
            (classified_df['solar_longitude_deg'] >= 0) & 
            (classified_df['solar_longitude_deg'] < 180)
        ).astype(int)
    
    # Time of day (Day vs Night/Early morning)
    if 'hour' in classified_df.columns:
        classified_df['daytime_activity'] = (
            classified_df['hour'].between(6, 18)
        ).astype(int)
    
    # Weekend activities (Earth time - just for reference)
    if 'weekday' in classified_df.columns:
        classified_df['earth_weekend'] = (
            classified_df['weekday'].isin([5, 6])
        ).astype(int)
    
    # Save to CSV
    try:
        classified_df.to_csv(output_csv_path, index=False)
        print(f"✓ Saved classified events to {output_csv_path}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return None
    
    # Generate summary report
    generate_classification_report(classified_df, output_csv_path)
    
    return classified_df

def generate_classification_report(df, csv_path):
    """Generate a summary report of the classifications"""
    
    print(f"\n{'='*60}")
    print("MARS INSIGHT MISSION EVENT CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Events: {len(df)}")
    print(f"Output File: {csv_path}")
    
    # Binary classification columns
    binary_cols = {
        'operational_status': 'Operational (1) vs Non-Operational (0)',
        'activity_type': 'Science (1) vs Engineering (0)', 
        'operation_phase': 'Deployment (1) vs Routine Ops (0)',
        'power_status': 'Power Issues (1) vs Normal Ops (0)',
        'criticality': 'Critical (1) vs Routine (0)',
        'mission_phase': 'Late Mission (1) vs Early Mission (0)',
        'season_warm': 'Warm Season (1) vs Cold Season (0)',
        'daytime_activity': 'Daytime (1) vs Night/Early (0)',
        'earth_weekend': 'Earth Weekend (1) vs Weekday (0)'
    }
    
    print(f"\nCLASSIFICATION BREAKDOWN:")
    print("-" * 40)
    
    for col, description in binary_cols.items():
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            total = len(df)
            
            print(f"\n{description}:")
            for value in [0, 1]:
                count = counts.get(value, 0)
                percentage = (count / total) * 100
                print(f"  {value}: {count:3d} events ({percentage:5.1f}%)")
    
    # Show examples for main categories
    print(f"\nSAMPLE CLASSIFICATIONS:")
    print("-" * 40)
    
    main_categories = ['operational_status', 'activity_type', 'criticality']
    
    for category in main_categories:
        if category in df.columns:
            print(f"\n{category.upper().replace('_', ' ')}:")
            
            for value in [0, 1]:
                sample_activities = df[df[category] == value]['activity'].unique()[:3]
                value_name = "YES" if value == 1 else "NO"
                print(f"  {value_name} ({value}):")
                for activity in sample_activities:
                    print(f"    • {activity}")
    
    # Mission timeline analysis
    if 'sol' in df.columns and 'criticality' in df.columns:
        critical_events = df[df['criticality'] == 1]
        early_critical = critical_events[critical_events['sol'] <= 100]
        late_critical = critical_events[critical_events['sol'] > 400]
        
        print(f"\nMISSION TIMELINE:")
        print("-" * 40)
        print(f"Critical events in first 100 sols: {len(early_critical)}")
        print(f"Critical events after sol 400: {len(late_critical)}")
        
        if len(early_critical) > 0:
            print("Early mission critical events:")
            for _, row in early_critical.head(3).iterrows():
                print(f"  Sol {row['sol']:3d}: {row['activity']}")

def create_classification_summary():
    """Create a summary of what each binary category represents"""
    
    summary_text = """
MARS INSIGHT EVENT CLASSIFICATION SYSTEM
========================================

BINARY CATEGORIES EXPLANATION:

1. OPERATIONAL_STATUS (0/1)
   0 = Non-Operational: Safe mode, runouts, recovery, monitoring only
   1 = Operational: Active science or engineering activities

2. ACTIVITY_TYPE (0/1) 
   0 = Engineering: System maintenance, checkouts, troubleshooting
   1 = Science: Data collection, imaging, measurements, observations

3. OPERATION_PHASE (0/1)
   0 = Routine Operations: Regular mission activities
   1 = Deployment/Setup: Major instrument deployments, first-time activities

4. POWER_STATUS (0/1)
   0 = Normal Operations: Regular activities
   1 = Power Issues: Solar array problems, thermal management, power-related

5. CRITICALITY (0/1) 
   0 = Routine: Standard mission operations
   1 = Critical: Mission-critical events (landing, major deployments, emergencies)

6. MISSION_PHASE (0/1)
   0 = Early Mission: First 400 sols
   1 = Late Mission: After sol 400

7. SEASON_WARM (0/1)
   0 = Cold Season: Autumn/Winter (Ls 180-360°)
   1 = Warm Season: Spring/Summer (Ls 0-180°)

8. DAYTIME_ACTIVITY (0/1)
   0 = Night/Early Morning: Hours 0-5, 19-23
   1 = Daytime: Hours 6-18

9. EARTH_WEEKEND (0/1)
   0 = Earth Weekday: Monday-Friday
   1 = Earth Weekend: Saturday-Sunday

USAGE:
- Load the CSV and filter by any binary category
- Example: df[df['criticality'] == 1] gets all critical events
- Example: df[df['power_status'] == 1] gets all power-related issues
"""
    
    with open('classification_guide.txt', 'w') as f:
        f.write(summary_text)
    
    print("✓ Created classification_guide.txt with detailed explanations")

# Main execution function
def main():
    """Main function to run the classification"""
    
    print("Mars InSight Event Classifier")
    print("=" * 40)
    
    # You can modify these file paths
    input_file = input("Enter input CSV filename (or press Enter for 'mars_data.csv'): ").strip()
    if not input_file:
        input_file = 'mars_data.csv'
    
    output_file = input("Enter output CSV filename (or press Enter for 'mars_events_classified.csv'): ").strip()
    if not output_file:
        output_file = 'mars_events_classified.csv'
    
    # Run the classification
    classified_df = categorize_mars_events(input_file, output_file)
    
    if classified_df is not None:
        create_classification_summary()
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION COMPLETE!")
        print(f"{'='*60}")
        print(f"✓ Input file: {input_file}")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Guide file: classification_guide.txt")
        print(f"✓ Total events processed: {len(classified_df)}")
        
        print(f"\nNext steps:")
        print(f"1. Open {output_file} in Excel or load with pandas")
        print(f"2. Filter by binary categories for analysis")
        print(f"3. Read classification_guide.txt for category explanations")
    else:
        print("\nClassification failed. Please check your input file.")

if __name__ == "__main__":
    main()