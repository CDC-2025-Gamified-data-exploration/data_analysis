import pandas as pd
import numpy as np

def analyze_missing_data(csv_file):
    """Load dataset and analyze missing values"""
    
    try:
        # Load the dataset
        df = pd.read_csv(csv_file)
        print(f"Loaded dataset: {len(df)} rows × {len(df.columns)} columns")
        
        # Calculate missing data for each column
        missing_data = []
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_data.append({
                'column_name': column,
                'total_count': len(df),
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
                'data_type': str(df[column].dtype)
            })
        
        # Create DataFrame
        na_analysis = pd.DataFrame(missing_data)
        
        # Sort by missing percentage (highest first)
        na_analysis = na_analysis.sort_values('missing_percentage', ascending=False)
        
        # Save results
        output_file = f"{csv_file.replace('.csv', '')}_missing_analysis.csv"
        na_analysis.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nMissing Data Summary:")
        print(f"=" * 40)
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns with no missing data: {(na_analysis['missing_count'] == 0).sum()}")
        print(f"Columns with missing data: {(na_analysis['missing_count'] > 0).sum()}")
        print(f"Total missing data points: {na_analysis['missing_count'].sum():,}")
        
        # Show columns with missing data
        missing_cols = na_analysis[na_analysis['missing_count'] > 0]
        if len(missing_cols) > 0:
            print(f"\nColumns with missing data:")
            for _, row in missing_cols.iterrows():
                print(f"  {row['column_name']}: {row['missing_count']} missing ({row['missing_percentage']}%)")
        else:
            print("\nNo missing data found!")
            
        print(f"\nDetailed analysis saved to: {output_file}")
        
        return na_analysis
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        return None
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None

if __name__ == "__main__":
    # Try to find your dataset file
    possible_files = [
        'mars_mission_master_dataset.csv',
        'mars_activities_dataframe.csv', 
        'mars_weather_15min_with_solar_longitude.csv'
    ]
    
    for file in possible_files:
        try:
            result = analyze_missing_data(file)
            if result is not None:
                break
        except:
            continue
    else:
        print("No dataset file found. Please specify the correct filename:")
        filename = input("Enter CSV filename: ")
        analyze_missing_data(filename)