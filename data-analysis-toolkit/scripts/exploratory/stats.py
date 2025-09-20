# Load and process Mars mission data to create activity dictionary

import pandas as pd
import numpy as np

def read_csv_with_encoding(filepath):
    """Try to read CSV with different encodings and error handling"""
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            print(f"Trying to read {filepath} with encoding: {encoding}")
            # Try with different parameters that often help with problematic files
            df = pd.read_csv(filepath, encoding=encoding, encoding_errors='ignore')
            print(f"Successfully loaded {filepath} with {encoding} encoding")
            print(f"Shape: {df.shape}, Columns: {list(df.columns)}")
            return df
        except UnicodeDecodeError as e:
            print(f"Failed with {encoding} encoding: {e}")
            continue
        except Exception as e:
            print(f"Other error with {encoding}: {e}")
            continue
    
    # Last resort - try with error handling
    try:
        print(f"Last resort: trying with utf-8 and error replacement")
        df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='replace')
        print(f"Loaded with character replacement - some characters may be corrupted")
        return df
    except Exception as e:
        print(f"Final attempt failed: {e}")
    
    raise ValueError(f"Could not read {filepath} with any method. Please check the file format.")

def process_mars_data():
    """Process Mars mission data and return activity dictionaries"""
    try:
        # Load the datasets with proper file paths and encoding handling
        print("Loading Mars surface activity data...")
        surface_activity_df = read_csv_with_encoding('csv/INS_InSight__Surface_Activity_Overview.csv')
        
        # Clean up the surface activity dataframe - remove empty columns and fix headers
        # Drop columns that are completely empty or unnamed
        surface_activity_df = surface_activity_df.dropna(axis=1, how='all')
        
        # Remove any unnamed columns that might be artifacts
        cols_to_keep = []
        for col in surface_activity_df.columns:
            if not (str(col).startswith('Unnamed') and surface_activity_df[col].isna().all()):
                cols_to_keep.append(col)
        
        surface_activity_df = surface_activity_df[cols_to_keep]
        
        print("Loading Mars weather data...")
        try:
            mars_weather_df = read_csv_with_encoding('csv/mars_weather_15min.csv')
        except:
            print("Weather data not available or failed to load - continuing with activity data only")
            mars_weather_df = pd.DataFrame()  # Empty dataframe
        
        # Filter surface activity data to cap dates within weather data range
        surface_activity_df = surface_activity_df[surface_activity_df['UTC'] <= '2020-12-31']
        print(f"Filtered surface activity data to dates <= 2020-12-31")
        
        print("Loaded datasets successfully!")
        print(f"Surface activity shape: {surface_activity_df.shape}")
        print(f"Mars weather shape: {mars_weather_df.shape}")
        
        # Display column names to verify structure
        print("\nSurface activity columns:", surface_activity_df.columns.tolist())
        if not mars_weather_df.empty:
            print("Mars weather columns:", mars_weather_df.columns.tolist())
        
        # Check if timestamp columns exist - adjust column names based on actual Mars data
        timestamp_col_surface = None
        timestamp_col_weather = None
        
        # For Mars data, look for specific column names
        mars_timestamp_cols = ['UTC', 'timestamp', 'datetime', 'date', 'time']
        mars_activity_cols = ['Activity', 'activity', 'event', 'surface_event_type', 'mission_activity']
        
        # Find timestamp column in surface activity data
        for col in mars_timestamp_cols:
            if col in surface_activity_df.columns:
                timestamp_col_surface = col
                print(f"Found timestamp column in surface data: {col}")
                break
        
        # Find timestamp column in weather data  
        if not mars_weather_df.empty:
            print(f"\nLooking for timestamp column in weather data...")
            print(f"Weather columns: {list(mars_weather_df.columns)}")
            
            for col in mars_timestamp_cols:
                if col in mars_weather_df.columns:
                    timestamp_col_weather = col
                    print(f"Found timestamp column in weather data: {col}")
                    break
            
            # If no exact match, look for any column that might be a timestamp
            if timestamp_col_weather is None:
                print("No exact timestamp match found, checking for datetime-like columns...")
                for col in mars_weather_df.columns:
                    # Check if column contains datetime-like data
                    sample_val = mars_weather_df[col].iloc[0] if not mars_weather_df[col].empty else None
                    if sample_val is not None:
                        col_str = str(sample_val)
                        # Look for datetime patterns like YYYY-MM-DD or timestamps
                        if any(pattern in col_str for pattern in ['2018', '2019', '2020', '-', ':']):
                            timestamp_col_weather = col
                            print(f"Found potential timestamp column: {col} (sample: {sample_val})")
                            break
                        # Also check if it's already a datetime type
                        if mars_weather_df[col].dtype.name.startswith('datetime'):
                            timestamp_col_weather = col
                            print(f"Found datetime column: {col}")
                            break
            
            # Last resort - check the first column if it looks like a timestamp
            if timestamp_col_weather is None and len(mars_weather_df.columns) > 0:
                first_col = mars_weather_df.columns[0]
                print(f"Checking first column as potential timestamp: {first_col}")
                try:
                    # Try to convert first column to datetime
                    test_conversion = pd.to_datetime(mars_weather_df[first_col].iloc[0])
                    timestamp_col_weather = first_col
                    print(f"Successfully using first column as timestamp: {first_col}")
                except:
                    print(f"First column {first_col} is not a valid timestamp")
            
            if timestamp_col_weather is None:
                print("Warning: Could not find any timestamp column in weather data")
                print("Available columns:", list(mars_weather_df.columns))
            else:
                print(f"Final timestamp column selection: {timestamp_col_weather}")
        
        if timestamp_col_surface is None:
            print("Warning: No timestamp column found in surface activity data")
            print("Available columns:", surface_activity_df.columns.tolist())
            # Use the UTC column from your sample data
            if 'UTC' in surface_activity_df.columns:
                timestamp_col_surface = 'UTC'
        
        # Convert timestamp columns to datetime format
        if timestamp_col_surface:
            surface_activity_df[timestamp_col_surface] = pd.to_datetime(surface_activity_df[timestamp_col_surface])
            print(f"Converted {timestamp_col_surface} to datetime")
        
        if timestamp_col_weather and not mars_weather_df.empty:
            mars_weather_df[timestamp_col_weather] = pd.to_datetime(mars_weather_df[timestamp_col_weather])
            print(f"Converted {timestamp_col_weather} to datetime")
        
        # Calculate avg_temp and avg_windspeed from specific Mars weather columns
        mars_temp_cols = ['BMY_BASE_ROD_TEMP', 'BMY_MID_ROD_TEMP', 'BMY_TIP_ROD_TEMP']
        mars_wind_cols = ['BMY_HORIZONTAL_WIND_SPEED']
        
        # Initialize flags to track if we successfully create weather columns
        weather_processing_success = False
        
        if not mars_weather_df.empty:
            print(f"\nProcessing weather data...")
            print(f"Weather dataframe shape: {mars_weather_df.shape}")
            print(f"All weather columns: {list(mars_weather_df.columns)}")
            
            # Check which temperature columns are available
            available_temp_cols = [col for col in mars_temp_cols if col in mars_weather_df.columns]
            available_wind_cols = [col for col in mars_wind_cols if col in mars_weather_df.columns]
            
            print(f"Available temperature columns: {available_temp_cols}")
            print(f"Available wind columns: {available_wind_cols}")
            
            # Calculate avg_temp from the three rod temperature measurements
            if available_temp_cols:
                # Check if temperature columns have valid data
                temp_data_check = mars_weather_df[available_temp_cols].notna().sum().sum()
                print(f"Temperature data points available: {temp_data_check}")
                
                if temp_data_check > 0:
                    mars_weather_df['avg_temp'] = mars_weather_df[available_temp_cols].mean(axis=1, skipna=True)
                    print(f"Calculated avg_temp from {len(available_temp_cols)} temperature columns")
                    print(f"Sample avg_temp values: {mars_weather_df['avg_temp'].head().tolist()}")
                else:
                    print("Warning: Temperature columns exist but contain no valid data")
                    mars_weather_df['avg_temp'] = None
            else:
                print("Warning: No temperature columns found for avg_temp calculation")
                print("Available columns that might be temperature:", [col for col in mars_weather_df.columns if 'temp' in col.lower() or 'TEMP' in col])
                mars_weather_df['avg_temp'] = None
            
            # Use horizontal wind speed as avg_windspeed
            if available_wind_cols:
                # Check if wind column has valid data
                wind_data_check = mars_weather_df[available_wind_cols[0]].notna().sum()
                print(f"Wind data points available: {wind_data_check}")
                
                if wind_data_check > 0:
                    mars_weather_df['avg_windspeed'] = mars_weather_df[available_wind_cols[0]]  # Use first (and likely only) wind column
                    print(f"Using {available_wind_cols[0]} as avg_windspeed")
                    print(f"Sample avg_windspeed values: {mars_weather_df['avg_windspeed'].head().tolist()}")
                else:
                    print("Warning: Wind column exists but contains no valid data")
                    mars_weather_df['avg_windspeed'] = None
            else:
                print("Warning: No wind speed column found")
                print("Available columns that might be wind:", [col for col in mars_weather_df.columns if 'wind' in col.lower() or 'WIND' in col])
                mars_weather_df['avg_windspeed'] = None
            
            # Check if we successfully created the required columns
            if 'avg_temp' in mars_weather_df.columns and 'avg_windspeed' in mars_weather_df.columns:
                # Check if they have any valid data
                temp_valid = mars_weather_df['avg_temp'].notna().sum()
                wind_valid = mars_weather_df['avg_windspeed'].notna().sum()
                
                print(f"Valid temperature readings: {temp_valid}")
                print(f"Valid wind readings: {wind_valid}")
                
                if temp_valid > 0 and wind_valid > 0:
                    weather_processing_success = True
                    missing_weather_cols = []
                    print("Successfully created avg_temp and avg_windspeed columns with valid data")
                    # Show sample of calculated values
                    valid_temp = mars_weather_df['avg_temp'].dropna()
                    valid_wind = mars_weather_df['avg_windspeed'].dropna()
                    if len(valid_temp) > 0 and len(valid_wind) > 0:
                        print(f"Temperature range: {valid_temp.min():.1f}°C to {valid_temp.max():.1f}°C")
                        print(f"Wind speed range: {valid_wind.min():.1f} to {valid_wind.max():.1f} m/s")
                else:
                    print("Error: Created weather columns but they contain no valid data")
                    missing_weather_cols = ['avg_temp', 'avg_windspeed']
            else:
                print("Error: Could not create required weather columns")
                missing_weather_cols = ['avg_temp', 'avg_windspeed']
        else:
            print("No weather data loaded - dataframe is empty")
            missing_weather_cols = ['avg_temp', 'avg_windspeed']
            weather_processing_success = False
        
        print(f"Weather processing success: {weather_processing_success}")
        print(f"Missing weather columns: {missing_weather_cols}")
        
        # Check for activity type column in Mars data
        activity_col = None
        
        for col in mars_activity_cols:
            if col in surface_activity_df.columns:
                activity_col = col
                print(f"Found activity column: {col}")
                break
        
        if activity_col is None:
            print("Warning: No activity type column found")
            print("Available surface activity columns:", surface_activity_df.columns.tolist())
            # Try to find Activity column specifically for Mars data
            if 'Activity' in surface_activity_df.columns:
                activity_col = 'Activity'
                print("Using 'Activity' column for Mars mission activities")
        
        # Also check for Sol column (Mars day) which is common in Mars datasets
        sol_col = None
        if 'Sol' in surface_activity_df.columns:
            sol_col = 'Sol'
            print("Found Sol (Mars day) column")
        
        # Display sample of the cleaned data
        print("\nSample of surface activity data:")
        print(surface_activity_df.head())
        
        if not mars_weather_df.empty:
            print("\nSample of weather data:")
            print(mars_weather_df.head())
        
        # Debug: Check weather data state before processing activities
        print(f"\n{'='*50}")
        print("WEATHER DATA STATE CHECK:")
        print(f"{'='*50}")
        print(f"Weather dataframe empty: {mars_weather_df.empty}")
        print(f"Weather processing success: {weather_processing_success}")
        print(f"Timestamp column found: {timestamp_col_weather}")
        if not mars_weather_df.empty:
            print(f"Weather data shape: {mars_weather_df.shape}")
            print(f"Weather columns: {list(mars_weather_df.columns)}")
            if 'avg_temp' in mars_weather_df.columns:
                print(f"avg_temp column exists, non-null values: {mars_weather_df['avg_temp'].notna().sum()}")
            if 'avg_windspeed' in mars_weather_df.columns:
                print(f"avg_windspeed column exists, non-null values: {mars_weather_df['avg_windspeed'].notna().sum()}")
        print(f"{'='*50}")
        
        # Create dictionary of surface activities with datetime keys
        if timestamp_col_surface and activity_col:
            
            print("Creating Mars surface activity dictionary...")
            
            # Convert UTC to datetime if not already done
            if surface_activity_df[timestamp_col_surface].dtype == 'object':
                surface_activity_df[timestamp_col_surface] = pd.to_datetime(surface_activity_df[timestamp_col_surface])
            
            # Create the dictionary with datetime as keys
            mars_activities_dict = {}
            
            # Iterate through each row to build the dictionary
            for index, row in surface_activity_df.iterrows():
                datetime_key = row[timestamp_col_surface]
                activity_event = row[activity_col]
                
                # Extract the date from the activity datetime (define this early to avoid scoping issues)
                activity_date = datetime_key.date()
                
                # Create entry for this datetime
                activity_data = {
                    'event': activity_event,
                    'datetime': datetime_key,
                }
                
                # Add Sol information if available
                if sol_col and pd.notna(row[sol_col]):
                    activity_data['sol'] = int(row[sol_col])
                
                # Add Solar longitude if available
                if 'Solar longitude (deg)' in surface_activity_df.columns and pd.notna(row['Solar longitude (deg)']):
                    activity_data['solar_longitude_deg'] = float(row['Solar longitude (deg)'])
                
                # Add any other columns from the surface activity data
                for col in surface_activity_df.columns:
                    if col not in [timestamp_col_surface, activity_col, sol_col, 'Solar longitude (deg)']:
                        if pd.notna(row[col]) and str(row[col]).strip() != '':
                            # Clean up column name for dictionary key
                            clean_col = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                            activity_data[clean_col] = row[col]
                
                # Always include weather data using intelligent matching strategy
                if weather_processing_success and not mars_weather_df.empty:
                    weather_data = None
                    
                    if timestamp_col_weather:
                        # We have timestamp data - use intelligent strategy selection
                        
                        # Pre-analysis: Check time ranges and gaps to choose best strategy
                        activity_date_only = activity_date
                        weather_same_date = mars_weather_df[mars_weather_df[timestamp_col_weather].dt.date == activity_date_only]
                        
                        # Calculate time difference to closest weather reading
                        mars_weather_df['time_diff_hours'] = abs((mars_weather_df[timestamp_col_weather] - datetime_key).dt.total_seconds()) / 3600
                        closest_time_gap = mars_weather_df['time_diff_hours'].min()
                        
                        print(f"Analyzing best strategy for {datetime_key}:")
                        print(f"  - Closest time gap: {closest_time_gap:.1f} hours")
                        print(f"  - Same date weather readings: {len(weather_same_date)}")
                        
                        # Strategy selection logic
                        strategy_used = None
                        
                        # Strategy 1: Use closest time if gap is reasonable (<12 hours)
                        if closest_time_gap <= 12:
                            print(f"  → Choosing Strategy 1: Closest time gap ({closest_time_gap:.1f}h) is acceptable")
                            
                            closest_readings = mars_weather_df.nsmallest(2, 'time_diff_hours')
                            weather_data = {
                                'avg_temp': round(closest_readings['avg_temp'].mean(), 2),
                                'avg_windspeed': round(closest_readings['avg_windspeed'].mean(), 2)
                            }
                            strategy_used = "1 - Closest Time"
                            
                            closest_datetime = closest_readings.iloc[0][timestamp_col_weather]
                            print(f"    ✓ Using 2 closest readings (nearest: {closest_datetime})")
                        
                        # Strategy 2: Use same date if we have good coverage (and Strategy 1 wasn't ideal)
                        elif len(weather_same_date) >= 2:
                            print(f"  → Choosing Strategy 2: {len(weather_same_date)} readings available for same date")
                            
                            weather_data = {
                                'avg_temp': round(weather_same_date['avg_temp'].mean(), 2),
                                'avg_windspeed': round(weather_same_date['avg_windspeed'].mean(), 2)
                            }
                            strategy_used = "2 - Same Date"
                            print(f"    ✓ Averaging {len(weather_same_date)} readings from {activity_date_only}")
                        
                        # Strategy 3: Try both following year and previous year
                        elif weather_data is None:
                            following_year = activity_date_only.year + 1
                            previous_year = activity_date_only.year - 1
                            
                            weather_next_year = pd.DataFrame()
                            weather_prev_year = pd.DataFrame()
                            
                            # Check following year
                            try:
                                following_year_date = activity_date_only.replace(year=following_year)
                                weather_next_year = mars_weather_df[mars_weather_df[timestamp_col_weather].dt.date == following_year_date]
                                print(f"  - Following year ({following_year_date}): {len(weather_next_year)} readings")
                            except ValueError:
                                print(f"  - Following year ({following_year}): Date error (likely Feb 29)")
                            
                            # Check previous year
                            try:
                                previous_year_date = activity_date_only.replace(year=previous_year)
                                weather_prev_year = mars_weather_df[mars_weather_df[timestamp_col_weather].dt.date == previous_year_date]
                                print(f"  - Previous year ({previous_year_date}): {len(weather_prev_year)} readings")
                            except ValueError:
                                print(f"  - Previous year ({previous_year}): Date error (likely Feb 29)")
                            
                            # Choose the year with more data, or prefer following year if tied
                            best_year_data = pd.DataFrame()
                            year_used = None
                            
                            if len(weather_next_year) > 0 and len(weather_prev_year) > 0:
                                if len(weather_next_year) >= len(weather_prev_year):
                                    best_year_data = weather_next_year
                                    year_used = f"following year ({following_year_date})"
                                    print(f"  → Choosing Strategy 3: Following year has {len(weather_next_year)} readings vs {len(weather_prev_year)} from previous year")
                                else:
                                    best_year_data = weather_prev_year
                                    year_used = f"previous year ({previous_year_date})"
                                    print(f"  → Choosing Strategy 3: Previous year has {len(weather_prev_year)} readings vs {len(weather_next_year)} from following year")
                            
                            elif len(weather_next_year) > 0:
                                best_year_data = weather_next_year
                                year_used = f"following year ({following_year_date})"
                                print(f"  → Choosing Strategy 3: Using {len(weather_next_year)} readings from following year (no previous year data)")
                            
                            elif len(weather_prev_year) > 0:
                                best_year_data = weather_prev_year
                                year_used = f"previous year ({previous_year_date})"
                                print(f"  → Choosing Strategy 3: Using {len(weather_prev_year)} readings from previous year (no following year data)")
                            
                            # Apply the best year data if found
                            if len(best_year_data) > 0:
                                weather_data = {
                                    'avg_temp': round(best_year_data['avg_temp'].mean(), 2),
                                    'avg_windspeed': round(best_year_data['avg_windspeed'].mean(), 2)
                                }
                                strategy_used = f"3 - {year_used.title()}"
                                print(f"    ✓ Using seasonal pattern from {year_used}")
                            else:
                                print(f"  → Strategy 3 failed: No data available in adjacent years")
                        
                        # Strategy 1 Fallback: If other strategies failed, use closest time despite large gap
                        if weather_data is None:
                            print(f"  → Fallback to Strategy 1: Using closest time despite {closest_time_gap:.1f}h gap")
                            
                            closest_readings = mars_weather_df.nsmallest(2, 'time_diff_hours')
                            weather_data = {
                                'avg_temp': round(closest_readings['avg_temp'].mean(), 2),
                                'avg_windspeed': round(closest_readings['avg_windspeed'].mean(), 2)
                            }
                            strategy_used = "1 - Closest Time (Forced)"
                            
                            closest_datetime = closest_readings.iloc[0][timestamp_col_weather]
                            print(f"    ⚠ Using 2 closest readings with large gap (nearest: {closest_datetime})")
                        
                        # Clean up temporary column
                        mars_weather_df.drop('time_diff_hours', axis=1, inplace=True)
                        
                        # Strategy 4: Absolute fallback - use overall average
                        if weather_data is None:
                            print(f"  → Strategy 4: Using overall averages as last resort")
                            weather_data = {
                                'avg_temp': round(mars_weather_df['avg_temp'].mean(), 2),
                                'avg_windspeed': round(mars_weather_df['avg_windspeed'].mean(), 2)
                            }
                            strategy_used = "4 - Overall Average"
                        
                        print(f"    Final: Strategy {strategy_used} selected")
                        activity_data['weather'] = weather_data
                
                else:
                    # This should never happen if weather processing was successful
                    print(f"Critical error: Weather processing failed - debug info:")
                    print(f"  timestamp_col_weather: {timestamp_col_weather}")
                    print(f"  weather_processing_success: {weather_processing_success}")
                    print(f"  mars_weather_df.empty: {mars_weather_df.empty}")
                    
                    activity_data['weather'] = {
                        'avg_temp': None,
                        'avg_windspeed': None
                    }
                
                # Use datetime as key (convert to string for better readability)
                datetime_str = datetime_key.strftime('%Y-%m-%d %H:%M:%S')
                mars_activities_dict[datetime_str] = activity_data
            
            # Display the dictionary
            print(f"\nCreated dictionary with {len(mars_activities_dict)} entries")
            print("\nSample entries from Mars Activities Dictionary:")
            print("=" * 50)
            
            # Show first few entries
            for i, (datetime_key, data) in enumerate(mars_activities_dict.items()):
                if i < 5:  # Show first 5 entries
                    print(f"\nDatetime: {datetime_key}")
                    print(f"Event: {data['event']}")
                    if 'sol' in data:
                        print(f"Sol: {data['sol']}")
                    if 'solar_longitude_deg' in data:
                        print(f"Solar Longitude: {data['solar_longitude_deg']}°")
                    if 'weather' in data:
                        weather = data['weather']
                        temp = weather.get('avg_temp')
                        wind = weather.get('avg_windspeed')
                        
                        if temp is not None and wind is not None:
                            print(f"Weather: Temp={temp}°C, Wind={wind} m/s")
                        else:
                            print(f"Weather: No data available")
                    print("-" * 30)
            
            if len(mars_activities_dict) > 5:
                print(f"... and {len(mars_activities_dict) - 5} more entries")
            
            # Summary statistics
            print(f"\nMars Mission Summary:")
            activities_count = {}
            for data in mars_activities_dict.values():
                event = data['event']
                activities_count[event] = activities_count.get(event, 0) + 1
            
            print("Activity counts:")
            for activity, count in activities_count.items():
                print(f"  {activity}: {count}")
            
            if sol_col:
                sols = [data['sol'] for data in mars_activities_dict.values() if 'sol' in data]
                if sols:
                    print(f"Sol range: {min(sols)} to {max(sols)}")
            
            # Return the dictionary for further use
            print(f"\nDictionary stored in variable: mars_activities_dict")
            print("You can access specific activities by datetime key, e.g.:")
            first_key = list(mars_activities_dict.keys())[0]
            print(f"mars_activities_dict['{first_key}']")
            
            # Also create a simpler version organized by event type
            events_by_type = {}
            for datetime_str, data in mars_activities_dict.items():
                event_type = data['event']
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append({
                    'datetime': datetime_str,
                    'data': data
                })
            
            print(f"\nAlso created events_by_type dictionary with {len(events_by_type)} event categories")
            
            return mars_activities_dict, events_by_type
        
        else:
            print("\nCannot create dictionary due to missing essential columns.")
            print("Required: timestamp and activity columns in surface data")
            if not timestamp_col_surface:
                print("Missing timestamp column in surface activity data")
            if not activity_col:
                print("Missing activity column in surface activity data")
            return None, None
    
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        print("Please check that the CSV files exist in the 'csv' folder")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: One of the CSV files is empty")
        return None, None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your data files and try again")
        return None, None

# Run the function and get the results
if __name__ == "__main__":
    mars_activities_dict, events_by_type = process_mars_data()
    
    # Example of how to use the returned dictionaries:
    if mars_activities_dict:
        print(f"\n{'='*50}")
        print("USAGE EXAMPLES:")
        print(f"{'='*50}")
        
        # Show how to access data
        first_datetime = list(mars_activities_dict.keys())[0]
        print(f"\n1. Access specific activity by datetime:")
        print(f"   activity_data = mars_activities_dict['{first_datetime}']")
        print(f"   Result: {mars_activities_dict[first_datetime]}")
        
        # Show event types available
        if events_by_type:
            print(f"\n2. Available event types:")
            for event_type, events in events_by_type.items():
                print(f"   '{event_type}': {len(events)} occurrences")
            
            # Show how to get all events of specific type
            first_event_type = list(events_by_type.keys())[0]
            print(f"\n3. Get all events of specific type:")
            print(f"   {first_event_type.lower().replace(' ', '_')}_events = events_by_type['{first_event_type}']")
            print(f"   Number of {first_event_type} events: {len(events_by_type[first_event_type])}")