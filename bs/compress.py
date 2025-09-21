import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

def parse_ps_data(file_path, interpolate_gaps=True, max_gap_minutes=60):
    """
    Parse InSight pressure sensor data file and handle missing values.
    
    Parameters:
    -----------
    file_path : Path or str
        Path to the CSV file
    interpolate_gaps : bool
        Whether to interpolate small gaps in the data
    max_gap_minutes : int
        Maximum gap size (in minutes) to interpolate
    
    Returns:
    --------
    pandas.DataFrame
        Parsed and processed data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Check if the file has data
        if df.empty:
            print(f"  File is empty, skipping.")
            return None
            
        # Look for datetime columns - common names in PS data
        datetime_cols = []
        for col in df.columns:
            if any(keyword in col.upper() for keyword in ['TIME', 'UTC', 'DATE']):
                datetime_cols.append(col)
        
        if not datetime_cols:
            print(f"  No datetime column found, skipping.")
            return None
            
        # Use the first datetime column found
        time_col = datetime_cols[0]
        
        # Parse datetime - handle different formats
        try:
            # Try common InSight format first
            df[time_col] = pd.to_datetime(df[time_col], format="%Y-%jT%H:%M:%S.%fZ")
        except:
            try:
                # Try standard ISO format
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                print(f"  Could not parse datetime format, skipping.")
                return None
        
        # Sort by time
        df = df.sort_values(time_col)
        
        # Identify numeric columns and important metadata columns
        numeric_cols = []
        important_metadata = []
        
        for col in df.columns:
            if col != time_col:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # Keep all numeric columns except file size
                    if 'SIZE' not in col.upper() and 'MB' not in col.upper():
                        numeric_cols.append(col)
                elif col.upper() in ['FILE NAME', 'URN', 'SOL_NUMBER']:
                    # Keep important string metadata
                    important_metadata.append(col)
        
        if not numeric_cols:
            print(f"  No numeric sensor columns found, skipping.")
            return None
            
        # Create working dataframe with time, numeric, and important metadata columns
        all_keep_cols = [time_col] + numeric_cols + important_metadata
        work_df = df[all_keep_cols].copy()
        
        # Remove duplicate timestamps first
        print(f"  Original data points: {len(work_df)}")
        duplicates_before = work_df.duplicated(subset=[time_col]).sum()
        if duplicates_before > 0:
            print(f"  Found {duplicates_before} duplicate timestamps, removing...")
            # Keep first occurrence of duplicate timestamps, average the values
            work_df = work_df.groupby(time_col)[numeric_cols].mean().reset_index()
            print(f"  After duplicate removal: {len(work_df)} data points")
        
        # Handle missing values if interpolation is enabled
        if interpolate_gaps and len(work_df) > 1:
            # Set time column as index for interpolation
            work_df = work_df.set_index(time_col)
            
            for col in numeric_cols:
                # Only interpolate if there are some valid values and missing values
                if work_df[col].notna().any() and work_df[col].isna().any():
                    # Calculate typical time step
                    time_diffs = work_df.index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        median_diff = time_diffs.median()
                        
                        # Only interpolate gaps smaller than max_gap_minutes
                        max_gap = pd.Timedelta(minutes=max_gap_minutes)
                        
                        # Interpolate with limit based on gap size
                        if pd.notna(median_diff) and median_diff.total_seconds() > 0:
                            max_gap_periods = max_gap.total_seconds() / median_diff.total_seconds()
                            
                            # Ensure limit is at least 1
                            limit_periods = max(1, int(max_gap_periods))
                            
                            try:
                                work_df[col] = work_df[col].interpolate(
                                    method='time', 
                                    limit=limit_periods
                                )
                            except Exception as e:
                                print(f"    Warning: Could not interpolate column {col}: {e}")
                                # Fall back to simple linear interpolation
                                work_df[col] = work_df[col].interpolate(method='linear', limit=limit_periods)
            
            # Reset index back to column
            work_df = work_df.reset_index()
        
        return work_df
        
    except Exception as e:
        print(f"  Error processing file: {str(e)}")
        return None

def aggregate_to_15min(df, time_col, numeric_cols, metadata_cols=None):
    """
    Aggregate data to 15-minute intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    time_col : str
        Name of the time column
    numeric_cols : list
        List of numeric column names
    metadata_cols : list
        List of metadata column names to preserve
    
    Returns:
    --------
    pandas.DataFrame
        Aggregated data
    """
    # Floor timestamps to 15-minute intervals
    df["time_bin"] = df[time_col].dt.floor("15min")
    
    # Define aggregation functions for different types of data
    agg_dict = {}
    for col in numeric_cols:
        # Use mean for most measurements, but could customize based on column name
        if 'DIRECTION' in col.upper() or 'DIR' in col.upper():
            # For directional data, use circular mean (simplified here as regular mean)
            # In a more sophisticated version, you'd handle circular statistics properly
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'mean'
    
    # For metadata columns, take the first value in each bin
    if metadata_cols:
        for col in metadata_cols:
            agg_dict[col] = 'first'
    
    # Aggregate data
    agg_df = df.groupby("time_bin").agg(agg_dict).reset_index()
    agg_df.rename(columns={"time_bin": time_col}, inplace=True)
    
    # Remove intervals with all NaN values in numeric columns only
    agg_df = agg_df.dropna(how='all', subset=numeric_cols)
    
    return agg_df

def process_ps_files(data_pattern="ps_data_*.csv", output_file="mars_ps_15min.csv", 
                    interpolate_gaps=True, max_gap_minutes=60):
    """
    Process multiple InSight pressure sensor data files.
    
    Parameters:
    -----------
    data_pattern : str
        File pattern to match (supports wildcards)
    output_file : str
        Output CSV file name
    interpolate_gaps : bool
        Whether to interpolate small data gaps
    max_gap_minutes : int
        Maximum gap size to interpolate (minutes)
    
    Returns:
    --------
    pandas.DataFrame
        Combined processed data
    """
    
    # Get list of files matching pattern
    file_paths = list(Path(".").glob(data_pattern))
    
    if not file_paths:
        print(f"No files found matching pattern: {data_pattern}")
        return None
    
    print(f"Found {len(file_paths)} files matching pattern: {data_pattern}")
    
    output = []
    total_processed = 0
    
    for file_path in sorted(file_paths):
        print(f"Processing file {file_path.name}...")
        
        # Parse the file
        df = parse_ps_data(file_path, interpolate_gaps, max_gap_minutes)
        
        if df is None:
            continue
            
        # Get column names
        time_col = None
        numeric_cols = []
        metadata_cols = []
        
        for col in df.columns:
            if df[col].dtype.name.startswith('datetime'):
                time_col = col
            elif df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if 'SIZE' not in col.upper() and 'MB' not in col.upper():
                    numeric_cols.append(col)
            elif col.upper() in ['FILE NAME', 'URN', 'SOL_NUMBER']:
                metadata_cols.append(col)
        
        if time_col is None:
            print(f"  No datetime column found, skipping.")
            continue
            
        if not numeric_cols:
            print(f"  No numeric sensor columns found, skipping.")
            continue
            
        # Print what columns we're keeping
        print(f"  Time column: {time_col}")
        print(f"  Numeric columns ({len(numeric_cols)}): {numeric_cols}")
        if metadata_cols:
            print(f"  Metadata columns: {metadata_cols}")
            
        # Aggregate to 15-minute intervals
        agg_df = aggregate_to_15min(df, time_col, numeric_cols, metadata_cols)
        
        if agg_df.empty:
            print(f"  No data after aggregation, skipping.")
            continue
            
        # Add file identifier
        file_id = re.search(r'(\d+)', file_path.stem)
        if file_id:
            agg_df["file_id"] = file_id.group(1).zfill(4)
        else:
            agg_df["file_id"] = file_path.stem
            
        output.append(agg_df)
        total_processed += len(agg_df)
        print(f"  Added {len(agg_df)} 15-minute intervals")
    
    if not output:
        print("No data processed successfully.")
        return None
        
    # Concatenate all files
    final_df = pd.concat(output, ignore_index=True)
    
    # Sort by time
    time_cols = [col for col in final_df.columns if final_df[col].dtype.name.startswith('datetime')]
    if time_cols:
        final_df = final_df.sort_values(time_cols[0])
        final_df = final_df.reset_index(drop=True)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"\nProcessing complete!")
    print(f"Files processed: {len(output)}")
    print(f"Total 15-minute intervals: {total_processed}")
    print(f"Saved aggregated data to {output_file}")
    print(f"Final dataset shape: {final_df.shape}")
    
    if time_cols:
        print(f"Time range: {final_df[time_cols[0]].min()} to {final_df[time_cols[0]].max()}")
    
    # Print summary statistics
    numeric_cols = [col for col in final_df.columns 
                   if final_df[col].dtype in ['float64', 'int64', 'float32', 'int32'] 
                   and not any(skip in col.upper() for skip in ['ID', 'SOL', 'YEAR'])]
    
    if numeric_cols:
        print(f"\nData completeness for sensor columns:")
        for col in numeric_cols[:10]:  # Show first 10 numeric columns
            completeness = (final_df[col].notna().sum() / len(final_df)) * 100
            print(f"  {col}: {completeness:.1f}% complete")
        
        if len(numeric_cols) > 10:
            print(f"  ... and {len(numeric_cols) - 10} more columns")
    
    return final_df

# Example usage:
if __name__ == "__main__":
    # Process files matching your pattern
    result = process_ps_files(
        data_pattern="ps_data_*.csv",  # Adjust pattern as needed
        output_file="mars_ps_15min.csv",
        interpolate_gaps=True,
        max_gap_minutes=60  # Interpolate gaps up to 1 hour
    )
    
    # If you have specific file ranges like in your original code:
    # You could also process specific numbered files:
    """
    for i in range(1, 15):  # Adjust range as needed
        file_pattern = f"ps_data_*{i:04d}*.csv"  # or whatever your naming convention is
        # Process each file...
    """