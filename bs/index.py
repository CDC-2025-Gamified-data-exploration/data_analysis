import pandas as pd
import numpy as np
from datetime import datetime
import re
from pathlib import Path

class InSightURNFilter:
    def __init__(self):
        """
        Filter existing InSight index files to get URNs for 2018-2020 data
        """
        # Define date range for filtering (2 years from landing)
        self.start_date = datetime(2018, 11, 26)  # InSight landing date
        self.end_date = datetime(2020, 12, 31)    # End of 2020
        
        # Input file paths (your existing index files)
        self.twins_file = "Twins_Calibrated_2022-08-27.csv"
        self.ps_file = "insight_ps_data_calibrated_del_1-14.csv"
    
    def load_index_files(self):
        """Load the existing index CSV files"""
        print("📂 Loading existing index files...")
        
        indices = {}
        
        # Load TWINS index
        if Path(self.twins_file).exists():
            try:
                twins_df = pd.read_csv(self.twins_file)
                indices['twins'] = twins_df
                print(f"   ✅ TWINS: {len(twins_df)} entries loaded from {self.twins_file}")
            except Exception as e:
                print(f"   ❌ Error loading TWINS file: {e}")
        else:
            print(f"   ❌ TWINS file not found: {self.twins_file}")
        
        # Load PS index
        if Path(self.ps_file).exists():
            try:
                ps_df = pd.read_csv(self.ps_file)
                indices['ps'] = ps_df
                print(f"   ✅ PS: {len(ps_df)} entries loaded from {self.ps_file}")
            except Exception as e:
                print(f"   ❌ Error loading PS file: {e}")
        else:
            print(f"   ❌ PS file not found: {self.ps_file}")
        
        return indices
    
    def show_column_info(self, df, name):
        """Display information about DataFrame columns"""
        print(f"\n📊 {name.upper()} INDEX STRUCTURE:")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Shape: {df.shape}")
        
        # Show sample rows
        print(f"   Sample entries:")
        for i in range(min(3, len(df))):
            print(f"   Row {i}: {dict(df.iloc[i])}")
    
    def extract_sol_from_filename(self, filename):
        """Extract Sol number from filename"""
        if pd.isna(filename) or filename == '':
            return None
            
        filename_str = str(filename).lower()
        
        # Try different patterns to extract Sol number
        patterns = [
            r'sol_(\d+)',           # sol_001, sol_0001
            r'sol(\d+)',            # sol001, sol0001
            r'(\d{3,4})\.csv',      # 001.csv, 1234.csv
            r'del_(\d+)',           # del_1 (from del_1-14)
            r'_(\d+)[-_\.]',        # _123_, _123-, _123.
            r'(\d+)[-_](\d+)',      # range like 1-14, use first number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_str)
            if match:
                return int(match.group(1))
        
        return None
    
    def convert_sol_to_date(self, sol_number):
        """Convert Sol number to Earth date (approximate)"""
        if sol_number is None:
            return None
        
        # Sol 1 = Nov 26, 2018 (landing date)
        # 1 Sol ≈ 24h 39m 35s ≈ 1.02749 Earth days
        sol_length_days = 1.02749
        landing_date = datetime(2018, 11, 26)
        
        earth_date = landing_date + pd.Timedelta(days=(sol_number - 1) * sol_length_days)
        return earth_date
    
    def parse_date_from_columns(self, row):
        """Try to extract date from various possible date columns"""
        date_columns = ['Start Time', 'Start_Time', 'Date', 'UTC', 'Time', 'Timestamp']
        
        for col in date_columns:
            if col in row and pd.notna(row[col]):
                try:
                    # Try different date formats
                    date_str = str(row[col]).strip()
                    
                    # Handle ISO format with Z suffix
                    if date_str.endswith('Z'):
                        date_str = date_str[:-1]  # Remove Z
                    
                    date_formats = [
                        '%Y-%m-%dT%H:%M:%S.%f',  # 2018-11-30T14:44:27.755
                        '%Y-%m-%dT%H:%M:%S',     # 2018-11-30T14:44:27
                        '%Y-%m-%d',              # 2018-11-30
                        '%Y%m%d',                # 20181130
                        '%m/%d/%Y',              # 11/30/2018
                    ]
                    
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, date_format)
                            return parsed_date
                        except ValueError:
                            continue
                            
                    # Try pandas to_datetime as fallback
                    try:
                        parsed_date = pd.to_datetime(date_str)
                        return parsed_date.to_pydatetime()
                    except:
                        continue
                        
                except Exception as e:
                    continue
        
        return None
    
    def filter_by_date_range(self, df, source_name):
        """Filter DataFrame to 2018-2020 date range"""
        print(f"\n🔍 Filtering {source_name} data for 2018-2020...")
        
        # Show structure first
        self.show_column_info(df, source_name)
        
        filtered_rows = []
        date_method_used = []
        
        # Debug: Check a few sample dates
        print(f"   🔍 Debug - Testing date parsing on first 3 rows:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            test_date = self.parse_date_from_columns(row)
            sol_date = None
            if 'sol_number' in row:
                sol_date = self.convert_sol_to_date(row['sol_number'])
            print(f"      Row {i}: Start Time='{row.get('Start Time', 'N/A')}' -> Parsed: {test_date}, Sol Date: {sol_date}")
        
        for idx, row in df.iterrows():
            row_date = None
            method = ""
            
            # Method 1: Try to parse date from date columns
            row_date = self.parse_date_from_columns(row)
            if row_date:
                method = "date_column"
            
            # Method 2: Try to use sol_number column directly (most reliable)
            if not row_date and 'sol_number' in row and pd.notna(row['sol_number']):
                sol_num = int(row['sol_number'])
                row_date = self.convert_sol_to_date(sol_num)
                method = f"sol_column_{sol_num}"
            
            # Method 3: Try to extract Sol number from filename and convert to date
            if not row_date:
                filename_candidates = []
                for col in ['URN', 'Filename', 'File', 'Product_Id', 'File Name']:
                    if col in row and pd.notna(row[col]):
                        filename_candidates.append(str(row[col]))
                
                for filename in filename_candidates:
                    sol_num = self.extract_sol_from_filename(filename)
                    if sol_num:
                        row_date = self.convert_sol_to_date(sol_num)
                        method = f"sol_extraction_{sol_num}"
                        break
            
            # Check if date falls within our range
            if row_date and self.start_date <= row_date <= self.end_date:
                filtered_rows.append(idx)
                date_method_used.append(method)
        
        filtered_df = df.iloc[filtered_rows].copy()
        
        print(f"   📅 Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"   📊 Filtered from {len(df)} to {len(filtered_df)} entries")
        print(f"   🔧 Date extraction methods used: {set(date_method_used)}")
        
        if len(filtered_df) > 0:
            print(f"   📋 Sample filtered entries:")
            for i in range(min(5, len(filtered_df))):
                row = filtered_df.iloc[i]
                urn = row.get('URN', 'N/A')
                start_time = row.get('Start Time', 'N/A')
                sol_num = row.get('sol_number', 'N/A')
                print(f"      {i+1}: Sol {sol_num} ({start_time[:10]}) - {urn}")
        else:
            print(f"   ❌ No entries found in date range. Sample dates from data:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                parsed_date = self.parse_date_from_columns(row)
                sol_date = self.convert_sol_to_date(row.get('sol_number')) if 'sol_number' in row else None
                print(f"      Row {i}: {parsed_date} (sol: {sol_date})")
        
        return filtered_df
    
    def extract_urns(self, df):
        """Extract URN values from filtered DataFrame"""
        urn_columns = ['URN', 'urn', 'Uniform_Resource_Name', 'Product_URN']
        
        for col in urn_columns:
            if col in df.columns:
                urns = df[col].dropna().tolist()
                print(f"   📝 Found {len(urns)} URNs in column '{col}'")
                return urns
        
        print(f"   ❌ No URN column found. Available columns: {list(df.columns)}")
        return []
    
    def save_urn_file(self, urns, filename):
        """Save URNs to a file that can be uploaded"""
        if not urns:
            print(f"   ❌ No URNs to save for {filename}")
            return None
            
        output_path = Path(filename)
        
        # Save as simple text file (one URN per line)
        with open(output_path, 'w') as f:
            for urn in urns:
                f.write(f"{urn}\n")
        
        print(f"   💾 Saved {len(urns)} URNs to {output_path}")
        return output_path
    
    def process_all_indices(self):
        """Main function to process both index files and generate URN files"""
        print("🚀 InSight 2018-2020 URN Filter")
        print("="*40)
        
        # Load existing index files
        indices = self.load_index_files()
        
        if not indices:
            print("❌ No index files could be loaded!")
            return
        
        total_urns = 0
        
        # Process TWINS data
        if 'twins' in indices:
            twins_filtered = self.filter_by_date_range(indices['twins'], 'twins')
            twins_urns = self.extract_urns(twins_filtered)
            
            if twins_urns:
                self.save_urn_file(twins_urns, 'twins_2018_2020_urns.txt')
                total_urns += len(twins_urns)
        
        # Process PS data  
        if 'ps' in indices:
            ps_filtered = self.filter_by_date_range(indices['ps'], 'ps')
            ps_urns = self.extract_urns(ps_filtered)
            
            if ps_urns:
                self.save_urn_file(ps_urns, 'ps_2018_2020_urns.txt')
                total_urns += len(ps_urns)
        
        # Create combined file
        all_urns = []
        if 'twins' in indices:
            twins_filtered = self.filter_by_date_range(indices['twins'], 'twins')
            all_urns.extend(self.extract_urns(twins_filtered))
        
        if 'ps' in indices:
            ps_filtered = self.filter_by_date_range(indices['ps'], 'ps')
            all_urns.extend(self.extract_urns(ps_filtered))
        
        if all_urns:
            self.save_urn_file(all_urns, 'insight_2018_2020_all_urns.txt')
        
        print(f"\n✅ SUMMARY:")
        print(f"   📊 Total URNs extracted: {total_urns}")
        print(f"   📁 Files created:")
        print(f"      - twins_2018_2020_urns.txt")
        print(f"      - ps_2018_2020_urns.txt") 
        print(f"      - insight_2018_2020_all_urns.txt (combined)")
        print(f"\n📤 Upload these files to the NASA PDS bulk download system!")

def main():
    """Run the URN filtering process"""
    filter_tool = InSightURNFilter()
    filter_tool.process_all_indices()

if __name__ == "__main__":
    main()