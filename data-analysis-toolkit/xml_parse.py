import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
csv_path = Path("mars_weather_15min.csv")
xml_folder = Path("../datasets/xml")  # adjust if needed

# Load the aggregated CSV
df = pd.read_csv(csv_path, parse_dates=["bin"])

# Function to parse solar longitude from an XML file
def get_solar_longitude(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Use the namespace from the XML
    ns = {'insight': 'http://pds.nasa.gov/pds4/mission/insight/v1'}
    # Find the element
    sl_element = root.find(".//insight:start_solar_longitude", ns)
    if sl_element is not None:
        return float(sl_element.text)
    else:
        return None

# Map file_id to solar longitude
solar_longitudes = {}

# Loop over XML files
for xml_file in xml_folder.glob("twins_calib_*.xml"):
    # Extract file_id from filename
    file_id = xml_file.stem.split("_")[2]  # '0736' from twins_calib_0736_01.xml
    solar_longitudes[file_id] = get_solar_longitude(xml_file)

# Add a new column to the CSV based on file_id
df["file_id_str"] = df["file_id"].apply(lambda x: f"{int(x):04d}")  # pad to 4 digits
df["solar_longitude"] = df["file_id_str"].map(solar_longitudes)

# Save updated CSV
df.to_csv("mars_weather_15min_with_solar_longitude.csv", index=False)

print("Added solar_longitude column successfully!")
