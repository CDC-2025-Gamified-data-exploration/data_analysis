import pandas as pd
from pathlib import Path

input_dir = Path("../datasets/twins_calib")   # folder with calib files
output = []

# files 14 to 741, any earlier is patchy data
for i in range(14, 741):
    # some files are _02.csv or _01.csv
    for suffix in ["01", "02"]:
        file_name = f"twins_calib_{i:04d}_{suffix}.csv"
        file_path = input_dir / file_name
        
        print(f"Processing file {file_name}...")

        if not file_path.exists():
            print(f"  File not found, skipping.")
            continue

        df = pd.read_csv(file_path)

        # Parse datetime from UTC column
        df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%jT%H:%M:%S.%fZ")

        # Floor timestamps to 15 min
        df["bin"] = df["UTC"].dt.floor("15min")

        # Select and aggregate
        cols = [
            "BMY_BASE_ROD_TEMP",
            "BMY_MID_ROD_TEMP",
            "BMY_TIP_ROD_TEMP",
            "BMY_HORIZONTAL_WIND_SPEED",
            "BMY_WIND_DIRECTION",
        ]
        agg = df.groupby("bin")[cols].mean().reset_index()

        # Drop rows with all NaN values (gaps)
        agg = agg.dropna(how="all", subset=cols)

        # Add file_id
        agg["file_id"] = f"{i:04d}"

        output.append(agg)

# Concatenate all days
final = pd.concat(output, ignore_index=True)

# output to csv
final.to_csv("mars_weather_15min.csv", index=False)
print("Saved aggregated file with shape:", final.shape)
