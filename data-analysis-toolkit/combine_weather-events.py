import pandas as pd
import numpy as np

# Load datasets
weather = pd.read_csv("mars_weather_15min.csv")
events = pd.read_csv("rover_events.csv")

def wind_dir_volatility(directions):
    # Remove NaNs
    directions = directions.dropna().values
    if len(directions) < 2:
        return np.nan  # not enough data to compute volatility
    # Compute differences between consecutive readings
    diffs = np.diff(directions)
    # Adjust for circular wrap-around to range [-180, 180]
    diffs = (diffs + 180) % 360 - 180
    # Return average absolute change
    return np.mean(np.abs(diffs))

# Step 1: Compute average rod temp for each row
weather["avg_rod_temp"] = weather[["BMY_BASE_ROD_TEMP", "BMY_MID_ROD_TEMP", "BMY_TIP_ROD_TEMP"]].mean(axis=1)

# Step 2: Group by sol (file_id)
agg_weather = weather.groupby("file_id").agg(
    min_rod_temp=("avg_rod_temp", "min"),
    avg_rod_temp=("avg_rod_temp", "mean"),
    max_rod_temp=("avg_rod_temp", "max"),
    min_wind_speed=("BMY_HORIZONTAL_WIND_SPEED", "min"),
    avg_wind_speed=("BMY_HORIZONTAL_WIND_SPEED", "mean"),
    max_wind_speed=("BMY_HORIZONTAL_WIND_SPEED", "max"),
    wind_dir_volatility=("BMY_WIND_DIRECTION", wind_dir_volatility)
).reset_index()

# Step 3: Clean rover events
events_small = events[["Sol", "Activity", "Solar longitude (deg)"]]

# Step 4: Merge on file_id <-> Sol
final = agg_weather.merge(events_small, left_on="file_id", right_on="Sol", how="inner")

#######################################

#file_id is same as sol
final = final.drop(columns=["file_id"])

#create Runout or no Runout
final["Activity"] = final["Activity"].apply(lambda x: "Runout" if x == "Runout" else "No runout")

# Drop sols > 83
final = final[final["Sol"] <= 83]

# Create age_factor between sol 14–83
mask = (final["Sol"] >= 14) & (final["Sol"] <= 83)
sols_range = final.loc[mask, "Sol"].sort_values().unique()

# map sols 14–83 -> evenly spaced 0–1
age_map = {sol: i / (len(sols_range) - 1) for i, sol in enumerate(sols_range)}
final.loc[mask, "age_factor"] = final.loc[mask, "Sol"].map(age_map)

#rearrange
cols = ["Sol", "Activity"] + [c for c in final.columns if c not in ["Sol", "Activity"]]
final = final[cols]

# Save
final.to_csv("mars_weather_daily_summary.csv", index=False)

print("Saved daily summary with weather + rover activities + solar longitude")
