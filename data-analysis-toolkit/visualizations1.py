import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 1️⃣ Load the aggregated CSV
df = pd.read_csv("mars_weather_15min.csv", parse_dates=["bin"])

# 2️⃣ Add a week column
df["week"] = df["bin"].dt.isocalendar().week

# 3️⃣ Weekly average temperature (mean of base, mid, tip)
df["avg_temp"] = df[["BMY_BASE_ROD_TEMP", "BMY_MID_ROD_TEMP", "BMY_TIP_ROD_TEMP"]].mean(axis=1)
weekly_temp = df.groupby("week")["avg_temp"].mean()


# 5️⃣ Wind speed visualization
# Line plot of daily average wind speed
df["date"] = df["bin"].dt.date
daily_wind = df.groupby("date")["BMY_HORIZONTAL_WIND_SPEED"].mean()

plt.figure(figsize=(12,6))
daily_wind.plot(kind="line", color="skyblue")
plt.title("Daily Average Horizontal Wind Speed on Mars")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.tight_layout()
plt.show()

# 6️⃣ Optional: histogram of wind speed
plt.figure(figsize=(8,5))
df["BMY_HORIZONTAL_WIND_SPEED"].hist(bins=30, color="lightgreen")
plt.title("Distribution of Mars Horizontal Wind Speeds")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



#################################

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("mars_weather_15min.csv", parse_dates=["bin"])

# Average temperature across rods
df["avg_temp"] = df[["BMY_BASE_ROD_TEMP", "BMY_MID_ROD_TEMP", "BMY_TIP_ROD_TEMP"]].mean(axis=1)

# Add month column
df["month"] = df["bin"].dt.to_period("M")

# Compute monthly mean and std for wind and temp
monthly_stats = df.groupby("month").agg({
    "BMY_HORIZONTAL_WIND_SPEED": ["mean", "std"],
    "avg_temp": ["mean", "std"]
})
monthly_stats.columns = ["wind_mean", "wind_std", "temp_mean", "temp_std"]
monthly_stats = monthly_stats.reset_index()

# Merge monthly stats back to daily-level data
df = df.merge(monthly_stats, on="month", how="left")

# Compute Z-scores for each day relative to its month
df["wind_z"] = (df["BMY_HORIZONTAL_WIND_SPEED"] - df["wind_mean"]) / df["wind_std"]
df["temp_z"] = (df["avg_temp"] - df["temp_mean"]) / df["temp_std"]

# Flag outliers (e.g., |z| > 2)
df["wind_outlier"] = df["wind_z"].abs() > 2
df["temp_outlier"] = df["temp_z"].abs() > 2

# Count of outliers per month
outlier_counts = df.groupby("month").agg({
    "wind_outlier": "sum",
    "temp_outlier": "sum"
})

# Plot number of outliers per month
plt.figure(figsize=(14,6))
plt.bar(outlier_counts.index.astype(str), outlier_counts["wind_outlier"], color="orange", alpha=0.7, label="Wind Outliers")
plt.bar(outlier_counts.index.astype(str), outlier_counts["temp_outlier"], color="blue", alpha=0.5, label="Temperature Outliers")
plt.xticks(rotation=90)
plt.xlabel("Month")
plt.ylabel("Number of Outliers")
plt.title("Monthly Counts of Outliers Based on Monthly SD/Mean")
plt.legend()
plt.tight_layout()
plt.show()