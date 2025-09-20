# mars_weather_viz.py

import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Load the aggregated CSV
df = pd.read_csv("mars_weather_15min.csv", parse_dates=["bin"])

# 2️⃣ Add a week column
df["week"] = df["bin"].dt.isocalendar().week

# 3️⃣ Weekly average temperature (mean of base, mid, tip)
df["avg_temp"] = df[["BMY_BASE_ROD_TEMP", "BMY_MID_ROD_TEMP", "BMY_TIP_ROD_TEMP"]].mean(axis=1)
weekly_temp = df.groupby("week")["avg_temp"].mean()

# 4️⃣ Plot: bar graph of weekly average temperature
plt.figure(figsize=(12,6))
weekly_temp.plot(kind="bar", color="tomato")
plt.title("Average Mars Rod Temperature by Week")
plt.xlabel("Week of Year")
plt.ylabel("Temperature (°C or K)")
plt.tight_layout()
plt.show()

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
