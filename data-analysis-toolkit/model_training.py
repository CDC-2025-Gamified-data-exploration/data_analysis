import pandas as pd
import statsmodels.api as sm

final = pd.read_csv("mars_weather_daily_summary.csv")

# Features
X = final[[
    "min_rod_temp", "avg_rod_temp", "max_rod_temp",
    "min_wind_speed", "avg_wind_speed", "max_wind_speed",
    "wind_dir_volatility", "Solar longitude (deg)", "age_factor"
]]

# Target: binary numeric
y = final["Activity"].map({"Runout": 1, "No runout": 0})

# Add constant for intercept
X = sm.add_constant(X)

logit_model = sm.Logit(y, X, missing='drop')  # drops rows with NaN
result = logit_model.fit()

# Print summary
print(result.summary())

import numpy as np

odds_ratios = pd.DataFrame({
    "OR": np.exp(result.params),
    "Lower CI": np.exp(result.conf_int()[0]),
    "Upper CI": np.exp(result.conf_int()[1])
})

print(odds_ratios)
