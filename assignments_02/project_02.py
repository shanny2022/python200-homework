"""Week 2 mini-project: predict New York City's daily high in 2023."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# --- Task 1: Fetch the Data ---
# City: New York City, NY; one calendar year, January–December 2023.
# Source: https://open-meteo.com/en/docs/historical-weather-api
# These are historical reanalysis data, rather than a future weather forecast.
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
}
response = requests.get(url, params=params, timeout=60)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print("Task 1 — New York City, 2023")
print("Dataset shape:", df.shape)
print(df.head())

# --- Task 2: Look at the Features ---
print("\nTask 2 — Feature exploration")
print("Missing values by column:\n", df.isna().sum())
# If a day has a missing measurement, drop the whole row rather than invent
# a temperature or precipitation value. Report the count before modeling.
rows_before = len(df)
df = df.dropna().copy()
print("Rows dropped for missing values:", rows_before - len(df))
if len(df) < 10:
    raise ValueError("Too few complete daily observations to train and evaluate.")
print(df.describe())
correlations = df.select_dtypes(include="number").corr()[
    "temperature_2m_max"
].drop("temperature_2m_max").sort_values(ascending=False)
print("Feature correlations with the daily high:\n", correlations)
print("Strongest absolute correlation:", correlations.abs().idxmax())
plt.figure(figsize=(7, 5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.6)
plt.title("New York City: Daily High vs Low (2023)")
plt.xlabel("Daily low (°C)")
plt.ylabel("Daily high (°C)")
plt.tight_layout()
plt.savefig(OUTPUTS / "high_vs_low_project.png", dpi=150)
plt.close()
# The scatter shows a strong upward trend with some spread: warmer lows tend
# to go with warmer highs. The low has the strongest correlation, about 0.934.

# --- Task 3: Engineer a Feature ---
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)
print("\nTask 3 — Summer feature")
print("Summer days:", int(df["is_summer"].sum()))
print("Non-summer days:", int((df["is_summer"] == 0).sum()))
# I expect summer to help because June–August generally have higher daily
# highs. The low temperature already captures much of this seasonal pattern.
df.to_csv(OUTPUTS / "nyc_weather_2023.csv", index=False)

# --- Task 4: Baseline Model ---
X_baseline = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)
baseline = LinearRegression()
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = baseline.score(X_test, y_test)
print("\nTask 4 — Baseline model")
print(f"Slope: {baseline.coef_[0]:.3f}")
print(f"Test RMSE: {baseline_rmse:.3f} °C")
print(f"Test R²: {baseline_r2:.3f}")
# RMSE describes the size of prediction errors in Celsius and gives larger
# misses extra weight; it is not the mean absolute error or an error bound.
# The baseline RMSE is 3.356 °C: a typical error scale is about 3.4 degrees.

# --- Task 5: Full Model ---
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# The same rows enter the baseline and full test sets because the row order,
# split fraction, and random seed match. This makes the comparison fair.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nTask 5 — Full model")
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Test RMSE: {rmse:.3f} °C")
print(f"Test R² change from baseline: {test_r2 - baseline_r2:+.3f}")
print(f"Test RMSE reduction: {baseline_rmse - rmse:+.3f} °C")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")
# Each coefficient describes a one-unit change holding other features fixed:
# 1 °C of low temperature, 1 mm precipitation, 1 km/h wind, or switching
# from non-summer to summer. Associations do not prove causation.
# In the verified run, a 1 °C warmer low raises the predicted high by 1.019 °C;
# 1 mm more precipitation lowers it by 0.103 °C. A 1 km/h wind increase raises
# it by 0.015 °C, while summer lowers it by only 0.015 °C at fixed other inputs.
# The tiny negative summer sign is surprising, but the low already captures
# seasonality; this coefficient describes its remaining, conditional effect.
# Wind's positive conditional sign also differs from its negative correlation.
# Train R² (0.884) and test R² (0.897) are close, with no obvious overfitting gap.
# Extra features improve test R² by 0.028 and reduce RMSE by 0.383 °C.
# Raw coefficient sizes have different units; compare coefficient × feature
# standard deviation for an approximate effect over a typical feature change.
scaled_effects = model.coef_ * X_train.std(axis=0)
print("Effects for a one-standard-deviation feature increase (°C):")
for name, effect in zip(feature_cols, scaled_effects):
    print(f"{name:22s}: {effect:+.3f}")

# --- Task 6: Evaluate and Summarize ---
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, y_test, alpha=0.7)
limits = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted daily high (°C)")
plt.ylabel("Actual daily high (°C)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "predicted_vs_actual_high.png", dpi=150)
plt.close()
# Above the diagonal means the model underestimated the high; below means
# it overestimated. Compare error across thirds of the actual temperatures.
evaluation = pd.DataFrame({"actual": y_test, "predicted": y_pred})
evaluation["range"] = pd.qcut(evaluation["actual"], 3, labels=["low", "middle", "high"])
print("\nTask 6 — Evaluation")
print("Dataset days:", len(df), "Training days:", len(y_train), "Test days:", len(y_test))
for name, group in evaluation.groupby("range", observed=True):
    group_rmse = np.sqrt(mean_squared_error(group["actual"], group["predicted"]))
    print(f"{name} temperature range RMSE: {group_rmse:.3f} °C")
evaluation.to_csv(OUTPUTS / "test_predictions.csv", index=False)
# The low-temperature third has larger errors (RMSE 3.487 °C), compared with
# the middle (2.717 °C) and high (2.615 °C) thirds. Some cool days are noticeably
# overestimated in the plot; performance is somewhat worse at the low end.
#
# Summary: All 365 days have complete measurements. The split uses 292 days
# for training and 73 for testing; 92 days are summer and 273 are non-summer.
# Full-model RMSE is 2.973 °C, so the typical error scale is about 3 °C
# (a temperature difference of about 5.4 °F). Individual misses can be larger.
# Test R² is 0.897: the model explains about 89.7% of variation in test highs,
# relative to predicting the test mean; this does not mean 89.7% accuracy.
# The low temperature has the largest effect: a typical one-standard-deviation
# increase raises the predicted high by about 8.12 °C, far more than the other
# standardized feature effects. Its raw coefficient is positive, about 1.019.
# A surprising result is that the summer flag adds almost no conditional
# effect despite the warm summer climate; daily lows already encode seasonality.
# This random holdout evaluates same-year, same-day estimation. Because inputs
# include that day's measurements, these results do not test advance forecasts
# or generalization to a later year.
