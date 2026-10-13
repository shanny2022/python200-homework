"""Week 2 warmup: scikit-learn API and linear regression."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# --- scikit-learn API ---
# Q1
import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])
model = LinearRegression()
model.fit(temp_min, temp_max)
predictions = model.predict(np.array([[6], [18]]))
print("scikit-learn Q1 slope:", model.coef_[0])
print("scikit-learn Q1 intercept:", model.intercept_)
print("Predicted high for low of 6 degrees:", predictions[0])
print("Predicted high for low of 18 degrees:", predictions[1])

# Q2
x = np.array([10, 20, 30, 40, 50])
print("scikit-learn Q2 original shape:", x.shape)
x = x.reshape(-1, 1)
print("scikit-learn Q2 reshaped shape:", x.shape)
# X needs rows for observations and columns for features. Even one feature
# needs its own column, so the model can tell samples and features apart.

# --- Linear Regression ---
# Generate the specified dataset once and reuse it for Q1–Q5.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.figure(figsize=(7, 5))
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.colorbar(ticks=[0, 1], label="Rain status (0 = dry, 1 = rainy)")
plt.title("Daily High vs Low")
plt.xlabel("Daily low (degrees)")
plt.ylabel("Daily high (degrees)")
plt.tight_layout()
plt.savefig(OUTPUTS / "high_vs_low.png", dpi=150)
plt.close()
# Two overlapping upward bands are visible. Rainy days tend to have lower
# highs at the same low, suggesting the rain flag adds useful information.

# Q2
X = temp_min.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, temp_max, test_size=0.2, random_state=42
)
print("\nLinear Regression Q2 X_train shape:", X_train.shape)
print("Linear Regression Q2 X_test shape:", X_test.shape)
print("Linear Regression Q2 y_train shape:", y_train.shape)
print("Linear Regression Q2 y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression Q3 slope:", model.coef_[0])
print("Linear Regression Q3 intercept:", model.intercept_)
y_pred = model.predict(X_test)
baseline_r2 = model.score(X_test, y_test)
print("Linear Regression Q3 test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Linear Regression Q3 test R²:", baseline_r2)
# For each one-degree rise in the daily low, the predicted high rises by
# about 0.944 degrees in this fitted model.

# Q4
X_full = np.column_stack([temp_min, is_rainy])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, temp_max, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
full_r2 = model_full.score(X_test_full, y_test_full)
print("\nLinear Regression Q4 one-feature test R²:", baseline_r2)
print("Linear Regression Q4 two-feature test R²:", full_r2)
print("Test R² change from adding rain:", full_r2 - baseline_r2)
print("Does adding the rain flag help?", full_r2 > baseline_r2)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])
# At the same low temperature, switching from dry (0) to rainy (1) changes
# the predicted high by the rain coefficient. A negative value means a
# lower predicted high on rainy days, consistent with the data generator.
# Here rain lowers the predicted high by about 2.919 degrees, and adding it
# improves test R² from 0.894 to 0.945 on the same 24 held-out observations.

# Q5
y_pred_full = model_full.predict(X_test_full)
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_full, y_test_full, alpha=0.7)
limits = [min(y_pred_full.min(), y_test_full.min()),
          max(y_pred_full.max(), y_test_full.max())]
plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted daily high (degrees)")
plt.ylabel("Actual daily high (degrees)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "predicted_vs_actual_warmup.png", dpi=150)
plt.close()
# Above the diagonal: the actual high is higher than predicted, so the model
# underestimated. Below: it predicted too high, so the model overestimated.
print("\nWarmup plots saved to:", OUTPUTS)
