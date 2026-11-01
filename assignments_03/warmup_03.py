"""Week 3 warmup reference; review and rework these exercises independently."""
from pathlib import Path
import os

# Make the assignment's exact relative-path setup work from any directory.
os.chdir(Path(__file__).resolve().parent)
Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
# Q1: Fit only on training data to prevent test-set information leaking into training.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled training column means:", X_train_scaled.mean(axis=0))

# Q2: Four columns represent the four distinct seasons; no category is dropped.
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])
print("Encoded season shape:", season_encoded.shape)
print("Category names:", encoder.get_feature_names_out())

# Q3: The pipeline automatically fits preprocessing on training data and reuses
# those same transformations for prediction, keeping inputs consistent.
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])
season_pipeline = Pipeline([("preprocess", preprocessor), ("logreg", LogisticRegression(max_iter=1000))])
X_season = df[numeric + ["season"]]
season_pipeline.fit(X_season.loc[X_train.index], y_train)
print("Numeric + season logistic accuracy:", season_pipeline.score(X_season.loc[X_test.index], y_test))

# --- KNN ---
# Q1
knn_pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
knn_pipeline.fit(X_train, y_train)
knn_predictions = knn_pipeline.predict(X_test)
print("Scaled KNN accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions, labels=[0, 1], target_names=["skip", "good"], zero_division=0))

# Q2
cv_scores = {}
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    candidate = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
    cv_scores[k] = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy").mean()
    print(f"k={k:2d}, mean CV accuracy={cv_scores[k]:.4f}")
print("Chosen k:", max(cv_scores, key=cv_scores.get))
# Choose k=11: it had the highest training CV accuracy (0.9417),
# without using test performance to select k.

# Q3
raw_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("Unscaled KNN accuracy:", raw_knn.score(X_test, y_test))
print("Scaled KNN accuracy:", knn_pipeline.score(X_test, y_test))
# Scaling helped: accuracy rose from 0.9250 to 0.9583.
# Scaling prevents large numeric ranges or units from dominating Euclidean distance.

# --- Logistic Regression ---
# Q1
pipeline = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print("Logistic accuracy:", accuracy_score(y_test, predictions))
print("Logistic precision:", precision_score(y_test, predictions, zero_division=0))
print("Logistic recall:", recall_score(y_test, predictions, zero_division=0))
print("Logistic F1:", f1_score(y_test, predictions, zero_division=0))

# Q2
for feature, coefficient in zip(numeric, pipeline.named_steps["logreg"].coef_[0]):
    print(f"{feature}: {coefficient:+.4f}")
# Wind (-2.4438) and precipitation (-1.8891) push most strongly toward
# skip per standard deviation, matching the discomfort of windy, rainy runs.

# Q3: A probability near 0.5 means the model has little preference between classes;
# it does not mean the probability estimate is guaranteed to be well calibrated.
good_index = list(pipeline.classes_).index(1)
print("First five P(good):", np.round(pipeline.predict_proba(X_test.iloc[:5])[:, good_index], 2))

# --- Evaluation ---
# Q1: Rows are actual classes and columns are predicted classes, in order skip, good.
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
print("Confusion matrix:\n", cm)
print("False positives:", cm[0, 1], "False negatives:", cm[1, 0])
ConfusionMatrixDisplay(cm, display_labels=["skip", "good"]).plot()
plt.title("Logistic regression: running weather")
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png", dpi=150)
plt.close()
# The model made 9 false positives and 7 false negatives on this split.
# False positives matter more: recommending a bad day can expose runners to
# unpleasant conditions; false negatives mean missing an enjoyable run.

# --- joblib ---
# Q1: Saving just the classifier would apply standardized-feature coefficients
# to raw units at inference, leading to incorrect scores and predictions.
joblib.dump(pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")
