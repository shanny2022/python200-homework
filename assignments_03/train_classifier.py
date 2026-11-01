"""Fetch NYC weather, compare classifiers, and save a complete fitted pipeline."""
from pathlib import Path
import json
import platform

import joblib
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
LABEL_RULE = "7 <= high <= 26 Celsius; low >= 0 Celsius; precipitation < 3 mm; wind < 30 km/h"


def evaluate(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    print(f"\n{name}: {scores}")
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["skip", "good"], zero_division=0))
    return scores


def main():
    (BASE / "models").mkdir(exist_ok=True)
    (BASE / "outputs").mkdir(exist_ok=True)
    # Task 1: New York City, 2023; explicit units match the label thresholds.
    params = {
        "latitude": 40.7128, "longitude": -74.0060,
        "start_date": "2023-01-01", "end_date": "2023-12-31",
        "daily": FEATURES, "timezone": "America/New_York",
        "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df.pop("time"))
    if len(df) != 365 or df[FEATURES].isna().any().any():
        raise ValueError("Expected 365 complete daily observations for 2023.")
    print("Weather shape:", df.shape)
    print(df.head())

    # Task 2: Use the assignment's unchanged thresholds.
    df["good_for_running"] = (
        df.temperature_2m_max.between(7, 26)
        & df.temperature_2m_min.ge(0)
        & df.precipitation_sum.lt(3)
        & df.wind_speed_10m_max.lt(30)
    ).astype(int)
    print("Class balance (0=skip, 1=good):", df.good_for_running.value_counts().sort_index().to_dict())
    print(f"Fraction good: {df.good_for_running.mean():.1%}")
    df.to_csv(BASE / "outputs/nyc_weather_2023.csv", index=False)
    # 137/365 days (37.5%) are good: plausible given NYC winters, summer
    # heat, and rain. The 62.5% skip share is moderately imbalanced; inspect
    # good-class precision/recall/F1 rather than trusting accuracy alone.

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df.good_for_running, test_size=0.2, random_state=42, stratify=df.good_for_running,
    )
    # Task 3: The scaler is fitted separately inside each CV training fold.
    cv_scores = {}
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        candidate = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        cv_scores[k] = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy").mean()
        print(f"k={k:2d}, mean CV accuracy={cv_scores[k]:.4f}")
    best_k = max(cv_scores, key=cv_scores.get)
    print("Selected k:", best_k)
    knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k))])
    knn.fit(X_train, y_train)
    knn_scores = evaluate("KNN", knn, X_test, y_test)

    # Task 4
    logreg = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    logreg_scores = evaluate("Logistic regression", logreg, X_test, y_test)
    for feature, coefficient in zip(FEATURES, logreg.named_steps["logreg"].coef_[0]):
        print(f"{feature}: {coefficient:+.4f}")
    # Rain (-2.8761) pushes most strongly toward skip per standard deviation;
    # high temperature (-0.4328) and wind (-0.3425) also push toward skip.
    # Warmer lows have a positive coefficient. These are associations, not
    # separate causal effects: high and low temperatures are correlated.
    # Task 5: KNN accuracy was 0.9178 vs logistic regression's 0.6164;
    # good-class precision was 0.8889 vs 0.4783, and F1 was 0.8889 vs 0.4400.
    # For a running app I prioritize precision: a false positive recommends
    # poor weather, whereas a false negative misses a suitable running day.
    # Deploy KNN (k=1 selected by CV). Its substantially better scores justify
    # storing training examples and doing distance searches at prediction time.
    # Logistic regression has a smaller artifact, faster inference as the
    # dataset grows, and interpretable coefficients; with just 292 training
    # days KNN's storage and inference costs are small enough here.
    # Both models provide predict_proba, but k=1 produces only 0 or 1: these
    # are neighbor votes, not reliable confidence estimates. A future version
    # needs validation/calibration if confidence will guide recommendations.
    # The good-weather label has both lower and upper temperature bounds;
    # a single linear decision boundary struggles to represent this rule.
    # Scores measure this deterministic weather rule on a random holdout,
    # not future-season performance or the safety of an actual run.

    # Tasks 5–6: Deploy KNN to favor precision when recommending an outdoor run.
    chosen = knn
    metadata = {
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "feature_names": FEATURES, "label_rule": LABEL_RULE, "city": "New York City",
        "year": 2023, "model_type": "KNeighborsClassifier", "selected_k": best_k,
        "test_accuracy": knn_scores["accuracy"], "test_f1": knn_scores["f1"],
        "test_precision": knn_scores["precision"], "test_recall": knn_scores["recall"],
        "comparison": {"knn": knn_scores, "logistic_regression": logreg_scores},
        "cv_accuracy_by_k": cv_scores, "test_size": 0.2, "random_state": 42,
        "source_url": response.url,
    }
    joblib.dump(chosen, BASE / "models/weather_classifier.pkl")
    (BASE / "models/weather_classifier_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved models/weather_classifier.pkl and models/weather_classifier_metadata.json")


if __name__ == "__main__":
    main()
