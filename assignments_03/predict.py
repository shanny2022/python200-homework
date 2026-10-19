"""Load the saved weather pipeline and predict; run train_classifier.py first."""
from pathlib import Path
import json

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    model_path = BASE / "models/weather_classifier.pkl"
    metadata_path = BASE / "models/weather_classifier_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise SystemExit("Missing saved model or metadata. Run python assignments_03/train_classifier.py first.")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    features = metadata["feature_names"]
    print("City:", metadata["city"])
    print("Features:", features)
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
    days = pd.DataFrame([
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 2, "temperature_2m_min": -5, "precipitation_sum": 15, "wind_speed_10m_max": 45},
        {"temperature_2m_max": 26, "temperature_2m_min": 15, "precipitation_sum": 2.9, "wind_speed_10m_max": 29},
        {"temperature_2m_max": 22, "temperature_2m_min": 12, "precipitation_sum": 1, "wind_speed_10m_max": 15},
    ], columns=features)
    predictions = model.predict(days)
    good_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(days)[:, good_index]
    for name, (_, day), label, probability in zip(
        ["Clearly good", "Clearly bad", "Borderline", "Mild"], days.iterrows(), predictions, probabilities,
    ):
        print(f"{name}: {day.to_dict()} -> {'good' if label == 1 else 'skip'}; P(good)={probability:.2f}")


# Task 8: The borderline day (high=26, rain=2.9, wind=29) received
# P(good)=0.00 and was labeled skip, although it satisfies the exact rule.
# Its nearest training example was a skip day. With k=1, 0.00 looks confident
# but is only one neighbor's vote; I would treat this near-threshold case
# cautiously and show the actual conditions rather than promise confidence.
# If a future model returns 0.52, I would call it uncertain, show that clearly,
# and ask the runner to review conditions instead of strongly recommending it.
# Running this before training would fail because the model and metadata do not
# exist. The explicit file checks above explain which training command to run.
# Week 10 must replace these hypothetical rows with each day's real weather from
# the pipeline, and write structured predictions for downstream tasks.

if __name__ == "__main__":
    main()
