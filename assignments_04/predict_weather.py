"""Demonstrate batch predictions with the saved Week 3 model."""
from pathlib import Path
from weather_model import WeatherClassifier


def main() -> None:
    """Print predictions for four varied hypothetical weather days."""
    classifier = WeatherClassifier(Path(__file__).resolve().parent / "models/weather_classifier.pkl")
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 2, "temperature_2m_min": -5, "precipitation_sum": 15, "wind_speed_10m_max": 45},
        {"temperature_2m_max": 26, "temperature_2m_min": 15, "precipitation_sum": 2.9, "wind_speed_10m_max": 29},
        {"temperature_2m_max": 22, "temperature_2m_min": 12, "precipitation_sum": 1, "wind_speed_10m_max": 15},
    ]
    for name, day, prediction in zip(
        ["Clearly good", "Clearly bad", "Borderline", "Mild"], days, classifier.predict(days),
    ):
        print(f"{name}: {day} -> {prediction.label}; P(good)={prediction.probability:.2f}")


# Without the guard, importing this script to reuse a helper would also load
# the model and print all demo predictions as an unintended side effect.
if __name__ == "__main__":
    main()

# Reflection:
# 1. A caller might insert wind first and temperature last in a dictionary.
#    Using dictionary values directly would feed wind into the temperature
#    column and change the prediction. Explicit feature order prevents this.
# 2. A fixture creates known training examples and expected outputs without
#    depending on the real model or changing training data. Tests run in
#    isolation even if the production artifact is missing or retrained.
# 3. Loading once and returning structured batch results in input order lets a
#    database pipeline reuse one instance. Before production I would validate
#    numeric types, finite values, and plausible weather ranges. The current
#    k=1 probabilities are neighbor votes, not calibrated confidence.
