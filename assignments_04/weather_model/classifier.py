"""Reusable interface for the saved binary weather classifier."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

FEATURES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
)


@dataclass
class Prediction:
    """Running-weather label and probability that conditions are good."""

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved pipeline once and reuse it for batches of weather days."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the pipeline.

        Raises:
            FileNotFoundError: If model_path does not identify an existing file.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Weather model file not found: {path}")
        self.model = joblib.load(path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Predict each day in input order, using Celsius, mm, and km/h.

        Extra dictionary keys are ignored.

        Raises:
            ValueError: If a day is missing any required weather features.
        """
        if not days:
            return []
        for index, day in enumerate(days):
            missing = [feature for feature in FEATURES if feature not in day]
            if missing:
                raise ValueError(f"Day {index} missing required features: {', '.join(missing)}")
        frame = pd.DataFrame(days, columns=list(FEATURES))
        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]
        return [
            Prediction("good" if label == 1 else "skip", float(probability))
            for label, probability in zip(labels, probabilities)
        ]
