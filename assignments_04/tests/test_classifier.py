"""Tests train their own tiny pipeline rather than requiring the real model."""
from unittest.mock import patch
import joblib
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from weather_model import Prediction, WeatherClassifier

FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
GOOD = dict(zip(FEATURES, [20.0, 10.0, 0.0, 10.0]))
BAD = dict(zip(FEATURES, [0.0, -8.0, 20.0, 50.0]))


@pytest.fixture
def model_path(tmp_path):
    model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))])
    model.fit(pd.DataFrame([BAD, GOOD], columns=FEATURES), [0, 1])
    path = tmp_path / "tiny_weather.pkl"
    joblib.dump(model, path)
    return path


@pytest.fixture
def classifier(model_path):
    return WeatherClassifier(model_path)


@pytest.mark.parametrize("day, expected", [(GOOD, "good"), (BAD, "skip")])
def test_clear_day_labels(classifier, day, expected):
    assert classifier.predict([day])[0].label == expected


def test_probability_range(classifier):
    for result in classifier.predict([GOOD, BAD]):
        assert isinstance(result.probability, float)
        assert 0 <= result.probability <= 1


def test_three_predictions_in_order(classifier):
    results = classifier.predict([GOOD, BAD, GOOD])
    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    assert [result.label for result in results] == ["good", "skip", "good"]
    assert [result.probability for result in results] == pytest.approx([1, 0, 1])


def test_empty_input(classifier):
    assert classifier.predict([]) == []


def test_missing_features(classifier):
    with pytest.raises(ValueError, match="precipitation_sum, wind_speed_10m_max"):
        classifier.predict([GOOD, {"temperature_2m_max": 20, "temperature_2m_min": 10}])


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Weather model file not found:.*missing.pkl"):
        WeatherClassifier(tmp_path / "missing.pkl")


def test_feature_order_and_extra_keys(classifier):
    reordered = dict(reversed(list(GOOD.items())))
    reordered["date"] = "2026-10-27"
    assert classifier.predict([reordered]) == classifier.predict([GOOD])


def test_loads_model_only_once(model_path):
    with patch("weather_model.classifier.joblib.load", wraps=joblib.load) as load:
        classifier = WeatherClassifier(model_path)
        classifier.predict([GOOD])
        classifier.predict([BAD])
        load.assert_called_once_with(model_path)


# Mutation check: always returning "skip" was caught by
# test_clear_day_labels for GOOD. The correct implementation was restored.
