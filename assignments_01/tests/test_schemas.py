import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def sample_data():
    return {"latitude": 40, "longitude": -74, "hourly": {
        "time": ["2026-04-08T00:00"], "temperature_2m": [10.0],
        "precipitation": [0.0]}}


def test_original_weather_file():
    # A plain relative path depends on the terminal's current directory;
    # this path is anchored to the test file.
    path = Path(__file__).parent.parent / "weather_raw.json"
    response = WeatherResponse.model_validate(json.loads(path.read_text()))
    assert len(response.hourly.time) == 168


@pytest.mark.parametrize("latitude", [-91, 91])
def test_bad_latitude(latitude):
    raw = sample_data()
    raw["latitude"] = latitude
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


@pytest.mark.parametrize("longitude", [-181, 181])
def test_bad_longitude(longitude):
    raw = sample_data()
    raw["longitude"] = longitude
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


@pytest.mark.parametrize("field", ["time", "temperature_2m", "precipitation"])
def test_unequal_lengths(field):
    raw = sample_data()
    raw["hourly"][field] = []
    with pytest.raises(ValidationError, match="matching lengths"):
        WeatherResponse.model_validate(raw)


def test_null_temperature():
    raw = sample_data()
    raw["hourly"]["temperature_2m"] = [None]
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)
