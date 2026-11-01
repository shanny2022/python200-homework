import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def sample_data():
    return {"latitude": 40, "longitude": -74, "timezone": "GMT",
        "elevation": 254.0, "hourly": {
        "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
        "temperature_2m": [10.0, 11.0, 12.0],
        "precipitation": [0.0, 0.5, 1.0]}}


def test_original_weather_file():
    # A plain relative path depends on the terminal's current directory;
    # this path is anchored to the test file.
    path = Path(__file__).parent.parent / "weather_raw.json"
    response = WeatherResponse.model_validate(json.loads(path.read_text()))
    assert len(response.hourly.time) == 168


@pytest.mark.parametrize("latitude", [-91, 91, 200.0])
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
    raw["hourly"][field] = raw["hourly"][field][:-1]
    with pytest.raises(ValidationError, match="matching lengths"):
        WeatherResponse.model_validate(raw)


def test_null_temperature():
    raw = sample_data()
    raw["hourly"]["temperature_2m"][1] = None
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


@pytest.mark.parametrize("field", ["timezone", "elevation"])
def test_required_metadata(field):
    raw = sample_data()
    del raw[field]
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)
