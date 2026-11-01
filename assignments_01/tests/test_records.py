from weatherkit import HourlyReading, WeatherResponse, to_readings


def sample_response():
    response = WeatherResponse.model_validate({
        "latitude": 40, "longitude": -74, "timezone": "GMT",
        "elevation": 254.0, "hourly": {
            "time": ["2026-04-09T01:00", "2026-04-08T02:00", "2026-04-10T03:00"],
            "temperature_2m": [12.5, -3.0, 21.0],
            "precipitation": [0.0, 1.5, 4.0]}})
    return response


def test_count_and_timestamp_order():
    response = sample_response()
    readings = to_readings(response)
    assert len(readings) == 3
    assert [r.timestamp for r in readings] == response.hourly.time
    assert readings[0].timestamp == response.hourly.time[0]
    assert readings[-1].timestamp == response.hourly.time[-1]


def test_index_pairing():
    response = sample_response()
    readings = to_readings(response)
    for i, reading in enumerate(readings):
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]


def test_dataclass_equality():
    assert HourlyReading("2026-04-08T00:00", 10, 1) == HourlyReading("2026-04-08T00:00", 10, 1)
    assert HourlyReading("2026-04-08T00:00", 10, 1) != HourlyReading("2026-04-08T00:00", 11, 1)
