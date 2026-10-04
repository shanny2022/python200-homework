from weatherkit import HourlyReading, WeatherResponse, to_readings


def test_count_order_and_index_pairing():
    response = WeatherResponse.model_validate({
        "latitude": 40, "longitude": -74, "hourly": {
            "time": ["2026-04-09T01:00", "2026-04-08T02:00", "2026-04-10T03:00"],
            "temperature_2m": [12.5, -3.0, 21.0],
            "precipitation": [0.0, 1.5, 4.0]}})
    readings = to_readings(response)
    assert len(readings) == 3
    assert [r.timestamp for r in readings] == response.hourly.time
    for i, reading in enumerate(readings):
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]


def test_dataclass_equality():
    assert HourlyReading("2026-04-08T00:00", 10, 1) == HourlyReading("2026-04-08T00:00", 10, 1)
    assert HourlyReading("2026-04-08T00:00", 10, 1) != HourlyReading("2026-04-08T00:00", 11, 1)
