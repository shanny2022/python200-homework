"""Convert validated weather data into internal records."""

from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """One observation: temperature in Celsius, precipitation in millimeters."""

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Combine corresponding hourly values without changing their order.

    Args:
        response: A validated weather response with matching hourly list lengths.

    Returns:
        Hourly records in the same order as the response timestamps.
    """
    # Pydantic checks external JSON at entry; dataclasses store validated data.
    hourly = response.hourly
    return [HourlyReading(time, temperature, precipitation)
            for time, temperature, precipitation in
            zip(hourly.time, hourly.temperature_2m, hourly.precipitation)]
