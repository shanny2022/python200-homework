"""Warmup examples inferred from the supplied guide; check against course questions."""

from dataclasses import dataclass, field, FrozenInstanceError

import pytest
from pydantic import BaseModel, ValidationError, model_validator

# --- Classes ---
# Q1
class Thermometer:
    """Keep an independent collection of Celsius readings."""

    def __init__(self, readings: list[float] | None = None) -> None:
        self.readings = list(readings) if readings is not None else []

    def add_reading(self, temperature: float) -> None:
        self.readings.append(temperature)

    def __repr__(self) -> str:
        return f"Thermometer(readings={self.readings!r})"


class TemperatureAlert(Thermometer):
    """Identify readings exceeding a configured Celsius threshold."""

    def __init__(self, threshold: float, readings: list[float] | None = None) -> None:
        super().__init__(readings)
        self.threshold = threshold

    def alerts(self) -> list[float]:
        return [value for value in self.readings if value > self.threshold]


# --- Dataclasses, Type Hints, and Docstrings ---
# Q1
@dataclass(frozen=True)
class Station:
    """An immutable weather station name and geographic coordinates."""

    name: str
    latitude: float
    longitude: float


@dataclass
class StationBatch:
    """A batch with a separate station list for each instance."""

    stations: list[Station] = field(default_factory=list)


# --- Pydantic ---
# Q1
class Reading(BaseModel):
    """A Celsius reading whose sensor must be present when temperature is given."""

    temperature: float
    sensor: str | None = None

    @model_validator(mode="after")
    def require_sensor(self) -> "Reading":
        if not self.sensor or not self.sensor.strip():
            raise ValueError("A temperature reading requires a sensor")
        return self


# --- pytest ---
# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32


def mean(values: list[float]) -> float:
    """Return the arithmetic mean; reject empty input."""
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list")
    return sum(values) / len(values)


@pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (-40, -40)])
def test_conversion(celsius, expected):
    assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)


@pytest.mark.parametrize("values, expected", [([1, 2, 3], 2), ([-4, 2], -1), ([5], 5)])
def test_mean(values, expected):
    assert mean(values) == pytest.approx(expected)


def test_empty_mean():
    with pytest.raises(ValueError, match="empty"):
        mean([])


def demonstrate() -> None:
    """Run examples and catch the intentional errors."""
    thermometer = Thermometer()
    thermometer.add_reading(23)
    print(thermometer)
    print("Independent default lists:", Thermometer().readings == [])
    print("Temperature alerts:", TemperatureAlert(25, [20, 30]).alerts())
    station = Station("Home", 40, -74)
    print("Station equality:", station == Station("Home", 40, -74))
    try:
        station.name = "Changed"
    except FrozenInstanceError as error:
        print("Frozen station:", error)
    try:
        @dataclass
        class IncorrectStationBatch:
            stations: list[Station] = []
    except ValueError as error:
        print("Mutable default:", error)
    first, second = StationBatch(), StationBatch()
    first.stations.append(station)
    print("Independent batches:", first, second)
    print("Validated reading:", Reading(temperature="12.5", sensor="outdoor"))
    for invalid in ({"temperature": "bad", "sensor": "outdoor"},
                    {"temperature": 12.5}):
        try:
            Reading.model_validate(invalid)
        except ValidationError as error:
            print("Expected validation failure:", error)
    print("Conversions and mean:", celsius_to_fahrenheit(0), mean([1, 2, 3]))


if __name__ == "__main__":
    demonstrate()

# Intentional mutation verification (correct implementation restored):
# FFF....                                                                  [100%]
# =================================== FAILURES ===================================
# ____________________________ test_conversion[0-32] _____________________________
#
# celsius = 0, expected = 32
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert 0.0 == 32 ± 3.2e-05
# E
# E         comparison failed
# E         Obtained: 0.0
# E         Expected: 32 ± 3.2e-05
#
# warmup_01.py:83: AssertionError
# ___________________________ test_conversion[100-212] ___________________________
#
# celsius = 100, expected = 212
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert 180.0 == 212 ± 2.1e-04
# E
# E         comparison failed
# E         Obtained: 180.0
# E         Expected: 212 ± 2.1e-04
#
# warmup_01.py:83: AssertionError
# ___________________________ test_conversion[-40--40] ___________________________
#
# celsius = -40, expected = -40
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert -72.0 == -40 ± 4.0e-05
# E
# E         comparison failed
# E         Obtained: -72.0
# E         Expected: -40 ± 4.0e-05
#
# warmup_01.py:83: AssertionError
# =========================== short test summary info ============================
# FAILED warmup_01.py::test_conversion[0-32] - assert 0.0 == 32 ± 3.2e-05
# FAILED warmup_01.py::test_conversion[100-212] - assert 180.0 == 212 ± 2.1e-04
# FAILED warmup_01.py::test_conversion[-40--40] - assert -72.0 == -40 ± 4.0e-05
# 3 failed, 4 passed in 0.04s

# Actual restored test summary: 7 passed in 0.04s.
