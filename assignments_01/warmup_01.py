"""Warmup examples inferred from the supplied guide; check against course questions."""

from dataclasses import dataclass, field, FrozenInstanceError

import pytest
from pydantic import BaseModel, ValidationError, model_validator

# --- Classes ---
# Q1
class Thermometer:
    """Store independent Celsius readings for a named location."""

    def __init__(self, location: str, readings: list[float] | None = None) -> None:
        self.location = location
        self.readings = list(readings) if readings is not None else []

    def add(self, temperature: float) -> None:
        """Append one Celsius observation."""
        self.readings.append(temperature)

    def average(self) -> float | None:
        """Return the mean temperature, or None when there are no readings."""
        return sum(self.readings) / len(self.readings) if self.readings else None

    def hottest(self) -> float | None:
        """Return the highest temperature, or None when there are no readings."""
        return max(self.readings) if self.readings else None

    # Q2
    def __repr__(self) -> str:
        return f"Thermometer(location={self.location!r}, readings={self.readings!r})"


# Q3
class TemperatureAlert:
    """Inspect an existing thermometer for readings above a Celsius threshold."""

    def __init__(self, threshold: float = 30.0) -> None:
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        """Return the thermometer readings strictly exceeding the threshold."""
        return [value for value in thermometer.readings if value > self.threshold]


# --- Dataclasses, Type Hints, and Docstrings ---
# Q1; Q2 adds frozen=True so equal stations can be members of a set.
@dataclass(frozen=True)
class Station:
    """Identify a weather station and its elevation in meters."""

    station_id: str
    name: str
    elevation: float


# Q3
@dataclass
class StationBatch:
    """A regional collection with a separate station list for each batch."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Append a station to the region's batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        return max(self.stations, key=lambda station: station.elevation) if self.stations else None


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


@pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (37, 98.6), (-40, -40)])
def test_conversion(celsius, expected):
    assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)


@pytest.mark.parametrize("values, expected", [([1, 2, 3], 2), ([-4, 2], -1), ([5], 5), ([0, 0, 0], 0)])
def test_mean(values, expected):
    assert mean(values) == pytest.approx(expected)


def test_empty_mean():
    with pytest.raises(ValueError, match="empty"):
        mean([])


def demonstrate() -> None:
    """Run examples and catch the intentional errors."""
    thermometer = Thermometer("Charlotte")
    print("Empty average and hottest:", thermometer.average(), thermometer.hottest())
    for value in [23, 30, 35]:
        thermometer.add(value)
    print(thermometer)
    print("Average and hottest:", thermometer.average(), thermometer.hottest())
    print("Independent default lists:", Thermometer("Other").readings == [])
    print("Temperature breaches:", TemperatureAlert().breaches(thermometer))
    station = Station("CLT", "Charlotte", 254)
    identical = Station("CLT", "Charlotte", 254)
    other = Station("AVL", "Asheville", 650)
    print("Station equality:", station == identical)
    print("Set of three stations length:", len({station, identical, other}))
    try:
        station.name = "Changed"
    except FrozenInstanceError as error:
        print("Frozen station:", error)
    try:
        @dataclass
        class IncorrectStationBatch:
            region: str
            stations: list[Station] = []
    except ValueError as error:
        print("Mutable default:", error)
    first, second = StationBatch("North Carolina"), StationBatch("Virginia")
    print("Empty highest:", first.highest())
    first.add(station)
    first.add(other)
    print("Highest station:", first.highest())
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



def test_thermometer_and_alert():
    empty = Thermometer("Empty")
    assert empty.average() is None
    assert empty.hottest() is None
    thermometer = Thermometer("Charlotte")
    for value in [20, 30, 40]:
        thermometer.add(value)
    assert thermometer.average() == pytest.approx(30)
    assert thermometer.hottest() == 40
    assert TemperatureAlert().breaches(thermometer) == [40]
    assert TemperatureAlert(25).breaches(thermometer) == [30, 40]
    assert empty.readings == []


def test_station_batch_and_frozen_equality():
    low = Station("CLT", "Charlotte", 254)
    same = Station("CLT", "Charlotte", 254)
    high = Station("AVL", "Asheville", 650)
    assert len({low, same, high}) == 2
    first, second = StationBatch("NC"), StationBatch("VA")
    assert first.highest() is None
    first.add(low)
    first.add(high)
    assert first.highest() == high
    assert second.stations == []
    with pytest.raises(FrozenInstanceError):
        low.elevation = 300

# Intentional conversion mutation; correct implementation restored:
# FFFF.......                                                              [100%]
# =================================== FAILURES ===================================
# ____________________________ test_conversion[0-32] _____________________________
#
# celsius = 0, expected = 32
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (37, 98.6), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert 0.0 == 32 ± 3.2e-05
# E
# E         comparison failed
# E         Obtained: 0.0
# E         Expected: 32 ± 3.2e-05
#
# warmup_01.py:105: AssertionError
# ___________________________ test_conversion[100-212] ___________________________
#
# celsius = 100, expected = 212
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (37, 98.6), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert 180.0 == 212 ± 2.1e-04
# E
# E         comparison failed
# E         Obtained: 180.0
# E         Expected: 212 ± 2.1e-04
#
# warmup_01.py:105: AssertionError
# ___________________________ test_conversion[37-98.6] ___________________________
#
# celsius = 37, expected = 98.6
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (37, 98.6), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert 66.6 == 98.6 ± 9.9e-05
# E
# E         comparison failed
# E         Obtained: 66.6
# E         Expected: 98.6 ± 9.9e-05
#
# warmup_01.py:105: AssertionError
# ___________________________ test_conversion[-40--40] ___________________________
#
# celsius = -40, expected = -40
#
#     @pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (37, 98.6), (-40, -40)])
#     def test_conversion(celsius, expected):
# >       assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)
# E       assert -72.0 == -40 ± 4.0e-05
# E
# E         comparison failed
# E         Obtained: -72.0
# E         Expected: -40 ± 4.0e-05
#
# warmup_01.py:105: AssertionError
# =========================== short test summary info ============================
# FAILED warmup_01.py::test_conversion[0-32] - assert 0.0 == 32 ± 3.2e-05
# FAILED warmup_01.py::test_conversion[100-212] - assert 180.0 == 212 ± 2.1e-04
# FAILED warmup_01.py::test_conversion[37-98.6] - assert 66.6 == 98.6 ± 9.9e-05
# FAILED warmup_01.py::test_conversion[-40--40] - assert -72.0 == -40 ± 4.0e-05
# 4 failed, 7 passed in 0.08s

# Actual mutable-default traceback from a separate demonstration:
# Traceback (most recent call last):
#   File "<string>", line 2, in <module>
#     @dataclass
#      ^^^^^^^^^
#   File "/Users/shuntoriareid/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/lib/python3.13/dataclasses.py", line 1353, in dataclass
#     return wrap(cls)
#   File "/Users/shuntoriareid/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/lib/python3.13/dataclasses.py", line 1343, in wrap
#     return _process_class(cls, init, repr, eq, order, unsafe_hash,
#                           frozen, match_args, kw_only, slots,
#                           weakref_slot)
#   File "/Users/shuntoriareid/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/lib/python3.13/dataclasses.py", line 1008, in _process_class
#     cls_fields.append(_get_field(cls, name, type, kw_only))
#                       ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/shuntoriareid/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/lib/python3.13/dataclasses.py", line 860, in _get_field
#     raise ValueError(f'mutable default {type(f.default)} for field '
#                      f'{f.name} is not allowed: use default_factory')
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

# Actual restored warmup test summary: 11 passed in 0.05s.
