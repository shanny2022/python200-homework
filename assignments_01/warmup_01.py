"""Week 1 demonstrations and explicitly runnable warmup tests."""

from dataclasses import dataclass, field, FrozenInstanceError

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

# --- Classes ---
# Q1
class Thermometer:
    """Store independent Celsius readings for a named location."""

    def __init__(self, location: str, readings: list[float] | None = None) -> None:
        self.location = location
        self.readings = list(readings) if readings is not None else []

    def add(self, reading: float) -> None:
        """Append one Celsius reading."""
        self.readings.append(reading)

    def average(self) -> float | None:
        """Return the mean, or None if no readings are available."""
        # Without the empty check, division by zero would raise ZeroDivisionError.
        return sum(self.readings) / len(self.readings) if self.readings else None

    def hottest(self) -> float | None:
        """Return the highest temperature, or None if no readings exist."""
        return max(self.readings) if self.readings else None

    # Q2
    def __repr__(self) -> str:
        """Show the location, observation count, and mean for debugging."""
        return (f"Thermometer(location={self.location!r}, "
                f"n_readings={len(self.readings)}, average={self.average()!r})")

# Without __repr__, Python shows the object type and memory address, which do
# not reveal its readings or location when debugging a collection of objects.

# Q3
class TemperatureAlert:
    """Check existing thermometers against a reusable Celsius threshold."""

    def __init__(self, threshold: float = 30.0) -> None:
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        """Return readings strictly above the configured threshold."""
        return [reading for reading in thermometer.readings if reading > self.threshold]

# The alert stores its threshold as configuration. One alert can check twenty
# thermometers consistently without passing the same threshold twenty times.

# --- Dataclasses, Type Hints, and Docstrings ---
# Q1; Q2 adds frozen=True.
@dataclass(frozen=True)
class Station:
    """A weather station's identity, coordinates, and elevation in meters."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

# Dataclass equality compares field values. The original hand-written class
# would compare object identity, so two separately constructed stations differ.
# frozen=True also generates a hash for these hashable fields, letting a set
# deduplicate equal stations as well as preventing field reassignment.

# Q3
@dataclass
class StationBatch:
    """A regional collection with an independent list of weather stations."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Append a station to this batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station of greatest elevation, or None for an empty batch."""
        return max(self.stations, key=lambda station: station.elevation) if self.stations else None

# Python refuses a list default because every instance would share that list.
# default_factory creates a fresh list whenever a batch is constructed.

# --- Pydantic ---
# Q1
class Reading(BaseModel):
    """Validate a station observation and reject a known sensor failure pattern."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4
    @model_validator(mode="after")
    def reject_failed_sensor(self) -> "Reading":
        """Reject zero humidity combined with temperature below -40 Celsius."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Failed sensor: zero humidity with temperature below -40")
        return self

# Field constraints validate each value separately; this rule depends on the
# combination of two otherwise valid values, so it requires a model validator.
# Q2: Numeric strings such as "21.5" can be converted to floats. "very humid"
# has no numeric interpretation, so Pydantic rejects it rather than guessing.
# Q3: The multi-error demo reports three errors. Reporting them together lets
# the caller repair every invalid field in one pass rather than retrying each.

# --- pytest ---
# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# Floating-point arithmetic can represent 98.6 with a tiny rounding difference;
# approx accepts that difference rather than demanding identical binary values.

# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean; reject an empty collection."""
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# raises(ValueError) alone would accept an unrelated ValueError; match also
# checks that the message describes the expected empty-input problem.

# Q3
@pytest.mark.parametrize("values, expected", [
    ([1, 2, 3], 2), ([-4, 2], -1), ([5], 5), ([0, 0, 0], 0)])
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)

# Parametrization shares the assertion while giving each case its own result;
# four near-identical functions would duplicate code that must be maintained.

# Q4: The intentional failure below reports actual and expected values (for
# example, 0.0 versus 32), showing the missing offset rather than only saying
# "assertion failed". The correct conversion is restored after the experiment.


def demonstrate() -> None:
    """Print every warmup example, catching the intentional validation errors."""
    # Classes Q1–Q3
    thermometer = Thermometer("Charlotte")
    print("Empty average and hottest:", thermometer.average(), thermometer.hottest())
    for reading in [20, 25, 30, 35]:
        thermometer.add(reading)
    print("Average and hottest:", thermometer.average(), thermometer.hottest())
    print(thermometer)
    print([thermometer, Thermometer("Asheville", [12, 15])])
    for alert in [TemperatureAlert(), TemperatureAlert(24)]:
        print("Threshold and breaches:", alert.threshold, alert.breaches(thermometer))

    # Dataclass Q1–Q3
    station = Station("CLT", "Charlotte", 35.2, -80.8, 254)
    same = Station("CLT", "Charlotte", 35.2, -80.8, 254)
    high = Station("AVL", "Asheville", 35.6, -82.6, 650)
    print("Station equality:", station == same)
    try:
        station.name = "Changed"
    except FrozenInstanceError as error:
        print("Frozen station:", error)
    print("Set of three stations length:", len({station, same, high}))
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
    first.add(high)
    print("Highest station:", first.highest())
    print("Independent batches:", first, second)

    # Pydantic Q1
    valid = {"station_id": "CLT", "timestamp": "2026-04-08T12:00",
             "temperature_c": 21.5, "humidity": 40.0}
    print("Valid reading:", Reading(**valid))
    # Q2: three separate constructions, each with its own caught error.
    missing = dict(valid)
    del missing["timestamp"]
    for invalid in [missing, {**valid, "temperature_c": 150.0},
                    {**valid, "humidity": "very humid"}]:
        try:
            Reading.model_validate(invalid)
        except ValidationError as error:
            print("Expected validation failure:", error)
    coerced = Reading(**{**valid, "temperature_c": "21.5", "humidity": 40})
    print("Coerced reading:", coerced)
    print("Coerced field types:", type(coerced.temperature_c), type(coerced.humidity))
    # Q3
    try:
        Reading(station_id="X", temperature_c="not numeric", humidity=40)
    except ValidationError as error:
        print("Multiple errors:", len(error.errors()))
        for detail in error.errors():
            print(detail["loc"], detail["msg"])
    # Q4
    print("Still valid:", Reading(**valid))
    try:
        Reading(**{**valid, "temperature_c": -45, "humidity": 0.0})
    except ValidationError as error:
        print("Sensor failure:", error)
    print("Conversion and mean:", celsius_to_fahrenheit(37), mean([1, 2, 3]))


if __name__ == "__main__":
    demonstrate()

# Q4: Actual intentional failure output (correct function restored):
# F.....                                                                   [100%]
# =================================== FAILURES ===================================
# __________________________ test_celsius_to_fahrenheit __________________________
#
#     def test_celsius_to_fahrenheit():
# >       assert celsius_to_fahrenheit(0) == 32
# E       assert 0.0 == 32
# E        +  where 0.0 = celsius_to_fahrenheit(0)
#
# warmup_01.py:122: AssertionError
# =========================== short test summary info ============================
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 0.0 == 32
# 1 failed, 5 passed in 0.06s

# Dataclass Q3: Actual mutable-default error traceback:
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

# pytest Q3: Actual restored summary: 6 passed in 0.04s.
