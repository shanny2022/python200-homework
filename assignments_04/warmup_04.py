"""Week 4 warmups: cached JSON, dataclasses, and error-handling tests."""
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


# Question 1
class ModelInfo:
    """Read model information once for repeated queries."""

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        """Return the number of stored features."""
        return len(self.data["features"])


# Loading in __init__ avoids repeated disk reads and JSON parsing on each call.

# Question 2
@dataclass
class DayForecast:
    """A day's temperature and rain forecast."""

    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Return a readable description of this forecast."""
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain}"


# @dataclass generates __init__ (and __repr__ and __eq__) for us.

# Question 3
def parse_temperature(text: str) -> float:
    """Parse numeric temperature text; raise ValueError for invalid text."""
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid temperature: {text!r}") from error


def test_parse_temperature_valid() -> None:
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, an unrelated ValueError with an unhelpful message could pass.
def main() -> None:
    """Print demonstrations without creating files at import time."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "model_info.json"
        path.write_text(json.dumps({"features": ["high", "low", "rain", "wind"]}))
        print("Feature count:", ModelInfo(path).feature_count())
    for forecast in [
        DayForecast("2026-10-26", 24.1, True),
        DayForecast("2026-10-27", 18.0, False),
    ]:
        print(forecast.summary())
    print("Parsed temperature:", parse_temperature("24.1"))


if __name__ == "__main__":
    main()

