"""Group hourly readings into daily summaries."""

from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Daily Celsius temperature extremes and total millimeter precipitation."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the highest and lowest temperature."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Summarize dates meeting the minimum number of hourly observations."""

    def __init__(self, min_hours: int = 24) -> None:
        self.min_hours = min_hours

    def _groups(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        groups: dict[str, list[HourlyReading]] = {}
        for reading in readings:
            groups.setdefault(reading.timestamp[:10], []).append(reading)
        return groups

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Return qualifying daily summaries in ascending date order."""
        summaries = []
        for date, group in sorted(self._groups(readings).items()):
            if len(group) < self.min_hours:
                continue
            temperatures = [reading.temperature_c for reading in group]
            summaries.append(DailySummary(
                date, max(temperatures), min(temperatures),
                sum(reading.precipitation_mm for reading in group), len(group)))
        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return sorted dates excluded by the observation threshold."""
        return sorted(date for date, group in self._groups(readings).items()
                      if len(group) < self.min_hours)

# Intentional mutation verification (correct implementation restored):
# F.....                                                                   [100%]
# =================================== FAILURES ===================================
# ________________ test_grouping_extremes_precipitation_and_range ________________
#
# readings = [HourlyReading(timestamp='2026-04-09T00:00', temperature_c=20, precipitation_mm=3.0), HourlyReading(timestamp='2026-04...rature_c=8, precipitation_mm=4.0), HourlyReading(timestamp='2026-04-08T02:00', temperature_c=16, precipitation_mm=2.0)]
#
#     def test_grouping_extremes_precipitation_and_range(readings):
#         summaries = DailyAggregator(min_hours=2).summarize(readings)
#         assert [day.date for day in summaries] == ["2026-04-08", "2026-04-09"]
#         first, second = summaries
# >       assert (first.temp_max, first.temp_min, first.hours_observed) == (16, -2, 3)
# E       assert (-2, -2, 3) == (16, -2, 3)
# E
# E         At index 0 diff: -2 != 16
# E         Use -v to get more diff
#
# tests/test_summarize.py:20: AssertionError
# =========================== short test summary info ============================
# FAILED tests/test_summarize.py::test_grouping_extremes_precipitation_and_range
# 1 failed, 5 passed in 0.06s
