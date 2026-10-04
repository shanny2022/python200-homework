import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def readings():
    # Later date first to expose accidental dependence on input date order.
    return [HourlyReading("2026-04-09T00:00", 20, 3.0),
            HourlyReading("2026-04-08T00:00", 10, 0.5),
            HourlyReading("2026-04-08T01:00", -2, 1.5),
            HourlyReading("2026-04-09T01:00", 8, 4.0),
            HourlyReading("2026-04-08T02:00", 16, 2.0)]


def test_grouping_extremes_precipitation_and_range(readings):
    summaries = DailyAggregator(min_hours=2).summarize(readings)
    assert [day.date for day in summaries] == ["2026-04-08", "2026-04-09"]
    first, second = summaries
    assert (first.temp_max, first.temp_min, first.hours) == (16, -2, 3)
    assert first.precipitation_total == pytest.approx(4.0)
    assert first.temp_range() == 18
    assert (second.temp_max, second.temp_min, second.hours) == (20, 8, 2)
    assert second.precipitation_total == pytest.approx(7.0)
    assert second.temp_range() == 12


@pytest.mark.parametrize("min_hours, kept, dropped", [
    (1, ["2026-04-08", "2026-04-09"], []),
    (2, ["2026-04-08", "2026-04-09"], []),
    (3, ["2026-04-08"], ["2026-04-09"]),
    (24, [], ["2026-04-08", "2026-04-09"]),
])
def test_incomplete_threshold(readings, min_hours, kept, dropped):
    aggregator = DailyAggregator(min_hours)
    assert [day.date for day in aggregator.summarize(readings)] == kept
    assert aggregator.incomplete_days(readings) == dropped


def test_empty_input():
    aggregator = DailyAggregator()
    assert aggregator.summarize([]) == []
    assert aggregator.incomplete_days([]) == []
