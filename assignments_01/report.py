"""Print a daily weather report from the course's original JSON data."""

import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def main() -> None:
    """Load, validate, convert, summarize, and print weather observations."""
    data_path = Path(__file__).resolve().parent / "weather_raw.json"
    response = WeatherResponse.model_validate(json.loads(data_path.read_text()))
    readings = to_readings(response)
    aggregator = DailyAggregator()
    print(f"{'Date':<12} {'High °C':>8} {'Low °C':>8} {'Rain mm':>9} {'Range °C':>9} {'Hours':>6}")
    for day in aggregator.summarize(readings):
        print(f"{day.date:<12} {day.temp_max:8.1f} {day.temp_min:8.1f} "
              f"{day.precipitation_total:9.1f} {day.temp_range():9.1f} {day.hours:6d}")
    incomplete = aggregator.incomplete_days(readings)
    if incomplete:
        print("Warning: incomplete dates dropped: " + ", ".join(incomplete))


# Without this guard, importing this module would execute the report.
if __name__ == "__main__":
    main()

# Reflection draft: review and rewrite in your own words before submission.
# 1. Rejecting missing temperatures prevents a summary from silently treating an
#    unknown observation as real data. Allowing gaps helps process imperfect
#    sensor feeds, but list[float | None] also requires downstream filtering,
#    explicit counts of valid temperatures, and handling all-missing days.
# 2. At noon, today's group has fewer than 24 observations, so the default
#    threshold excludes it. incomplete_days() exposes the excluded date so the
#    user can distinguish a partial day from a day with no observations.
# 3. A Week 10 pipeline can import WeatherResponse, to_readings, and
#    DailyAggregator from weatherkit. Package imports do not print anything,
#    and report.py's main guard also prevents a report on import.
