# python200-homework

Week 1 work is in `assignments_01/`, on branch `assignments_01`.

Requires Python 3.10+ and Pydantic 2:

```sh
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
cd assignments_01
python warmup_01.py
pytest warmup_01.py -v
pytest -v
python report.py
```

The pipeline validates JSON with Pydantic, converts parallel hourly lists into
`HourlyReading` dataclasses, and produces sorted daily summaries. Days with
fewer than 24 observations are excluded and reported by `incomplete_days()`.

## Pending before course submission

- Check the inferred warmup implementations and model/dataclass field names
  against the full course assignment questions, which were not provided.
- Review and rewrite the reflection draft in your own words.
- Request mentor review once the mentor's GitHub username is known. Merge only
  after approval, and submit the open PR URL to the course.

Current verification: 7 warmup tests and all 17 package tests pass. The supplied
original JSON is included unchanged and validates with 168 hourly observations.
The report prints seven days of 24 observations each; April 8 has high 16.8 °C,
low 7.8 °C, and 0.0 mm precipitation, and April 9 has high 19.3 °C, low 3.7 °C,
and 0.0 mm precipitation, matching the guide. Intentional conversion and aggregation mutations
were caught by tests; exact failure output is recorded in source comments and
the working implementations were restored. `conftest.py` is empty.
