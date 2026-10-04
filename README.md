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

- Add the original course `weather_raw.json` to `assignments_01/`. It was not
  included with the request; synthetic data has not been substituted.
- Check the inferred warmup implementations and model/dataclass field names
  against the full course assignment questions, which were not provided.
- Review and rewrite the reflection draft in your own words.
- Run the full checks above with the original JSON. Confirm 168 hourly readings,
  seven daily rows, and the first two expected daily results from the guide.
- Request mentor review once the mentor's GitHub username is known. Merge only
  after approval, and submit the open PR URL to the course.

Current verification: 7 warmup tests and 16 package tests pass. The original-data
test requires the missing JSON. Intentional conversion and aggregation mutations
were caught by tests; exact failure output is recorded in source comments and
the working implementations were restored. `conftest.py` is empty.
