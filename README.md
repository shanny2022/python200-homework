# python200-homework

Week 2's warmup exercises are in `assignments_02/warmup_02.py`, and its
six-task mini-project is in `assignments_02/project_02.py`.
It uses New York City's 2023 daily weather from Open-Meteo to compare
a low-temperature baseline with a four-feature linear regression model.

```sh
uv pip install -r requirements.txt
.venv/bin/python assignments_02/warmup_02.py
.venv/bin/python assignments_02/project_02.py
```

Plots, the reusable weather CSV, and test predictions are saved in
`assignments_02/outputs/` regardless of the working directory. Running the
script requires internet access. `run_results.txt` records the verified run.
`warmup_results.txt` records the verified warmup run.

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

## Week 1 follow-up

- Review and rewrite the reflection draft in your own words.
- Request mentor review once the mentor's GitHub username is known. Merge only
  after approval, and submit the open PR URL to the course.

Current verification: 6 warmup cases and all 21 package tests pass. The supplied
original JSON is included unchanged and validates with 168 hourly observations.
The report prints seven days of 24 observations each; April 8 has high 16.8 °C,
low 7.8 °C, and 0.0 mm precipitation, and April 9 has high 19.3 °C, low 3.7 °C,
and 0.0 mm precipitation, matching the guide. Intentional conversion and aggregation mutations
were caught by tests; exact failure output is recorded in source comments and
the working implementations were restored. `conftest.py` is empty.
Week 3 classification work is in [assignments_03](assignments_03/README.md).
It includes preprocessing and classifier warmups, NYC weather model training,
and a separate script that loads the committed model to predict new days.
See the assignment README for setup, verified results, and practice guidance.

Week 4's reusable classifier, warmups, tests, and batch prediction demo are in
[assignments_04](assignments_04/README.md).
