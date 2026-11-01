# Week 4: Reusable weather model

The `weather_model` package exports `WeatherClassifier` and `Prediction`.
It loads the saved Week 3 pipeline once and predicts batches in a fixed feature
order. Inputs use Celsius, millimeters of precipitation, and km/h of wind.
The model and metadata are copied unchanged from `assignments_03/models/`.

From the repository root, install the model-compatible dependencies:

```sh
uv pip install -r assignments_04/requirements.txt
source .venv/bin/activate
cd assignments_04
python warmup_04.py
pytest warmup_04.py -v
pytest -v
python predict_weather.py
```

Verified: 2 warmup tests and 9 package test cases pass from `assignments_04/`.
The tests train a tiny temporary pipeline, independent of the saved artifact.
They also check feature order, ignored extra keys, and loading only once.
For the required mutation check, temporarily returning only `skip` caused
`test_clear_day_labels[day0-good]` to fail with `assert 'skip' == 'good'`.
The implementation was restored and the complete Week 4 suite passed again.

The real-model demo labels the four examples good, skip, skip, good, with
probabilities 1.00, 0.00, 0.00, 1.00. With k=1, these are neighbor votes rather
than calibrated confidence. The borderline example illustrates that the model
can disagree with the original weather rule.

The required warmups, package, tests, demo, and reflection are included; optional
extensions are omitted. Review the reflection comments in your own words before
submitting the open PR URL to the course.
