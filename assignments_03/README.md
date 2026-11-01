# Week 3: Weather classification and deployment

This AI-assisted reference includes the warmups and a NYC 2023 mini-project.
The course recommends practicing warmups independently; review and rework them
before submitting. Review the interpretations and reflection in your own words.

From the repository root (Python 3.10+):

```sh
uv venv
uv pip install -r assignments_03/requirements.txt
MPLBACKEND=Agg .venv/bin/python assignments_03/warmup_03.py
.venv/bin/python assignments_03/train_classifier.py
.venv/bin/python assignments_03/predict.py
```

All scripts also work from inside `assignments_03/`. Training requires internet
access; prediction uses only the committed model and metadata. The saved models
were created with Python and scikit-learn versions recorded in the metadata.
Install the pinned scikit-learn version before loading them. The training
pipeline includes its fitted scaler; prediction never trains or fetches data.

`weather_classification.csv` is copied unchanged from
[the official course resource](https://github.com/Code-the-Dream-School/python-200-v1/blob/e538f67c6667afe62d4569b48b36c7b1845fef1e/lessons/03_ML_classification/resources/weather_classification.csv).
This is the warmup dataset; the project separately fetches historical data from
Open-Meteo using Celsius, mm, and km/h and the assignment's exact label rule.
The fetched snapshot is saved in `outputs/nyc_weather_2023.csv`.

Verified run results are in `outputs/training_results.txt`,
`outputs/warmup_results.txt`, and `outputs/prediction_results.txt`.
The warmup confusion matrix is `outputs/logreg_confusion_matrix.png`.

| NYC model | Accuracy | Precision (good) | Recall (good) | F1 (good) |
| --- | ---: | ---: | ---: | ---: |
| KNN, k=1 | 0.9178 | 0.8889 | 0.8889 | 0.8889 |
| Logistic regression | 0.6164 | 0.4783 | 0.4074 | 0.4400 |

The selected KNN uses k=1, which had the highest mean five-fold training CV
accuracy (0.9385). Its saved artifact is about 27 KB. Higher precision supports
avoiding poor-weather recommendations, though its one-neighbor probabilities
are votes rather than calibrated confidence. The borderline example satisfies
the label rule but KNN predicts skip: the model approximates the rule imperfectly.
The linear logistic model struggles with the bounded temperature interval.

The project includes all required coding tasks. The mindset assignment asks for
personal interests and job exploration; those answers should reflect your own
experience. No optional extensions are included.
