# Week 7: AI agents

The warmups extend the course's minimal ReAct agent and CSV tools, then compare
smolagents ToolCallingAgent and CodeAgent. The project uses four tools and a
CodeAgent to explore ten years of World Happiness data.

From the repository root:

```sh
uv pip install --python .venv/bin/python -r requirements.txt
cd assignments_07
../.venv/bin/python warmup_07.py
../.venv/bin/python project_07.py
```

Put `OPENAI_API_KEY=...` in the repository root `.env`. That file is ignored by
Git. Running these scripts makes paid API calls. The scripts use a noninteractive
matplotlib backend and save plots to `outputs/`.

The project passes its loaded DataFrame into the code interpreter as
`happiness_df`; the loader tool still returns only shape and columns. This is
necessary because tool globals and the agent interpreter have separate namespaces.
Regional lines represent country means within each region and year, without
population weighting. Dataset coverage can change over time. Correlation over
pooled country-year rows does not establish causality or independent observations.

The original bike CSV, ten happiness CSVs, and copied/adapted lesson setup come
from [Code the Dream's course repository](https://github.com/Code-the-Dream-School/python-200-v1),
under `lessons/07_AI_agents/` and `assignments/resources/happiness_project/`.
The manual tool loop follows the [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).

Run logs in `outputs/warmup_run.txt` and `outputs/project_run.txt` record actual
model behavior. Outputs can vary on subsequent runs. Reflection comments are
drafts based on those runs; review and personalize them before class submission.

Verified: both scripts completed, all seven project queries returned answers,
and the regional chart was saved and visually inspected. Direct tool checks
covered the 1,502-row merge, all ten years, ranking, summary statistics, missing
columns/data, invalid year, constant inputs, and finite paired correlation.
The project agent recovered from an initial metadata-access error. In the warmup,
both agents used the fixed plotting tool and failed to produce green dots; the
tool agent incorrectly claimed it had. These outcomes are discussed in comments.
