# Week 5: Introduction to AI

Install dependencies with `uv pip install -r requirements.txt`. Create `.env` at
repository root containing `OPENAI_API_KEY=your-key-here` (never commit the key).

Run from repository root:

```sh
.venv/bin/python assignments_05/warmup_05.py
.venv/bin/python assignments_05/project_05.py --demo
.venv/bin/python assignments_05/project_05.py
```

These commands make live API requests and use API credits. The demo runs the
starter bullet rewrite, two cover letter examples, and safe/flagged moderation
examples. The chatbot moderates every input, including nested prompts, and keeps
feature outputs in conversation history. Type quit/exit at any prompt to leave.

Before submission, run the API examples with your course key, update the warmup
comments with your observed results, and personalize the ethics reflection.
Ollama is not installed in the current environment: run the command in the
warmup's Ollama section, paste the actual output there, and compare both replies.
No model output or live verification has been fabricated.
