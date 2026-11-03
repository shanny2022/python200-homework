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
Ollama was installed and the required qwen3:0.6b command completed successfully.
Its actual final answer is recorded in the warmup comments. The OpenAI comparison
still requires the course API key; compare both replies after running it.
No OpenAI output or live API verification has been fabricated.
