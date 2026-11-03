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

Live verification completed with the course key: the warmup and project demo ran
successfully, JSON fields parsed, the delimiter fallback matched exactly, and
moderation accepted the safe example and rejected the violent example. A three-turn
live chat correctly recalled the supplied name and previous career and exited cleanly.
Ollama was installed and the required qwen3:0.6b command completed successfully;
its actual final answer and comparison observations are in the warmup comments.
The ethics reflection remains a draft for you to review and personalize.

The structured JSON exercise uses JSON mode because a first live run returned
Markdown fences; its required parsing fallback remains in place.
