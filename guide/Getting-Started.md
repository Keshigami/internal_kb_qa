# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv.
- No API key is required: without one the app runs in a deterministic offline mode and says so on screen. Set `ANTHROPIC_API_KEY` (or `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, or `gemini` with `GOOGLE_API_KEY`) in your shell for real model answers.

## Three commands
```bash
git clone https://github.com/Keshigami/internal_kb_qa.git && cd internal_kb_qa
just setup
just up           # http://127.0.0.1:8113/  (just down to stop)
```
Everything the app needs is in the repo (synthetic sample data where a dataset is used).

## Commands
`just setup`, `just train` where a model is trained, `just up` / `just down`, `just test`, `just demo`.

## What you see
- Run: `just internal_kb_qa-setup && just kb-index && just internal_kb_qa-up`, open `http://localhost:8113/`.
- Two-minute walkthrough: (1) sidebar: badge, `minilm`, 18 chunks, four files; (2) Use sample -> Ask, read the
  "Leave of absence" passage and its citation; (3) open the citation expanders and show the scores; (4) click the
  "work from another country" sample to show a hit inside the PDF.
- Video clip: `demo/clips/portfolio__internal_kb_qa.webm`.
- Sample inputs: `SAMPLE_QUESTIONS` in `kb.py`; expected top files are `leave_policy.md`, `expense_policy.md`,
  `remote_work_policy.pdf`, `it_security_policy.md`.

## Try this
Try it: Use sample, Ask, open the citation.
