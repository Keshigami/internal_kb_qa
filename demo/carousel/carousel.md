# Internal knowledge-base Q&A · carousel

Spec: film/specs/portfolio__internal_kb_qa.json · 8 slides, 1080x1350.

## 1/8 Hook

- The answer, and the file it came from.

## 2/8 The question

- Four policies, eighteen chunks.
- Which chunk holds the thirty days?
- leave_policy.md 5

## 3/8 What it does

- Internal knowledge-base Q&A
- Portfolio · Applied AI
- Local embeddings, cited answers, no vector service
- max@baluyot ~ $ uv run python kb.py --query "How do I request a leave of absence?"

## 4/8 How it is built

- 1. docs/: 3 Markdown policies + 1 generated PDF, 18 — sentence-transformers
- 2. common/llm.py for the answer — sentence-transformers
- 3. sentence-transformers all-MiniLM-L6-v2 (TF-IDF f — sentence-transformers · models/st/ · scikit-learn
- 4. Streamlit with common/streamlit_theme.py — Streamlit
- 5. Dashboard — Streamlit
- An exact phrase beats a paraphrase

## 5/8 The result

- Top passage per sample question · MiniLM
- business expense · expense_policy.md · chunk 0 · 0.774 (highlighted)
- work from another country · remote_work_policy.pdf · chunk 1 · 0.767
- laptop stolen · it_security_policy.md · chunk 1 · 0.381

## 6/8 The proof

- Ask: the passage, leave_policy.md chunk 3
- screen: portfolio/internal_kb_qa/demo/poster.png

## 7/8 The answer and the limit

- 4/4
- sample questions · right section on top · MiniLM
- Offline, the answer is the passage, not prose.
- And it ships.

## 8/8 Try it

- Try it: Use sample, Ask, open the citation.
- max@baluyot ~ $ git clone https://github.com/Keshigami/internal_kb_qa.git
- max@baluyot ~ $ just internal_kb_qa-setup
- max@baluyot ~ $ just internal_kb_qa-up
- github.com/Keshigami/internal_kb_qa
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
