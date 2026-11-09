---
title: Internal Knowledge-Base Q&A
group: portfolio
role: Applied AI
port: 8113
status: working
one_liner: Ask a policy question and get an answer that cites the file and chunk it came from, with retrieval done locally by MiniLM embeddings and a numpy cosine index, no hosted vector service.
---
# Internal Knowledge-Base Q&A

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (67 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8113/.

Two-minute walkthrough:
1. Internal knowledge base: the sidebar lists the indexed documents and their chunks
2. Use sample: how do I request a leave of absence?
3. Ask: the answer cites the file and chunk it came from
4. Open the cited passage
5. A second question: submitting a business expense
6. Every answer is grounded: no citation, no claim
<!-- /demo:auto -->

## Brief
Employees lose time hunting through policy PDFs and wikis for a number or a deadline, and the answers they get from
colleagues are often wrong. This app indexes the policy documents, finds the passages that match a question and
answers with citations so the reader can verify every fact. Solved means a question about expenses, leave, security
or remote work returns the right section in under a second, with the source next to it.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | same as the rest of the portfolio |
| Model or core logic | sentence-transformers `all-MiniLM-L6-v2` (TF-IDF fallback) + numpy cosine index + lexical bonus; `common/llm.py` for the answer | local, free, offline after one download; provider by env for the LLM step |
| Interface | Streamlit with `common/streamlit_theme.py` | question in, answer and citation expanders out; `key="input"`, `key="run"` |
| Data | `docs/`: 3 Markdown policies + 1 generated PDF, 18 chunks | small, checkable, exercises both loaders |
| Tests | pytest (5) incl. `AppTest` smoke | loaders, chunking, index round trip, retrieval hits, offline citations, UI |

## Libraries used
`sentence-transformers` (MiniLM embeddings, cached in `models/st/`), `scikit-learn` (TF-IDF fallback), `numpy`
(index and cosine search), `pypdf` (PDF text), `streamlit` (UI), `anthropic` / `openai` / `google-genai` (answer
providers), `reportlab` (dev only: generated the sample PDF), `pytest`, `ruff`.

## How it was built
1. Read the original `rag_pipeline.py`: LangChain `PyPDFLoader` + `RecursiveCharacterTextSplitter`, OpenAI
   embeddings, an in-memory Chroma store rebuilt on every question, `RetrievalQA`, and a hard-coded mock answer when
   `docs/` was empty (it was empty).
2. Wrote the four sample policies (expense, leave, IT security as Markdown; remote work generated as a PDF with
   reportlab) so there is something real to retrieve, with concrete numbers to check.
3. Rewrote the pipeline in `kb.py` without LangChain: `load_docs` (md/txt/pdf), paragraph-aware `chunk_text` with
   word-boundary overlap, two embedders behind one interface, an `Index` that saves to `models/kb_index/` and loads
   back, and `answer`, which prompts for `[file chunk N]` citations through `common/llm.py`.
4. Added a lexical bonus to the cosine score after noticing MiniLM tied the "Leave of absence" section with a
   neighbouring chunk; the exact-phrase section now ranks first.
5. Built the Streamlit page: provider and embedder badges, indexed documents with chunk counts, Use sample, Ask,
   answer, citation expanders with scores, Rebuild index.
6. Tests run with TF-IDF in a temporary index dir so they need no download; `just kb-index` builds the MiniLM index.

## Results
On this machine: 18 chunks from 4 files, MiniLM, index build about 15 s including model load, search under 50 ms.
Retrieval on the four sample questions is 4/4 (top chunk is the right section for leave of absence, expense
submission, stolen laptop, working abroad), checked by the tests with TF-IDF and by hand with MiniLM. This is a
sanity check, not the >85% precision claim: that needs a labelled question set of a few hundred. The hallucination
rate cannot be measured without a key; the prompt restricts the model to the excerpts and the page flags answers
without citation markers.

## Demo details
- Run: `just internal_kb_qa-setup && just kb-index && just internal_kb_qa-up`, open `http://localhost:8113/`.
- Two-minute walkthrough: (1) sidebar: badge, `minilm`, 18 chunks, four files; (2) Use sample -> Ask, read the
  "Leave of absence" passage and its citation; (3) open the citation expanders and show the scores; (4) click the
  "work from another country" sample to show a hit inside the PDF.
- Video clip: `demo/clips/portfolio__internal_kb_qa.webm`.
- Sample inputs: `SAMPLE_QUESTIONS` in `kb.py`; expected top files are `leave_policy.md`, `expense_policy.md`,
  `remote_work_policy.pdf`, `it_security_policy.md`.

## Limits and next steps
- Offline, the answer is the best passage, not generated text; a key turns on the written, cited answer.
- Four fictional documents; a real deployment points `KB_DOCS_DIR` at the policy folder and rebuilds.
- Scores are not calibrated: an off-topic question still returns passages, and only the prompt stops the model from
  answering from them.
- Text only; tables and images in PDFs are skipped. A reranker and a labelled eval set are the next steps.
