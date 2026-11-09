# How it was built

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

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | same as the rest of the portfolio |
| Model or core logic | sentence-transformers `all-MiniLM-L6-v2` (TF-IDF fallback) + numpy cosine index + lexical bonus; `common/llm.py` for the answer | local, free, offline after one download; provider by env for the LLM step |
| Interface | Streamlit with `common/streamlit_theme.py` | question in, answer and citation expanders out; `key="input"`, `key="run"` |
| Data | `docs/`: 3 Markdown policies + 1 generated PDF, 18 chunks | small, checkable, exercises both loaders |
| Tests | pytest (5) incl. `AppTest` smoke | loaders, chunking, index round trip, retrieval hits, offline citations, UI |

## Libraries
`sentence-transformers` (MiniLM embeddings, cached in `models/st/`), `scikit-learn` (TF-IDF fallback), `numpy`
(index and cosine search), `pypdf` (PDF text), `streamlit` (UI), `anthropic` / `openai` / `google-genai` (answer
providers), `reportlab` (dev only: generated the sample PDF), `pytest`, `ruff`.
