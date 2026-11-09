<!-- public-top -->
# Internal Knowledge-Base Q&A

Ask a policy question and get an answer that cites the file and chunk it came from, with retrieval done locally by MiniLM embeddings and a numpy cosine index, no hosted vector service. Portfolio, Applied AI.

**Watch it first:** [the 90-second film](demo/film.mp4) (what it does, how it is built, the proof, the answer) or the [60-second cut](demo/film-60s.mp4). The plain walkthrough with captions is [demo/demo.mp4](demo/demo.mp4).

## Try it in three commands

```bash
git clone https://github.com/Keshigami/internal_kb_qa.git && cd internal_kb_qa
just setup        # creates the venv with uv (Python 3.11)
just up           # http://127.0.0.1:8113/  (just down to stop)
```

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (`brew install uv just` on a Mac). Everything the app needs is in the repo (synthetic sample data where a dataset is used). No API key is required: without one the app runs in a deterministic offline mode and says so on screen. Set `ANTHROPIC_API_KEY` (or `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, or `gemini` with `GOOGLE_API_KEY`) in your shell for real model answers. `just test` runs the lint and the offline tests.

Step by step: [guide/Getting-Started.md](guide/Getting-Started.md), then [how it was built](guide/How-It-Was-Built.md), [results](guide/Results.md), [limits](guide/Limits-and-Next-Steps.md) and [the film](guide/The-Film.md). The same pages are on the [wiki](https://github.com/Keshigami/internal_kb_qa/wiki).
<!-- /public-top -->
# Internal Knowledge-Base Q&A (RAG)

## Demo walkthrough
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

Port 8113. Streamlit. Real retrieval with local embeddings: the policy documents in `docs/` (Markdown and PDF) are
chunked, embedded with `all-MiniLM-L6-v2` (sentence-transformers, downloaded once into `models/st/`, TF-IDF as the
fallback) and searched with a numpy cosine index cached in `models/kb_index/`. Every answer cites file and chunk.
The answer step goes through the repo's shared LLM door (`common/llm.py`); with no key the app shows the best
matching passage instead of generated text.

## 1. Problem Statement & Business Impact

- **The Problem**: Employees spend hours searching through disparate PDFs and Notion docs for company processes.
- **Goal**: Provide an instant Q&A interface that answers questions with 100% grounded citations.
- **Metric of Success**: Retrieval precision > 85%, Hallucination rate < 1%.

## 2. Technical Solution

- **Approach**: `kb.py` loads `.md`, `.txt` and `.pdf` files, splits them on paragraph boundaries into ~700-character
  chunks with a 120-character overlap, embeds them, and ranks by cosine similarity plus a small lexical bonus (query
  terms and bigrams present in the chunk, so the exact "leave of absence" section beats a paraphrase). The top-k
  passages go to the LLM with a prompt that requires `[file chunk N]` citations and forbids guessing.
- **Stack**: Python 3.11 (uv), sentence-transformers (MiniLM) with scikit-learn TF-IDF fallback, numpy, pypdf,
  Streamlit, `common/llm.py`, pytest with `AppTest`, ruff. No LangChain, no hosted embedding API, no vector DB server.
- **Diagram**: [docs/*.md, *.pdf] -> [chunk] -> [MiniLM or TF-IDF] -> [numpy cosine + lexical bonus] -> [LLM with citations]

## 3. Evaluation & Results

Index on this machine: 18 chunks from 4 files (`expense_policy.md` 5, `it_security_policy.md` 6, `leave_policy.md` 5,
`remote_work_policy.pdf` 2), MiniLM, built in about 15 s including model load.

Retrieval checks that pass offline with TF-IDF (tests) and were repeated by hand with MiniLM:

| Question | Top chunk | Right? |
|---|---|---|
| How do I request a leave of absence? | `leave_policy.md` chunk 3 ("Leave of absence" section) | yes |
| Process for submitting a business expense? | `expense_policy.md` | yes |
| Stolen laptop? | `it_security_policy.md` | yes |
| Work from another country? | `remote_work_policy.pdf` | yes |

That is 4/4 on the sample questions; a real precision number needs a larger labelled question set. Hallucination
cannot be measured without a key; the prompt restricts the model to the excerpts and the page flags answers with no
citation markers.

- **Tradeoffs**: numpy over ChromaDB because 18 chunks (or 18,000) fit in memory and the index is a file; MiniLM over
  a hosted embedding API so the demo costs nothing and works offline after the first download.

## 4. How to run

```bash
just internal_kb_qa-setup
just kb-index                          # builds models/kb_index (MiniLM download on first run, ~90 MB)
just internal_kb_qa-test               # ruff + 5 pytest tests, offline, TF-IDF, temporary index dir
just internal_kb_qa-up                 # background on :8113, pid in .pid, log in app.log
curl -s localhost:8113/_stcore/health  # ok
just internal_kb_qa-down
```

`uv run python kb.py --query "..."` builds and answers from the shell. Env: `KB_EMBEDDER=auto|minilm|tfidf`,
`KB_INDEX_DIR`, `KB_DOCS_DIR`. Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` / `GOOGLE_API_KEY`) before `-up` for a
written answer; the app starts and builds the index itself if `just kb-index` was skipped.

## 5. Two-minute interviewer walkthrough

1. Open `http://localhost:8113/`. The sidebar shows the provider badge, the embedder (`minilm`), the chunk count and
   the four indexed documents.
2. Press **Use sample** ("How do I request a leave of absence?") then **Ask**. Offline: the "Leave of absence" passage
   with its `[leave_policy.md chunk 3]` citation. With a key: a written answer with citations after each fact.
3. Open the citation expanders and show the scores; the PDF question ("work from another country") lands on
   `remote_work_policy.pdf`.
4. Drop another `.md` into `docs/`, press **Rebuild index**, ask about it.

## 6. What is real vs mocked

- Real: document loading (including PDF via pypdf), chunking, local embeddings, cosine retrieval with scores,
  citations, the LLM answer when a key is set, the index cache and rebuild.
- Offline (no key): the answer is the top passage verbatim, labelled, with the same citations. Nothing is generated.
- Sample docs: the four policies were written for this demo (the original project shipped an empty `docs/`); they
  are fictional but internally consistent so the questions have checkable answers.

## 7. Case Study Narrative

- **Context**: Demonstrates the "gold standard" of business AI: RAG with verifiable citations.
- **Implementation**: A chunking and embedding pipeline that turns unstructured documents into a searchable vector
  space with no external services.
- **Limitations**: Text only; tables and images in PDFs are not handled. Chunk scores are not calibrated, so a
  question outside the corpus still returns passages (the prompt tells the model to say so).
- **Next Steps**: A labelled question set for precision/recall, a reranker, and an agentic layer that can choose
  between the KB and web search.

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
