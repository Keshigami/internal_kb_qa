"""Internal Knowledge-Base Q&A (RAG), Streamlit UI. Run: uv run streamlit run app.py --server.port 8113"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import llm  # noqa: E402
from common.streamlit_theme import apply  # noqa: E402
from kb import DOCS_DIR, INDEX_DIR, SAMPLE_QUESTIONS, answer, build_index, load_or_build  # noqa: E402

st.set_page_config(page_title="Internal KB Q&A", page_icon=":material/menu_book:", layout="wide")
provider = llm.provider()
badge = "offline · extractive" if provider == "offline" else f"{provider} · {llm.model()}"
apply("Internal Knowledge-Base Q&A", "Ask a question about company policy; the answer cites the file and chunk "
      "it came from. Retrieval is real and local (MiniLM or TF-IDF embeddings, numpy cosine index). Port 8113.",
      badge=badge)


@st.cache_resource(show_spinner="Loading the index (first run builds it)...")
def get_index(_rebuild: int = 0):
    return load_or_build()


if "rebuild" not in st.session_state:
    st.session_state["rebuild"] = 0
index = get_index(st.session_state["rebuild"])

with st.sidebar:
    st.subheader("Provider")
    st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)
    if provider == "offline":
        st.info("No API key found: the app returns the best matching passage instead of a written answer. "
                "Set ANTHROPIC_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY) and restart for a written, cited answer.")
    st.subheader("Index")
    st.markdown(f'<span class="badge">{index.embedder_note}</span>', unsafe_allow_html=True)
    st.caption(f"{len(index.chunks)} chunks · {INDEX_DIR.relative_to(HERE)}")
    st.subheader("Indexed documents")
    for f, n in index.files.items():
        st.markdown(f"- `{f}` · {n} chunks")
    st.caption(f"Drop .md, .txt or .pdf files into `{DOCS_DIR.relative_to(HERE)}/` and rebuild.")
    if st.button("Rebuild index", key="rebuild_btn"):
        with st.spinner("Rebuilding..."):
            build_index()
        st.session_state["rebuild"] += 1
        st.cache_resource.clear()
        st.rerun()
    st.subheader("What is real")
    st.caption("Real: chunking, local embeddings, cosine retrieval, citations, the LLM call when a key is set. "
               "Offline: the answer is the top passage, not generated text.")

if "input" not in st.session_state:
    st.session_state["input"] = ""


def _use_sample(i: int = 0):
    st.session_state["input"] = SAMPLE_QUESTIONS[i]


left, right = st.columns([1, 1])
with left:
    st.subheader("Question")
    st.text_input("Ask about company policy", key="input", placeholder="e.g. How do I request a leave of absence?")
    c1, c2, _ = st.columns([1, 1, 2])
    run = c1.button("Ask", key="run", type="primary")
    c2.button("Use sample", key="use_sample", on_click=_use_sample, kwargs={"i": 0})
    st.caption("More samples:")
    for i, q in enumerate(SAMPLE_QUESTIONS[1:], start=1):
        st.button(q, key=f"sample_{i}", on_click=_use_sample, kwargs={"i": i})
    k = st.slider("Passages to retrieve", 1, 8, 4, key="k")

with right:
    st.subheader("Answer")
    if run:
        q = st.session_state.get("input", "").strip()
        if not q:
            st.warning("Type a question or press Use sample first.")
        else:
            try:
                with st.spinner("Retrieving and answering..."):
                    res = answer(q, index, k=k)
            except Exception as e:  # noqa: BLE001 provider errors surface to the user
                st.error(f"Answer failed: {e}. Check the provider key or run offline.")
            else:
                if res["provider"] == "offline":
                    st.warning("Offline mode: showing the best matching passage, not a generated answer.")
                st.markdown(res["answer"])
                st.caption(f"{res['provider']} · {res['model']} · "
                           + ("citations present" if res["grounded"] else "no citation markers in the answer"))
                st.markdown("**Citations**")
                for c in res["citations"]:
                    with st.expander(f"{c['file']} · chunk {c['chunk']} · score {c['score']:.3f}"):
                        st.write(c["text"])
    else:
        st.markdown('<div style="color:#7c869d;padding:24px;text-align:center;border:1px dashed '
                    'rgba(226,232,245,.12);border-radius:12px">Press <b>Use sample</b> then <b>Ask</b>. '
                    'Every answer cites the file and chunk it came from.</div>', unsafe_allow_html=True)
