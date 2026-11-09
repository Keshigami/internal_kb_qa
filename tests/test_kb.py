"""Offline tests: TF-IDF embedder, a temporary index dir, no network, no key."""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
TMP_INDEX = Path(os.environ.setdefault("KB_INDEX_DIR", str(HERE / "models" / "kb_index_test")))
os.environ["KB_EMBEDDER"] = "tfidf"
os.environ["LLM_PROVIDER"] = "offline"

import kb  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("KB_EMBEDDER", "tfidf")
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def index():
    return kb.build_index(out_dir=TMP_INDEX, embedder_kind="tfidf")


def test_docs_load_including_pdf():
    docs = kb.load_docs()
    names = [n for n, _ in docs]
    assert len(docs) >= 4 and any(n.endswith(".pdf") for n in names) and any(n.endswith(".md") for n in names)
    pdf_text = dict(docs)["remote_work_policy.pdf"]
    assert "30 days per calendar year" in pdf_text


def test_chunking_is_deterministic_and_bounded():
    text = "\n\n".join(f"Paragraph {i}. " + ("word " * 60) for i in range(12))
    a = kb.chunk_text(text, size=500, overlap=80)
    b = kb.chunk_text(text, size=500, overlap=80)
    assert a == b and len(a) > 3
    assert all(len(t) <= 500 * 1.6 for _, t in a)
    assert a[0][0] == 0 and all(s >= 0 for s, _ in a)


def test_index_roundtrip_and_search(index):
    assert index.vectors.shape[0] == len(index.chunks) >= 8
    loaded = kb.Index.load(TMP_INDEX)
    assert len(loaded.chunks) == len(index.chunks) and loaded.embedder.name == "tfidf"
    hits = loaded.search("How do I request a leave of absence?", k=3)
    assert hits[0][0].file == "leave_policy.md" and "leave-of-absence form" in hits[0][0].text
    hits = loaded.search("stolen laptop", k=3)
    assert hits[0][0].file == "it_security_policy.md"
    hits = loaded.search("work from another country", k=3)
    assert hits[0][0].file == "remote_work_policy.pdf"


def test_answer_offline_cites_file_and_chunk(index):
    res = kb.answer("What is the process for submitting a business expense?", index)
    assert res["provider"] == "offline" and res["grounded"]
    assert res["citations"][0]["file"] == "expense_policy.md"
    assert f"[{res['citations'][0]['file']} chunk {res['citations'][0]['chunk']}]" in res["answer"]
    assert {"file", "chunk", "score", "text"} <= set(res["citations"][0])
    with pytest.raises(ValueError):
        kb.answer("", index)


def test_app_smoke(index):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.button(key="use_sample").click().run()
    assert at.session_state["input"] == kb.SAMPLE_QUESTIONS[0]
    at.button(key="run").click().run()
    assert not at.exception
    assert any("leave_policy.md" in e.label for e in at.expander)
    assert any("leave-of-absence form" in m.value for m in at.markdown)
