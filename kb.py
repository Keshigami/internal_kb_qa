"""Internal knowledge-base RAG with local embeddings: load docs -> chunk -> embed -> numpy cosine index -> answer.

Embedders (env KB_EMBEDDER = auto | minilm | tfidf, default auto):
  minilm  sentence-transformers all-MiniLM-L6-v2 (384-d, ~90 MB, downloaded once into models/st/)
  tfidf   scikit-learn TF-IDF, no download, used by the tests and as the fallback when MiniLM is unavailable
Index: KB_INDEX_DIR (default models/kb_index/) holds vectors.npy + chunks.json (+ tfidf.pkl).
Answers go through common/llm.py; offline mode returns the best passage, labelled, with the same citations.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import llm  # noqa: E402  (sys.path set above)

DOCS_DIR = Path(os.getenv("KB_DOCS_DIR", HERE / "docs"))
INDEX_DIR = Path(os.getenv("KB_INDEX_DIR", HERE / "models" / "kb_index"))
ST_HOME = Path(os.getenv("SENTENCE_TRANSFORMERS_HOME", HERE / "models" / "st"))
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = int(os.getenv("KB_CHUNK_CHARS", "700"))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "120"))

SYSTEM_PROMPT = (
    "You answer employee questions using only the policy excerpts provided. Quote the exact numbers, deadlines and "
    "names from the excerpts. After each fact, cite its source in square brackets as [file chunk N]. "
    "If the excerpts do not contain the answer, say so in one sentence and do not guess."
)


@dataclass
class Chunk:
    file: str
    chunk: int
    text: str
    start: int


# ---------- loading and chunking ----------

def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def load_docs(docs_dir: Path = DOCS_DIR) -> list[tuple[str, str]]:
    """[(file name, text)] for every .md/.txt/.pdf in docs_dir, sorted by name."""
    out = []
    for p in sorted(Path(docs_dir).glob("*")):
        if p.suffix.lower() in (".md", ".txt", ".pdf") and p.is_file():
            text = read_document(p)
            if text.strip():
                out.append((p.name, text))
    return out


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    """Split on paragraph boundaries into pieces of about `size` chars with `overlap` chars carried over.
    Returns [(start_offset, chunk_text)]. Deterministic."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[tuple[int, str]] = []
    buf, buf_start, pos = "", 0, 0
    offsets = []
    for p in paras:
        start = text.find(p, pos)
        pos = start + len(p)
        offsets.append((start, p))
    for start, p in offsets:
        if buf and len(buf) + len(p) + 2 > size:
            chunks.append((buf_start, buf))
            tail = buf[-overlap:] if overlap else ""
            if " " in tail:  # snap the carried-over tail to a word boundary
                tail = tail[tail.find(" ") + 1:]
            buf = (tail + "\n\n" + p) if tail else p
            buf_start = start - len(tail) - 2 if tail else start
        else:
            if not buf:
                buf_start = start
            buf = (buf + "\n\n" + p) if buf else p
        while len(buf) > size * 1.6:  # one very long paragraph
            chunks.append((buf_start, buf[:size]))
            buf_start += size - overlap
            buf = buf[size - overlap:]
    if buf:
        chunks.append((buf_start, buf))
    return chunks


def chunk_docs(docs: list[tuple[str, str]]) -> list[Chunk]:
    out = []
    for name, text in docs:
        for i, (start, piece) in enumerate(chunk_text(text)):
            out.append(Chunk(file=name, chunk=i, text=piece, start=start))
    return out


# ---------- embedders ----------

class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
        self.vectorizer.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2(m)

    def save(self, d: Path):
        (d / "tfidf.pkl").write_bytes(pickle.dumps(self.vectorizer))

    @classmethod
    def load(cls, d: Path):
        return cls(pickle.loads((d / "tfidf.pkl").read_bytes()))  # noqa: S301 our own file under models/


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        ST_HOME.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.model = SentenceTransformer(MINILM, cache_folder=str(ST_HOME), device="cpu")

    def fit(self, texts):
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def save(self, d: Path):
        pass

    @classmethod
    def load(cls, d: Path):
        return cls()


_STOP = {"the", "a", "an", "of", "to", "for", "in", "on", "is", "are", "do", "i", "my", "how", "what", "can",
         "and", "or", "it", "be", "if", "we", "you", "your", "with", "at", "by", "from", "that", "this", "me"}


def _terms(text: str) -> tuple[set[str], set[str]]:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    terms = {w[:6] for w in words if w not in _STOP and len(w) > 2}
    bigrams = {f"{a} {b}" for a, b in zip(words, words[1:], strict=False)}
    return terms, bigrams


def _overlap(terms: set[str], bigrams: set[str], chunk: str) -> float:
    words = re.sub(r"[^a-z0-9]+", " ", chunk.lower()).split()
    stems = {w[:6] for w in words}
    cb = {f"{a} {b}" for a, b in zip(words, words[1:], strict=False)}
    t = len(terms & stems) / len(terms) if terms else 0.0
    b = len(bigrams & cb) / len(bigrams) if bigrams else 0.0
    return t + b


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def make_embedder(kind: str | None = None):
    """auto: MiniLM when it loads (cached or downloadable), else TF-IDF. Returns (embedder, note)."""
    kind = (kind or os.getenv("KB_EMBEDDER", "auto")).lower()
    if kind == "tfidf":
        return TfidfEmbedder(), "tfidf"
    try:
        return MiniLMEmbedder(), "minilm"
    except Exception as e:  # noqa: BLE001 import or download failure -> fallback
        if kind == "minilm":
            raise
        return TfidfEmbedder(), f"tfidf (MiniLM unavailable: {type(e).__name__})"


# ---------- index ----------

class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder, embedder_note: str = ""):
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder
        self.embedder_note = embedder_note or embedder.name

    @property
    def files(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.file] = out.get(c.file, 0) + 1
        return out

    def search(self, query: str, k: int = 4, lexical_bonus: float = 0.1) -> list[tuple[Chunk, float]]:
        """Cosine similarity plus a small lexical bonus (query terms and bigrams present in the chunk), so an exact
        phrase such as 'leave of absence' beats a paraphrase with the same embedding score."""
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        if lexical_bonus:
            terms, bigrams = _terms(query)
            if terms:
                bonus = np.array([_overlap(terms, bigrams, c.text) for c in self.chunks], dtype=np.float32)
                scores = scores + lexical_bonus * bonus
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def save(self, d: Path = INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder.name, "embedder_note": self.embedder_note,
                "chunks": [asdict(c) for c in self.chunks]}
        (d / "chunks.json").write_text(json.dumps(meta, indent=1))
        self.embedder.save(d)

    @classmethod
    def load(cls, d: Path = INDEX_DIR) -> Index:
        meta = json.loads((d / "chunks.json").read_text())
        emb = MiniLMEmbedder.load(d) if meta["embedder"] == "minilm" else TfidfEmbedder.load(d)
        return cls([Chunk(**c) for c in meta["chunks"]], np.load(d / "vectors.npy"), emb, meta.get("embedder_note", ""))


def build_index(docs_dir: Path = DOCS_DIR, out_dir: Path = INDEX_DIR, embedder_kind: str | None = None) -> Index:
    docs = load_docs(docs_dir)
    if not docs:
        raise FileNotFoundError(f"no .md/.txt/.pdf documents in {docs_dir}")
    chunks = chunk_docs(docs)
    emb, note = make_embedder(embedder_kind)
    emb.fit([c.text for c in chunks])
    vectors = emb.encode([c.text for c in chunks])
    idx = Index(chunks, vectors, emb, note)
    idx.save(out_dir)
    return idx


def load_or_build(docs_dir: Path = DOCS_DIR, out_dir: Path = INDEX_DIR, embedder_kind: str | None = None) -> Index:
    if (out_dir / "chunks.json").exists() and (out_dir / "vectors.npy").exists():
        try:
            return Index.load(out_dir)
        except Exception:  # noqa: BLE001 corrupt or incompatible cache -> rebuild
            pass
    return build_index(docs_dir, out_dir, embedder_kind)


# ---------- answering ----------

def answer(question: str, index: Index, k: int = 4) -> dict:
    """{"answer", "citations": [{file, chunk, score, text}], "provider", "model", "grounded"}"""
    q = (question or "").strip()
    if not q:
        raise ValueError("question is empty")
    hits = index.search(q, k)
    citations = [{"file": c.file, "chunk": c.chunk, "score": round(s, 3), "text": c.text} for c, s in hits]
    provider = llm.provider()
    if provider == "offline":
        top = citations[0]
        text = (f"Offline mode (no API key): the best matching passage is shown instead of a written answer.\n\n"
                f"{top['text']}\n\n[{top['file']} chunk {top['chunk']}]")
        return {"answer": text, "citations": citations, "provider": provider, "model": "offline", "grounded": True}
    context = "\n\n".join(f"[{c['file']} chunk {c['chunk']}]\n{c['text']}" for c in citations)
    text = llm.complete(SYSTEM_PROMPT, f"Question: {q}\n\nExcerpts:\n{context}", max_tokens=600, temperature=0.1)
    grounded = bool(re.search(r"\[[^\]]+ chunk \d+\]", text))
    return {"answer": text.strip(), "citations": citations, "provider": provider, "model": llm.model(),
            "grounded": grounded}


SAMPLE_QUESTIONS = [
    "How do I request a leave of absence?",
    "What is the process for submitting a business expense?",
    "How many days can I work from another country?",
    "What should I do if my laptop is stolen?",
]


if __name__ == "__main__":  # just kb-index
    import argparse

    ap = argparse.ArgumentParser(description="Build the KB index")
    ap.add_argument("--embedder", choices=["auto", "minilm", "tfidf"], default=None)
    ap.add_argument("--query", default=None, help="run one query after building")
    args = ap.parse_args()
    ix = build_index(embedder_kind=args.embedder)
    print(f"indexed {len(ix.chunks)} chunks from {len(ix.files)} files with {ix.embedder_note} -> {INDEX_DIR}")
    for f, n in ix.files.items():
        print(f"  {f}: {n} chunks")
    if args.query:
        res = answer(args.query, ix)
        print("\n" + res["answer"])
        for c in res["citations"]:
            print(f"  [{c['file']} chunk {c['chunk']}] score={c['score']}")
