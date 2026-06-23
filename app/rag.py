"""Local CPU embeddings + FAISS index. No GPU, no network."""
import os
import pickle
from pathlib import Path

import numpy as np

from loaders import chunk_text

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
META_PATH = VECTORSTORE_DIR / "meta.pkl"


_model_cache = {}


def get_model(name: str | None = None):
    """Load (and cache) the sentence-transformers model on CPU."""
    from sentence_transformers import SentenceTransformer
    name = name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name, device="cpu")
    return _model_cache[name]


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def embed(texts: list[str], model_name: str | None = None, is_query: bool = False) -> np.ndarray:
    """Embed a list of texts. e5 models need 'query:'/'passage:' prefixes."""
    model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = get_model(model_name)
    if _is_e5(model_name):
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    vecs = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    )
    return vecs.astype("float32")


def build_index(cv_docs: list[dict], jd_docs: list[dict], model_name: str | None = None) -> dict:
    """Chunk all docs, embed on CPU, build a FAISS index, persist to disk.

    Returns a stats dict.
    """
    import faiss

    model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    records = []  # one per chunk: {kind, name, chunk}

    for kind, docs in (("cv", cv_docs), ("jd", jd_docs)):
        for doc in docs:
            for chunk in chunk_text(doc["text"]):
                records.append({"kind": kind, "name": doc["name"], "chunk": chunk})

    if not records:
        raise ValueError("No text found to index. Add files to data/cv and data/jd.")

    vectors = embed([r["chunk"] for r in records], model_name=model_name, is_query=False)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vectors
    index.add(vectors)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"records": records, "model_name": model_name}, f)

    return {
        "chunks": len(records),
        "cv_files": len(cv_docs),
        "jd_files": len(jd_docs),
        "dim": dim,
        "model": model_name,
    }


def index_exists() -> bool:
    return INDEX_PATH.exists() and META_PATH.exists()


def _load_index():
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query: str, k: int = 5, kind: str | None = None) -> list[dict]:
    """Return top-k chunks for a query, optionally filtered to 'cv' or 'jd'."""
    if not index_exists():
        return []
    index, meta = _load_index()
    records = meta["records"]
    model_name = meta.get("model_name")
    qvec = embed([query], model_name=model_name, is_query=True)

    # Over-fetch then filter, so kind filtering still returns k results.
    fetch = min(len(records), k * 5 if kind else k)
    scores, idxs = index.search(qvec, fetch)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        rec = records[idx]
        if kind and rec["kind"] != kind:
            continue
        out.append({**rec, "score": float(score)})
        if len(out) >= k:
            break
    return out


def cv_text_summary(max_chars: int = 6000) -> str:
    """Reconstruct CV text from the index (for sending to the LLM as context)."""
    if not index_exists():
        return ""
    _, meta = _load_index()
    parts = [r["chunk"] for r in meta["records"] if r["kind"] == "cv"]
    return "\n".join(parts)[:max_chars]
