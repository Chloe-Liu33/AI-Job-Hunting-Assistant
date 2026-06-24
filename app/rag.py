"""Local CPU embeddings + FAISS index. No GPU, no network."""
import os
import pickle
from pathlib import Path

import numpy as np

import paths
from loaders import chunk_text

VECTORSTORE_DIR = paths.DATA_ROOT / "vectorstore"  # legacy/shared default
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
META_PATH = VECTORSTORE_DIR / "meta.pkl"


def _paths(vectorstore_dir: Path | str | None = None) -> tuple[Path, Path, Path]:
    """Resolve (dir, index.faiss, meta.pkl). A per-user dir keeps each user's
    index isolated; None falls back to the shared/legacy location."""
    d = Path(vectorstore_dir) if vectorstore_dir else VECTORSTORE_DIR
    return d, d / "index.faiss", d / "meta.pkl"


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


def build_index(
    cv_docs: list[dict],
    jd_docs: list[dict],
    model_name: str | None = None,
    vectorstore_dir: Path | str | None = None,
) -> dict:
    """Chunk all docs, embed on CPU, build a FAISS index, persist to disk.

    Pass `vectorstore_dir` to write into a per-user location. Returns stats.
    """
    import faiss

    model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    vdir, index_path, meta_path = _paths(vectorstore_dir)
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

    vdir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"records": records, "model_name": model_name}, f)

    return {
        "chunks": len(records),
        "cv_files": len(cv_docs),
        "jd_files": len(jd_docs),
        "dim": dim,
        "model": model_name,
    }


def index_exists(vectorstore_dir: Path | str | None = None) -> bool:
    _, index_path, meta_path = _paths(vectorstore_dir)
    return index_path.exists() and meta_path.exists()


def _load_index(vectorstore_dir: Path | str | None = None):
    import faiss
    _, index_path, meta_path = _paths(vectorstore_dir)
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(
    query: str,
    k: int = 5,
    kind: str | None = None,
    name: str | None = None,
    vectorstore_dir: Path | str | None = None,
) -> list[dict]:
    """Return top-k chunks for a query, optionally filtered to a kind ('cv'/'jd')
    and/or a specific source file `name` (e.g. one CV among several)."""
    if not index_exists(vectorstore_dir):
        return []
    index, meta = _load_index(vectorstore_dir)
    records = meta["records"]
    model_name = meta.get("model_name")
    qvec = embed([query], model_name=model_name, is_query=True)

    # Over-fetch then filter, so kind/name filtering still returns k results.
    # A name filter is restrictive, so scan the whole (small) corpus for it.
    if name:
        fetch = len(records)
    elif kind:
        fetch = min(len(records), k * 5)
    else:
        fetch = k
    scores, idxs = index.search(qvec, min(len(records), fetch))
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        rec = records[idx]
        if kind and rec["kind"] != kind:
            continue
        if name and rec["name"] != name:
            continue
        out.append({**rec, "score": float(score)})
        if len(out) >= k:
            break
    return out


def cv_text_summary(max_chars: int = 6000, vectorstore_dir: Path | str | None = None) -> str:
    """Reconstruct CV text from the index (for sending to the LLM as context)."""
    if not index_exists(vectorstore_dir):
        return ""
    _, meta = _load_index(vectorstore_dir)
    parts = [r["chunk"] for r in meta["records"] if r["kind"] == "cv"]
    return "\n".join(parts)[:max_chars]


def retrieve_context(
    query: str,
    k: int = 6,
    kind: str | None = None,
    name: str | None = None,
    vectorstore_dir: Path | str | None = None,
) -> tuple[str, list[dict]]:
    """True-RAG helper: retrieve the top-k chunks for `query` and format them as
    a numbered, citable context block to inject into a prompt.

    `name` restricts retrieval to a single source file (e.g. one chosen CV).
    Returns (context_string, hits). This is what makes the pipeline
    *retrieval-augmented*: instead of dumping a whole document into the prompt,
    we ground generation on only the passages the retriever judged relevant —
    cheaper, more focused, and the [n] markers let the LLM cite its sources.
    """
    hits = search(query, k=k, kind=kind, name=name, vectorstore_dir=vectorstore_dir)
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] (source: {h['kind']}/{h['name']})\n{h['chunk']}")
    return "\n\n".join(blocks), hits
