"""A LlamaIndex RAG query engine over the CV + JD corpus.

LlamaIndex is the other big RAG/agent framework JDs name (alongside LangChain).
This shows the same retrieval-augmented Q&A, built with LlamaIndex's
abstractions instead of hand-rolled FAISS:

    Document -> VectorStoreIndex -> query_engine.query(...)

It stays free + no-GPU: embeddings are a LOCAL HuggingFace model on CPU
(`HuggingFaceEmbedding`), and only the final synthesis call hits the free
Gemini/Groq LLM. Heavy deps are optional — install with:

    pip install -r requirements-extras.txt
"""
from __future__ import annotations

import os
from pathlib import Path

from loaders import load_dir

ROOT = Path(__file__).resolve().parent.parent


def _configure_settings():
    """Wire LlamaIndex's global Settings to local embeddings + a free LLM."""
    try:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        raise ImportError(
            "LlamaIndex not installed. Run: pip install -r requirements-extras.txt"
        ) from e

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "groq" if os.getenv("GROQ_API_KEY") else ""
    if provider == "gemini":
        from llama_index.llms.gemini import Gemini

        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not model.startswith("models/"):
            model = f"models/{model}"
        Settings.llm = Gemini(model=model, api_key=os.environ["GEMINI_API_KEY"])
    elif provider == "groq":
        from llama_index.llms.groq import Groq

        Settings.llm = Groq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.environ["GROQ_API_KEY"],
        )
    else:
        raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or GROQ_API_KEY.")


def build_query_engine(kind: str | None = None):
    """Build a LlamaIndex query engine over the CV (+optionally JD) corpus."""
    from llama_index.core import Document, VectorStoreIndex

    _configure_settings()
    docs = []
    for k in (("cv", "jd") if kind is None else (kind,)):
        for d in load_dir(ROOT / "data" / k):
            docs.append(Document(text=d["text"], metadata={"kind": k, "name": d["name"]}))
    if not docs:
        raise RuntimeError("No CV/JD documents found in data/.")
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine(similarity_top_k=5)


def query(question: str, kind: str | None = "cv") -> dict:
    """Answer a question via LlamaIndex RAG. Returns {answer, sources}. Traced."""
    import observability as obs

    with obs.span("llamaindex.query", question=question[:200]):
        qe = build_query_engine(kind=kind)
        resp = qe.query(question)
    sources = []
    for node in getattr(resp, "source_nodes", []) or []:
        sources.append(
            {
                "name": node.metadata.get("name", "?"),
                "score": round(float(node.score), 3) if node.score is not None else None,
                "text": node.get_content()[:300],
            }
        )
    return {"answer": str(resp), "sources": sources}
