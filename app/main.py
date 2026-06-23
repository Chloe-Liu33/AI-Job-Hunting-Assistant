"""Job-hunting agent — Streamlit UI. CPU-only, free cloud LLM."""
import sys
from pathlib import Path

# Make sibling modules importable when run via `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from dotenv import load_dotenv

import llm
import prompts
import rag
from loaders import load_dir

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CV_DIR = ROOT / "data" / "cv"
JD_DIR = ROOT / "data" / "jd"

st.set_page_config(page_title="求职 Agent", page_icon="🎯", layout="wide")


def rank_jds_for_cv(cv_docs, jd_docs, model_name):
    """Score each JD against the combined CV text via mean chunk similarity."""
    import numpy as np

    cv_full = "\n".join(d["text"] for d in cv_docs)
    cv_vec = rag.embed([cv_full], model_name=model_name, is_query=True)
    results = []
    for jd in jd_docs:
        from loaders import chunk_text
        chunks = chunk_text(jd["text"])
        if not chunks:
            continue
        jd_vecs = rag.embed(chunks, model_name=model_name, is_query=False)
        sims = (jd_vecs @ cv_vec.T).ravel()
        results.append({"name": jd["name"], "score": float(np.mean(sims)), "top": float(np.max(sims))})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎯 求职 Agent")
    provider, status = llm.provider_status()
    if provider:
        st.success(f"LLM: {status}")
    else:
        st.error(status)
        st.caption("Add a free key in `.env` — see README.")

    st.divider()
    st.subheader("Corpus")
    cv_docs = load_dir(CV_DIR)
    jd_docs = load_dir(JD_DIR)
    st.write(f"📄 CVs: **{len(cv_docs)}**")
    st.write(f"📋 JDs: **{len(jd_docs)}**")
    with st.expander("Files"):
        for d in cv_docs:
            st.caption(f"CV · {d['name']}")
        for d in jd_docs:
            st.caption(f"JD · {d['name']}")

    st.divider()
    if st.button("🔁 (Re)build index", use_container_width=True):
        if not cv_docs and not jd_docs:
            st.warning("Add files to data/cv and data/jd first.")
        else:
            with st.spinner("Embedding on CPU…"):
                import os
                stats = rag.build_index(
                    cv_docs, jd_docs, model_name=os.getenv("EMBED_MODEL")
                )
            st.success(f"Indexed {stats['chunks']} chunks (dim {stats['dim']}).")
    st.caption(f"Embed model: `{__import__('os').getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')}`")


# ---------------- Main tabs ----------------
st.header("Match your CV against job descriptions")

if not cv_docs:
    st.info("👈 Drop your CV into `data/cv/` (.pdf, .docx, .md, .txt), then click **(Re)build index**.")
if not jd_docs:
    st.info("👈 Drop job descriptions into `data/jd/` (one per file), then rebuild the index.")

tab_match, tab_analyze, tab_letter, tab_search = st.tabs(
    ["📊 Rank JDs", "🔍 Fit analysis", "✉️ Cover letter", "🔎 Semantic search"]
)

with tab_match:
    st.subheader("Which jobs fit you best?")
    if cv_docs and jd_docs:
        if st.button("Rank job descriptions", type="primary"):
            import os
            with st.spinner("Scoring on CPU…"):
                ranked = rank_jds_for_cv(cv_docs, jd_docs, os.getenv("EMBED_MODEL"))
            for i, r in enumerate(ranked, 1):
                pct = round(r["score"] * 100, 1)
                st.write(f"**{i}. {r['name']}** — similarity {pct}%")
                st.progress(min(max(r["score"], 0.0), 1.0))
    else:
        st.caption("Need at least one CV and one JD.")

with tab_analyze:
    st.subheader("Deep-dive on one role")
    if not provider:
        st.warning("Set an LLM API key to use analysis.")
    if cv_docs and jd_docs:
        jd_names = [d["name"] for d in jd_docs]
        choice = st.selectbox("Pick a job description", jd_names, key="analyze_jd")
        if st.button("Analyze fit", type="primary", disabled=not provider):
            jd = next(d for d in jd_docs if d["name"] == choice)
            cv_text = "\n".join(d["text"] for d in cv_docs)
            with st.spinner("Asking the LLM…"):
                out = llm.complete(
                    prompts.fit_analysis_prompt(cv_text, jd["text"]),
                    system=prompts.SYSTEM,
                )
            st.markdown(out)

with tab_letter:
    st.subheader("Draft a tailored cover letter")
    if not provider:
        st.warning("Set an LLM API key to generate letters.")
    if cv_docs and jd_docs:
        jd_names = [d["name"] for d in jd_docs]
        choice = st.selectbox("Pick a job description", jd_names, key="letter_jd")
        extra = st.text_area(
            "Optional: anything to emphasize (tone, specific project, company detail)…",
            height=80,
        )
        if st.button("Generate draft", type="primary", disabled=not provider):
            jd = next(d for d in jd_docs if d["name"] == choice)
            cv_text = "\n".join(d["text"] for d in cv_docs)
            with st.spinner("Writing…"):
                out = llm.complete(
                    prompts.cover_letter_prompt(cv_text, jd["text"], extra),
                    system=prompts.SYSTEM,
                )
            st.markdown(out)
            st.download_button("⬇️ Download as .md", out, file_name="cover_letter.md")

with tab_search:
    st.subheader("Search across your CVs + JDs")
    if not rag.index_exists():
        st.caption("Build the index first (sidebar).")
    else:
        q = st.text_input("Query", placeholder="e.g. spatio-temporal prediction experience")
        scope = st.radio("Scope", ["all", "cv", "jd"], horizontal=True)
        if q:
            kind = None if scope == "all" else scope
            hits = rag.search(q, k=6, kind=kind)
            for h in hits:
                st.write(f"**{h['kind'].upper()} · {h['name']}** — score {round(h['score'], 3)}")
                st.caption(h["chunk"][:400] + ("…" if len(h["chunk"]) > 400 else ""))
                st.divider()

st.caption("All embeddings & search run locally on CPU. Only analysis/letter text is sent to the cloud LLM.")
