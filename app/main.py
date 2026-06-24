"""Job-hunting agent — Streamlit UI. CPU-only, free cloud LLM.

Multi-user: each account has isolated data. Storage is pluggable (see store.py)
— local files + FAISS by default, or an external Qdrant vector DB when
QDRANT_URL is set, so data can persist across restarts. CVs are uploaded in the
browser (PDF, DOCX, MD or TXT).
"""
import sys
from pathlib import Path

# Make sibling modules importable when run via `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from dotenv import load_dotenv

import auth
import llm
import prompts
import store

load_dotenv()

st.set_page_config(page_title="求职 Agent", page_icon="🎯", layout="wide")


# ---------------- Auth gate ----------------
st.session_state.setdefault("user", None)


def _auth_view():
    st.title("🎯 求职 Agent")
    st.caption("Sign in to manage your private CVs and job matches.")
    tab_login, tab_register = st.tabs(["Log in", "Register"])
    with tab_login:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Password", type="password", key="login_p")
        if st.button("Log in", type="primary"):
            if auth.verify(u, p):
                st.session_state["user"] = u.strip()
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tab_register:
        u = st.text_input("Choose a username (min 3 chars)", key="reg_u")
        p = st.text_input("Choose a password (min 6 chars)", type="password", key="reg_p")
        p2 = st.text_input("Confirm password", type="password", key="reg_p2")
        if st.button("Create account"):
            if p != p2:
                st.error("Passwords don't match.")
            else:
                ok, msg = auth.register(u, p)
                (st.success if ok else st.error)(msg)


if not st.session_state["user"]:
    _auth_view()
    st.stop()

USER = st.session_state["user"]
USER_ID = auth.get_user_id(USER)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎯 求职 Agent")
    st.caption(f"👤 Signed in as **{USER}**")
    if st.button("Log out", use_container_width=True):
        st.session_state["user"] = None
        st.rerun()

    provider, status = llm.provider_status()
    if provider:
        st.success(f"LLM: {status}")
    else:
        st.error(status)
        st.caption("Add a free key in `.env` — see README.")

    st.divider()
    st.subheader("Upload")
    # A bump-counter in the widget key resets the uploader after a save, so files
    # aren't re-submitted on every rerun and don't fight with deletes.
    cvk = st.session_state.get("cvk", 0)
    up_cv = st.file_uploader(
        "Add CV — PDF recommended",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        key=f"up_cv_{cvk}",
    )
    if up_cv and st.button("Save CV(s)", use_container_width=True):
        with st.spinner("Saving & embedding…"):
            for f in up_cv:
                store.add_document(USER_ID, "cv", Path(f.name).name, f.getvalue())
        st.session_state["cvk"] = cvk + 1
        st.toast(f"Saved {len(up_cv)} CV file(s).")
        st.rerun()

    jdk = st.session_state.get("jdk", 0)
    up_jd = st.file_uploader(
        "Add JD — PDF / DOCX / MD / TXT",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        key=f"up_jd_{jdk}",
    )
    if up_jd and st.button("Save JD(s)", use_container_width=True):
        with st.spinner("Saving & embedding…"):
            for f in up_jd:
                store.add_document(USER_ID, "jd", Path(f.name).name, f.getvalue())
        st.session_state["jdk"] = jdk + 1
        st.toast(f"Saved {len(up_jd)} JD file(s).")
        st.rerun()

    st.divider()
    st.subheader("Corpus")
    cv_list = store.list_documents(USER_ID, "cv")
    jd_list = store.list_documents(USER_ID, "jd")
    shared_jd_n = sum(1 for d in jd_list if d.get("shared"))
    st.write(f"📄 CVs: **{len(cv_list)}**")
    st.write(f"📋 JDs: **{len(jd_list)}**  ·  {shared_jd_n} shared")
    with st.expander("Files (🗑 to remove · 🔒 = shared sample)"):
        for d in cv_list:
            col1, col2 = st.columns([5, 1])
            col1.caption(f"CV · {d['name']}")
            if col2.button("🗑", key=f"delcv_{d['name']}"):
                store.delete_document(USER_ID, "cv", d["name"])
                st.rerun()
        for d in jd_list:
            col1, col2 = st.columns([5, 1])
            col1.caption(f"JD · {d['name']}" + ("  ·  sample" if d.get("shared") else ""))
            if d.get("shared"):
                col2.caption("🔒")
            elif col2.button("🗑", key=f"deljd_{d['name']}"):
                store.delete_document(USER_ID, "jd", d["name"])
                st.rerun()

    # Pick which CV to match with (keep several tailored versions side by side).
    st.divider()
    if cv_list:
        cv_options = ["All (combined)"] + [d["name"] for d in cv_list]
        selected_cv = st.selectbox(
            "Active CV",
            cv_options,
            key="active_cv",
            help="Which CV to use for ranking, fit analysis, cover letters and RAG. "
            "'All (combined)' merges every uploaded CV.",
        )
        if selected_cv == "All (combined)":
            active_cv_names, active_cv_name = None, None
        else:
            active_cv_names, active_cv_name = [selected_cv], selected_cv
    else:
        active_cv_names, active_cv_name, selected_cv = None, None, None

    st.divider()
    if st.button("🔁 Rebuild index", use_container_width=True):
        with st.spinner("Re-embedding on CPU…"):
            info = store.reindex(USER_ID)
        st.success(f"Index rebuilt ({info.get('backend')}).")
    import os as _os
    st.caption(
        f"Storage: `{store.backend()}` · Embed: `{_os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')}`"
    )


# ---------------- Main tabs ----------------
st.header("Match your CV against job descriptions")

if not cv_list:
    st.info("👈 Upload your CV (PDF recommended) in the sidebar to get started.")
if not jd_list:
    st.info("👈 Upload one or more job descriptions in the sidebar.")

tab_match, tab_analyze, tab_letter, tab_search, tab_agent, tab_ops = st.tabs(
    [
        "📊 Rank JDs",
        "🔍 Fit analysis",
        "✉️ Cover letter",
        "🔎 Semantic search",
        "🤖 Agent",
        "📈 LLMOps",
    ]
)

with tab_match:
    st.subheader("Which jobs fit you best?")
    if cv_list and jd_list:
        st.caption(f"Using CV: **{selected_cv}**")
        if st.button("Rank job descriptions", type="primary"):
            with st.spinner("Scoring on CPU…"):
                ranked = store.rank_jobs(USER_ID, active_cv_names)
            for i, r in enumerate(ranked, 1):
                pct = round(r["score"] * 100, 1)
                tag = " · shared" if r.get("shared") else ""
                st.write(f"**{i}. {r['name']}**{tag} — similarity {pct}%")
                st.progress(min(max(r["score"], 0.0), 1.0))
    else:
        st.caption("Need at least one CV and one JD.")

with tab_analyze:
    st.subheader("Deep-dive on one role")
    if not provider:
        st.warning("Set an LLM API key to use analysis.")
    use_rag = st.toggle(
        "Use RAG (ground analysis on retrieved CV passages)",
        value=store.has_index(USER_ID),
        help="On: retrieve only the CV chunks relevant to this JD and cite them. "
        "Off: send the full CV.",
    )
    if cv_list and jd_list:
        st.caption(f"Using CV: **{selected_cv}**")
        jd_names = [d["name"] for d in jd_list]
        choice = st.selectbox("Pick a job description", jd_names, key="analyze_jd")
        if st.button("Analyze fit", type="primary", disabled=not provider):
            jd_text = store.get_combined_text(USER_ID, "jd", [choice])
            cv_text = store.get_combined_text(USER_ID, "cv", active_cv_names)
            prompt = None
            if use_rag and store.has_index(USER_ID):
                cv_ctx, hits = store.retrieve_context(
                    USER_ID, jd_text, k=6, kind="cv", name=active_cv_name
                )
                if hits:
                    prompt = prompts.fit_analysis_rag_prompt(cv_ctx, jd_text)
                    with st.expander(f"🔎 Retrieved {len(hits)} CV passages (RAG context)"):
                        for i, h in enumerate(hits, 1):
                            st.caption(f"[{i}] {h['name']} · score {round(h['score'], 3)}")
                            st.text(h["chunk"][:300])
                else:
                    st.info("No retrievable CV passages — using full CV text.")
            if prompt is None:
                prompt = prompts.fit_analysis_prompt(cv_text, jd_text)
            with st.spinner("Asking the LLM…"):
                out = llm.complete(prompt, system=prompts.SYSTEM)
            st.markdown(out)

with tab_letter:
    st.subheader("Draft a tailored cover letter")
    if not provider:
        st.warning("Set an LLM API key to generate letters.")
    if cv_list and jd_list:
        st.caption(f"Using CV: **{selected_cv}**")
        jd_names = [d["name"] for d in jd_list]
        choice = st.selectbox("Pick a job description", jd_names, key="letter_jd")
        extra = st.text_area(
            "Optional: anything to emphasize (tone, specific project, company detail)…",
            height=80,
        )
        if st.button("Generate draft", type="primary", disabled=not provider):
            jd_text = store.get_combined_text(USER_ID, "jd", [choice])
            cv_text = store.get_combined_text(USER_ID, "cv", active_cv_names)
            with st.spinner("Writing…"):
                out = llm.complete(
                    prompts.cover_letter_prompt(cv_text, jd_text, extra),
                    system=prompts.SYSTEM,
                )
            st.markdown(out)
            st.download_button("⬇️ Download as .md", out, file_name="cover_letter.md")

with tab_search:
    st.subheader("Search across your CVs + JDs")
    if not store.has_index(USER_ID):
        st.caption("Upload a CV or JD first.")
    else:
        q = st.text_input("Query", placeholder="e.g. spatio-temporal prediction experience")
        scope = st.radio("Scope", ["all", "cv", "jd"], horizontal=True)
        if q:
            kind = None if scope == "all" else scope
            hits = store.search(USER_ID, q, k=6, kind=kind)
            for h in hits:
                st.write(f"**{h['kind'].upper()} · {h['name']}** — score {round(h['score'], 3)}")
                st.caption(h["chunk"][:400] + ("…" if len(h["chunk"]) > 400 else ""))
                st.divider()

with tab_agent:
    st.subheader("Tool-calling agent (LangChain)")
    st.caption(
        "Ask a question; the agent decides which tools to call — `search_cv`, "
        "`search_jd`, `rank_jobs` — and reasons over the results."
    )
    if not provider:
        st.warning("Set an LLM API key to use the agent.")
    if not store.has_index(USER_ID):
        st.info("Upload a CV/JD first so the agent's RAG tools have data.")
    q = st.text_input(
        "Question for the agent",
        placeholder="Which job fits me best, and what are my two weakest gaps for it?",
        key="agent_q",
    )
    if st.button("Run agent", type="primary", disabled=not provider) and q:
        try:
            import agent as job_agent

            with st.spinner("Agent is reasoning & calling tools…"):
                res = job_agent.run_agent(q, user_id=USER_ID)
            st.markdown(res["answer"])
            if res["steps"]:
                with st.expander(f"🛠️ Tool calls ({len(res['steps'])})"):
                    for i, s in enumerate(res["steps"], 1):
                        st.markdown(f"**{i}. `{s['tool']}`** — input: `{s['input']}`")
                        st.caption(s["observation"][:300])
        except ImportError as e:
            st.error(str(e))
            st.code("pip install -r requirements.txt")

with tab_ops:
    st.subheader("LLMOps — observability")
    import observability as obs

    s = obs.summary()
    if s.get("calls", 0) == 0:
        st.info("No LLM calls traced yet. Run an analysis, letter, or the agent.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM calls", s["calls"])
        c2.metric("Total tokens", f"{s['total_tokens']:,}")
        c3.metric("Est. cost (USD)", f"${s['est_cost_usd']:.4f}")
        c4.metric("Avg latency", f"{s['avg_latency_ms']:.0f} ms")
        st.caption(f"p95 latency: {s['p95_latency_ms']:.0f} ms · errors: {s['errors']}")
        st.write("**Calls by model**")
        st.json(s["by_model"])
        st.write("**Recent traces**")
        recent = [r for r in obs.load_traces(limit=200) if r.get("kind") == "complete"][-15:]
        st.dataframe(
            [
                {
                    "provider": r.get("provider"),
                    "model": r.get("model"),
                    "ms": r.get("latency_ms"),
                    "in_tok": r.get("in_tokens"),
                    "out_tok": r.get("out_tokens"),
                    "cost$": r.get("est_cost_usd"),
                    "ok": r.get("ok"),
                }
                for r in reversed(recent)
            ],
            use_container_width=True,
        )
        st.caption("System-wide operational telemetry (latency/tokens/cost only — no CV content).")

st.caption("Embeddings & search run locally on CPU. Only analysis/letter text is sent to the cloud LLM.")
