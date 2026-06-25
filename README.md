---
title: Job Agent
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app/main.py
python_version: "3.11"
pinned: false
---

# Job-Hunting Agent

> ⚠️ **Work in progress — not officially launched.** This project is still being
> tested and organized. Features, data, and the live demo may change or break at
> any time. Treat it as a preview / portfolio demo, not a finished product.

A **zero-cost, no-GPU** job-hunting assistant. Match your CV against job descriptions, get tailored fit analysis, generate cover-letter drafts, and surface gaps — all running on free cloud LLMs + local CPU embeddings.

> The YAML block above is the [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-config-reference) config (Streamlit SDK, entry point `app/main.py`). It's ignored by GitHub.

## Stack (the "no-GPU free stack")

| Layer | Tool | Where it runs |
|---|---|---|
| **Brain (LLM)** | Gemini Flash *or* Groq | Free cloud tier |
| **Retrieval (RAG)** | sentence-transformers + FAISS | Local CPU |
| **Agent frameworks** | LangChain · LlamaIndex · OpenAI Agents SDK | Free cloud tier |
| **Prompt benchmark** | LLM-as-judge harness | Free cloud tier |
| **LLMOps** | built-in tracing (latency / tokens / cost) | Local CPU |
| **UI** | Streamlit | Local CPU |

Everything is CPU + free. Deploys to Hugging Face Spaces' free CPU tier (2 vCPU + 16 GB) without a paid GPU Space.

## GenAI capabilities (and how to try each)

This project demonstrates the GenAI tooling stack commonly listed in JDs, all on the free no-GPU setup:

| Capability | Where | How to try it |
|---|---|---|
| **RAG** (true retrieval-augmented generation) | [app/rag.py](app/rag.py) `retrieve_context()` + Fit-analysis tab | Build the index, open **🔍 Fit analysis**, keep **Use RAG** on — the LLM is grounded on retrieved CV passages with `[n]` citations, not the whole CV. |
| **Agent · LangChain** | [app/agent.py](app/agent.py) | **🤖 Agent** tab. A tool-calling agent picks among `search_cv` / `search_jd` / `rank_jobs` and reasons over the results. |
| **Agent · LlamaIndex** | [app/agent_llamaindex.py](app/agent_llamaindex.py) | `python -c "import sys; sys.path.insert(0,'app'); import agent_llamaindex as a; print(a.query('summarize my strengths')['answer'])"` |
| **Agent · OpenAI Agents SDK** | [app/agent_openai_sdk.py](app/agent_openai_sdk.py) | Runs the OpenAI SDK against **Groq's** OpenAI-compatible endpoint (no paid OpenAI key). Needs `GROQ_API_KEY`. |
| **Prompt-engineering benchmark** | [app/benchmark.py](app/benchmark.py) | `python app/benchmark.py` — races prompt variants, scores each with an LLM-as-judge, writes `eval/results/benchmark.md`. |
| **LLMOps** | [app/observability.py](app/observability.py) | **📈 LLMOps** tab — every LLM call is traced (latency, token estimate, cost estimate, errors) to `data/traces/llm_traces.jsonl`. |

The **LangChain Agent tab** works out of the box (`requirements.txt`). The two
*alternative* framework demos (LlamaIndex, OpenAI Agents SDK) are not wired into
the UI and need the optional deps:

```bash
pip install -r requirements-extras.txt
```

### Persistent storage (deploys)

Storage is **pluggable** via [app/store.py](app/store.py):

- **file (default):** accounts in a JSON file, CVs/JDs as files, a per-user FAISS
  index — all under a writable root. By default that's the repo's `data/`; set
  `DATA_DIR` to redirect it onto a persistent disk (e.g. `DATA_DIR=/data` on
  Hugging Face persistent storage).
- **qdrant (recommended for permanent storage):** set `QDRANT_URL`
  (+ `QDRANT_API_KEY`) and accounts + CV/JD chunks + embeddings move to an
  external [Qdrant](https://cloud.qdrant.io) vector DB (free managed tier).
  Everything is external and survives restarts; per-user isolation is enforced
  by a `user_id` filter on every query. Embeddings are still computed locally on
  CPU — Qdrant only stores and searches the vectors.

```bash
# pick ONE persistence strategy
DATA_DIR=/data                                  # file backend on a persistent disk
# — or —
QDRANT_URL=https://YOUR-CLUSTER.cloud.qdrant.io:6333
QDRANT_API_KEY=...
```

Shipped samples (the sample CV and the shared seed JDs) always stay in the repo
and are read-only; the shared JDs are surfaced to every user (in Qdrant they're
stored once under a reserved `_shared` id).

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/Chloe-Liu33/AI-Job-Hunting-Assistant.git
cd AI-Job-Hunting-Assistant
pip install -r requirements.txt

# 2. Add your free API key (pick ONE)
cp .env.example .env
# then edit .env and paste your key

# 3. Run
streamlit run app/main.py
```

Then in the browser: **Register** an account → **Log in** → upload your CV (PDF
recommended) and job descriptions from the sidebar → **(Re)build index**.

> **Multi-user:** each account's data is isolated, so users only ever see their
> own **CVs**. **JDs are hybrid:** everyone sees the shared seed library
> (read-only, marked 🔒) plus their own uploaded JDs; a user upload overrides a
> shared one of the same name. Passwords are stored only as salted PBKDF2
> hashes. Where that data lives depends on the storage backend — local files by
> default, or external Qdrant when `QDRANT_URL` is set (see **Persistent
> storage** below).

## Getting a free API key

- **Gemini Flash** (recommended, generous free tier): https://aistudio.google.com/apikey
  → put it in `.env` as `GEMINI_API_KEY=...`
- **Groq** (very fast, free tier): https://console.groq.com/keys
  → put it in `.env` as `GROQ_API_KEY=...` and set `LLM_PROVIDER=groq`

You only need one. The app auto-detects which key is present.

## What it does

1. **Index** — reads your CVs + JDs, chunks them, embeds locally on CPU, stores in FAISS.
2. **Match** — ranks each JD against your CV by semantic similarity.
3. **Analyze** — for a chosen JD, the LLM gives you: fit score reasoning, matched strengths, gaps, and keywords to add.
4. **Draft** — generates a tailored cover-letter / outreach draft you can edit.

## Run it / Deploy it

Full step-by-step (local **and** Hugging Face) is in **[RUN_AND_DEPLOY.md](RUN_AND_DEPLOY.md)**.

- **Local:** clone → `pip install -r requirements.txt` → set keys in `.env` → `streamlit run app/main.py`.
- **Hugging Face Spaces (free CPU):** create a Streamlit Space → `git push` this repo → add `GROQ_API_KEY`, `LLM_PROVIDER=groq`, `QDRANT_URL`, `QDRANT_API_KEY` as Space **Secrets** (HF doesn't read `.env`). No code changes, no GPU.

Live demo: https://limei-liu-agent-job-search.hf.space

## Project layout

```
job-agent/
├── app/
│   ├── main.py                # Streamlit UI (tabs incl. Agent + LLMOps)
│   ├── llm.py                 # Cloud LLM wrapper (Gemini / Groq) + OpenAI-compat config + tracing
│   ├── rag.py                 # Local CPU embeddings + FAISS + retrieve_context() for true RAG
│   ├── loaders.py             # Read PDF / DOCX / TXT / MD
│   ├── prompts.py             # Prompt templates + benchmark variants + judge rubric
│   ├── agent.py               # LangChain tool-calling agent
│   ├── agent_llamaindex.py    # LlamaIndex RAG query engine
│   ├── agent_openai_sdk.py    # OpenAI Agents SDK (on Groq's OpenAI-compatible API)
│   ├── benchmark.py           # Prompt-engineering benchmark (LLM-as-judge)
│   ├── auth.py                # User registration/login (credentials only)
│   ├── store.py               # Storage abstraction: file/FAISS or Qdrant backend
│   ├── paths.py               # REPO_DATA (shipped) vs DATA_ROOT (writable) roots
│   └── observability.py       # LLMOps: per-call tracing / tokens / cost
├── data/
│   ├── cv/                    # sample CV (file backend: user CVs under data/users/)
│   ├── jd/                    # shared seed JDs (read-only library)
│   ├── users/                 # file backend: per-user private workspaces
│   ├── users.json            # file backend: accounts (salted password hashes)
│   ├── vectorstore/           # file backend: legacy/shared FAISS index
│   └── traces/                # auto-generated LLMOps trace log (JSONL)
├── eval/results/             # benchmark reports (.md / .csv)
├── .streamlit/config.toml
├── requirements.txt          # base app (incl. LangChain + qdrant-client)
├── requirements-extras.txt   # LlamaIndex + OpenAI Agents SDK demos
├── .env.example
└── README.md
```

## Notes

- First run downloads the embedding model (`all-MiniLM-L6-v2`, ~80 MB) once. For multilingual CVs/JDs (English + Chinese, etc.), switch `EMBED_MODEL` in `.env` to `intfloat/multilingual-e5-small`.
- The LLM call is the only network dependency; embeddings and search are fully offline.
- No data leaves your machine except the text you send to the LLM for analysis.
