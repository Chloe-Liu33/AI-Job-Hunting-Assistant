# 求职 Agent · Job-Hunting Agent

A **zero-cost, no-GPU** job-hunting assistant. Match your CV against job descriptions, get tailored fit analysis, generate cover-letter drafts, and surface gaps — all running on free cloud LLMs + local CPU embeddings.

## Stack (the "no-GPU free stack")

| Layer | Tool | Where it runs |
|---|---|---|
| **Brain (LLM)** | Gemini Flash *or* Groq | Free cloud tier |
| **Retrieval (RAG)** | sentence-transformers + FAISS | Local CPU |
| **UI** | Streamlit | Local CPU |

Everything is CPU + free. Deploys to Hugging Face Spaces' free CPU tier (2 vCPU + 16 GB) without a paid GPU Space.

## Quick start

```bash
# 1. Clone & install
git clone <your-repo-url>
cd job-agent
pip install -r requirements.txt

# 2. Add your free API key (pick ONE)
cp .env.example .env
# then edit .env and paste your key

# 3. Drop your files in
#   data/cv/  <- your CV(s) as .pdf, .docx, .md or .txt
#   data/jd/  <- job descriptions as .txt or .md (one per file)

# 4. Run
streamlit run app/main.py
```

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

## Deploy to Hugging Face Spaces (free)

1. Create a new **Streamlit** Space (CPU basic — free).
2. Push this repo to it.
3. In Space **Settings → Variables and secrets**, add your `GEMINI_API_KEY` (or `GROQ_API_KEY`).
4. Done. No GPU needed.

## Project layout

```
job-agent/
├── app/
│   ├── main.py          # Streamlit UI
│   ├── llm.py           # Cloud LLM wrapper (Gemini / Groq)
│   ├── rag.py           # Local CPU embeddings + FAISS
│   ├── loaders.py       # Read PDF / DOCX / TXT / MD
│   └── prompts.py       # Prompt templates
├── data/
│   ├── cv/              # your CVs
│   ├── jd/              # job descriptions
│   └── vectorstore/     # auto-generated FAISS index
├── .streamlit/config.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- First run downloads the embedding model (`all-MiniLM-L6-v2`, ~80 MB) once. For multilingual CVs/JDs (English + 中文), switch `EMBED_MODEL` in `.env` to `intfloat/multilingual-e5-small`.
- The LLM call is the only network dependency; embeddings and search are fully offline.
- No data leaves your machine except the text you send to the LLM for analysis.
