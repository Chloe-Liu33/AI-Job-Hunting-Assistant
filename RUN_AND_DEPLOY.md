# Run & Deploy Guide

> ⚠️ **Work in progress — not officially launched.** Still being tested and
> organized; things may change or break.

Zero-cost, no GPU. There are two paths — pick what you need:

- **A. Run locally** (sections 1–7): run it on your own machine; best for
  development / debugging / trying it out first.
- **B. Deploy to Hugging Face** (sections 8–10): put a public demo online and
  share the link.

Both paths **share the same configuration** (`.env` ↔ HF Secrets) and **the same
Qdrant**, so an account / CV you create locally also shows up online (shared
data). Requires **Python 3.10+**.

> Live demo: https://limei-liu-agent-job-search.hf.space

---

# A. Run locally

## 1. Install dependencies

```bash
git clone https://github.com/Chloe-Liu33/AI-Job-Hunting-Assistant.git
cd AI-Job-Hunting-Assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The base install already covers the in-app **🤖 Agent** tab (LangChain). The
> two *alternative* framework demos (LlamaIndex, OpenAI Agents SDK) and the
> benchmark only need the optional extras:
> `pip install -r requirements-extras.txt` (the main app runs without them).

## 2. Configure the API key and (optionally) Qdrant

### 2.1 Create `.env`

```bash
cp .env.example .env
```

### 2.2 Open `.env`

`.env` is a **hidden file** (its name starts with `.`). Open it with any of:

```bash
code .env      # VS Code (recommended)
# or
open -e .env   # macOS TextEdit
# or
nano .env      # edit in the terminal — Ctrl+O to save, Ctrl+X to exit
```

> VS Code's Explorer shows `.env` by default; in Finder press `Cmd+Shift+.` to
> reveal hidden files.

### 2.3 Fill in these values

| Variable | Required? | What to put |
|---|---|---|
| `GEMINI_API_KEY` | **Required** (pick one) | key from https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | pick one | key from https://console.groq.com/keys (remove the leading `#` to enable the line) |
| `QDRANT_URL` | only for cloud persistence | e.g. `https://xxx.cloud.qdrant.io:6333` (**must end with `:6333`**) |
| `QDRANT_API_KEY` | same as above | key from the Qdrant console |

**Formatting gotchas** (the most common mistakes):

- **No spaces** around the `=`: `GEMINI_API_KEY=AbC123` (not `GEMINI_API_KEY = AbC123`).
- **Do not** wrap the value in quotes.
- A leading `#` means the line is a comment = inactive; **delete the `#`** to enable it.
- Fill **one** of `GEMINI` / `GROQ`; the app auto-detects it (no need to set `LLM_PROVIDER`).
- Leave `QDRANT_URL` unset → local file mode; set it → cloud Qdrant.

### 2.4 Verify the values are actually read

After editing, run this to confirm the keys/URL loaded (keys are masked):

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
mask=lambda v:(v[:6]+'…'+v[-4:]) if v and len(v)>12 else (v or '(empty)')
print('GEMINI_API_KEY:', mask(os.getenv('GEMINI_API_KEY')))
print('GROQ_API_KEY  :', mask(os.getenv('GROQ_API_KEY')))
print('QDRANT_URL    :', os.getenv('QDRANT_URL') or '(empty)')
print('QDRANT_API_KEY:', mask(os.getenv('QDRANT_API_KEY')))
"
```

- `GEMINI_API_KEY` (or `GROQ_API_KEY`) not `(empty)` = the LLM will work.
- For cloud persistence, `QDRANT_URL` should print the full address **ending in `:6333`**.
- If something that should be set shows `(empty)`: usually a leftover `#`, a space around `=`, or you didn't save the file.

## 3. Launch

```bash
streamlit run app/main.py
```

The browser opens `http://localhost:8501` automatically.

## 4. Register and upload

1. On first open, **Register** an account, then **Log in**.
   - Each account's CVs / JDs / index are isolated — you only ever see your own.
2. Once logged in, **upload from the left sidebar**:
   - **Add CV** (PDF recommended; docx/md/txt also work) → click **Save CV(s)**
   - **Add JD** → click **Save JD(s)**
   - In the file list, 🗑 deletes your own; 🔒 marks a shared sample JD (read-only).

> **Isolation:** CVs are fully private, only in your own workspace. JDs are
> **hybrid** — everyone sees the shared samples (e.g. Northern Health, RCH) plus
> their own uploads; a same-named upload overrides the shared one. Passwords are
> stored only as salted hashes. Where the data lives depends on the backend (see
> "Persistence" below).

## 5. How to use it

1. Uploads are **embedded into the index automatically** (the first time
   downloads the ~80 MB embedding model, once). You only need **🔁 Rebuild
   index** if you change the embedding model.
2. With several CVs, use the sidebar **Active CV** to pick which one to match with.
3. The tabs:
   - **📊 Rank JDs** — which jobs fit you best
   - **🔍 Fit analysis** — deep dive on one role (turn on **Use RAG** to ground it on retrieved passages with citations)
   - **✉️ Cover letter** — generate a tailored cover letter
   - **🔎 Semantic search** — search across your CVs + JDs
   - **🤖 Agent** — a tool-calling agent
   - **📈 LLMOps** — per-call latency / tokens / cost

## 6. (Optional) Run the prompt benchmark

```bash
python app/benchmark.py            # full
python app/benchmark.py --quick    # quick version
```

Results are written to `eval/results/benchmark.md`.

## 7. Persistence (where data lives)

The storage backend is switchable (see [app/store.py](app/store.py)):

- **Default = local files + FAISS:** accounts/CVs/JDs/index live in `data/` (or
  the directory `DATA_DIR` points to). Fine locally; wiped on restart when
  deployed to an ephemeral container.
- **Recommended permanent storage = Qdrant:** set `QDRANT_URL`
  (+ `QDRANT_API_KEY`, free tier at https://cloud.qdrant.io) and accounts + CV/JD
  + vectors all go to an external Qdrant; survives restarts. Embeddings are still
  computed locally on CPU for free — Qdrant only stores/searches the vectors. If
  `QDRANT_URL` is unset, the app falls back to local mode automatically.

```bash
# .env (REST port is 6333)
QDRANT_URL=https://YOUR-CLUSTER.cloud.qdrant.io:6333
QDRANT_API_KEY=...
# — or, instead of Qdrant, just put local data on a persistent disk —
# DATA_DIR=/data
```

**After configuring, run a connection self-test** (doesn't touch the app; just
confirms the URL/key work):

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from qdrant_client import QdrantClient
c = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'])
print('existing collections:', [x.name for x in c.get_collections().collections])
print('✅ Qdrant connected')
"
```

- Printing `✅ Qdrant connected` means it works (an empty collection list on the first run is normal).
- timeout / connection error → check the `:6333` suffix and that the cluster is Running.
- 401/403 → wrong key or insufficient permission; regenerate it.

After launch, the **bottom of the sidebar** should show `Storage: qdrant` (not
`file`). Registering an account and uploading a CV auto-creates the `chunks` and
`accounts` collections; if the data is still there after a restart = success.

---

# B. Deploy to Hugging Face Spaces

The code is already on GitHub, and everything a Space needs (the YAML header at
the top of the README, `requirements.txt`, env-var reading) is in place —
**no code changes needed to deploy**. It's best to **get it running locally
(Part A) first** (local and HF share the same Qdrant, so the data is shared).

## 8. Create the Space and push the code

1. huggingface.co → top-right **New Space** → SDK **Streamlit**, Hardware
   **CPU basic (free)**, give it a name (e.g. `agent-job-search`).
2. From the project directory, push the code to the Space:
   ```bash
   # one-time login: paste a WRITE token from https://huggingface.co/settings/tokens
   # (answer Y to "Add token as git credential?"). Run it where huggingface_hub is installed.
   huggingface-cli login        # newer versions: hf auth login
   # add the Space remote and push
   git remote add space https://huggingface.co/spaces/<your-HF-username>/<space-name>
   git push space main          # if rejected (HF's auto README): git push space main --force
   ```

## 9. Set Secrets (important — HF does not read `.env`)

Space → **Settings → Variables and secrets**, add each:

| Name | Type | Value |
|---|---|---|
| `GROQ_API_KEY` | Secret | your Groq key |
| `QDRANT_API_KEY` | Secret | your Qdrant key |
| `QDRANT_URL` | Secret / Variable | `https://YOUR-CLUSTER...:6333` |
| `LLM_PROVIDER` | Variable | `groq` |

(Optionally add `GEMINI_API_KEY` and `GEMINI_MODEL=gemini-2.5-flash` as a
backup.) Saving triggers a rebuild: **Building → Running** (first build takes a
few minutes — it installs torch).

## 10. Verify + day-to-day

- Open the app at `https://<username>-<space-name>.hf.space` (or the **App** tab
  on the Space page). The sidebar should show `LLM: Groq` + `Storage: qdrant`.
- **Closing your browser has no effect** — it runs on HF's cloud, not your machine.
- It **sleeps after ~48 hours of inactivity** and **auto-wakes** on the next
  visit (~30s–1min). Because the data lives in Qdrant, **sleeping loses no data**.
- **Visibility** (Settings → Change visibility): Public = anyone can open the
  login page (they can't see your data — login + per-user isolation); set it to
  Private if you don't want strangers consuming your free quota.
- **Redeploy after code changes:** `git push space main` (with `--force` if needed).

> To keep "local testing" and "live production" data separate: create a second
> free Qdrant cluster and use cluster A in local `.env`, cluster B in HF Secrets.
> Sharing one is fine for now.

---

# Appendix

## Troubleshooting

### Configuration

- **Sidebar shows "No API key found":** `.env` has no LLM key, or you didn't start from the project root.
- **`.env` changes don't take effect:** you didn't restart the app; or there are spaces around `=`, quotes around the value, or a leftover `#` (comment = inactive). After editing, verify with the command in section 2.4.
- **`Storage` shows `file` instead of `qdrant`:** `QDRANT_URL` wasn't read — check `.env` is in the root, the `:6333` suffix is there, that line has no stray space, and you restarted the app.
- **Chinese CV retrieval is inaccurate:** set `EMBED_MODEL=intfloat/multilingual-e5-small` in `.env`, then click **🔁 Rebuild index**.
- **To stop:** press `Ctrl+C` in the terminal.

### LLM (errors in Fit analysis / Cover letter / Agent)

- **Gemini `429 ... limit: 0`:** this model's **free quota is literally 0** for your project (not throttling — waiting won't help).
  - Change the model: set `GEMINI_MODEL=gemini-2.5-flash` in `.env` (`gemini-2.0-flash` is often 0 on new projects).
  - Or switch to Groq (more generous): add `GROQ_API_KEY=...` and `LLM_PROVIDER=groq` to `.env`, restart.
  - To check per-model quota: sign in at https://ai.dev/rate-limit and see whether the RPM/RPD for `gemini-2.x-flash` is 0.
- **Groq `tool_use_failed` / "Failed to call a function"** (only in 🤖 Agent): Llama on Groq occasionally emits malformed tool-call JSON. The code already retries 3× and degrades gracefully; usually a re-run succeeds. If it persists: simplify the question, run again, or temporarily set `LLM_PROVIDER=gemini` for the Agent.
- **Gemini's 20/day used up** (`gemini-2.5-flash` free tier RPD=20): the Agent makes several calls per question. Switch to Groq's more generous free tier.

### Dependencies / versions

- **`LangChain not installed` but you installed it:** almost always installed into **a different Python environment**. Make sure streamlit uses the project `.venv`:
  ```bash
  source .venv/bin/activate
  which python && which streamlit          # both should be inside .venv
  python -c "import langchain; print(langchain.__version__)"
  pip install -r requirements.txt          # install inside the activated venv
  ```
- **`'QdrantClient' object has no attribute 'search'`:** old code + new `qdrant-client`. Already fixed (uses `query_points`) — `git pull` for the latest code.
- **`AttributeError: ... AgentExecutor` / other LangChain import errors:** LangChain 1.x changed the API. The code is already compatible (uses `create_agent`) — make sure you're on the latest code.
