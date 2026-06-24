"""Central path config so writable data can move to a persistent mount.

Two roots, kept deliberately separate:

- REPO_DATA: files SHIPPED with the code (the sample CV, the shared seed JDs).
  Always inside the repo; treated as read-only at runtime.
- DATA_ROOT: everything the app WRITES at runtime (accounts, per-user CVs/JDs,
  FAISS indexes, LLMOps traces). Defaults to the repo's data/, but set the
  DATA_DIR env var to redirect it onto a disk that survives restarts —
  e.g. on Hugging Face persistent storage:  DATA_DIR=/data
"""
import os
from pathlib import Path

REPO_DATA = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else REPO_DATA
