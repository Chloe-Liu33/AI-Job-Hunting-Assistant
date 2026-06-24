"""Storage abstraction: accounts + documents + vectors, two interchangeable
backends behind one API.

- **file** (default): accounts in a JSON file, documents as files on disk, and a
  per-user FAISS index — all under DATA_ROOT. Zero extra dependencies; great for
  local dev. Nothing survives an ephemeral container restart.
- **qdrant** (when `QDRANT_URL` is set): accounts and document chunks live in a
  Qdrant vector database. Everything is external and permanent, with per-user
  isolation enforced by a `user_id` payload filter on every query.

Embeddings are ALWAYS computed locally on CPU (`rag.embed`) — free, no GPU. The
backend only stores and searches the vectors plus the chunk text + metadata.

Shared seed JDs (the read-only library shipped in the repo at data/jd/) are
exposed to every user by both backends; in Qdrant they're stored once under the
reserved `_shared` user id.
"""
from __future__ import annotations

import json
import os

import paths
import rag
from loaders import chunk_text, load_dir, read_bytes

SHARED_USER = "_shared"
CHUNKS = "chunks"
ACCOUNTS = "accounts"


def backend() -> str:
    return "qdrant" if os.getenv("QDRANT_URL") else "file"


def _shared_jds() -> list[dict]:
    """The read-only seed JDs shipped with the code (always from the repo)."""
    return load_dir(paths.REPO_DATA / "jd")


def _shared_jd_text() -> dict[str, str]:
    return {d["name"]: d["text"] for d in _shared_jds()}


def format_hits(hits: list[dict]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    return "\n\n".join(
        f"[{i}] (source: {h['kind']}/{h['name']})\n{h['chunk']}"
        for i, h in enumerate(hits, 1)
    )


# ============================ FILE BACKEND ============================
class FileBackend:
    @property
    def users_file(self):
        return paths.DATA_ROOT / "users.json"

    def _udir(self, user_id):
        return paths.DATA_ROOT / "users" / user_id

    def _dir(self, user_id, kind):
        return self._udir(user_id) / kind

    def _vstore(self, user_id):
        return self._udir(user_id) / "vectorstore"

    # ---- accounts ----
    def load_users(self) -> dict:
        if not self.users_file.exists():
            return {}
        try:
            return json.loads(self.users_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save_users(self, users: dict) -> None:
        paths.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self.users_file.write_text(
            json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- documents ----
    def add_document(self, user_id, kind, name, data: bytes) -> None:
        d = self._dir(user_id, kind)
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(bytes(data))
        self._rebuild(user_id)

    def delete_document(self, user_id, kind, name) -> None:
        f = self._dir(user_id, kind) / name
        if f.exists():
            f.unlink()
        self._rebuild(user_id)

    def list_documents(self, user_id, kind) -> list[dict]:
        own = [{"name": d["name"], "shared": False} for d in load_dir(self._dir(user_id, kind))]
        if kind == "jd":
            own_names = {d["name"] for d in own}
            shared = [
                {"name": n, "shared": True}
                for n in _shared_jd_text()
                if n not in own_names
            ]
            return shared + own
        return own

    def get_combined_text(self, user_id, kind, names=None) -> str:
        text_by_name = {d["name"]: d["text"] for d in load_dir(self._dir(user_id, kind))}
        if kind == "jd":
            for n, t in _shared_jd_text().items():
                text_by_name.setdefault(n, t)
        if names is None:
            selected = list(text_by_name.values())
        else:
            selected = [text_by_name[n] for n in names if n in text_by_name]
        return "\n".join(selected)

    def _rebuild(self, user_id) -> None:
        cv_docs = load_dir(self._dir(user_id, "cv"))
        jd_own = load_dir(self._dir(user_id, "jd"))
        jd_names = {d["name"] for d in jd_own}
        jd_docs = jd_own + [d for d in _shared_jds() if d["name"] not in jd_names]
        if not cv_docs and not jd_docs:
            return
        rag.build_index(cv_docs, jd_docs, vectorstore_dir=self._vstore(user_id))

    def search(self, user_id, query, k=6, kind=None, name=None) -> list[dict]:
        return rag.search(query, k=k, kind=kind, name=name, vectorstore_dir=self._vstore(user_id))

    def has_index(self, user_id) -> bool:
        return rag.index_exists(self._vstore(user_id))

    def reindex(self, user_id) -> dict:
        self._rebuild(user_id)
        return {"backend": "file"}


# ============================ QDRANT BACKEND ============================
class QdrantBackend:
    def __init__(self):
        self.url = os.environ["QDRANT_URL"]
        self.api_key = os.getenv("QDRANT_API_KEY")
        self._client_ = None
        self._dim = None
        self._accounts_ready = False
        self._chunks_ready = False
        self._shared_ready = False

    # ---- lazy client + collections (created on demand, idempotently) ----
    def _client(self):
        if self._client_ is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise ImportError(
                    "qdrant-client not installed. Run: pip install qdrant-client"
                ) from e
            self._client_ = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        return self._client_

    def _has_collection(self, name) -> bool:
        # get_collections works across all client versions (collection_exists is newer).
        return name in {c.name for c in self._client().get_collections().collections}

    def _dimension(self) -> int:
        if self._dim is None:
            self._dim = int(rag.embed(["probe"], is_query=False).shape[1])
        return self._dim

    def _ensure_accounts(self):
        """Create the accounts collection if missing. No embedding needed, so
        registering / logging in never downloads the embedding model."""
        if self._accounts_ready:
            return
        from qdrant_client import models

        if not self._has_collection(ACCOUNTS):
            self._client().create_collection(
                collection_name=ACCOUNTS,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            )
        self._accounts_ready = True

    def _ensure_chunks(self):
        """Create the chunks collection if missing (needs the embedding dim)."""
        if self._chunks_ready:
            return
        from qdrant_client import models

        if not self._has_collection(CHUNKS):
            self._client().create_collection(
                collection_name=CHUNKS,
                vectors_config=models.VectorParams(
                    size=self._dimension(), distance=models.Distance.COSINE
                ),
            )
            # Indexed payload keys make user_id/kind/name filters fast. Best-effort.
            for key in ("user_id", "kind", "name"):
                try:
                    self._client().create_payload_index(
                        CHUNKS, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    pass
        self._chunks_ready = True

    def _ensure_shared(self):
        """Upsert the repo's seed JDs once under the reserved _shared user id."""
        if self._shared_ready:
            return
        self._ensure_chunks()
        for d in _shared_jds():
            self._upsert_doc(SHARED_USER, "jd", d["name"], d["text"])
        self._shared_ready = True

    def _point_id(self, *parts) -> str:
        from uuid import NAMESPACE_URL, uuid5

        return str(uuid5(NAMESPACE_URL, "|".join(str(p) for p in parts)))

    # ---- accounts ----
    def load_users(self) -> dict:
        self._ensure_accounts()
        client = self._client()
        users = {}
        offset = None
        while True:
            points, offset = client.scroll(
                ACCOUNTS, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                rec = p.payload or {}
                if rec.get("username"):
                    users[rec["username"].lower()] = rec
            if offset is None:
                break
        return users

    def save_users(self, users: dict) -> None:
        from qdrant_client import models

        self._ensure_accounts()
        client = self._client()
        points = [
            models.PointStruct(
                id=self._point_id("account", key),
                vector=[0.0],
                payload={**rec, "username": rec.get("username", key)},
            )
            for key, rec in users.items()
        ]
        if points:
            client.upsert(ACCOUNTS, points=points)

    # ---- documents ----
    def _upsert_doc(self, user_id, kind, name, text):
        from qdrant_client import models

        self._ensure_chunks()
        self.delete_document(user_id, kind, name)  # replace any prior version
        chunks = chunk_text(text)
        if not chunks:
            return
        vecs = rag.embed(chunks, is_query=False)
        points = [
            models.PointStruct(
                id=self._point_id(user_id, kind, name, i),
                vector=vecs[i].tolist(),
                payload={"user_id": user_id, "kind": kind, "name": name, "idx": i, "chunk": c},
            )
            for i, c in enumerate(chunks)
        ]
        self._client().upsert(CHUNKS, points=points)

    def add_document(self, user_id, kind, name, data: bytes) -> None:
        text = read_bytes(name, bytes(data))
        self._upsert_doc(user_id, kind, name, text)

    def delete_document(self, user_id, kind, name) -> None:
        from qdrant_client import models

        client = self._client()
        if not self._has_collection(CHUNKS):
            return
        client.delete(
            CHUNKS,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                        models.FieldCondition(key="kind", match=models.MatchValue(value=kind)),
                        models.FieldCondition(key="name", match=models.MatchValue(value=name)),
                    ]
                )
            ),
        )

    def _distinct_names(self, user_id, kind) -> list[str]:
        from qdrant_client import models

        client = self._client()
        if not self._has_collection(CHUNKS):
            return []
        names, offset = set(), None
        flt = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind)),
            ]
        )
        while True:
            points, offset = client.scroll(
                CHUNKS, scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                names.add(p.payload["name"])
            if offset is None:
                break
        return sorted(names)

    def list_documents(self, user_id, kind) -> list[dict]:
        own = [{"name": n, "shared": False} for n in self._distinct_names(user_id, kind)]
        if kind == "jd":
            own_names = {d["name"] for d in own}
            shared = [
                {"name": n, "shared": True} for n in _shared_jd_text() if n not in own_names
            ]
            return shared + own
        return own

    def _doc_text(self, user_id, kind, name) -> str:
        from qdrant_client import models

        client = self._client()
        if not self._has_collection(CHUNKS):
            return ""
        flt = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind)),
                models.FieldCondition(key="name", match=models.MatchValue(value=name)),
            ]
        )
        rows, offset = [], None
        while True:
            points, offset = client.scroll(
                CHUNKS, scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            rows.extend(points)
            if offset is None:
                break
        rows.sort(key=lambda p: p.payload.get("idx", 0))
        return "\n".join(p.payload["chunk"] for p in rows)

    def get_combined_text(self, user_id, kind, names=None) -> str:
        shared = _shared_jd_text() if kind == "jd" else {}
        if names is None:
            names = [d["name"] for d in self.list_documents(user_id, kind)]
        out = []
        for n in names:
            if n in shared:
                out.append(shared[n])
            else:
                out.append(self._doc_text(user_id, kind, n))
        return "\n".join(t for t in out if t)

    def search(self, user_id, query, k=6, kind=None, name=None) -> list[dict]:
        from qdrant_client import models

        if kind in ("jd", None):
            self._ensure_shared()
        else:
            self._ensure_chunks()
        client = self._client()
        if not self._has_collection(CHUNKS):
            return []
        qv = rag.embed([query], is_query=True)[0].tolist()
        must = []
        # JD search spans the user's own JDs plus the shared seed library.
        uids = [user_id, SHARED_USER] if kind == "jd" else [user_id]
        must.append(models.FieldCondition(key="user_id", match=models.MatchAny(any=uids)))
        if kind:
            must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
        if name:
            must.append(models.FieldCondition(key="name", match=models.MatchValue(value=name)))
        res = client.search(
            CHUNKS, query_vector=qv, query_filter=models.Filter(must=must), limit=k
        )
        return [
            {
                "kind": p.payload["kind"],
                "name": p.payload["name"],
                "chunk": p.payload["chunk"],
                "score": float(p.score),
            }
            for p in res
        ]

    def has_index(self, user_id) -> bool:
        from qdrant_client import models

        client = self._client()
        if not self._has_collection(CHUNKS):
            return False
        points, _ = client.scroll(
            CHUNKS,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0

    def reindex(self, user_id) -> dict:
        # Re-embed every stored chunk for this user (e.g. after an EMBED_MODEL change).
        from qdrant_client import models

        client = self._client()
        if not self._has_collection(CHUNKS):
            return {"backend": "qdrant", "reembedded": 0}
        offset, n = None, 0
        flt = models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        )
        while True:
            points, offset = client.scroll(
                CHUNKS, scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            if points:
                vecs = rag.embed([p.payload["chunk"] for p in points], is_query=False)
                client.upsert(
                    CHUNKS,
                    points=[
                        models.PointStruct(id=p.id, vector=vecs[i].tolist(), payload=p.payload)
                        for i, p in enumerate(points)
                    ],
                )
                n += len(points)
            if offset is None:
                break
        return {"backend": "qdrant", "reembedded": n}


# ============================ dispatch ============================
_file = None
_qdrant = None


def _active():
    global _file, _qdrant
    if backend() == "qdrant":
        if _qdrant is None:
            _qdrant = QdrantBackend()
        return _qdrant
    if _file is None:
        _file = FileBackend()
    return _file


def load_users() -> dict:
    return _active().load_users()


def save_users(users: dict) -> None:
    _active().save_users(users)


def add_document(user_id, kind, name, data) -> None:
    _active().add_document(user_id, kind, name, data)


def delete_document(user_id, kind, name) -> None:
    _active().delete_document(user_id, kind, name)


def list_documents(user_id, kind) -> list[dict]:
    return _active().list_documents(user_id, kind)


def get_combined_text(user_id, kind, names=None) -> str:
    return _active().get_combined_text(user_id, kind, names)


def search(user_id, query, k=6, kind=None, name=None) -> list[dict]:
    return _active().search(user_id, query, k=k, kind=kind, name=name)


def retrieve_context(user_id, query, k=6, kind=None, name=None):
    hits = search(user_id, query, k=k, kind=kind, name=name)
    return format_hits(hits), hits


def has_index(user_id) -> bool:
    return _active().has_index(user_id)


def reindex(user_id) -> dict:
    return _active().reindex(user_id)


def rank_jobs(user_id, cv_names=None) -> list[dict]:
    """Rank all JDs (own + shared) by similarity to the chosen CV(s)."""
    import numpy as np

    cv_text = get_combined_text(user_id, "cv", cv_names)
    if not cv_text.strip():
        return []
    cv_vec = rag.embed([cv_text], is_query=True)
    out = []
    for jd in list_documents(user_id, "jd"):
        text = get_combined_text(user_id, "jd", [jd["name"]])
        chunks = chunk_text(text)
        if not chunks:
            continue
        jd_vecs = rag.embed(chunks, is_query=False)
        sims = (jd_vecs @ cv_vec.T).ravel()
        out.append({"name": jd["name"], "score": float(np.mean(sims)), "shared": jd["shared"]})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out
