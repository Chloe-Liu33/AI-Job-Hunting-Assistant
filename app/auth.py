"""User registration / login. Accounts persist via the storage backend
(`store.py`) — a local JSON file by default, or Qdrant when `QDRANT_URL` is set.

Passwords are stored only as salted PBKDF2-SHA256 hashes (stdlib `hashlib`).
Per-user data isolation is handled by `store` (file dirs or a `user_id` payload
filter in Qdrant); auth just owns the credential check and the user id.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time

import store

_PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return dk.hex()


def _user_id(username: str) -> str:
    """A stable, filesystem/collection-safe id derived from the username."""
    slug = re.sub(r"[^a-z0-9_-]+", "_", username.strip().lower()).strip("_")
    return slug or "user"


def user_exists(username: str) -> bool:
    return username.strip().lower() in store.load_users()


def register(username: str, password: str) -> tuple[bool, str]:
    """Create a new user. Returns (ok, message)."""
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if _user_id(username) == store.SHARED_USER:
        return False, "That username is reserved."
    users = store.load_users()
    key = username.lower()
    if key in users:
        return False, "That username is already taken."
    salt = secrets.token_hex(16)
    users[key] = {
        "username": username,
        "salt": salt,
        "hash": _hash_password(password, salt),
        "user_id": _user_id(username),
        "created": time.time(),
    }
    store.save_users(users)
    return True, "Account created. You can log in now."


def verify(username: str, password: str) -> bool:
    """Constant-time check of username + password."""
    user = store.load_users().get(username.strip().lower())
    if not user:
        return False
    candidate = _hash_password(password, user["salt"])
    return hmac.compare_digest(user["hash"], candidate)


def get_user_id(username: str) -> str:
    user = store.load_users().get(username.strip().lower())
    return user["user_id"] if user else _user_id(username)
