"""LLMOps: lightweight, dependency-free observability for every LLM call.

This is the "ops" layer recruiters mean by *LLMOps*: instead of firing LLM
calls into the void, we record a structured trace for each one — provider,
model, latency, token estimate, cost estimate, success/error, and a hash of
the prompt — to an append-only JSONL file. We can then aggregate it (avg
latency, total spend, per-model breakdown) and render it in the UI.

Design choices that keep it free + portable:
- No external tracing SaaS, no tiktoken. Token count is a cheap heuristic
  (~4 chars/token) which is good enough for cost dashboards on free tiers.
- Append-only JSONL so traces survive restarts and are trivially greppable.
- A `span()` context manager + `@traced` decorator so *any* function (agent
  step, benchmark run, RAG retrieval) can be instrumented, not just `complete`.
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
from contextlib import contextmanager

import paths

TRACES_DIR = paths.DATA_ROOT / "traces"
TRACES_PATH = TRACES_DIR / "llm_traces.jsonl"

# Nominal prices (USD per 1M tokens). Gemini Flash / Groq have generous *free*
# tiers, so real cost is ~0 — but we keep a price table so the dashboard shows
# what the same traffic *would* cost on a paid plan. Update as needed.
PRICING = {
    # model substring -> (input_per_1m, output_per_1m)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.5-flash": (0.30, 2.50),
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "_default": (0.20, 0.60),
}


def estimate_tokens(text: str | None) -> int:
    """Cheap token estimate: ~4 chars/token. No tokenizer dependency."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _price_for(model: str) -> tuple[float, float]:
    model = (model or "").lower()
    for key, price in PRICING.items():
        if key != "_default" and key in model:
            return price
    return PRICING["_default"]


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    in_p, out_p = _price_for(model)
    return (in_tokens / 1_000_000) * in_p + (out_tokens / 1_000_000) * out_p


def prompt_hash(text: str | None) -> str:
    """Short stable hash — lets you group/version identical prompts in traces."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def record(event: dict) -> None:
    """Append one trace event as a JSON line. Never raises (best-effort)."""
    try:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRACES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:  # observability must never break the app
        pass


def record_llm_call(
    *,
    provider: str,
    model: str,
    prompt: str,
    system: str | None,
    output: str,
    latency_ms: float,
    error: str | None = None,
    kind: str = "complete",
) -> dict:
    """Build, persist, and return a trace event for a single LLM completion."""
    in_tokens = estimate_tokens(prompt) + estimate_tokens(system)
    out_tokens = estimate_tokens(output)
    event = {
        "ts": time.time(),
        "kind": kind,
        "provider": provider,
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "est_cost_usd": round(estimate_cost(model, in_tokens, out_tokens), 6),
        "prompt_hash": prompt_hash(prompt),
        "ok": error is None,
        "error": error,
    }
    record(event)
    return event


@contextmanager
def span(name: str, **fields):
    """Time an arbitrary block and record it as a trace span.

    Usage:
        with span("agent.run", question=q):
            ...
    """
    start = time.perf_counter()
    err = None
    try:
        yield
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        record(
            {
                "ts": time.time(),
                "kind": "span",
                "name": name,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "ok": err is None,
                "error": err,
                **fields,
            }
        )


def traced(name: str | None = None):
    """Decorator that records a span around any function call."""

    def deco(fn):
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def load_traces(limit: int | None = None) -> list[dict]:
    """Read all (or the last `limit`) trace events from disk, newest last."""
    if not TRACES_PATH.exists():
        return []
    rows = []
    with open(TRACES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


def summary() -> dict:
    """Aggregate stats across all recorded LLM completions (for the dashboard)."""
    rows = [r for r in load_traces() if r.get("kind") == "complete"]
    if not rows:
        return {"calls": 0}
    total_tokens = sum(r.get("total_tokens", 0) for r in rows)
    total_cost = sum(r.get("est_cost_usd", 0.0) for r in rows)
    latencies = [r.get("latency_ms", 0.0) for r in rows]
    errors = sum(1 for r in rows if not r.get("ok", True))
    by_model: dict[str, int] = {}
    for r in rows:
        by_model[r.get("model", "?")] = by_model.get(r.get("model", "?"), 0) + 1
    return {
        "calls": len(rows),
        "errors": errors,
        "total_tokens": total_tokens,
        "est_cost_usd": round(total_cost, 6),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "by_model": by_model,
    }
