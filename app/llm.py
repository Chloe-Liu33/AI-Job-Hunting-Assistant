"""Thin wrapper over free cloud LLMs (Gemini Flash / Groq).

Every completion is automatically traced by the LLMOps layer (see
`observability.py`), so latency / token / cost / error data is captured for
the dashboard without callers having to do anything.
"""
import os
import time

import observability as obs


def detect_provider() -> str | None:
    """Decide which provider to use based on env."""
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("gemini", "groq"):
        return explicit
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return None


def current_model(provider: str | None = None) -> str:
    provider = provider or detect_provider()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return "unknown"


def provider_status() -> tuple[str | None, str]:
    """Return (provider, human-readable status message)."""
    provider = detect_provider()
    if provider == "gemini":
        if os.getenv("GEMINI_API_KEY"):
            return "gemini", f"Gemini · {current_model('gemini')}"
        return None, "LLM_PROVIDER=gemini but GEMINI_API_KEY is missing."
    if provider == "groq":
        if os.getenv("GROQ_API_KEY"):
            return "groq", f"Groq · {current_model('groq')}"
        return None, "LLM_PROVIDER=groq but GROQ_API_KEY is missing."
    return None, "No API key found. Add GEMINI_API_KEY or GROQ_API_KEY to your .env."


# ---- OpenAI-compatible config ------------------------------------------------
# Both Groq and Gemini expose OpenAI-compatible endpoints. We surface that here
# so frameworks built around the OpenAI SDK (e.g. the OpenAI Agents SDK) can run
# on the SAME free keys without a paid OpenAI account.
def openai_compatible_config(provider: str | None = None) -> dict:
    """Return {base_url, api_key, model} for an OpenAI-compatible client."""
    provider = provider or detect_provider()
    if provider == "groq":
        return {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.environ.get("GROQ_API_KEY", ""),
            "model": current_model("groq"),
        }
    if provider == "gemini":
        return {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "model": current_model("gemini"),
        }
    raise RuntimeError("No provider configured for an OpenAI-compatible client.")


def _complete_gemini(prompt: str, system: str | None) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_name = current_model("gemini")
    model = genai.GenerativeModel(model_name, system_instruction=system)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def _complete_groq(prompt: str, system: str | None) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model_name = current_model("groq")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.4
    )
    return (resp.choices[0].message.content or "").strip()


def complete(prompt: str, system: str | None = None) -> str:
    """Send a prompt to whichever provider is configured, with LLMOps tracing."""
    provider = detect_provider()
    if provider not in ("gemini", "groq"):
        raise RuntimeError(
            "No LLM provider configured. Set GEMINI_API_KEY or GROQ_API_KEY in your .env."
        )

    model = current_model(provider)
    start = time.perf_counter()
    output, error = "", None
    try:
        output = _complete_gemini(prompt, system) if provider == "gemini" else _complete_groq(prompt, system)
        return output
    except Exception as e:  # noqa: BLE001 — record the failure, then re-raise
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        obs.record_llm_call(
            provider=provider,
            model=model,
            prompt=prompt,
            system=system,
            output=output,
            latency_ms=(time.perf_counter() - start) * 1000,
            error=error,
        )
