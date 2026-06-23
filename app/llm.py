"""Thin wrapper over free cloud LLMs (Gemini Flash / Groq)."""
import os


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


def provider_status() -> tuple[str | None, str]:
    """Return (provider, human-readable status message)."""
    provider = detect_provider()
    if provider == "gemini":
        if os.getenv("GEMINI_API_KEY"):
            return "gemini", f"Gemini · {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}"
        return None, "LLM_PROVIDER=gemini but GEMINI_API_KEY is missing."
    if provider == "groq":
        if os.getenv("GROQ_API_KEY"):
            return "groq", f"Groq · {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}"
        return None, "LLM_PROVIDER=groq but GROQ_API_KEY is missing."
    return None, "No API key found. Add GEMINI_API_KEY or GROQ_API_KEY to your .env."


def _complete_gemini(prompt: str, system: str | None) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name, system_instruction=system)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def _complete_groq(prompt: str, system: str | None) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.4
    )
    return (resp.choices[0].message.content or "").strip()


def complete(prompt: str, system: str | None = None) -> str:
    """Send a prompt to whichever provider is configured."""
    provider = detect_provider()
    if provider == "gemini":
        return _complete_gemini(prompt, system)
    if provider == "groq":
        return _complete_groq(prompt, system)
    raise RuntimeError(
        "No LLM provider configured. Set GEMINI_API_KEY or GROQ_API_KEY in your .env."
    )
