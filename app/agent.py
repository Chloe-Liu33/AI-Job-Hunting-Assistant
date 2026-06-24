"""A LangChain tool-calling agent for the job hunt.

This is the "agent" piece North-Health-style JDs ask for. Instead of one fixed
LLM call, the model is given TOOLS and decides, step by step, which to call:

    search_cv   -> RAG over the candidate's own CV
    search_jd   -> RAG over the job descriptions
    rank_jobs   -> which JDs best match the CV (semantic similarity)

So a question like "Which job fits me best and what are my two weakest gaps for
it?" makes the agent (1) call rank_jobs, (2) search the winning JD and the CV,
(3) reason over the results — a genuine ReAct / tool-calling loop, not a
template.

It runs on the SAME free keys as the rest of the app: LangChain's
`ChatGoogleGenerativeAI` (Gemini) or `ChatGroq` (Groq). LangChain ships in the
base requirements; if missing, install with:

    pip install -r requirements.txt
"""
from __future__ import annotations

import os

import observability as obs
import store


# ---- Tools (plain functions; wrapped as LangChain tools below) ---------------
# Each takes the logged-in `user_id` so the agent only ever reads that user's
# data — isolation is enforced by the storage backend (dirs or Qdrant filter).
def _tool_search_cv(query: str, user_id=None) -> str:
    """Search the candidate's CV for passages relevant to `query`."""
    ctx, _ = store.retrieve_context(user_id, query, k=5, kind="cv")
    return ctx or "No CV uploaded yet."


def _tool_search_jd(query: str, user_id=None) -> str:
    """Search the job descriptions for passages relevant to `query`."""
    ctx, _ = store.retrieve_context(user_id, query, k=5, kind="jd")
    return ctx or "No JDs available yet."


def _tool_rank_jobs(_: str = "", user_id=None) -> str:
    """Rank all jobs by semantic similarity to the CV. Input ignored."""
    ranked = store.rank_jobs(user_id)
    if not ranked:
        return "No CV uploaded yet."
    return "\n".join(
        f"{i}. {r['name']} — similarity {round(r['score'] * 100, 1)}%"
        for i, r in enumerate(ranked, 1)
    )


def _build_llm():
    """Return a LangChain chat model on the configured free provider."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "groq" if os.getenv("GROQ_API_KEY") else ""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.2,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
        )
    raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY or GROQ_API_KEY.")


SYSTEM = (
    "You are a job-application agent. Use the tools to gather evidence from the "
    "candidate's CV and the job descriptions before answering. Never invent "
    "experience that the tools don't surface. Cite which tool/source backs each "
    "claim. Be concise and concrete."
)


def build_agent(user_id=None):
    """Construct a tool-calling agent bound to one user. Supports LangChain 1.x
    (`create_agent`) and falls back to 0.3 (`AgentExecutor`). Returns
    (api_version, agent). Raises ImportError if LangChain isn't installed."""
    try:
        from langchain_core.tools import tool
    except ImportError as e:
        raise ImportError(
            "LangChain not installed. Run: pip install -r requirements.txt"
        ) from e

    llm = _build_llm()

    # Define tools as closures over user_id. They must be real functions (not
    # functools.partial) — LangChain 1.x introspects them with get_type_hints.
    @tool
    def search_cv(query: str) -> str:
        """Search the candidate's CV for passages relevant to the query."""
        return _tool_search_cv(query, user_id=user_id)

    @tool
    def search_jd(query: str) -> str:
        """Search the job descriptions for passages relevant to the query."""
        return _tool_search_jd(query, user_id=user_id)

    @tool
    def rank_jobs(query: str = "") -> str:
        """Rank all jobs by similarity to the candidate's CV. The input is ignored."""
        return _tool_rank_jobs(query, user_id=user_id)

    tools = [search_cv, search_jd, rank_jobs]

    # LangChain 1.x: create_agent returns a compiled (langgraph) runnable.
    try:
        from langchain.agents import create_agent
        return "v1", create_agent(llm, tools, system_prompt=SYSTEM)
    except ImportError:
        pass

    # LangChain 0.3.x fallback.
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as e:
        raise ImportError(
            "Unsupported LangChain version (need create_agent or AgentExecutor). "
            "Try: pip install -U langchain"
        ) from e
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return "v0", AgentExecutor(agent=agent, tools=tools, verbose=False,
                               return_intermediate_steps=True, max_iterations=6)


def _parse_v1(result) -> dict:
    """Pull the final answer + tool calls out of a LangChain 1.x message list."""
    msgs = result.get("messages", []) if isinstance(result, dict) else []
    answer = ""
    for m in reversed(msgs):
        c = getattr(m, "content", None)
        if isinstance(c, str) and c.strip():
            answer = c
            break
        if isinstance(c, list):  # some models return content as a list of parts
            text = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            if text.strip():
                answer = text
                break
    obs_by_id = {
        getattr(m, "tool_call_id", ""): str(getattr(m, "content", ""))[:500]
        for m in msgs
        if type(m).__name__ == "ToolMessage"
    }
    steps = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            get = tc.get if isinstance(tc, dict) else lambda k, d=None: getattr(tc, k, d)
            steps.append(
                {
                    "tool": get("name", "?"),
                    "input": get("args", ""),
                    "observation": obs_by_id.get(get("id", ""), ""),
                }
            )
    return {"answer": answer, "steps": steps}


def _parse_v0(result) -> dict:
    steps = [
        {
            "tool": getattr(action, "tool", "?"),
            "input": getattr(action, "tool_input", ""),
            "observation": str(observation)[:500],
        }
        for action, observation in result.get("intermediate_steps", [])
    ]
    return {"answer": result.get("output", ""), "steps": steps}


def run_agent(question: str, user_id=None) -> dict:
    """Run the agent on a question. Returns {answer, steps}. Traced for LLMOps."""
    with obs.span("agent.run", question=question[:200]):
        api, agent = build_agent(user_id=user_id)
        if api == "v1":
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            return _parse_v1(result)
        result = agent.invoke({"input": question})
        return _parse_v0(result)
