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
`ChatGoogleGenerativeAI` (Gemini) or `ChatGroq` (Groq). Heavy deps are imported
lazily so the base app still runs without them installed — install with:

    pip install -r requirements-extras.txt
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
    """Construct a tool-calling AgentExecutor bound to one user. Raises
    ImportError with guidance if the optional LangChain packages aren't installed."""
    try:
        from functools import partial

        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import Tool
    except ImportError as e:
        raise ImportError(
            "LangChain not installed. Run: pip install -r requirements-extras.txt"
        ) from e

    llm = _build_llm()
    tools = [
        Tool(name="search_cv", func=partial(_tool_search_cv, user_id=user_id),
             description="Search the candidate's CV. Input: a search query string."),
        Tool(name="search_jd", func=partial(_tool_search_jd, user_id=user_id),
             description="Search the job descriptions. Input: a search query string."),
        Tool(name="rank_jobs", func=partial(_tool_rank_jobs, user_id=user_id),
             description="Rank all jobs by fit to the CV. Input: ignored (pass '')."),
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False,
                         return_intermediate_steps=True, max_iterations=6)


def run_agent(question: str, user_id=None) -> dict:
    """Run the agent on a question. Returns {answer, steps}. Traced for LLMOps."""
    with obs.span("agent.run", question=question[:200]):
        executor = build_agent(user_id=user_id)
        result = executor.invoke({"input": question})
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append(
            {
                "tool": getattr(action, "tool", "?"),
                "input": getattr(action, "tool_input", ""),
                "observation": str(observation)[:500],
            }
        )
    return {"answer": result.get("output", ""), "steps": steps}
