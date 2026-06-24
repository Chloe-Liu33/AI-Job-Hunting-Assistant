"""An agent built with the OpenAI Agents SDK — running on FREE Groq keys.

The OpenAI Agents SDK (`openai-agents`) is a popular agent framework, but it's
normally tied to a paid OpenAI account. The trick here: Groq exposes an
OpenAI-COMPATIBLE endpoint, so we point the SDK's client at Groq's base URL and
run the exact same Agent/Runner/function_tool API for $0.

    AsyncOpenAI(base_url=<groq>, api_key=<groq>) -> OpenAIChatCompletionsModel
    Agent(tools=[search_cv]) -> Runner.run_sync(agent, question)

This demonstrates hands-on use of the OpenAI Agents SDK without an OpenAI bill.
Requires GROQ_API_KEY (the OpenAI-compatible path is cleanest on Groq). Install:

    pip install -r requirements-extras.txt
"""
from __future__ import annotations

import os

import llm as llm_cfg
import rag


def _make_model():
    """Build an OpenAIChatCompletionsModel backed by Groq's OpenAI-compatible API."""
    try:
        from agents import OpenAIChatCompletionsModel, set_tracing_disabled
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "openai-agents not installed. Run: pip install -r requirements-extras.txt"
        ) from e

    # The SDK's built-in tracing uploads to OpenAI's platform (needs an OpenAI
    # key). We're on Groq, so turn it off — our own LLMOps tracer handles spans.
    set_tracing_disabled(True)

    cfg = llm_cfg.openai_compatible_config("groq")
    if not cfg["api_key"]:
        raise RuntimeError("GROQ_API_KEY required for the OpenAI Agents SDK path.")
    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return OpenAIChatCompletionsModel(model=cfg["model"], openai_client=client)


def build_agent():
    """Construct an OpenAI Agents SDK Agent with a CV-search tool."""
    from agents import Agent, function_tool

    @function_tool
    def search_cv(query: str) -> str:
        """Search the candidate's CV for passages relevant to the query."""
        ctx, _ = rag.retrieve_context(query, k=5, kind="cv")
        return ctx or "No CV indexed."

    return Agent(
        name="JobFitAgent",
        instructions=(
            "You assess how well the candidate fits a role. Use the search_cv "
            "tool to gather evidence before answering. Never invent experience."
        ),
        model=_make_model(),
        tools=[search_cv],
    )


def run(question: str) -> str:
    """Run the OpenAI-SDK agent synchronously and return its final answer. Traced."""
    import observability as obs
    from agents import Runner

    with obs.span("openai_sdk.run", question=question[:200]):
        result = Runner.run_sync(build_agent(), question)
    return result.final_output
