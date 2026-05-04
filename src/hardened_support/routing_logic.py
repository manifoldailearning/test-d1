"""Structured ``RouteDecision`` (route + confidence + reason) for the supervisor step.

Uses OpenAI via LangChain structured output (Pattern 1). Requires ``OPENAI_API_KEY``.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import chat_openai
from .paths import project_root
from .model_policy import model_for_agent
from .prompts_io import load_system_prompt
from .schemas import RouteDecision

CONFIDENCE_THRESHOLD = 0.75


def _default_supervisor_prompt_path() -> Path:
    return project_root() / "prompts" / "supervisor_structured.yaml"


def structured_route_decision(query: str, prompt_path: Path | None = None) -> RouteDecision:
    """
    Supervisor routing via OpenAI structured output. ``prompt_path`` must exist when provided.
    """
    path = prompt_path or _default_supervisor_prompt_path()
    if not path.is_file():
        raise FileNotFoundError(path)

    system = load_system_prompt(path)
    llm = chat_openai(model=model_for_agent("supervisor"))
    structured = llm.with_structured_output(RouteDecision)
    msg = structured.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=query),
        ]
    )
    if isinstance(msg, RouteDecision):
        return msg
    return RouteDecision.model_validate(msg)


def supervisor_route_string(query: str, *, prompt_path: Path | None = None) -> str:
    """Returns route name only (used by routing eval)."""
    return structured_route_decision(query, prompt_path=prompt_path).route
