"""Specialist execution with contracts, tool behavior, limits, and structured OpenAI output.

Patterns 2, 4, 6–9: contracts, structured output, blast-radius style isolation, escalation.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .escalation import should_escalate
from .limits import limits_exceeded
from .llm import chat_openai
from .model_policy import model_for_agent
from .schemas import SpecialistOutput


def _specialist_system_prompt(contract: dict[str, Any]) -> str:
    contract_yaml = yaml.safe_dump(contract, default_flow_style=False, sort_keys=False)
    simulate_tool_fail = os.getenv("SIMULATE_TOOL_FAILURE") == "1"
    tool_note = ""
    if simulate_tool_fail:
        tool_note = (
            "\n\nSIMULATION: The first allowed tool call fails with a transient error. "
            "Follow the contract's fallback_behavior and set status/confidence accordingly; "
            "still list the tool you attempted in tools_used."
        )
    return f"""You are a customer support specialist. Obey the agent contract exactly.

CONTRACT (YAML):
{contract_yaml}

OUTPUT RULES:
- Produce JSON matching SpecialistOutput: status (resolved | needs_more_info | escalated),
  answer, confidence (0-1), requires_human, tools_used (names from allowed_tools only, or empty),
  risk_level (low | medium | high).
- Do not claim you performed forbidden_actions.
- Refunds or credits above $500 require status escalated, requires_human true, risk_level high.
- For ambiguous billing + order cases, prefer billing when duplicate/wrong charges are the main issue.
- Keep answers concise and professional.{tool_note}
"""


def run_specialist(state: dict, contracts: dict[str, dict[str, Any]]) -> dict:
    """
    Executes one specialist with contract loaded, OpenAI structured output,
    runtime limits, and model policy logging.
    """
    route = state["route_decision"]["route"]
    contract_key = f"{route}_agent"
    contract = contracts.get(contract_key)
    if not contract:
        raise KeyError(f"Missing contract for route={route} key={contract_key}")

    events: list[dict[str, Any]] = []
    handoff_id = f"handoff_{uuid.uuid4().hex[:10]}"
    events.append(
        {
            "trace_id": state["trace_id"],
            "session_id": state.get("session_id"),
            "agent_name": contract_key,
            "event": "handoff_created",
            "route": route,
            "handoff_id": handoff_id,
            "model": model_for_agent(contract_key),
        }
    )

    handoffs = state.get("handoffs", 0) + 1
    agent_steps = state.get("agent_steps", 0) + 1
    tool_calls = state.get("tool_calls", 0)

    exceeded, reason = limits_exceeded({**state, "handoffs": handoffs, "agent_steps": agent_steps, "tool_calls": tool_calls})
    if exceeded:
        out = SpecialistOutput(
            status="escalated",
            answer="We hit a runtime safety limit and stopped automated processing. A human will follow up.",
            confidence=0.5,
            requires_human=True,
            tools_used=[],
            risk_level="medium",
        )
        events.append(
            {
                "trace_id": state["trace_id"],
                "session_id": state.get("session_id"),
                "agent_name": contract_key,
                "event": "limit_exceeded",
                "route": route,
                "handoff_id": handoff_id,
                "limit": reason,
            }
        )
        return {
            "handoff_id": handoff_id,
            "handoffs": handoffs,
            "agent_steps": agent_steps,
            "tool_calls": tool_calls,
            "contract": contract,
            "specialist_output": out.model_dump(),
            "events": events,
        }

    user = state["user_request"]
    system = _specialist_system_prompt(contract)
    llm = chat_openai(model=model_for_agent(contract_key))
    structured = llm.with_structured_output(SpecialistOutput)
    raw = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    out = raw if isinstance(raw, SpecialistOutput) else SpecialistOutput.model_validate(raw)

    tools_used = list(out.tools_used)
    tool_calls += len(tools_used)
    for name in tools_used:
        events.append(_tool_event(state, contract_key, route, handoff_id, name))

    events.append(
        {
            "trace_id": state["trace_id"],
            "session_id": state.get("session_id"),
            "agent_name": contract_key,
            "event": "specialist_output",
            "route": route,
            "handoff_id": handoff_id,
            "requires_human": out.requires_human,
            "risk_level": out.risk_level,
            "confidence": out.confidence,
        }
    )

    return {
        "handoff_id": handoff_id,
        "handoffs": handoffs,
        "agent_steps": agent_steps,
        "tool_calls": tool_calls,
        "contract": contract,
        "specialist_output": out.model_dump(),
        "events": events,
    }


def _tool_event(state: dict, agent_name: str, route: str, handoff_id: str, tool_name: str) -> dict[str, Any]:
    return {
        "trace_id": state["trace_id"],
        "session_id": state.get("session_id"),
        "agent_name": agent_name,
        "event": "tool_call_completed",
        "route": route,
        "handoff_id": handoff_id,
        "tool_name": tool_name,
        "latency_ms": None,
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": None,
        "fallback_level": None,
        "escalated": False,
    }


def post_specialist_rules(state: dict) -> dict:
    """Applies deterministic rules from diagram (Pattern 4 / 9)."""
    out = SpecialistOutput.model_validate(state["specialist_output"])
    escalate = should_escalate(out)
    events = [
        {
            "trace_id": state["trace_id"],
            "session_id": state.get("session_id"),
            "agent_name": "orchestrator",
            "event": "escalation_evaluated",
            "route": state["route_decision"]["route"],
            "handoff_id": state.get("handoff_id"),
            "should_escalate": escalate,
        }
    ]
    return {"needs_escalation": escalate, "events": events}


def synthesize_final(state: dict) -> dict:
    out = SpecialistOutput.model_validate(state["specialist_output"])
    base = out.answer
    if state.get("needs_escalation"):
        base += "\n\n[Escalation] Routed to human review based on structured output rules."
    return {"final_response": base, "events": []}
