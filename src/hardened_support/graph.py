"""Production-hardening LangGraph: guardrail → structured route → confidence gate → specialist → trace.

Implements the layered flow in ``3- production_hardening_multi_agent_systems_class_script.md``
(Diagram 2 / recap architecture), with trace events for observability.
"""
from __future__ import annotations

import operator
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts_loader import load_agent_contracts
from .guardrails import input_guardrail_node
from .routing_logic import CONFIDENCE_THRESHOLD, structured_route_decision
from .specialists import post_specialist_rules, run_specialist, synthesize_final
from .trace_export import export_graph_trace_json


class HardenedState(TypedDict, total=False):
    user_request: str
    trace_id: str
    session_id: str
    blocked: bool
    block_reason: str | None
    route_decision: dict[str, Any]
    clarify_only: bool
    contract: dict[str, Any] | None
    specialist_output: dict[str, Any]
    handoff_id: str | None
    tool_calls: int
    handoffs: int
    agent_steps: int
    session_cost_usd: float
    needs_escalation: bool
    final_response: str | None
    trace_summary: str
    trace_json_path: str | None
    events: Annotated[list[dict[str, Any]], operator.add]


def build_graph(*, supervisor_prompt_path: Path, contracts_path: Path):
    contracts = load_agent_contracts(contracts_path)

    def init_run(state: HardenedState) -> dict:
        tid = state.get("trace_id") or f"trace_{uuid.uuid4().hex[:12]}"
        sid = state.get("session_id") or f"session_{uuid.uuid4().hex[:8]}"
        return {
            "trace_id": tid,
            "session_id": sid,
            "tool_calls": state.get("tool_calls", 0),
            "handoffs": state.get("handoffs", 0),
            "agent_steps": state.get("agent_steps", 0),
            "session_cost_usd": state.get("session_cost_usd", 0.0),
            "events": [
                {
                    "trace_id": tid,
                    "session_id": sid,
                    "agent_name": "orchestrator",
                    "event": "run_started",
                    "route": None,
                    "handoff_id": None,
                }
            ],
        }

    def guardrail(state: HardenedState) -> dict:
        gr = input_guardrail_node(state)
        events = []
        if gr.get("blocked"):
            events.append(
                {
                    "trace_id": state["trace_id"],
                    "session_id": state.get("session_id"),
                    "agent_name": "input_guardrail",
                    "event": "guardrail_result",
                    "route": None,
                    "handoff_id": None,
                    "blocked": True,
                    "block_reason": gr.get("block_reason"),
                }
            )
        else:
            events.append(
                {
                    "trace_id": state["trace_id"],
                    "session_id": state.get("session_id"),
                    "agent_name": "input_guardrail",
                    "event": "guardrail_result",
                    "route": None,
                    "handoff_id": None,
                    "blocked": False,
                }
            )
        return {**gr, "events": events}

    def supervisor(state: HardenedState) -> dict:
        rd = structured_route_decision(state["user_request"], prompt_path=supervisor_prompt_path)
        d = rd.model_dump()
        events = [
            {
                "trace_id": state["trace_id"],
                "session_id": state.get("session_id"),
                "agent_name": "supervisor",
                "event": "supervisor_route",
                "route": d["route"],
                "handoff_id": None,
                "confidence": d["confidence"],
                "reason": d["reason"],
            }
        ]
        return {"route_decision": d, "events": events}

    def clarify(state: HardenedState) -> dict:
        rd = state["route_decision"]
        msg = (
            "I’m not confident which team should handle this yet. "
            "Can you clarify whether this is about an order, a payment, or a technical issue with our app?"
        )
        events = [
            {
                "trace_id": state["trace_id"],
                "session_id": state.get("session_id"),
                "agent_name": "supervisor",
                "event": "low_confidence_clarify",
                "route": rd.get("route"),
                "handoff_id": None,
                "confidence": rd.get("confidence"),
            }
        ]
        return {"clarify_only": True, "final_response": msg, "events": events}

    def specialist(state: HardenedState) -> dict:
        return run_specialist({**state, "route_decision": state["route_decision"]}, contracts)

    def post_rules(state: HardenedState) -> dict:
        return post_specialist_rules(state)

    def synthesize(state: HardenedState) -> dict:
        return synthesize_final(state)

    def finalize_trace(state: HardenedState) -> dict:
        lines = [f"TRACE SUMMARY trace_id={state.get('trace_id')} session_id={state.get('session_id')}"]
        for ev in state.get("events", []):
            parts = [f"- {ev.get('event')}", f"agent={ev.get('agent_name')}"]
            if ev.get("route") is not None:
                parts.append(f"route={ev.get('route')}")
            if ev.get("handoff_id"):
                parts.append(f"handoff_id={ev.get('handoff_id')}")
            if "confidence" in ev:
                parts.append(f"confidence={ev.get('confidence')}")
            if ev.get("tool_name"):
                parts.append(f"tool={ev.get('tool_name')}")
            lines.append(" ".join(parts))
        summary = "\n".join(lines)
        out_path, err = export_graph_trace_json({**state, "trace_summary": summary})
        extra_events: list[dict[str, Any]] = []
        trace_json_path: str | None = str(out_path) if out_path else None
        if err:
            trace_json_path = None
            extra_events.append(
                {
                    "trace_id": state.get("trace_id"),
                    "session_id": state.get("session_id"),
                    "agent_name": "trace_export",
                    "event": "trace_export_failed",
                    "route": None,
                    "handoff_id": None,
                    "error": err,
                }
            )
        elif out_path:
            extra_events.append(
                {
                    "trace_id": state.get("trace_id"),
                    "session_id": state.get("session_id"),
                    "agent_name": "trace_export",
                    "event": "trace_export_written",
                    "route": None,
                    "handoff_id": None,
                    "path": str(out_path),
                }
            )
        return {"trace_summary": summary, "trace_json_path": trace_json_path, "events": extra_events}

    def route_after_guardrail(state: HardenedState) -> str:
        return "blocked" if state.get("blocked") else "continue"

    def route_after_confidence(state: HardenedState) -> str:
        rd = state.get("route_decision") or {}
        conf = float(rd.get("confidence", 0.0))
        return "clarify" if conf < CONFIDENCE_THRESHOLD else "specialist"

    g = StateGraph(HardenedState)
    g.add_node("init_run", init_run)
    g.add_node("guardrail", guardrail)
    g.add_node("blocked_finalize", finalize_trace)
    g.add_node("supervisor", supervisor)
    g.add_node("clarify", clarify)
    g.add_node("specialist", specialist)
    g.add_node("post_rules", post_rules)
    g.add_node("synthesize", synthesize)
    g.add_node("finalize", finalize_trace)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "guardrail")
    g.add_conditional_edges("guardrail", route_after_guardrail, {"blocked": "blocked_finalize", "continue": "supervisor"})
    g.add_edge("blocked_finalize", END)

    g.add_conditional_edges("supervisor", route_after_confidence, {"clarify": "clarify", "specialist": "specialist"})
    g.add_edge("clarify", "finalize")
    g.add_edge("specialist", "post_rules")
    g.add_edge("post_rules", "synthesize")
    g.add_edge("synthesize", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
