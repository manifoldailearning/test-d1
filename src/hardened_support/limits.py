"""Runtime limits (Pattern 7) — checked before unbounded agent/tool work."""
from __future__ import annotations

# Script "Runtime Limits"
runtime_limits = {
    "max_agent_steps": 5,
    "max_tool_calls": 3,
    "max_handoffs": 2,
    "max_latency_ms": 8000,
    "max_session_cost_usd": 0.05,
}


def limits_exceeded(state: dict) -> tuple[bool, str]:
    """Returns (exceeded, reason)."""
    if state.get("agent_steps", 0) >= runtime_limits["max_agent_steps"]:
        return True, "max_agent_steps"
    if state.get("tool_calls", 0) >= runtime_limits["max_tool_calls"]:
        return True, "max_tool_calls"
    if state.get("handoffs", 0) >= runtime_limits["max_handoffs"]:
        return True, "max_handoffs"
    if state.get("session_cost_usd", 0.0) >= runtime_limits["max_session_cost_usd"]:
        return True, "max_session_cost_usd"
    return False, ""
