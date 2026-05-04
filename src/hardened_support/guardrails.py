"""Input guardrail before supervisor (Pattern 3 in the production hardening script)."""
from __future__ import annotations

# Script: "Common Injection Patterns"
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "you are now",
    "system prompt",
    "developer message",
    "reveal your instructions",
    "bypass policy",
    "act as unrestricted",
]


def input_guardrail_node(state: dict) -> dict:
    """Script Pattern 3 — guardrail before supervisor."""
    user_request = state["user_request"].lower()

    for pattern in INJECTION_PATTERNS:
        if pattern in user_request:
            return {
                "blocked": True,
                "block_reason": "prompt_injection_detected",
                "final_response": "I can't help with requests that attempt to override system instructions.",
            }

    return {"blocked": False}
