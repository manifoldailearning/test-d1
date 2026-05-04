"""Reference solution for Assignment A7 (instructor / self-check). Copy into
``src/hardened_support/learner_extensions.py`` for a fully passing implementation.
"""
from __future__ import annotations

from .guardrails import input_guardrail_node


def deterministic_supervisor_route(query: str) -> str:
    q = query.lower()

    if any(k in q for k in ("business hours", "phone number", "hi there", "thanks!")):
        return "general"

    if any(
        k in q
        for k in (
            "crash",
            "freeze",
            "freezes",
            "login",
            "ios",
            "android",
            "bug:",
            "bug ",
            "error code",
            "checkout error",
            "dump attached",
        )
    ):
        return "technical"

    if any(
        k in q
        for k in (
            "charged",
            "refund",
            "invoice",
            "billed",
            "txn-",
            "duplicate charge",
            "payment failed",
            "card was charged",
            "subscription",
        )
    ):
        return "billing"

    if any(
        k in q
        for k in (
            "ord-",
            "track shipment",
            "return policy",
            "package not delivered",
            "change delivery address",
            "where is my order",
        )
    ):
        return "orders"

    return "general"


EXTRA_ROUTING_TESTS: list[dict] = [
    {"query": "My ORD-abc never shipped", "expected_route": "orders"},
    {"query": "Cancel subscription renewal charge", "expected_route": "billing"},
    {"query": "Error code 418 when syncing photos", "expected_route": "technical"},
]

EXTRA_INJECTION_SUBSTRINGS: list[str] = [
    "jailbreak mode",
    "disable all safety filters",
]


def combined_input_guardrail(state: dict) -> dict:
    base = input_guardrail_node(state)
    if base.get("blocked"):
        return base

    user_request = str(state.get("user_request", "")).lower()
    for pattern in EXTRA_INJECTION_SUBSTRINGS:
        if pattern.lower() in user_request:
            return {
                "blocked": True,
                "block_reason": "prompt_injection_detected",
                "final_response": (
                    "I can't help with requests that attempt to override system instructions."
                ),
            }
    return {"blocked": False}


def get_trace_schema_version() -> str:
    from .trace_export import SCHEMA_VERSION

    return SCHEMA_VERSION
