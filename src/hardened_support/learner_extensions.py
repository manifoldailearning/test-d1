"""
Assignment A7 — learner extensions (deterministic routing + combined guardrails).

Implement the functions and constants below. The auto-grader imports this module
as ``src.hardened_support.learner_extensions`` only — the file must live at
``src/hardened_support/learner_extensions.py`` (not at the repository root).

You may import from ``src.hardened_support.guardrails`` only (plus stdlib).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) Deterministic supervisor routing (golden set accuracy)
# ---------------------------------------------------------------------------


def deterministic_supervisor_route(query: str) -> str:
    """
    Return one of: ``\"orders\"``, ``\"billing\"``, ``\"technical\"``, ``\"general\"``.

    Must achieve **>= 85%** accuracy on the same 20 golden rows documented in
    ``Assignment-A7.MD`` (aligned with ``routing_eval.routing_tests`` in the starter).

    Tip: keyword / priority ordering (technical before generic billing terms) helps
    avoid misroutes like ``\"payment screen\"`` freezing cases.
    """
    raise NotImplementedError("Implement deterministic_supervisor_route per Assignment-A7.MD")


# ---------------------------------------------------------------------------
# 2) Extend the golden set (documentation + regression habit)
# ---------------------------------------------------------------------------

# Each item: {"query": str, "expected_route": str}
# ``expected_route`` must be one of: orders | billing | technical | general
# Minimum **3** items. Your ``deterministic_supervisor_route`` should classify
# these the same way for your own pytest / CI.
EXTRA_ROUTING_TESTS: list[dict] = []


# ---------------------------------------------------------------------------
# 3) Extra injection substrings + combined guardrail
# ---------------------------------------------------------------------------

# Minimum **2** entries. At least one entry must contain the substring ``jailbreak``
# (case-insensitive), e.g. ``\"jailbreak mode\"``.
EXTRA_INJECTION_SUBSTRINGS: list[str] = []


def combined_input_guardrail(state: dict) -> dict:
    """
    Run the **stock** ``input_guardrail_node`` first. If that does not block, apply
    ``EXTRA_INJECTION_SUBSTRINGS`` with the same semantics as the stock guardrail:
    lowercase match on ``state[\"user_request\"]`` using ``substring in text``.

    If blocked, return a dict with at least:
      - ``blocked``: True
      - ``block_reason``: ``\"prompt_injection_detected\"``
      - ``final_response``: non-empty string (may match the stock message)

    If not blocked, return ``{\"blocked\": False}`` (same shape as stock partial).
    """
    raise NotImplementedError("Implement combined_input_guardrail per Assignment-A7.MD")


# ---------------------------------------------------------------------------
# 4) Trace schema awareness (ties to trace_export.SCHEMA_VERSION)
# ---------------------------------------------------------------------------


def get_trace_schema_version() -> str:
    """Return the same value as ``trace_export.SCHEMA_VERSION`` (import from there)."""
    raise NotImplementedError("Implement get_trace_schema_version per Assignment-A7.MD")
