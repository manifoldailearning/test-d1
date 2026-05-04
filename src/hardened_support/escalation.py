"""Deterministic escalation rules on structured specialist output (Pattern 9)."""
from __future__ import annotations

from .schemas import SpecialistOutput


def should_escalate(output: SpecialistOutput) -> bool:
    """Script Pattern 9 — example escalation logic."""
    if output.requires_human:
        return True
    if output.confidence < 0.7:
        return True
    if output.risk_level == "high":
        return True
    if output.status == "escalated":
        return True
    return False
