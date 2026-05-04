"""Pydantic schemas: structured route output and specialist output (Patterns 1 & 4)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured supervisor output (script Pattern 1 + confidence diagram)."""

    route: Literal["orders", "billing", "technical", "general"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class SpecialistOutput(BaseModel):
    """Pydantic specialist output schema from the script (Pattern 4)."""

    status: Literal["resolved", "needs_more_info", "escalated"]
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool
    tools_used: list[str]
    risk_level: Literal["low", "medium", "high"]
