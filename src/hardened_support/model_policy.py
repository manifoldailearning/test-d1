"""Per-role model names for OpenAI calls and logging (Pattern 8)."""
from __future__ import annotations

import os

# OpenAI model ids (override per role with env HARDEN_MODEL_<ROLE> e.g. HARDEN_MODEL_SUPERVISOR).
_DEFAULT = "gpt-4o-mini"
MODEL_POLICY = {
    "supervisor": os.getenv("HARDEN_MODEL_SUPERVISOR", _DEFAULT),
    "general_agent": os.getenv("HARDEN_MODEL_GENERAL_AGENT", _DEFAULT),
    "orders_agent": os.getenv("HARDEN_MODEL_ORDERS_AGENT", _DEFAULT),
    "billing_agent": os.getenv("HARDEN_MODEL_BILLING_AGENT", _DEFAULT),
    "technical_agent": os.getenv("HARDEN_MODEL_TECHNICAL_AGENT", _DEFAULT),
    "synthesis": os.getenv("HARDEN_MODEL_SYNTHESIS", _DEFAULT),
}


def model_for_agent(agent_key: str) -> str:
    return MODEL_POLICY.get(agent_key, MODEL_POLICY["general_agent"])
