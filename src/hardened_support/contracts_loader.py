"""Load per-agent contracts from YAML (Pattern 2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_agent_contracts(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid contracts file: {p}")
    return data


def contract_for_route(route: str, contracts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    key = f"{route}_agent"
    return contracts.get(key)
