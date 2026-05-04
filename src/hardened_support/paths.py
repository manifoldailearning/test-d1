"""Project root and ``.env`` loading for this package (keep ``.env`` next to ``.env.example``)."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    """``production_hardening_multi_agent_working_code/`` (contains ``contracts/``, ``prompts/``, ``.env``)."""
    return Path(__file__).resolve().parents[2]


def load_app_dotenv() -> None:
    """Load ``<project_root>/.env`` into the process environment."""
    load_dotenv(project_root() / ".env")
