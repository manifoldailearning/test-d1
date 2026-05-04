"""Load ``<project_root>/.env``; isolate trace JSON writes to a temp directory."""
from __future__ import annotations

import os
import tempfile

from src.hardened_support.paths import load_app_dotenv

load_app_dotenv()

_trace_tmp = tempfile.mkdtemp(prefix="harden_traces_")
os.environ.setdefault("HARDEN_TRACE_DIR", _trace_tmp)
