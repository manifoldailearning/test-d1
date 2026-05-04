## Production hardening multi-agent systems (working code)

> **Bootcamp Assignment A7:** implement `src/hardened_support/learner_extensions.py` using the brief in the course pack (**`Assignment-A7/Assignment-A7.MD`**). Instructor reference: **`Assignment-A7/solution/learner_extensions.py`** (copy paths relative to the full `Learner-Deliverable` folder).

Runnable reference for `3- production_hardening_multi_agent_systems_class_script.md`: routing evaluation, route confidence, agent contracts, guardrails before supervisor, structured specialist output (Pydantic), in-graph events (`trace_id`, `handoff_id`), **JSON trace files** (`json.dump` per run), runtime limits, model policy, escalation rules, and a small integration test set.

**Docs**

- **[`SESSION.md`](SESSION.md)** — **live teaching spine**: product problem → runtime journey → layers → happy/failure paths → code map → deep dives → assignments (2+ hour friendly).
- **[`assignment.md`](assignment.md)** — **three graded-style tasks**: routing golden row, contract edit, escalation/guardrail/trace (submission checklist included).
- **[`SCRIPT_EXPLANATION.md`](SCRIPT_EXPLANATION.md)** — maps lecture patterns (1–10) → files and commands (compact).
- **[`script.md`](script.md)** — full walkthrough: every module, graph flow, each demo input, and trace/JSON behavior.
- **[`diagrams.md`](diagrams.md)** — Mermaid diagrams: high-level flow, data flow, and end-to-end sequence (main, graph branches, `routing_eval`, pytest).

### Setup

```bash
cd "production_hardening_multi_agent_working_code"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy **`.env.example`** to **`.env`** in this same directory (next to `.env.example`). Secrets and toggles are loaded with **`load_dotenv()`** via **`src.hardened_support.paths.load_app_dotenv()`**, which reads **`<this folder>/.env`**.

| Variable | Purpose |
|----------|---------|
| **`OPENAI_API_KEY`** | Required for supervisor + specialist (LangChain OpenAI). |
| **`OPENAI_BASE_URL`** | Optional (e.g. Azure OpenAI–compatible endpoint). |
| **`HARDEN_MODEL_*`** | Optional per-role model overrides; see `.env.example`. |
| **`HARDEN_TRACE_DIR`** | Directory for **structured trace JSON** (default: **`./traces/`**). In production, point this at a writable volume. |
| **`HARDEN_TRACE_EXPORT`** | Set to `0` / `false` to disable writing JSON files. |
| **`HARDEN_TRACE_JSON_PRETTY`** | Set to `1` for indented JSON (larger files). |
| **`SIMULATE_TOOL_FAILURE`** | Set to `1` to nudge the specialist LLM to simulate a first-tool failure (demo). |

**`.gitignore`** in this folder ignores **`.env`** and **`traces/`** so secrets and per-run JSON are not committed.

### Structured trace JSON (deployable)

Each completed **`graph.invoke()`** ends in **`finalize_trace`**, which:

1. Builds **`trace_summary`** (human-readable).
2. Writes **one UTF-8 JSON file** per run with **`json.dump`**: top-level **`schema_version`**, **`exported_at`** (UTC ISO), and **`record`** (pipeline fields including **`events`**, **`route_decision`**, **`final_response`**, etc.). See **`src/hardened_support/trace_export.py`**.
3. Sets **`trace_json_path`** on the returned state when the write succeeds, and appends **`trace_export_written`** or **`trace_export_failed`** to **`events`**.

The file under **`HARDEN_TRACE_DIR`** is named from **`trace_id`** (sanitized). Mount that directory in containers (example: Kubernetes **`emptyDir`** or a PVC at the same path you set in **`HARDEN_TRACE_DIR`**).

### Run demo (full graph + console trace + JSON path)

```bash
python -m src.hardened_support.main
```

For each demo query you get **`final_response`**, a **`TRACE_JSON:`** line with the file path when export succeeds, then **`trace_summary`**. JSON files appear under **`traces/`** by default.

### Run routing accuracy only (20 golden cases)

```bash
python -m src.hardened_support.routing_eval
```

This exercises **supervisor routing only** (no full graph, **no** trace JSON from `finalize_trace`).

### Simulate tool failure (graceful degradation path)

```bash
export SIMULATE_TOOL_FAILURE=1
python -m src.hardened_support.main
```

### Optional: pytest integration tests

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -q
```

**`tests/conftest.py`** sets **`HARDEN_TRACE_DIR`** to a temporary directory so tests do not litter `./traces/`, and still loads **`.env`** for **`OPENAI_API_KEY`** when you run LLM-backed tests.
