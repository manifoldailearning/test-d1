# Learner assignment: three small changes (full-stack confidence)

Each task is **small**, **reversible** (use `git checkout` or a branch), and touches a **different layer** of [production_hardening_multi_agent_working_code](.). Together they cover routing, domain contracts, and policy or observability.

**Prerequisites:** working `.env` with `OPENAI_API_KEY`, `pip install -r requirements.txt`, run from project root with `PYTHONPATH=.`.

---

## Assignment 1 — Own a routing case (supervisor + golden set)

**Goal:** Prove you understand **how the supervisor picks a route** and how we **regress-test** it without running the full graph.

**What you do**

1. Open [`src/hardened_support/routing_eval.py`](src/hardened_support/routing_eval.py) and find the list **`routing_tests`** (20 golden `{ "query", "expected_route" }` rows).
2. Add **one new row** with:
   - A query that is **unambiguous** for one of: `orders`, `billing`, `technical`, or `general` (your choice).
   - The matching **`expected_route`** you intend.
3. Optionally tighten the supervisor’s behavior by editing **one line** in [`prompts/supervisor_structured.yaml`](prompts/supervisor_structured.yaml) if the model disagrees with your label (keep the change minimal).

**How you verify**

```bash
PYTHONPATH=. python -m src.hardened_support.routing_eval
```

Accuracy should still meet your bar; if your new case fails, iterate on query wording or the prompt until it passes—or document in one sentence why the model is unstable and adjust `expected_route` only if you accept that label.

**Stretch:** Run the routing test in CI style:

```bash
PYTHONPATH=. pytest tests/test_hardening.py::test_routing_accuracy_20_cases -v
```

(requires `OPENAI_API_KEY`; skipped if missing.)

**You are confident when:** you can explain aloud how `supervisor_route_string` → `structured_route_decision` → OpenAI → `RouteDecision` connects to your new row.

---

## Assignment 2 — Change the law for one specialist (contracts → behavior)

**Goal:** Prove you understand **agent contracts** as the lever that shapes **specialist** output (still one LLM call, but bound by YAML).

**What you do**

1. Open [`contracts/agents.yaml`](contracts/agents.yaml) and pick **one** agent (e.g. `general_agent` or `billing_agent`).
2. Make **one concrete change**, for example:
   - Add a **bullet** under `forbidden_actions`, or  
   - Add or rename an item under `allowed_tools` (keep it consistent with what [`specialists.py`](src/hardened_support/specialists.py) already mentions in the system prompt if you want the model to use that name in `tools_used`), or  
   - Tweak **one sentence** in `responsibility` or `fallback_behavior`.
3. Run the full graph for a message that **routes to that agent** (use or copy a string from [`src/hardened_support/main.py`](src/hardened_support/main.py) `demos`, or add a one-line temporary `demos = ["your query"]`).

**How you verify**

```bash
PYTHONPATH=. python -m src.hardened_support.main
```

Open the latest file under **`traces/`** (see `TRACE_JSON:` in the console) and inspect **`record.specialist_output`** (and `record.route_decision`) **before vs after** your edit (stash/revert to compare if needed).

**You are confident when:** you can point to **one field** in `specialist_output` (e.g. `answer`, `tools_used`, `risk_level`, `requires_human`) that plausibly reflects your YAML change.

---

## Assignment 3 — Tighten policy or visibility (escalation, guardrail, or trace)

**Goal:** Prove you understand **deterministic rules** after the model, **input blocking**, or **what gets persisted**—without changing the graph topology.

**Pick exactly one sub-track:**

### 3A — Escalation rule

1. Open [`src/hardened_support/escalation.py`](src/hardened_support/escalation.py) and [`src/hardened_support/schemas.py`](src/hardened_support/schemas.py) (`SpecialistOutput`).
2. Change **`should_escalate`** in a **small, intentional** way (example ideas: adjust the confidence cutoff, or treat `status == "needs_more_info"` as escalation—choose one and implement clearly).
3. Run a query that still gets a specialist response (e.g. an order or billing line from `demos`) and observe whether the **`[Escalation]`** suffix on `final_response` appears more or less often, and how **`record.events`** (`escalation_evaluated`) aligns.

### 3B — Guardrail pattern

1. Open [`src/hardened_support/guardrails.py`](src/hardened_support/guardrails.py).
2. Add **one** new substring to **`INJECTION_PATTERNS`** (something you are comfortable blocking in class, e.g. a specific jailbreak phrase—keep it professional).
3. Run `main` with a **user_request** that contains that substring and confirm: **`blocked`** is true, **no** `supervisor_route` in `trace_summary` / JSON `record.events`, and a JSON file is still written.

### 3C — Trace export (schema + payload)

1. Open [`src/hardened_support/trace_export.py`](src/hardened_support/trace_export.py). Read **`_TRACE_KEYS`** and **`build_trace_record`**.
2. Do **both** of the following:
   - Add **one** new top-level field next to `schema_version` / `exported_at` / `record` in the dict returned by `build_trace_record`—for example `"assignment_marker": "your_initials_or_team"` (any short static string you choose).
   - In your write-up, explain **three** keys from `_TRACE_KEYS` in one sentence each: when is that key **non-null** in `record` for (i) a blocked request, (ii) a full specialist success path?

**How you verify (3C):** Run `main` once, open the new JSON under `traces/`, and confirm your `assignment_marker` (or chosen field) appears at the **root** of the file next to `record`.

**How you verify (3A / 3B)**

```bash
PYTHONPATH=. python -m src.hardened_support.main
```

and/or:

```bash
PYTHONPATH=. pytest tests/test_hardening.py::test_guardrail_blocks_injection -v
```

(if you changed guardrails, confirm this test still passes or update the test **only** if your new pattern is part of the agreed injection curriculum.)

**You are confident when:** you can explain the difference between **LLM output** and **code that runs after** it (3A/3B), or what is **serialized** for ops (3C).

---

## Submission (lightweight)

Turn in **any format** your bootcamp uses (thread, doc, PR):

1. **Assignment 1:** the new `routing_tests` line (query + expected_route) + paste of `routing_eval` last line or pytest result.  
2. **Assignment 2:** which agent you edited + before/after snippet of YAML + one sentence on what changed in `specialist_output` or `answer`.  
3. **Assignment 3:** which sub-track (A/B/C) + file changed + one screenshot or paste of `trace_summary` or JSON snippet showing the effect.

---

## Hints

- Use a **`git branch assignment`** so you can compare and revert quickly.  
- If OpenAI calls fail, check `.env` and [`paths.py`](src/hardened_support/paths.py) `project_root()`.  
- Session narrative order: [SESSION.md](SESSION.md); full file map: [script.md](script.md).
