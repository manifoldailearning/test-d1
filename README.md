# Agentic Day 4 – Multi-Agent Collaboration (Supervisor + Specialists)

This repo is the **reference structure / example implementation** for the Day 4 assignment of the Agentic AI Enterprise Mastery Bootcamp.

The goal is to build a **multi-agent customer support system** with:

- A **supervisor** that routes each request
- 4 **specialist agents** (orders, billing, technical, subscription) plus a general path
- **Structured handoffs** between agents
- **Graceful degradation** inside specialist agents
- A **session-level audit log** with approximate cost tracking

---

## Project Structure

```text
agentic-day4-multi-agent/
├── .gitignore
├── requirements.txt
├── README.md
├── app.py
└── prompts/
    └── supervisor_v1.yaml
```

Your own submission should at minimum follow this layout.

---

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root for your API keys (if you use OpenAI, etc.):

```bash
OPENAI_API_KEY="sk-..."
```

> **Important:** `.env` is listed in `.gitignore` and **must not be committed**.  
> Never push API keys or secrets to GitHub.

---

## Running the app

Once dependencies are installed and your `.env` is set up (if needed), run:

```bash
python app.py
```

The script will:

- Load the **supervisor classification prompt** from `prompts/supervisor_v1.yaml`
- Build a simple **LangGraph** with:
  - `supervisor_node`
  - 4 specialist nodes: `orders_agent_node`, `billing_agent_node`, `technical_agent_node`, `subscription_agent_node`
  - A `general_agent_node`
  - A `synthesize_response` node
- Use a `MultiAgentState` TypedDict to share state
- Run a few sample user requests and print:
  - The detected `route`
  - Which `agent_used` handled the request
  - The `final_response` text
- Log events into a `SessionAuditLog` and append a JSON line to `audit_log.jsonl`

This is a **minimal** example to show the architecture; your own implementation can add real tools and richer prompts.

---

## Submission Rules (Summary)

- Repository should be **public**
- `.env` must **not** be committed
- Default branch should be `main`
- Code must run via:

```bash
python app.py
```

Please read the full Day 4 assignment brief (`Assignment-A4.MD`) for detailed requirements and grading criteria.

