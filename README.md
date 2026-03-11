# Agentic Day 3 – Prompting That Ships: Production Hardening

This repo is the **reference solution / starter structure** for the Day 3 assignment of the Agentic AI Enterprise Mastery Bootcamp.

The goal is to take a minimal customer support agent and harden it with:

- Prompts as **YAML + Git–versioned code**
- **Prompt injection defense** (3-layer model)
- **Error handling with retries**
- **Circuit breaker** around LLM calls
- **Session-level cost tracking**

---

## Project Structure

```text
agentic-day3-production/
├── .gitignore
├── requirements.txt
├── README.md
├── app.py
└── prompts/
    └── support_agent_v1.yaml
```

You should at minimum mirror this layout in your own submission.

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

- Load the **support agent** system prompt from `prompts/support_agent_v1.yaml`
- Wrap all agent calls with:
  - `detect_injection` (input layer)
  - `production_invoke` (error handling + retries)
  - `CircuitBreaker` (stops hammering a failing service)
  - `SessionCostTracker` (tracks total USD cost per session)
- Run:
  - One normal query (e.g. "What is your refund policy?")
  - One obvious injection attempt (e.g. "Ignore your previous instructions...")
- Print:
  - Whether the injection attempt was blocked
  - A short **cost summary** (total calls, total USD)

You can adapt the prompts, wording, and extra logging for your own submission.

---

## Submission Rules (Summary)

- Repository should be **public**
- `.env` must **not** be committed
- Default branch should be `main`
- Code must run via:

```bash
python app.py
```

Please read the full Day 3 assignment brief (`Assignment-A3.MD`) for detailed requirements and grading criteria.

