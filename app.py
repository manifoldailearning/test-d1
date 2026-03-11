import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, TypedDict

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict):
    user_request: str
    route: str
    agent_used: str
    specialist_result: str
    final_response: str


@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {json.dumps(self.context, indent=2)}\n"
            f"Received at: {self.timestamp}"
        )


@dataclass
class SessionAuditLog:
    session_id: str
    events: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def log(self, agent: str, action: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        cost = (tokens_in * 0.000015 + tokens_out * 0.00006) / 1000
        self.total_cost_usd += cost
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "action": action,
                "cost_usd": round(cost, 6),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "events": self.events,
        }


def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path("audit_log.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")


def load_supervisor_prompt() -> str:
    data = yaml.safe_load(Path("prompts/supervisor_v1.yaml").read_text(encoding="utf-8"))
    for key in ["version", "created_by", "created_at", "description", "changelog", "system"]:
        if key not in data:
            raise ValueError(f"Missing key in supervisor_v1.yaml: {key}")
    return data["system"]


INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (your |all |previous )?instructions",
    r"system prompt.*disabled",
    r"you are now a",
    r"repeat.*system prompt",
    r"jailbreak",
]


def detect_injection(user_input: str) -> bool:
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "I can only assist with account and order support. (Request blocked.)"
    return user_input


VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}


def _llm():
    # Simple helper so tests don't depend on global symbols
    return init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)


def supervisor_node(state: MultiAgentState) -> dict:
    system_prompt = load_supervisor_prompt()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_request"]),
    ]
    response = _llm().invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}


def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")


def _simple_specialist(agent_name: str, state: MultiAgentState, audit: SessionAuditLog) -> dict:
    handoff = AgentHandoff(
        from_agent="supervisor",
        to_agent=agent_name,
        task=state["user_request"],
        context={"route": state["route"]},
        priority="normal",
        timestamp=datetime.utcnow().isoformat(),
    )
    content = handoff.to_prompt_context()
    audit.log(agent=agent_name, action="handled", tokens_in=100, tokens_out=50)
    text = f"[{agent_name}] {content}"
    return {
        "agent_used": agent_name,
        "specialist_result": text,
    }


def orders_agent_node(state: MultiAgentState) -> dict:
    audit = SessionAuditLog(session_id=os.getenv("SESSION_ID", "demo"))
    return _simple_specialist("orders_agent", state, audit)


def billing_agent_node(state: MultiAgentState) -> dict:
    audit = SessionAuditLog(session_id=os.getenv("SESSION_ID", "demo"))
    return _simple_specialist("billing_agent", state, audit)


def technical_agent_node(state: MultiAgentState) -> dict:
    audit = SessionAuditLog(session_id=os.getenv("SESSION_ID", "demo"))
    return _simple_specialist("technical_agent", state, audit)


def subscription_agent_node(state: MultiAgentState) -> dict:
    audit = SessionAuditLog(session_id=os.getenv("SESSION_ID", "demo"))
    return _simple_specialist("subscription_agent", state, audit)


def general_agent_node(state: MultiAgentState) -> dict:
    audit = SessionAuditLog(session_id=os.getenv("SESSION_ID", "demo"))
    return _simple_specialist("general_agent", state, audit)


def synthesize_response_node(state: MultiAgentState) -> dict:
    final = f"Final answer from {state.get('agent_used', 'unknown')}: {state.get('specialist_result', '')}"
    return {"final_response": final}


def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")
    workflow.add_conditional_edges("supervisor_node", route_to_specialist)

    for specialist in [
        "orders_agent_node",
        "billing_agent_node",
        "technical_agent_node",
        "subscription_agent_node",
        "general_agent_node",
    ]:
        workflow.add_edge(specialist, "synthesize_response")

    workflow.add_edge("synthesize_response", END)

    return workflow.compile()


def main() -> None:
    audit = SessionAuditLog(session_id="demo-session")
    graph = build_graph()

    requests = [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
    ]

    for req in requests:
        safe = guard_request(req)
        state: MultiAgentState = {
            "user_request": safe,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
        }
        result = graph.invoke(state)
        audit.log(agent=result.get("agent_used", "unknown"), action="completed")
        print("Request:", req)
        print("Route:", result.get("route"), "Agent used:", result.get("agent_used"))
        print("Final:", result.get("final_response"))
        print("---")

    print("Total cost (USD):", round(audit.total_cost_usd, 6))
    persist_audit_log(audit)


if __name__ == "__main__":
    main()

