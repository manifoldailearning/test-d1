"""Golden routing set (20 cases) and ``evaluate_routing()`` from the class script.

Pattern 1 in ``3- production_hardening_multi_agent_systems_class_script.md``.
Run: ``python -m src.hardened_support.routing_eval``.
"""
from __future__ import annotations

import json
import os

from .paths import load_app_dotenv
from .routing_logic import supervisor_route_string


def evaluate_routing(supervisor_fn, test_cases):
    """Exact structure from the class script."""
    correct = 0
    results = []

    for test in test_cases:
        actual_route = supervisor_fn(test["query"])
        expected_route = test["expected_route"]
        passed = actual_route == expected_route

        if passed:
            correct += 1

        results.append(
            {
                "query": test["query"],
                "expected_route": expected_route,
                "actual_route": actual_route,
                "passed": passed,
            }
        )

    accuracy = correct / len(test_cases) if test_cases else 0.0

    return {
        "accuracy": accuracy,
        "results": results,
    }


# Assignment: 20 routing test cases (golden set)
routing_tests = [
    {"query": "Where is my order ORD-123?", "expected_route": "orders"},
    {"query": "I was charged twice for my subscription.", "expected_route": "billing"},
    {"query": "The mobile app crashes during checkout.", "expected_route": "technical"},
    {"query": "What are your business hours?", "expected_route": "general"},
    {"query": "Track shipment for ORD-999", "expected_route": "orders"},
    {"query": "I need a refund for duplicate charge on TXN-1001", "expected_route": "billing"},
    {"query": "Login fails on iOS after update", "expected_route": "technical"},
    {"query": "Hi there", "expected_route": "general"},
    {"query": "Return policy for opened items?", "expected_route": "orders"},
    {"query": "Invoice PDF is wrong for last payment", "expected_route": "billing"},
    {"query": "App freezes at payment screen", "expected_route": "technical"},
    {"query": "ORD-555 still says processing", "expected_route": "orders"},
    {"query": "Why was I billed twice this month?", "expected_route": "billing"},
    {"query": "Bug: checkout error code 500", "expected_route": "technical"},
    {"query": "Thanks!", "expected_route": "general"},
    {"query": "Package not delivered after 2 weeks ORD-777", "expected_route": "orders"},
    {"query": "Payment failed but card was charged", "expected_route": "billing"},
    {"query": "Android app crash dump attached", "expected_route": "technical"},
    {"query": "What is your phone number?", "expected_route": "general"},
    {"query": "Change delivery address for ORD-321", "expected_route": "orders"},
]


def main() -> None:
    load_app_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run routing evaluation against OpenAI.")
    out = evaluate_routing(supervisor_route_string, routing_tests)
    print(f"Routing accuracy: {out['accuracy']:.2%} ({sum(1 for r in out['results'] if r['passed'])}/{len(out['results'])})")
    fails = [r for r in out["results"] if not r["passed"]]
    if fails:
        print("Failures:")
        print(json.dumps(fails, indent=2))


if __name__ == "__main__":
    main()
