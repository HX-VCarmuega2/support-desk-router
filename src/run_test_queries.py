"""Runs every question in test_queries.json through the orchestrator and
reports whether it was routed to the expected department.

This is the routing-accuracy evidence for the project: it is not enough
for each agent to answer well in isolation (steps 5-6 already showed
that) — the orchestrator also has to send each question to the right
agent in the first place.
"""

import json
from pathlib import Path

from src.agents.orchestrator import route

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_QUERIES_PATH = PROJECT_ROOT / "test_queries.json"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "test_results.json"


def load_test_queries() -> list[dict]:
    with TEST_QUERIES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_all() -> list[dict]:
    """
    Run every test case. A single case raising an exception (an API
    failure, a classification error, ...) is recorded as its own failed
    result instead of crashing the whole run and losing every result
    already computed.
    """
    test_queries = load_test_queries()
    results = []

    for case in test_queries:
        try:
            outcome = route(case["question"])

            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "category": case["category"],
                    "expected_domain": case["expected_domain"],
                    "actual_domain": outcome["domain"],
                    "correct": outcome["domain"] == case["expected_domain"],
                    "answer": outcome["answer"],
                    "trace_id": outcome.get("trace_id"),
                    "error": None,
                }
            )

        except Exception as error:
            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "category": case["category"],
                    "expected_domain": case["expected_domain"],
                    "actual_domain": None,
                    "correct": False,
                    "answer": None,
                    "trace_id": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return results


def save_results(results: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def print_report(results: list[dict]) -> None:
    print("=" * 70)
    print("ROUTING TEST RESULTS")
    print("=" * 70)

    for result in results:
        status = "ERROR" if result["error"] else ("PASS" if result["correct"] else "FAIL")
        actual = result["actual_domain"] or "-"
        print(
            f"[{status}] #{result['id']:<2} ({result['category']:<9}) "
            f"expected={result['expected_domain']:<7} "
            f"actual={actual:<7} "
            f"- {result['question']}"
        )
        if result["error"]:
            print(f"         -> {result['error']}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    print("-" * 70)
    print(f"Routing accuracy: {correct}/{total} ({correct / total:.0%})")

    failed = [r for r in results if not r["correct"]]

    if failed:
        print(f"\n{len(failed)} misrouted quer{'y' if len(failed) == 1 else 'ies'}:")
        for result in failed:
            print(
                f"  #{result['id']} expected {result['expected_domain']}, "
                f"got {result['actual_domain']}: \"{result['question']}\""
            )


def main():
    results = run_all()
    print_report(results)
    save_results(results)
    print(f"\nFull results (including answers) saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
