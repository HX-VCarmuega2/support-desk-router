"""Runs the evaluator (bonus) over outputs/test_results.json: scores each
generated answer on relevance/completeness/accuracy and attaches the
scores to that run's Langfuse trace via the Score API.

Run `python -m src.run_test_queries` first (or again) to (re)generate
outputs/test_results.json with fresh answers and trace_ids to evaluate.
"""

import json
from pathlib import Path

from src.evaluator import DIMENSIONS, evaluate_and_score
from src.observability import safe_flush

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "outputs" / "test_results.json"
EVALUATION_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.json"


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run 'python -m src.run_test_queries' first."
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_evaluations(evaluations: list[dict]) -> None:
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with EVALUATION_PATH.open("w", encoding="utf-8") as file:
        json.dump(evaluations, file, ensure_ascii=False, indent=2)


def main():
    results = load_results()
    evaluations = []

    print("=" * 70)
    print("EVALUATING RESPONSES")
    print("=" * 70)

    for result in results:
        # Skip cases that errored out (no answer/trace to judge) — see
        # run_test_queries.py, which records those as their own rows
        # rather than crashing the batch.
        if result.get("error") or not result.get("trace_id"):
            print(f"[SKIP] #{result['id']} - no answer/trace to evaluate")
            continue

        scores = evaluate_and_score(result["question"], result["answer"], result["trace_id"])

        evaluations.append(
            {
                "id": result["id"],
                "question": result["question"],
                "domain": result["actual_domain"],
                "trace_id": result["trace_id"],
                "relevance": scores.relevance,
                "completeness": scores.completeness,
                "accuracy": scores.accuracy,
                "rationale": scores.rationale,
            }
        )

        print(
            f"#{result['id']:<2} "
            f"relevance={scores.relevance} completeness={scores.completeness} accuracy={scores.accuracy} "
            f"- {result['question']}"
        )

    safe_flush()
    save_evaluations(evaluations)

    print("-" * 70)

    if evaluations:
        for dimension in DIMENSIONS:
            average = sum(e[dimension] for e in evaluations) / len(evaluations)
            print(f"Average {dimension}: {average:.1f}/10")

    print(f"\nEvaluated {len(evaluations)}/{len(results)} responses.")
    print(f"Saved to: {EVALUATION_PATH}")
    print("Scores were also attached to each response's Langfuse trace via the Score API.")


if __name__ == "__main__":
    main()
