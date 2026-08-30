"""Command-line entry point for asking the support desk a question.

Same interaction pattern used in the M1 and M2 projects: pass the
question as an argument, or run with no argument to be prompted for one.

By default the output is what an end user should see: just the answer
and the related topics it came from, with no internal debugging detail.
--debug adds the similarity score for each source and the Langfuse trace
ID, for whoever is troubleshooting a bad answer rather than just reading
it — the same underlying data is always visible in the Langfuse trace
either way, this flag only controls what's printed to the terminal.
"""

import argparse
import sys

from openai import OpenAIError

from src.agents.orchestrator import route
from src.errors import ClassificationError, InvalidQuestionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the Meridian Cloud support desk a question.")
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask. If omitted, you will be prompted for one.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also show each source's similarity score and the Langfuse trace ID.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    question = args.question.strip() if args.question else input("Enter your question: ").strip()

    try:
        result = route(question)

    except InvalidQuestionError as error:
        print(f"Invalid input: {error}")
        sys.exit(1)

    except ClassificationError as error:
        print(f"Could not classify the question: {error}")
        sys.exit(1)

    except OpenAIError as error:
        print(f"OpenAI API error: {error}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Routed to: {result['domain']}")
    print("=" * 60)

    print(f"\n{result['answer']}")

    if args.debug:
        print("\nSources:")
        for chunk in result["chunks"]:
            print(f"  [{chunk['similarity']:.3f}] {chunk['question']}")

        if result.get("trace_id"):
            print(f"\nTrace ID: {result['trace_id']}")
    else:
        print("\nRelated topics:")
        for chunk in result["chunks"]:
            print(f"  - {chunk['question']}")


if __name__ == "__main__":
    main()
