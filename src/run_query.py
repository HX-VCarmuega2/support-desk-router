"""Command-line entry point for asking the support desk a question.

Same interaction pattern used in the M1 and M2 projects: pass the
question as an argument, or run with no argument to be prompted for one.
"""

import sys

from openai import OpenAIError

from src.agents.orchestrator import route
from src.errors import ClassificationError, InvalidQuestionError


def main():
    if len(sys.argv) > 1:
        question = sys.argv[1].strip()
    else:
        question = input("Enter your question: ").strip()

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

    print("\nSources:")
    for chunk in result["chunks"]:
        print(f"  [{chunk['similarity']:.3f}] {chunk['question']}")

    if result.get("trace_id"):
        print(f"\nTrace ID: {result['trace_id']}")


if __name__ == "__main__":
    main()
