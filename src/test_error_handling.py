"""Exercises the error-handling paths, as opposed to test_queries.json /
run_test_queries.py, which only exercise the happy path (does it route
correctly?). This answers a different question: does the system fail
predictably, with the right exception type, when given bad input?

None of these cases call the OpenAI or Langfuse APIs — input validation
happens before any API call is made, so this suite is fast and free to run.
"""

from src.agents import hr_agent, orchestrator
from src.domains import get_domain
from src.errors import InvalidQuestionError


def expect_raises(exception_type: type[Exception], fn) -> None:
    try:
        fn()
    except exception_type:
        return
    except Exception as error:
        raise AssertionError(
            f"expected {exception_type.__name__}, got {type(error).__name__}: {error}"
        ) from error

    raise AssertionError(f"expected {exception_type.__name__}, but no exception was raised")


CASES = [
    (
        "orchestrator.route('') raises InvalidQuestionError",
        lambda: expect_raises(InvalidQuestionError, lambda: orchestrator.route("")),
    ),
    (
        "orchestrator.route('   ') raises InvalidQuestionError (whitespace-only)",
        lambda: expect_raises(InvalidQuestionError, lambda: orchestrator.route("   ")),
    ),
    (
        "orchestrator.route(None) raises InvalidQuestionError",
        lambda: expect_raises(InvalidQuestionError, lambda: orchestrator.route(None)),
    ),
    (
        "hr_agent.answer('') raises InvalidQuestionError",
        lambda: expect_raises(InvalidQuestionError, lambda: hr_agent.answer("")),
    ),
    (
        "domains.get_domain('legal') raises ValueError (domain not configured)",
        lambda: expect_raises(ValueError, lambda: get_domain("legal")),
    ),
]


def main():
    failures = 0

    for description, run_case in CASES:
        try:
            run_case()
            print(f"[PASS] {description}")
        except AssertionError as error:
            print(f"[FAIL] {description}\n       -> {error}")
            failures += 1

    print()
    if failures:
        print(f"{failures}/{len(CASES)} error-handling checks failed.")
    else:
        print(f"All {len(CASES)} error-handling checks passed.")


if __name__ == "__main__":
    main()
