"""Orchestrator: classifies an incoming support question into a department
(HR, Tech, or Finance) and routes it to that department's RAG agent.

Why LangGraph and not a plain if/elif: the routing decision here depends
on a value computed at runtime (the classifier's output), and LangGraph's
StateGraph models exactly that — a shared state, a node that produces a
value, and a conditional edge that picks the next node based on that
value. The state also carries the final answer and its source chunks, so
the whole flow (classification -> retrieval -> generation) is one
inspectable object instead of scattered variables.
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.agents import finance_agent, hr_agent, tech_agent
from src.errors import ClassificationError, InvalidQuestionError
from src.observability import get_callback_handler, safe_flush

load_dotenv()

Domain = Literal["hr", "tech", "finance"]


class IntentClassification(BaseModel):
    """Structured output for the classifier node — forces the LLM to pick
    exactly one of the three known departments instead of returning free
    text that would need brittle parsing."""

    domain: Domain = Field(
        description=(
            "The department best suited to answer the question: "
            "'hr' for benefits, leave, onboarding, performance reviews, compensation bands/raises, "
            "and workplace conduct; "
            "'tech' for account access, hardware, software, VPN, or security questions; "
            "'finance' for expenses, billing, payroll processing (direct deposit, pay schedule, "
            "tax forms, pay discrepancies), and procurement. "
            "Payroll mechanics (how/when someone gets paid) belong to finance, even though HR "
            "decides raise amounts."
        )
    )


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify employee support questions into exactly one department: "
            "hr, tech, or finance. Always choose the single best match, even if the "
            "question could loosely relate to more than one department.",
        ),
        ("human", "{question}"),
    ]
)


class OrchestratorState(TypedDict):
    question: str
    domain: str
    answer: str
    chunks: list[dict]


CLASSIFICATION_ATTEMPTS = 2


def classify_intent(state: OrchestratorState, config: RunnableConfig) -> dict:
    """
    Classify the question into a department.

    The LLM is expected to return structured output matching
    IntentClassification, but a malformed/incomplete response (the model
    fails to produce valid structured output) is retried once before
    giving up, rather than silently guessing a department or crashing
    with a raw parsing error from deep inside LangChain.
    """
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    classifier = CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification)

    last_error: Exception | None = None

    for attempt in range(1, CLASSIFICATION_ATTEMPTS + 1):
        try:
            result = classifier.invoke({"question": state["question"]}, config=config)
            return {"domain": result.domain}

        except Exception as error:  # noqa: BLE001 - deliberately broad: any
            # failure to produce valid structured output (bad tool call,
            # incomplete JSON, validation error) should trigger a retry,
            # not just a specific exception type that may vary by
            # LangChain/provider version.
            last_error = error

    raise ClassificationError(
        f"Could not classify question after {CLASSIFICATION_ATTEMPTS} attempts: "
        f"{state['question']!r}"
    ) from last_error


def run_hr_agent(state: OrchestratorState, config: RunnableConfig) -> dict:
    result = hr_agent.answer(state["question"], config=config)
    return {"answer": result["answer"], "chunks": result["chunks"]}


def run_tech_agent(state: OrchestratorState, config: RunnableConfig) -> dict:
    result = tech_agent.answer(state["question"], config=config)
    return {"answer": result["answer"], "chunks": result["chunks"]}


def run_finance_agent(state: OrchestratorState, config: RunnableConfig) -> dict:
    result = finance_agent.answer(state["question"], config=config)
    return {"answer": result["answer"], "chunks": result["chunks"]}


def route_by_domain(state: OrchestratorState) -> str:
    """Reads the domain the classifier node already wrote into state and
    tells LangGraph which node to run next."""
    return state["domain"]


def build_graph():
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify_intent)
    graph.add_node("hr", run_hr_agent)
    graph.add_node("tech", run_tech_agent)
    graph.add_node("finance", run_finance_agent)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_by_domain,
        {"hr": "hr", "tech": "tech", "finance": "finance"},
    )

    graph.add_edge("hr", END)
    graph.add_edge("tech", END)
    graph.add_edge("finance", END)

    return graph.compile()


def route(question: str) -> dict:
    """
    Run the full orchestrator for one question, traced end-to-end in
    Langfuse: classification, the routing decision, retrieval, and
    generation all nest under a single trace because the same callback
    handler is threaded through every node via `config`.

    Returns {"question", "domain", "answer", "chunks", "trace_id"}.
    `trace_id` is the Langfuse trace id for this run (or None if tracing
    is unavailable), so a specific result can be cross-referenced back to
    its full trace in the Langfuse dashboard for auditing.

    Raises InvalidQuestionError if `question` is empty or not a string,
    ClassificationError if the intent classifier fails after retrying.
    """
    if not isinstance(question, str) or not question.strip():
        raise InvalidQuestionError(
            f"question must be a non-empty string, got: {question!r}"
        )

    app = build_graph()
    handler = get_callback_handler()
    config = {"callbacks": [handler]}

    # app.stream() (instead of app.invoke()) yields each node's output as
    # soon as that node finishes, so progress can be reported while the
    # graph is still running. invoke() would only return once everything
    # is done, which — since classification and generation are each a
    # network call — can look like the program is stuck.
    result: dict = {"question": question}
    print("Classifying question...")

    for step in app.stream(result, config=config):
        for node_name, update in step.items():
            result.update(update)

            if node_name == "classify":
                print(f"Routed to: {result['domain']}. Retrieving context and generating answer...")

    result["trace_id"] = handler.last_trace_id

    safe_flush()

    return result


if __name__ == "__main__":
    sample_questions = [
        "How many weeks of paid parental leave do I get?",
        "My VPN keeps disconnecting, what should I do?",
        "What is the reimbursement limit for meals during business travel?",
    ]

    for question in sample_questions:
        print("=" * 60)
        print(f"Q: {question}")

        result = route(question)

        print(f"Routed to: {result['domain']}")
        print(f"\nA: {result['answer']}")
        print()
