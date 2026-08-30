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
from src.observability import get_callback_handler, get_langfuse_client

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


def classify_intent(state: OrchestratorState, config: RunnableConfig) -> dict:
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    classifier = CLASSIFIER_PROMPT | llm.with_structured_output(IntentClassification)

    result = classifier.invoke({"question": state["question"]}, config=config)

    return {"domain": result.domain}


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

    Returns {"question", "domain", "answer", "chunks"}.
    """
    app = build_graph()
    config = {"callbacks": [get_callback_handler()]}

    result = app.invoke({"question": question}, config=config)

    get_langfuse_client().flush()

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
