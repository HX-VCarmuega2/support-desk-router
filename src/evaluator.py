"""LLM-as-judge evaluator (bonus): scores a RAG response on relevance,
completeness, and accuracy (1-10 each), based only on the original
question and the final answer — per the assignment brief — then writes
each score onto that run's Langfuse trace via the Score API.

This runs as a separate, offline step over already-generated
(question, answer, trace_id) triples (see run_evaluation.py), not inside
orchestrator.route(). A production system would run this as a scheduled
batch job against recent traffic, not add another LLM call's worth of
latency and cost to every live user request.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.observability import get_langfuse_client

load_dotenv()

DIMENSIONS = ("relevance", "completeness", "accuracy")


class EvaluationScores(BaseModel):
    relevance: int = Field(ge=1, le=10, description="Does the answer address what was actually asked?")
    completeness: int = Field(
        ge=1,
        le=10,
        description="Does the answer cover the relevant specifics (numbers, conditions), not just a vague partial answer?",
    )
    accuracy: int = Field(
        ge=1,
        le=10,
        description="Does the answer look internally consistent, specific, and free of invented-sounding details?",
    )
    rationale: str = Field(description="One or two sentences explaining the three scores.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a quality reviewer for an internal support desk's answers. "
            "Score the ANSWER to the QUESTION on three 1-10 dimensions: "
            "relevance (does it address what was actually asked), "
            "completeness (does it cover the relevant specifics, not just a vague partial answer), "
            "and accuracy (does it look internally consistent and specific, without invented-sounding details). "
            "A clear 'I don't have enough information to answer that' should score high on accuracy "
            "and completeness when the question is genuinely outside a normal support scope — "
            "score it low only if the answer looks like it dodged a question it should have been able to answer.",
        ),
        ("human", "QUESTION:\n{question}\n\nANSWER:\n{answer}"),
    ]
)


def build_judge(model: str | None = None):
    llm = ChatOpenAI(model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    return JUDGE_PROMPT | llm.with_structured_output(EvaluationScores)


def evaluate(question: str, answer: str) -> EvaluationScores:
    judge = build_judge()
    return judge.invoke({"question": question, "answer": answer})


def score_trace(trace_id: str, scores: EvaluationScores) -> None:
    """
    Attach each dimension as its own named Langfuse score on the given
    trace, via the Score API, so the Langfuse UI can show, filter, and
    aggregate quality per trace and across many traces over time.
    """
    client = get_langfuse_client()

    for dimension in DIMENSIONS:
        client.create_score(
            trace_id=trace_id,
            name=dimension,
            value=getattr(scores, dimension),
            data_type="NUMERIC",
            comment=scores.rationale,
        )


def evaluate_and_score(question: str, answer: str, trace_id: str | None) -> EvaluationScores:
    scores = evaluate(question, answer)

    if trace_id:
        score_trace(trace_id, scores)

    return scores
