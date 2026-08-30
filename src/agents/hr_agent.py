"""HR RAG agent: retrieves from the HR knowledge base and answers grounded
in that context.

This is a LangChain LCEL chain, not a plain Python function calling the
retriever. Wrapping retrieval as a RunnableLambda step (instead of calling
search_chunks() before invoking the chain) means retrieval shows up as its
own traced step once Langfuse instrumentation is added, alongside the
prompt and the LLM call.
"""

import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.errors import InvalidQuestionError
from src.observability import get_callback_handler, safe_flush
from src.retriever import search_chunks

load_dotenv()

DOMAIN = "hr"
TOP_K = 3

SYSTEM_PROMPT = """You are the HR support assistant for Meridian Cloud, a fictional SaaS company.
Answer the employee's question using ONLY the context below, taken from the internal HR knowledge base.
If the context does not contain enough information to answer, say so explicitly instead of guessing.
Be concise and specific: include the exact numbers, timeframes, or thresholds from the context when present.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"Q: {c['question']}\nA: {c['text']}" for c in chunks)


def build_chain(model: str | None = None):
    """
    Build the HR agent as a single LCEL runnable:

      1. retrieve  -> top-k chunks from the HR FAISS index
      2. context   -> format those chunks into a text block
      3. generate  -> prompt the LLM with the question + context
      4. output    -> {"answer": str, "chunks": list[dict]}
    """
    llm = ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    generation = prompt | llm | StrOutputParser()

    retrieve = RunnableLambda(
        lambda inputs: search_chunks(inputs["question"], DOMAIN, k=TOP_K),
        name="retrieve_hr_chunks",
    )

    return (
        {
            "chunks": retrieve,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=lambda x: _format_context(x["chunks"]))
        | {
            "answer": generation,
            "chunks": itemgetter("chunks"),
        }
    )


def answer(question: str, config: RunnableConfig | None = None) -> dict:
    """
    Run the HR agent for one question.

    If `config` is provided (by the orchestrator), tracing callbacks in it
    are inherited so this run nests under the caller's trace. If not (the
    agent is run standalone), it creates and flushes its own trace.

    Returns {"domain": "hr", "answer": str, "chunks": list[dict]}.

    Raises InvalidQuestionError if `question` is empty or not a string.
    """
    if not isinstance(question, str) or not question.strip():
        raise InvalidQuestionError(
            f"question must be a non-empty string, got: {question!r}"
        )

    standalone_run = config is None

    if standalone_run:
        config = {"callbacks": [get_callback_handler()]}

    chain = build_chain()
    result = chain.invoke({"question": question}, config=config)

    if standalone_run:
        safe_flush()

    return {
        "domain": DOMAIN,
        "answer": result["answer"],
        "chunks": result["chunks"],
    }


if __name__ == "__main__":
    sample_questions = [
        "How many weeks of paid parental leave do I get?",
        "What is the laptop refresh cycle?",  # deliberately off-domain
    ]

    for question in sample_questions:
        print("=" * 60)
        print(f"Q: {question}")
        result = answer(question)
        print(f"\nA: {result['answer']}")
        print("\nRetrieved chunks:")
        for chunk in result["chunks"]:
            print(f"  [{chunk['similarity']:.3f}] {chunk['question']}")
        print()
