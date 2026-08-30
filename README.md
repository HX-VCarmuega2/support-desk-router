# Support Desk Router

Support Desk Router is a multi-agent orchestration system for a fictional SaaS support desk. An **orchestrator** classifies the intent of an incoming customer question (HR, IT/Tech, or Finance) and conditionally routes it to a specialized **RAG agent** for that domain, which answers grounded in that domain's internal documentation.

This project is being built incrementally, one working commit at a time. This README grows alongside the code — sections below get filled in as each part is implemented.

## Why this architecture

* **Orchestrator + specialized agents, not one generalist agent**: a single agent with all the documentation mixed together would retrieve irrelevant chunks across domains (an HR question pulling in Finance content, for example). Splitting retrieval by domain keeps each agent's knowledge base focused and its answers grounded.
* **LangGraph for the orchestrator**: routing here is conditional — the path taken depends on a classification result computed at runtime. LangGraph models this natively as a graph with a state and conditional edges, instead of encoding the branching as nested `if/else` logic.
* **LangChain for each RAG agent**: each domain agent is a standard retrieve-then-generate chain (retriever + prompt + LLM), which is exactly what LangChain's chain abstractions are built for.
* **Langfuse for tracing**: every run (classification → routing decision → retrieval → generation) is instrumented end-to-end, so misrouted questions or bad retrievals can be inspected after the fact instead of guessed at.

## Project Structure

```text
support-desk-router/
│
├── data/
│   ├── hr_docs/
│   ├── tech_docs/
│   └── finance_docs/
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── hr_agent.py
│   │   ├── tech_agent.py
│   │   └── finance_agent.py
│   └── multi_agent_system.py
│
├── outputs/
│
├── test_queries.json
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Status

- [x] Project scaffold
- [x] Dependencies and environment setup
- [x] Domain knowledge bases (HR, Tech, Finance)
- [x] Vector stores per domain
- [x] HR RAG agent
- [x] Tech and Finance RAG agents
- [ ] Orchestrator with conditional routing (LangGraph)
- [ ] Test query suite
- [ ] Langfuse tracing
- [ ] Technical decisions writeup
- [ ] (Bonus) Evaluator agent with Langfuse Score API

## Knowledge Bases

Each domain's knowledge base is a single FAQ-style Markdown document (`data/<domain>_docs/<domain>_faq.md`), following the same FAQ-aware format used in the M2 project (`peopleflow-rag-support`): a `SECTION` header groups related FAQs, and each `FAQ` entry pairs one question with one self-contained answer paragraph. This format was reused because it already proved reliable for chunking — each question/answer pair becomes exactly one retrieval chunk, so chunk count is predictable and every chunk is a complete semantic unit.

All three domains describe the same fictional company, **Meridian Cloud**, a mid-size B2B SaaS company, so that cross-domain test queries (e.g. a question that could plausibly belong to more than one department) are meaningful.

| Domain  | File                                | FAQ entries |
| ------- | ------------------------------------ | ----------: |
| HR      | `data/hr_docs/hr_faq.md`             |          54 |
| Tech/IT | `data/tech_docs/tech_faq.md`         |          54 |
| Finance | `data/finance_docs/finance_faq.md`   |          54 |

## Vector Stores

Each domain has its own FAISS index — HR, Tech, and Finance are never mixed into a shared index. This is what makes retrieval domain-aware: a question routed to the HR agent can only ever retrieve HR chunks, even if a Finance chunk happens to be semantically close.

Build (or rebuild, after editing a knowledge base) all three indices:

```bash
python -m src.vector_store
```

This pipeline, per domain:

1. Loads `data/<domain>_docs/<domain>_faq.md`
2. Splits it into FAQ-aware chunks (one chunk per question/answer pair)
3. Generates embeddings with `text-embedding-3-small`
4. Normalizes the vectors and builds a `faiss.IndexFlatIP` index (inner product on normalized vectors = cosine similarity)
5. Saves the index and chunk metadata under `data/<domain>_docs/index/`

Generated files (committed to the repo so the project runs without rebuilding):

```text
data/hr_docs/index/{faiss.index, chunks.json}
data/tech_docs/index/{faiss.index, chunks.json}
data/finance_docs/index/{faiss.index, chunks.json}
```

Current chunk counts: 54 per domain.

## RAG Agents

Each domain agent (`src/agents/<domain>_agent.py`) is a LangChain LCEL chain with the same shape:

```text
retrieve (top-k chunks from that domain's FAISS index)
    -> format context
    -> prompt (system instructions + context + question)
    -> LLM
    -> {"answer": str, "chunks": [...]}
```

Retrieval is wrapped as a `RunnableLambda` step inside the chain, rather than called as a plain Python function before the chain runs. This keeps retrieval as part of the LangChain execution graph, so once Langfuse tracing is added it appears as its own traced span — needed to debug failed retrievals, not just bad final answers.

The prompt instructs the model to answer strictly from the retrieved context and to say so explicitly when the context is insufficient, instead of guessing. Try any of them directly:

```bash
python -m src.agents.hr_agent
python -m src.agents.tech_agent
python -m src.agents.finance_agent
```

## Setup

Python 3.13 was used for this project.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```env
OPENAI_API_KEY=your-key-here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```
