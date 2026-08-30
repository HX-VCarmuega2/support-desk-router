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
- [ ] Vector stores per domain
- [ ] HR RAG agent
- [ ] Tech and Finance RAG agents
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
