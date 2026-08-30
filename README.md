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
- [ ] Dependencies and environment setup
- [ ] Domain knowledge bases (HR, Tech, Finance)
- [ ] Vector stores per domain
- [ ] HR RAG agent
- [ ] Tech and Finance RAG agents
- [ ] Orchestrator with conditional routing (LangGraph)
- [ ] Test query suite
- [ ] Langfuse tracing
- [ ] Technical decisions writeup
- [ ] (Bonus) Evaluator agent with Langfuse Score API

## Setup

_To be documented once dependencies are defined (next step)._
