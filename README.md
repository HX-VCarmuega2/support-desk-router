# Support Desk Router

Support Desk Router is a multi-agent orchestration system for a fictional mid-size SaaS company, **Meridian Cloud**. An **orchestrator** classifies the intent of an incoming employee support question (HR, IT/Tech, or Finance) and conditionally routes it to a specialized **RAG agent** for that domain, which answers grounded in that domain's internal documentation.

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
│   ├── hr_docs/{hr_faq.md, index/}
│   ├── tech_docs/{tech_faq.md, index/}
│   └── finance_docs/{finance_faq.md, index/}
│
├── src/
│   ├── domains.py           # domain -> document/index path registry
│   ├── chunking.py          # FAQ-aware chunking
│   ├── embeddings.py        # OpenAI embeddings client
│   ├── vector_store.py      # builds the 3 FAISS indices
│   ├── retriever.py         # searches one domain's index
│   ├── observability.py     # Langfuse client/handler
│   ├── errors.py            # project-specific exceptions
│   ├── run_query.py         # CLI: ask the system a question
│   ├── run_test_queries.py  # routing-accuracy test runner
│   ├── test_error_handling.py # error-path test suite
│   └── agents/
│       ├── orchestrator.py # LangGraph: classify + conditional routing
│       ├── hr_agent.py
│       ├── tech_agent.py
│       └── finance_agent.py
│
├── outputs/
│   └── test_results.json
│
├── test_queries.json
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Python 3.13 was used for this project.

1. Clone this repository and move into it:

   ```bash
   git clone https://github.com/HX-VCarmuega2/support-desk-router.git
   cd support-desk-router
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Git Bash on Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and **replace every placeholder value with your own real credentials** — the project will not run with the placeholders as-is:

   ```env
   OPENAI_API_KEY=your-key-here        # <- replace with your real OpenAI API key
   LLM_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small

   LANGFUSE_PUBLIC_KEY=pk-lf-xxx        # <- replace with your real Langfuse public key
   LANGFUSE_SECRET_KEY=sk-lf-xxx        # <- replace with your real Langfuse secret key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

   - `OPENAI_API_KEY` is required for everything (embeddings + generation) — get one at [platform.openai.com](https://platform.openai.com/api-keys).
   - Langfuse keys come from **Settings → API Keys** in a [Langfuse Cloud](https://cloud.langfuse.com) project (free tier is enough).
   - **Configuration gotcha:** the variable must be named exactly `LANGFUSE_HOST`, not `LANGFUSE_BASE_URL` (used in some Langfuse tutorials). If it's misnamed, `python-dotenv` silently loads `None` for the host and requests fail instead of raising a clear "missing config" error.

## How to Run

The project is multiple Python modules rather than a single notebook, so "cell order" becomes "run order". The vector indices are already built and committed to the repo, so steps 1 is only needed if you edit a knowledge base and want to regenerate its index.

1. **(Optional) Rebuild the vector stores**, only needed after editing a file under `data/*_docs/*.md`:

   ```bash
   python -m src.vector_store
   ```

2. **Ask the support desk a question** — this is the main entry point:

   ```bash
   python -m src.run_query "How many weeks of paid parental leave do I get?"
   ```

   Or run it with no argument to be prompted interactively:

   ```bash
   python -m src.run_query
   # Enter your question: How many weeks of paid parental leave do I get?
   ```

   By default this prints an end-user-facing view: the department, the answer, and the related topic titles it was grounded in — no internal detail. Add `--debug` to also see each source's similarity score and the Langfuse trace ID, for troubleshooting a specific answer:

   ```bash
   python -m src.run_query "How many weeks of paid parental leave do I get?" --debug
   ```

   The same full detail is always in the Langfuse trace regardless of this flag — `--debug` only controls what gets printed to the terminal.

3. **(Development) Try one domain agent directly**, bypassing the orchestrator (retrieval + generation for a single domain only, no routing):

   ```bash
   python -m src.agents.hr_agent
   ```

4. **Run the full test query suite** against the orchestrator and get a routing-accuracy report:

   ```bash
   python -m src.run_test_queries
   ```

5. **Run the error-handling checks** (does the system fail predictably on bad input, not just succeed on good input):

   ```bash
   python -m src.test_error_handling
   ```

## Usage Example

End-user view (default):

```text
$ python -m src.run_query "How many weeks of paid parental leave do I get?"

Classifying question...
Routed to: hr. Retrieving context and generating answer...

============================================================
Routed to: hr
============================================================

You are eligible for 16 weeks of paid parental leave if you are the
birthing parent, and 10 weeks if you are the non-birthing parent.

Related topics:
  - How many weeks of paid parental leave are offered?
  - Is parental leave available to non-birthing parents?
  - Can an employee take unpaid leave beyond their PTO balance?
```

Debug view (`--debug`) — same run, with similarity scores and the trace ID for troubleshooting:

```text
Sources:
  [0.679] How many weeks of paid parental leave are offered?
  [0.617] Is parental leave available to non-birthing parents?
  [0.486] Can an employee take unpaid leave beyond their PTO balance?

Trace ID: 98e664a6d1e0d8c222a14df1a65d781f
```

To use the orchestrator from your own code instead of the CLI:

```python
from src.agents.orchestrator import route

result = route("How many weeks of paid parental leave do I get?")

result["domain"]     # "hr"
result["answer"]     # the grounded answer
result["chunks"]     # source chunks used (id, section, question, similarity)
result["trace_id"]   # Langfuse trace id for this run
```

## Technical Decisions

### Knowledge bases

Each domain's knowledge base is a single FAQ-style Markdown document (`data/<domain>_docs/<domain>_faq.md`), following the same FAQ-aware format used in the M2 project (`peopleflow-rag-support`): a `SECTION` header groups related FAQs, and each `FAQ` entry pairs one question with one self-contained answer paragraph. This format was reused because it already proved reliable for chunking — each question/answer pair becomes exactly one retrieval chunk, so chunk count is predictable and every chunk is a complete semantic unit.

All three domains describe the same fictional company, Meridian Cloud, so that cross-domain test queries (a question that could plausibly belong to more than one department) are meaningful.

| Domain  | File                                | FAQ entries |
| ------- | ------------------------------------ | ----------: |
| HR      | `data/hr_docs/hr_faq.md`             |          54 |
| Tech/IT | `data/tech_docs/tech_faq.md`         |          54 |
| Finance | `data/finance_docs/finance_faq.md`   |          54 |

### Vector stores

Each domain has its own FAISS index (`faiss.IndexFlatIP` over L2-normalized `text-embedding-3-small` vectors, i.e. cosine similarity) — HR, Tech, and Finance are never mixed into a shared index. This is what makes retrieval domain-aware: a question routed to the HR agent can only ever retrieve HR chunks, even if a Finance chunk happens to be semantically close. The indices are committed to the repo (`data/*_docs/index/`) so the project runs without needing to pay for re-embedding just to try it.

### RAG agents

Each domain agent (`src/agents/<domain>_agent.py`) is a LangChain LCEL chain with the same shape:

```text
retrieve (top-k chunks from that domain's FAISS index)
    -> format context
    -> prompt (system instructions + context + question)
    -> LLM
    -> {"answer": str, "chunks": [...]}
```

Retrieval is wrapped as a `RunnableLambda` step inside the chain, rather than called as a plain Python function before the chain runs. This keeps retrieval as part of the LangChain execution graph, so it appears as its own traced span in Langfuse — needed to debug failed retrievals, not just bad final answers.

The prompt instructs the model to answer strictly from the retrieved context and to say so explicitly when the context is insufficient, instead of guessing.

### Orchestrator

`src/agents/orchestrator.py` classifies each question into a department and routes it to that department's agent, using LangGraph:

```text
START -> classify -> (conditional edge, based on classified domain) -> hr | tech | finance -> END
```

**Why LangGraph instead of an if/elif in Python:** the branch taken depends on a value computed at runtime (the classifier's output), which is exactly what LangGraph's conditional edges model — a shared `OrchestratorState`, a node that writes a value into it, and an edge function that reads that value to pick the next node. The state also carries the question, domain, answer, and source chunks together as one object, instead of separate variables threaded through function calls.

**Classification** uses `.with_structured_output()` with a Pydantic model restricted to `Literal["hr", "tech", "finance"]`, rather than asking the LLM to output free text and parsing it — this makes an invalid/unroutable classification structurally impossible instead of something to defend against.

**Progress feedback:** `route()` runs the graph with `app.stream(...)` instead of `app.invoke(...)`. `invoke()` blocks until the entire graph finishes, which — since classification and generation are each a network call — can take several seconds with zero output, making the CLI look frozen. `stream()` yields each node's result as soon as that node finishes, so `route()` can print progress ("Classifying question...", then "Routed to: hr. Retrieving context and generating answer...") while the graph is still running.

### Observability (Langfuse)

Every call to `orchestrator.route()` is traced end-to-end: classification, the routing decision, retrieval, and generation all appear nested under a single trace, not as separate disconnected traces per step.

**How tracing is threaded through the graph:** `route()` builds one `CallbackHandler` and passes it in `config={"callbacks": [...]}` to `app.invoke()`. Each LangGraph node declares a second `config: RunnableConfig` parameter — LangGraph injects the run's config automatically — and forwards that same `config` into the agent it calls (`hr_agent.answer(question, config=config)`), which forwards it again into its own LCEL chain (`chain.invoke(..., config=config)`). Because it is the same config object all the way down, every step reports to the same trace instead of starting a new one.

Each domain agent can also run standalone (`python -m src.agents.hr_agent`) — in that case it builds its own handler and flushes its own trace, so tracing works whether an agent is invoked directly or through the orchestrator.

Verified trace structure (one real trace, `finance` domain question):

```text
LangGraph (root)
└─ classify
   └─ ChatOpenAI (classification call)
└─ route_by_domain
└─ finance
   └─ retrieve_finance_chunks   <- retrieval as its own named, inspectable step
   └─ ChatPromptTemplate
   └─ ChatOpenAI (generation call)
   └─ StrOutputParser
```

Check the **Traces** view in your Langfuse project after running the orchestrator to see this structure and inspect inputs/outputs at every step.

### Test queries

`test_queries.json` has 16 questions: 4 per domain that are unambiguous ("clear"), plus 4 "edge_case" questions that specifically probe the HR/Finance payroll boundary and phrasing that could plausibly point to the wrong domain (e.g. "I lost my corporate card" vs. "I lost my laptop"). `python -m src.run_test_queries` runs all of them against the orchestrator and saves full results — including each run's Langfuse `trace_id`, for cross-referencing a specific result back to its full trace — to `outputs/test_results.json`.

Current result: **16/16 (100%)** routed to the expected domain.

### Error handling

`src/errors.py` defines the project's own exception types, kept separate from third-party exceptions (`openai.OpenAIError`, LangChain's parsing errors) so calling code can tell "our logic rejected this" apart from "the provider had a problem":

* **`InvalidQuestionError`** — raised by `orchestrator.route()` and every agent's `answer()` before any API call is made, if `question` is empty, whitespace-only, or not a string. Failing fast here avoids spending an API call on input that was never going to work.
* **`ClassificationError`** — `classify_intent()` retries once if the LLM fails to produce valid structured output (a malformed tool call, an incomplete response), then raises this instead of either crashing with a raw parser error or silently guessing a department.
* **Tracing never blocks a good answer.** `observability.safe_flush()` wraps the Langfuse flush call in its own `try/except`; if Langfuse is unreachable, a warning is emitted but the already-generated answer is still returned. An observability outage should never be the reason a correct answer fails to reach the caller.
* **`run_test_queries.py` isolates failures per question** — one question raising an exception is recorded as its own error row (with the exception type and message) instead of crashing the whole batch and losing every result already computed.

`python -m src.test_error_handling` is a small standalone suite covering the error paths specifically (empty/`None`/whitespace-only questions, an unconfigured domain name) — separate from `test_queries.json`, which only covers routing *correctness*, not failure behavior. None of these cases call the OpenAI or Langfuse APIs, since input validation happens first.

## Known Limitations

* **HR/Finance payroll boundary is inherently fuzzy.** HR covers compensation *decisions* (raises, pay bands); Finance covers payroll *mechanics* (direct deposit, pay schedule, tax forms). Initial testing misrouted a direct-deposit question to HR; the classifier's prompt was tightened to state the boundary explicitly, which fixed the observed cases in `test_queries.json`, but a differently-phrased question could still land on the wrong side.
* **The classifier always picks one of the three domains** — there is no "none of the above" path. A completely unrelated question (e.g. about the weather) still gets routed to whichever domain the LLM judges closest, and that agent will typically respond that it lacks relevant context, but the routing itself doesn't surface "this isn't a support question" as a distinct outcome.
* **Scoped to 3 domains (HR, Tech, Finance), not 4.** The project brief's scenario also mentions Legal; the deliverable requirements only require a minimum of 3 specialized agents, so Legal was left out of scope rather than added as a fourth shallow domain.
* **No conversation memory.** Each call to `route()` is independent — there is no multi-turn context, so a follow-up question like "and how do I request it?" would not know what "it" refers to.
* **Retries are limited to classification.** `classify_intent()` retries once on a malformed structured-output response, but embedding calls and answer generation have no retry/backoff — a transient rate limit or timeout there still propagates as an error. Acceptable for this project's scope, but a gap for a production deployment.
* **Knowledge bases are synthetic.** All FAQ content describes a fictional company (Meridian Cloud) generated for this project, not real internal documentation.

## Status

- [x] Project scaffold
- [x] Dependencies and environment setup
- [x] Domain knowledge bases (HR, Tech, Finance)
- [x] Vector stores per domain
- [x] HR RAG agent
- [x] Tech and Finance RAG agents
- [x] Orchestrator with conditional routing (LangGraph)
- [x] Test query suite
- [x] Langfuse tracing
- [x] Technical decisions writeup
- [ ] (Bonus) Evaluator agent with Langfuse Score API
