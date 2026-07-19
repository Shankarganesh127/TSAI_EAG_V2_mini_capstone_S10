# Momentum

Momentum is a learning-focused agent architecture that turns questions into validated answers through explicit cognitive, execution, and reflection stages. It combines semantic session memory, local document RAG, reusable web-search knowledge, MCP tools, a Markdown web chat, and background logging.

## Features

- Async state-driven agent with query enrichment, validation, reflection, and replanning
- Pydantic-validated stage outputs with one controlled malformed-JSON repair retry
- Persistent FAISS answer memory built from session logs
- Automatic index rebuilds when session logs or documents change
- Local PDF, Office, and text RAG with source metadata
- Web results and downloaded pages saved into the RAG corpus
- MCP servers for math, documents, PDFs, webpage conversion, and web search
- Gemini, OpenAI, Anthropic, and local OpenAI-compatible chat
- Independently configurable chat and embedding providers
- Momentum FastAPI UI with sanitized Markdown, plus an interactive CLI
- Rotating application logs, structured session logs, and automated tests

## Requirements and setup

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- A hosted-provider API key or local OpenAI-compatible endpoint
- Ollama with `nomic-embed-text` for the default document embedding setup
- Optional `ffmpeg` only for media handled by MarkItDown

```powershell
uv sync
```

Create a `.env` file and never commit secrets:

```dotenv
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.5-flash
GEMINI_API_KEY=your-key
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-2
```

Chat and embeddings are independent, so local chat can use hosted embeddings:

```dotenv
LLM_PROVIDER=local
LLM_MODEL=openai/gpt-oss:20b
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_API_KEY=dummy
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-2
GEMINI_API_KEY=your-key
```

Project overrides live in `config/project_config.yaml`; package defaults live beside their libraries.

## Running Momentum

### Web interface

```powershell
uv run agent-web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Answers render as sanitized Markdown, including headings, lists, tables, links, and code blocks. Runtime logs stay in the background.

| Endpoint | Method | Purpose |
|---|---:|---|
| `/` | GET | Momentum chat |
| `/api/query` | POST | Submit a query and receive text plus rendered HTML |
| `/api/health` | GET | Health check |

### CLI

```powershell
uv run python main.py
```

Use `exit`, `quit`, or `q` to stop. The LLM library also exposes `uv run llm-api`.

## End-to-end flow

```text
User query
   |
   +-- Semantic answer-memory search
   |      +-- qualifying hit -> return stored answer
   |
   +-- Local document RAG search
   |      +-- evidence -> attach source-labelled context
   |
   +-- Query enrichment
   |
   +-- perception -> context -> decision -> planning
   |      -> action -> observation -> validation
   |                         ^              |
   |                         +-- replan <- reflection
   |
   +-- Store successful answer in memory
   +-- Return Markdown to CLI or web
```

Memory and RAG failures are logged and execution continues without that source. Agent failures are surfaced rather than replaced with placeholder answers.

## How this architecture evolved from S10Share

This project was reviewed against the course reference supplied as `S10Share.zip`. The reference establishes perception, decision-making, iterative execution, MCP tools, session history, and memory-assisted reasoning. Momentum retains those ideas but reorganizes them around explicit package boundaries, typed state transitions, persistent semantic indexes, and shared CLI/web runtime services.

### Architecture comparison

| Concern | S10Share reference | Momentum implementation | Why it changed |
|---|---|---|---|
| Main orchestrator | A large `AgentLoop` manually controls memory, perception, decision, code execution, and repeated perception | `BaseAgent` advances an `AgentContext` through an explicit state machine | Makes transitions predictable, independently testable, and easier to extend |
| Cognitive model | Perception and Decision are the principal LLM components | Perception, Context Retrieval, Decision, and Planning are separate sub-agents | Separates query understanding, context selection, action choice, and plan construction |
| Execution model | Decision emits `CODE`, `CONCLUDE`, or `NOP`; generated Python is passed to a custom executor that can call MCP tools | Action produces a typed answer, followed by a separate Observation stage | Removes generated-code execution from the normal answer path and makes assessment explicit |
| Completion check | Perception judges both initial queries and step results with `original_goal_achieved` and `local_goal_achieved` | Validation has its own contract: `SUCCESS`, `NEED_REPLAN`, or `FAILED` | Completion is no longer mixed into perception |
| Retry loop | Nested loop logic creates new plan versions after unhelpful steps | Validation routes through `REFLECTION -> REPLAN -> ACTION`, bounded by `max_loops` | Failure analysis and revised planning become named states |
| Runtime data | `AgentSession`, `Step`, and `PerceptionSnapshot` are dataclasses | Pydantic models validate stage inputs and outputs | Rejects malformed cross-stage data early |
| Structured output | JSON is split or extracted with regex; missing fields get defaults; some failures enter `pdb` | JSON is schema-validated; fences are handled safely; malformed JSON gets one controlled repair retry | Avoids debugger stops and prevents invalid data silently entering the pipeline |
| LLM access | Perception and Decision instantiate Gemini directly; `ModelManager` separately supports Gemini/Ollama | One client supports Gemini, OpenAI, Anthropic, and local OpenAI-compatible endpoints | Centralizes providers, errors, temperature, and credentials |
| Historical memory | Every query scans JSON logs and ranks text with RapidFuzz, without a minimum acceptance threshold | Successful turns are embedded once in persistent FAISS and filtered by cosine similarity | Provides semantic matching and avoids rescanning every log per query |
| Memory freshness | Session files are read directly during every search | Logs are fingerprinted at startup; changes atomically rebuild the answer-memory database | Keeps fast persistent retrieval synchronized with its sources |
| Current-run failures | Up to three failed steps are kept in a temporary list | Failures live in `AgentContext`, reflection state, and structured session events | Uses one state and audit model |
| Session persistence | A complete mutable session snapshot is repeatedly overwritten | Append-style stage events are stored by date and session | Preserves a chronological trace of memory, RAG, results, and errors |
| Document retrieval | A document MCP and prebuilt FAISS files ship with the reference | The corpus is fingerprinted and automatically rebuilt after additions, edits, or deletions | Makes indexing reproducible and current |
| Web knowledge | Search and fetched pages feed the active tool step | Results are also saved under `mcp_lib/Documents/web` and indexed | Turns one-time research into reusable local knowledge |
| MCP configuration | Machine-specific absolute `cwd` values and numbered server scripts | Named servers and parameters use package-relative paths | Makes the project portable |
| User experience | Terminal prints the complete live trace | CLI and Momentum web chat share a runtime; web logs remain in the background | Gives users a clean answer surface without losing diagnostics |
| Output | Terminal prints `solution_summary` as plain text | API returns plain Markdown and sanitized rendered HTML | Supports readable code, tables, headings, and links safely |
| Configuration | Settings span profiles, model JSON, MCP YAML, constants, and hard-coded models | Library defaults combine with project YAML and environment secrets | Separates reusable defaults from deployment choices |
| Tests | Component scripts plus perception/decision tests | Agent, memory, RAG, query, and web regression suites | Verifies boundaries and failure paths end to end |

### Reference control loop

The active S10Share entry point selects `agent_loop2.py`:

```text
Create AgentSession
  -> scan session JSON with RapidFuzz
  -> run Perception(query + memory)
  -> if already solved: complete
  -> run Decision to produce CODE / CONCLUDE / NOP
  -> execute generated code or accept conclusion
  -> run Perception again on the step result
  -> continue, replan, or stop
  -> overwrite the live session snapshot
```

This is an iterative **Perception–Decision–Action–Perception** loop. Perception both understands input and judges local/original goal completion. Decision combines planning with selection of the next executable step.

### Momentum control loop

Momentum splits those responsibilities and adds retrieval before cognition:

```text
Synchronize document and session indexes at startup
  -> search semantic answer memory
  -> search local document RAG on a memory miss
  -> enrich and ground the query
  -> Perception
  -> Context Retrieval
  -> Decision
  -> Planning
  -> Action
  -> Observation
  -> Validation
  -> Output, or Reflection and Replanning
  -> persist the successful answer and stage events
```

Orchestration policy is represented by states and transition rules rather than nested `if`/`while` logic. Each stage can be replaced or tested without rewriting the whole loop.

### Data model changes

S10Share stores a rich snapshot containing perception, plan versions, executable steps, and final state. Momentum uses two complementary records:

1. `AgentContext` is the in-memory workflow record: query, stage products, loop count, state history, output, and error.
2. `SessionLogger` is the durable event record: `query_received`, `memory_search`, `memory_hit`, `local_rag_search`, `agent_result`, `memory_store`, and errors.

This separates mutable execution state from the durable audit log. Event logs can be reprocessed into semantic memory without understanding each internal plan structure.

### Retrieval changes

The reference `MemorySearch` uses `rapidfuzz.partial_ratio` against historical queries and summaries. It is simple and transparent, but favors shared wording and reads all JSON files for every request. Momentum embeds successful query/answer pairs and uses FAISS, so differently worded questions with similar meaning can match. A threshold rejects weak candidates.

Momentum also separates **answer memory** from **knowledge RAG**:

- Answer memory retrieves a previously successful answer and may short-circuit execution.
- Document RAG retrieves source-labelled chunks that ground a new answer.
- Saved web content joins the document corpus rather than the answer cache.

The two indexes have separate fingerprints and rebuild lifecycles because their source data and retrieval roles differ.

### Execution and safety changes

S10Share executes Python generated by Decision through a custom AST-based executor, including wrappers that translate generated calls into MCP calls. Momentum's normal query path does not execute arbitrary LLM-generated Python. Instead, runtime startup discovers configured MCP tools and their exact schemas. A typed Tool Selection sub-agent chooses a registered tool and constructs its arguments; Execution calls it directly through `MultiMCP`, normalizes the result, and gives the evidence to Action. The selected name and raw result remain in `AgentContext` and are written as an `mcp_tool` session event.

The reference includes `QueryHeuristics` for URLs, file paths, sentence length, and a blacklist, but that class is not called by its active `main.py` or `agent_loop2.py`. Momentum therefore does not claim those dormant heuristics as an inherited runtime feature. It instead enforces API input limits, Pydantic schemas, sanitized Markdown, explicit provider errors, and bounded retries.

### Portability and maintainability changes

The reference MCP YAML contains absolute course-machine paths and numbered scripts. Momentum resolves scripts, documents, and indexes relative to package configuration. Runtime assembly is split into reusable functions for document synchronization, memory synchronization, cache lookup, RAG retrieval, grounding, execution, and storage. The CLI and web interface share these functions, preventing their behavior from drifting apart.
## Agent architecture

The core in `agent_base_lib` is an asynchronous state machine:

```text
START -> INPUT_RECEIVED -> PERCEPTION -> CONTEXT_RETRIEVAL -> DECISION
      -> PLANNING -> ACTION -> OBSERVATION -> VALIDATION
```

Validation chooses:

- `SUCCESS -> OUTPUT -> END`
- `NEED_REPLAN -> REFLECTION -> REPLAN -> ACTION`
- `FAILED -> ERROR -> END`

A maximum loop count prevents endless reflection.

| Stage | Responsibility |
|---|---|
| Query enrichment | Correct language and extract intent, entities, sub-goals, and assumptions |
| Perception | Normalize the request and classify intent |
| Context retrieval | Identify relevant knowledge, memory, and tools |
| Decision | Select an action and approach |
| Planning | Create concise execution steps |
| Action | Generate the direct answer or use a capability |
| Observation | Assess completeness |
| Validation | Decide whether the answer satisfies the request |
| Reflection | Explain a validation failure |
| Replanning | Revise the approach before retrying |

Every sub-agent declares a Pydantic output model. Data must validate before entering the next stage. An outer JSON Markdown fence is supported, while Markdown code fences inside answers are preserved.

If JSON is malformed, the framework makes one repair request that asks the model to fix syntax only, preserve the answer, and match the schema. The repaired result must still validate.

`BaseAgent(llm_client=None)` provides deterministic defaults for tests. Configured provider errors remain visible in normal execution.

## Semantic answer memory

Successful turns are embedded into FAISS under `memory_lib/vector_store`. A query searches historical turns first. A hit above the threshold is returned immediately; otherwise RAG and agent execution continue.

| Setting | Default | Meaning |
|---|---:|---|
| `base_dir` | `vector_store` | Memory directory |
| `similarity_threshold` | `0.88` | Minimum cosine similarity |
| `top_k` | `3` | Maximum candidates |

At startup, the session-log tree is content-fingerprinted. Added, edited, or deleted logs cause successful turns to be re-embedded into a fresh database that atomically replaces the old index. Unchanged data reuses the current index.

## Local document RAG

Put files under `mcp_lib/Documents/`. Supported PDF, Office, and text-like content is recursively extracted, chunked with overlap, embedded, and stored under `mcp_lib/faiss_index`.

The corpus is fingerprinted at startup. Changes trigger a fresh index. On each memory miss, qualifying source-labelled chunks are attached to the agent request with grounding instructions.

Defaults in `mcp_lib/default_mcp_config.yaml`:

- Ollama endpoint: `http://localhost:11434/api/embeddings`
- Model: `nomic-embed-text`
- Top K: 5; minimum similarity: 0.35
- Chunk size: 256 words; overlap: 40 words

## Live MCP execution

At startup, `create_runtime()` initializes `MultiMCP`, scans all configured servers, rejects duplicate names, and builds a catalog containing each tool's name, owning server, description, and exact input schema. The same registry is injected into `BaseAgent` for both CLI and web requests.

During Action:

1. `ToolSelectionAgent` receives the normalized query, chosen action, plan, and complete discovered catalog.
2. It returns a schema-validated tool name and argument object, or `null` when no tool is useful.
3. Execution rejects invented or unavailable names.
4. `MultiMCP.call_tool()` invokes the owning server over MCP stdio with a 45-second overall timeout.
5. MCP content blocks are normalized and supplied to `ActionAgent`.
6. Action grounds its answer in the result and preserves URLs or document Source fields.
7. The tool and result are retained in `action_result` and logged through MCP audit events.
8. Web tools save their content immediately, return the result, and let the host schedule document indexing as a tracked background task.

When Decision chooses `search_web`, Execution guarantees a fallback to `duckduckgo_search_results` if the selector returns no tool. This prevents a current-information request from silently becoming an answer based only on model knowledge.

The discovered catalog currently contains 22 tools:

- Math and time: `add`, `subtract`, `multiply`, `divide`, `power`, `cbrt`, `factorial`, `remainder`, `sin`, `cos`, `tan`, `mine`, and `current_time`
- Transformations: `strings_to_chars_to_int`, `int_list_to_exponential_sum`, and `fibonacci_numbers`
- Images: `create_thumbnail`
- Documents: `search_stored_documents_rag`, `convert_webpage_url_into_markdown`, and `extract_pdf`
- Web: `duckduckgo_search_results` and `download_raw_html_from_url`

Tool discovery failure is non-fatal at startup: it is logged and the agent continues without MCP tools. A selected-tool failure is surfaced as an execution error rather than hidden by an ungrounded answer.
## Web knowledge and MCP

The web MCP searches DuckDuckGo and downloads page content. Results are stored in `mcp_lib/Documents/web/` and become reusable local RAG knowledge.

| Server | Capabilities |
|---|---|
| `math` | Arithmetic, roots, trigonometry, factorial, Fibonacci, and conversions |
| `documents` | Local RAG, PDF extraction, and webpage-to-Markdown |
| `websearch` | DuckDuckGo results and webpage fetching |

Run smoke checks with `uv run python mcp_lib/test_mcp_servers.py`. Some require network access, Ollama, or an existing index.

## Provider variables

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini`, `openai`, `anthropic`, or `local` |
| `LLM_MODEL` / `LLM_TEMPERATURE` | Chat model and sampling |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini credentials |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI credentials and endpoint |
| `ANTHROPIC_API_KEY` | Anthropic credentials |
| `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_API_KEY` | Local endpoint |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | Embedding configuration |

## Logs and configuration

Runtime diagnostics go to `logs/app.log`, rotating at 10 MB with five backups. Structured events go to:

```text
session_lib/session_logs/YYYY/MM/DD/<session-id>.json
```

Events include queries, memory searches and hits, RAG evidence, answers, memory storage, and errors. They are also the source used to rebuild answer memory.

Every call to `AgentContext.transition_to()` produces a `state_transition` entry in both `logs/app.log` and the active session JSON. Each entry records `from_state`, `to_state`, `loop_count`, and the active query. This includes successful paths, reflection/replanning paths, and transitions through `ERROR` and `END`.

MCP execution produces paired audit events:

- `mcp_tool_call` records the exact registered tool, schema-shaped arguments, and selected agent action before invocation.
- `mcp_tool_result` records the tool, execution duration in milliseconds, and normalized result.
- `mcp_tool_error` records the tool, arguments, elapsed duration, and exception when invocation fails.

Events are persisted immediately through `SessionLogger`, so an MCP failure remains visible even when it prevents Action from producing a final response.

| File | Responsibility |
|---|---|
| `config/project_config.yaml` | Project logging and LLM overrides |
| `llm_api_lib/default_config.yaml` | LLM defaults |
| `logging_lib/default_config.yaml` | Rotating-log defaults |
| `session_lib/default_config.yaml` | Session-log settings |
| `memory_lib/default_config.yaml` | Answer-memory settings |
| `mcp_lib/default_mcp_config.yaml` | MCP, RAG, and web settings |

## Project layout

```text
agent_base_lib/    State machine, stages, models, and tests
config/            Project configuration
llm_api_lib/       Chat and embedding clients
logging_lib/       Rotating logging
memory_lib/        Semantic memory and RAG helpers
mcp_lib/           MCP servers, corpus, and index
query_lib/         Query enrichment and context models
session_lib/       JSON session history
web/               Momentum frontend and tests
main.py            Runtime and CLI
web_app.py         FastAPI app and Markdown rendering
```

## Tests

```powershell
uv run pytest -q
```

Tests cover state transitions, replanning, provider errors, code-fence preservation, malformed-JSON recovery, vector synchronization, document RAG, query enrichment, and web endpoints.

## Troubleshooting

### Current time for “my location”

The browser sends its IANA timezone, obtained from `Intl.DateTimeFormat().resolvedOptions().timeZone`, with every query. The API adds that location context without changing the visible user message.

Queries containing phrases such as “current time”, “local time”, “what time is it”, or “time now” bypass semantic answer memory, document RAG, and web-search selection before invoking the deterministic `current_time` MCP tool. Time answers are not stored in semantic memory because they become stale immediately. The tool uses Python `zoneinfo` to return the exact date, time, timezone abbreviation, daylight-saving state, and UTC offset. Common comparison cities are mapped to IANA zones, including London/Sheffield, New York, Paris, Tokyo, and Sydney.

CLI requests cannot infer the user's physical location. Specify an IANA timezone or city when using the terminal, for example: “What is the current time in Europe/London?”
### `ActionAgent returned invalid structured output`

The model violated its JSON schema. One syntax-repair attempt is made. Restart `agent-web` after updating code. If both attempts fail, inspect `logs/app.log`.

### `Could not store answer in vector memory: 404 Not Found`

The embedding endpoint, provider, or model is unavailable. Verify `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, credentials, and base URL. Chat can succeed while storage fails because the configurations are independent.

### Ollama connection failure

```powershell
ollama pull nomic-embed-text
ollama serve
```

### `ffmpeg` warning

This is optional media support. Text, PDF, Office, RAG, and chat features do not require it.

### Web search remains on `mcp_tool_call`

A web tool previously rebuilt the complete document index before returning its search result. If the configured Ollama embedding endpoint was unavailable or slow, the webpage appeared to hang even though DuckDuckGo had already completed.

Web MCP tools now only persist their result before returning. The CLI/web host schedules FAISS synchronization in the background, and every MCP invocation has a 45-second overall timeout. Check for `mcp_tool_result` or `mcp_tool_error` after `mcp_tool_call`. A `rag_sync_scheduled` event confirms that saved web content was queued for indexing.
### Web HTTP 500

Check the newest `agent_error` event in session JSON and the matching details in `logs/app.log`.

## Design principles

- Keep cognitive responsibilities explicit and testable.
- Validate data at every stage boundary.
- Treat memory and RAG as context, not hidden truth.
- Preserve sources for grounded answers.
- Rebuild indexes only when source data changes.
- Surface provider failures instead of hiding them.
- Share small reusable functions between CLI and web.
