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
