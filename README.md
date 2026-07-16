# EAG v2 Mini Capstone

An asynchronous, staged agent framework that separates query enrichment,
cognition, execution, observation, validation, reflection, and replanning. It
also includes a multi-server MCP tool registry, provider-neutral LLM access,
rotating application logs, and JSON session logs.

## Requirements

- Python 3.13 or newer
- `uv`
- An API key for the provider selected in `config/project_config.yaml`, or a
  local OpenAI-compatible endpoint

## Setup

```powershell
uv sync
Copy-Item .env.example .env  # if an example file is present
uv run python main.py
```

The agent classes can run without an LLM when constructed with
`BaseAgent(llm_client=None)`; this deterministic mode is intended for tests and
local development. When a client is configured, provider and structured-output
errors are surfaced as agent errors instead of being replaced with stub output.

## Tests

```powershell
uv run pytest -q
```

The MCP smoke suite includes network- and local-index-dependent cases:

```powershell
uv run python mcp_lib/test_mcp_servers.py
```

## Architecture

The normal state flow is:

`input -> perception -> context -> decision -> planning -> action -> observation -> validation`

Failed validation produces reflection and a revised plan before another action.
Successful validation produces the final output; unrecoverable failures end in
the error state.

Successful CLI query/answer turns are also written to the session JSON log and
embedded into a persistent FAISS index under `memory_lib/vector_store`. Before
running the agent, a new query searches this index and reuses the highest-scoring
answer when its cosine similarity meets the configured threshold. Memory hits,
misses, storage operations, and errors are recorded in the active session log.
The threshold and search depth are configured in `memory_lib/default_config.yaml`.
At application startup, the session-log tree is content-fingerprinted. If any
source log was added, changed, or removed, all successful historical turns are
re-embedded and a fresh FAISS database atomically replaces the previous index.
When the fingerprint is unchanged, the existing database is reused.

Local RAG sources live under `mcp_lib/Documents`. Files are recursively
content-fingerprinted and chunked; additions, edits, and deletions cause a fresh
document FAISS index to be built. Supported content includes PDF, Office and
text-like files. Web-search results and downloaded page text produced by the web
MCP are saved under `mcp_lib/Documents/web` and indexed immediately. Each query
searches this corpus before the agent runs, and qualifying chunks are supplied
to the agent with source metadata for grounded answers.

Configuration defaults live beside each library. Project-level overrides are in
`config/project_config.yaml`, and MCP servers are declared in
`mcp_lib/default_mcp_config.yaml`.
