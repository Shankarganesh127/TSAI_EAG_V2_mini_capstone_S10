import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent_base_lib import BaseAgent
from config import configure_project
from memory_lib import MemoryHit, SessionVectorMemory
from mcp_lib import MultiMCP
from query_lib import enrich_query
from session_lib import SessionLogger, get_default_session_config


@dataclass
class RuntimeServices:
    agent: BaseAgent
    llm_client: Any
    logger: logging.Logger
    session: SessionLogger
    memory: SessionVectorMemory | None
    mcp: MultiMCP | None
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)


WEB_CONTENT_TOOLS = {
    "duckduckgo_search_results",
    "download_raw_html_from_url",
}


def schedule_document_sync(
    context: Any,
    services: RuntimeServices,
) -> None:
    """Refresh RAG in the host process without delaying the user response."""
    tool = context.action_result.get("tool_used")
    if tool not in WEB_CONTENT_TOOLS:
        return
    task = asyncio.create_task(synchronize_document_corpus(services.logger))
    services.background_tasks.add(task)
    task.add_done_callback(services.background_tasks.discard)
    services.session.log(
        "rag_sync_scheduled",
        {"tool": tool, "background": True},
    )


async def initialize_mcp_tools(logger: logging.Logger) -> MultiMCP | None:
    """Discover every configured MCP tool without making startup brittle."""
    registry = MultiMCP()
    try:
        report = await registry.initialize()
    except Exception as exc:
        logger.warning("MCP initialization failed; continuing without tools: %s", exc)
        return None
    logger.info(
        "MCP ready: %d tools across %d servers",
        report["tools"],
        report["servers"],
    )
    return registry


async def synchronize_document_corpus(logger: logging.Logger) -> None:
    """Synchronize local files and saved web content with the RAG index."""
    try:
        from mcp_lib.mcp_server_documents import process_documents

        report = await asyncio.to_thread(process_documents)
    except Exception as exc:
        logger.warning("Document RAG synchronization failed: %s", exc)
        return

    if report["changed"]:
        logger.info(
            "Document corpus changed; rebuilt RAG DB from %d files (%d chunks)",
            report["files"],
            report["chunks"],
        )
    else:
        logger.info("Document RAG DB is current (%d chunks)", report["chunks"])


async def synchronize_session_memory(
    memory: SessionVectorMemory | None,
    logger: logging.Logger,
) -> None:
    """Synchronize successful session turns with the answer-memory index."""
    if memory is None:
        return

    logs_dir = get_default_session_config().base_dir
    report = await memory.sync_session_logs(logs_dir)
    if report.changed:
        logger.info(
            "Session data changed; rebuilt vector DB from %d files (%d records)",
            report.source_files,
            report.indexed_records,
        )
    else:
        logger.info("Session data unchanged; existing vector DB is current")


async def find_cached_answer(
    query: str,
    services: RuntimeServices,
) -> MemoryHit | None:
    """Return the strongest qualifying answer-memory hit, if one exists."""
    if services.memory is None:
        return None

    try:
        hits = await services.memory.search(query)
        serialized = [services.memory.describe_hit(hit) for hit in hits]
        services.session.log(
            "memory_search",
            {"query": query, "result_count": len(hits), "results": serialized},
        )
        if not hits:
            return None
        services.session.log("memory_hit", serialized[0])
        return hits[0]
    except Exception as exc:
        services.logger.warning(
            "Vector memory search failed; continuing with agent: %s", exc
        )
        services.session.log(
            "memory_error", {"operation": "search", "error": str(exc)}
        )
        return None


async def retrieve_rag_evidence(
    query: str,
    services: RuntimeServices,
) -> list[str]:
    """Retrieve source-labelled chunks from the local document corpus."""
    try:
        from mcp_lib.mcp_server_documents import search_documents

        results = await asyncio.to_thread(search_documents, query)
        evidence = [item for item in results if not item.startswith("ERROR:")]
        services.session.log(
            "local_rag_search",
            {"query": query, "result_count": len(evidence), "results": evidence},
        )
        return evidence
    except Exception as exc:
        services.logger.warning(
            "Local RAG search failed; continuing without document context: %s", exc
        )
        services.session.log(
            "rag_error", {"operation": "search", "error": str(exc)}
        )
        return []


def build_grounded_query(query: str, evidence: list[str]) -> str:
    """Combine a user query with retrieved evidence and grounding rules."""
    if not evidence:
        return query
    context = "\n\n---\n\n".join(evidence)
    return (
        f"{query}\n\nUse the following retrieved local evidence when answering. "
        "Cite its Source fields and do not invent unsupported details:\n\n"
        f"{context}"
    )


async def execute_agent_query(
    original_query: str,
    grounded_query: str,
    services: RuntimeServices,
) -> str | None:
    """Enrich and execute a query, recording either its answer or error."""
    enriched = await enrich_query(grounded_query, services.llm_client)
    if enriched.elaborated_query != original_query.strip():
        services.logger.info("Enriched: %s", enriched.elaborated_query)

    context = await services.agent.run(enriched.elaborated_query)
    schedule_document_sync(context, services)
    if context.error:
        services.logger.error("%s", context.error)
        services.session.log(
            "agent_error", {"query": original_query, "error": context.error}
        )
        return None

    answer = str(context.final_output)
    services.logger.info("Result: %s", answer)
    services.session.log(
        "agent_result", {"query": original_query, "answer": answer}
    )
    return answer


async def store_answer_memory(
    query: str,
    answer: str,
    services: RuntimeServices,
) -> None:
    """Persist a successful answer in semantic session memory."""
    if services.memory is None:
        return
    try:
        await services.memory.add(
            query,
            answer,
            services.session.session_id,
            str(services.session.path),
        )
        services.session.log("memory_store", {"query": query, "stored": True})
    except Exception as exc:
        services.logger.warning("Could not store answer in vector memory: %s", exc)
        services.session.log(
            "memory_error", {"operation": "store", "error": str(exc)}
        )


def is_current_time_query(query: str) -> bool:
    """Return true for clock queries whose answers become stale immediately."""
    lowered = query.lower()
    return any(
        phrase in lowered
        for phrase in ("current time", "local time", "what time is it", "time now")
    )


async def run_agent(query: str, services: RuntimeServices) -> str | None:
    """Resolve one user query through cache, local RAG, and agent execution."""
    services.session.log("query_received", {"query": query})
    volatile = is_current_time_query(query)

    if volatile:
        services.session.log(
            "memory_bypass",
            {"query": query, "reason": "current-time query"},
        )
        cached = None
    else:
        cached = await find_cached_answer(query, services)
    if cached is not None:
        services.logger.info(
            "Result (memory, similarity %.3f): %s", cached.score, cached.answer
        )
        return cached.answer

    if volatile:
        services.session.log(
            "rag_bypass",
            {"query": query, "reason": "current-time query"},
        )
        evidence = []
    else:
        evidence = await retrieve_rag_evidence(query, services)
    grounded_query = build_grounded_query(query, evidence)
    answer = await execute_agent_query(query, grounded_query, services)
    if answer is not None and not volatile:
        await store_answer_memory(query, answer, services)
    elif answer is not None:
        services.session.log(
            "memory_store_skipped",
            {"query": query, "reason": "current-time query"},
        )
    return answer


async def create_runtime(background_logs: bool = False) -> RuntimeServices:
    """Configure dependencies and synchronize persistent stores."""
    config = configure_project(log_console_enabled=not background_logs)
    logger = config["logger"]
    llm_client = config.get("llm_client")
    if llm_client is None:
        logger.warning("No LLM configured - using stub responses")
    else:
        logger.info(
            "LLM ready: %s / %s",
            llm_client.config.provider,
            llm_client.config.model,
        )

    await synchronize_document_corpus(logger)
    mcp = await initialize_mcp_tools(logger)
    memory = SessionVectorMemory(llm_client) if llm_client is not None else None
    await synchronize_session_memory(memory, logger)

    session = SessionLogger(str(uuid.uuid4()), "interactive session")
    return RuntimeServices(
        agent=BaseAgent(
            llm_client=llm_client,
            tool_registry=mcp,
            event_handler=session.log,
        ),
        llm_client=llm_client,
        logger=logger,
        session=session,
        memory=memory,
        mcp=mcp,
        background_tasks=set(),
    )


async def main() -> None:
    services = await create_runtime()
    services.logger.info("Agent ready. Type 'exit' to quit.")

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            await run_agent(query, services)


if __name__ == "__main__":
    asyncio.run(main())
