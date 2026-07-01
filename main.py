import asyncio
from config import configure_project
from agent_base_lib import BaseAgent, AgentState
from query_lib import enrich_query

config     = configure_project()
logger     = config["logger"]
llm_client = config.get("llm_client")

if llm_client:
    logger.info(f"LLM ready: {llm_client.config.provider} / {llm_client.config.model}")
else:
    logger.warning("No LLM configured — using stub responses")

agent = BaseAgent(llm_client=llm_client)


async def run_agent(raw_query: str) -> None:
    # Step 1: enrich the raw input before sending to the agent
    enriched = await enrich_query(raw_query, llm_client)
    if enriched.elaborated_query != raw_query.strip():
        logger.info(f"Enriched: {enriched.elaborated_query}")

    # Step 2: run the agent on the elaborated query
    ctx = await agent.run(enriched.elaborated_query)
    if ctx.error:
        logger.error(f"{ctx.error}")
    else:
        logger.info(f"Result: {ctx.final_output}")


async def main():
    logger.info("Agent ready. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            await run_agent(query)


if __name__ == "__main__":
    asyncio.run(main())
