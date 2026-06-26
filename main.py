
import asyncio
from config import configure_project

# Configure all project components using defaults from config/project_config.yaml
config = configure_project()

logger = config["logger"]
llm_client = config["llm_client"]


async def main():
    """Main entry point - demonstrates local LLM setup"""
    logger.info("Hello from eag-v2-mini-capstone-s10!")
    
    # Test LLM connection
    try:
        logger.info("Testing local LLM connection...")
        response = llm_client.chat("Say 'Hello from local LLM!' in one sentence.")
        logger.info(f"LLM Response: {response}\n")
    except Exception as e:
        logger.error(f"Failed to connect to LLM: {e}")
        logger.error("Ensure your local LLM server is running (e.g., Ollama, vLLM, LM Studio)")
        logger.error("Check that it's accessible at the configured base URL")


if __name__ == "__main__":
    asyncio.run(main())
