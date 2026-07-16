import sys
from pathlib import Path
from typing import Any

import yaml


def load_mcp_config(config_path: Path) -> dict:
    """Load the shared MCP YAML configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_mcp_server(server: Any) -> None:
    """Run an MCP server with the transport selected by CLI arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        server.run()
        return
    server.run(transport="stdio")
