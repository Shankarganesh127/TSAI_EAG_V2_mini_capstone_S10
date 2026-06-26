import os
import sys
import logging
from pathlib import Path
from typing import Any

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).with_name("default_mcp_config.yaml")


def load_server_configs(config_path: Path = _CONFIG_PATH) -> list[dict]:
    """Load MCP server configs from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    servers = data.get("mcp_servers", [])
    # Resolve relative script paths relative to the config file's directory
    config_dir = str(config_path.parent)
    for server in servers:
        if not os.path.isabs(server["script"]):
            server["script"] = str(config_path.parent / server["script"])
        if "cwd" not in server:
            server["cwd"] = config_dir
    return servers


class MultiMCP:
    """Manages multiple MCP servers and routes tool calls to the right one."""

    def __init__(self, server_configs: list[dict] | None = None):
        self.server_configs = server_configs or load_server_configs()
        self.tool_map: dict[str, dict] = {}  # tool_name -> {config, tool}

    def _server_params(self, config: dict) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=[config["script"]],
            cwd=config.get("cwd", os.getcwd()),
        )

    async def initialize(self):
        """Discover all tools from all configured servers."""
        for config in self.server_configs:
            try:
                params = self._server_params(config)
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        for tool in tools.tools:
                            self.tool_map[tool.name] = {"config": config, "tool": tool}
                logger.info(f"[OK] MCP server '{config['id']}' initialized — tools: {[t.name for t in tools.tools]}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize MCP server '{config.get('id')}': {e}")

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool by name on the appropriate server."""
        entry = self.tool_map.get(tool_name)
        if not entry:
            raise ValueError(f"Tool '{tool_name}' not found on any server.")
        params = self._server_params(entry["config"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)

    def list_all_tools(self) -> list[dict]:
        """Return a list of all discovered tools with their server id."""
        return [
            {"name": name, "server": entry["config"]["id"], "tool": entry["tool"]}
            for name, entry in self.tool_map.items()
        ]
