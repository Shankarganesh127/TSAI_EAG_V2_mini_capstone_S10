import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).with_name("default_mcp_config.yaml")


def load_server_configs(config_path: Path = _CONFIG_PATH) -> list[dict]:
    """Load server definitions and resolve their working paths."""
    if not config_path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    servers = data.get("mcp_servers", [])
    for server in servers:
        server["script"] = _resolve_script_path(server["script"], config_path.parent)
        server.setdefault("cwd", str(config_path.parent))
    return servers


def _resolve_script_path(script: str, config_dir: Path) -> str:
    return script if os.path.isabs(script) else str(config_dir / script)


class MultiMCP:
    """Discover MCP tools and route calls to their owning server."""

    def __init__(self, server_configs: list[dict] | None = None):
        self.server_configs = (
            load_server_configs() if server_configs is None else server_configs
        )
        self.tool_map: dict[str, dict] = {}

    @staticmethod
    def _server_params(config: dict) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=[config["script"]],
            cwd=config.get("cwd", os.getcwd()),
        )

    async def initialize(self) -> dict:
        """Discover tools, rejecting required failures and name collisions."""
        discovered: dict[str, dict] = {}
        failures: list[str] = []
        required_failures: list[str] = []

        for config in self.server_configs:
            try:
                tools = await self._discover_server_tools(config)
                self._merge_discovered_tools(discovered, config, tools)
                self._log_initialized_server(config, tools)
            except Exception as exc:
                failure = self._format_initialization_failure(config, exc)
                failures.append(failure)
                if config.get("required", True):
                    required_failures.append(failure)

        if required_failures:
            self.tool_map.clear()
            raise RuntimeError(
                "Required MCP initialization failed: " + "; ".join(required_failures)
            )

        self.tool_map = discovered
        return self._initialization_report(failures)

    async def _discover_server_tools(self, config: dict) -> list[Any]:
        params = self._server_params(config)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                return list(response.tools)

    @staticmethod
    def _merge_discovered_tools(
        discovered: dict[str, dict], config: dict, tools: list[Any]
    ) -> None:
        for tool in tools:
            if tool.name in discovered:
                owner = discovered[tool.name]["config"]["id"]
                raise ValueError(
                    f"Duplicate tool '{tool.name}' from '{owner}' and '{config['id']}'"
                )
            discovered[tool.name] = {"config": config, "tool": tool}

    @staticmethod
    def _log_initialized_server(config: dict, tools: list[Any]) -> None:
        logger.info(
            "[OK] MCP server '%s' initialized - tools: %s",
            config["id"],
            [tool.name for tool in tools],
        )

    @staticmethod
    def _format_initialization_failure(config: dict, exc: Exception) -> str:
        server_id = config.get("id", "<unknown>")
        logger.error("[ERROR] Failed to initialize MCP server '%s': %s", server_id, exc)
        return f"{server_id}: {exc}"

    def _initialization_report(self, failures: list[str]) -> dict:
        return {
            "servers": len(self.server_configs),
            "tools": len(self.tool_map),
            "failures": failures,
        }

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on its owning server."""
        entry = self.tool_map.get(tool_name)
        if entry is None:
            raise ValueError(f"Tool '{tool_name}' not found on any server.")
        params = self._server_params(entry["config"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)

    def list_all_tools(self) -> list[dict]:
        """Return discovered tools with their owning server IDs."""
        return [
            {"name": name, "server": entry["config"]["id"], "tool": entry["tool"]}
            for name, entry in self.tool_map.items()
        ]

    def describe_tools(self) -> list[dict]:
        """Return a prompt-safe catalog containing exact MCP input schemas."""
        return [
            {
                "name": name,
                "server": entry["config"]["id"],
                "description": entry["tool"].description or "",
                "input_schema": entry["tool"].inputSchema,
            }
            for name, entry in self.tool_map.items()
        ]
