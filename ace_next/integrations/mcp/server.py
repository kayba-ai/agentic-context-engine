from __future__ import annotations

import asyncio
import logging
import sys
from importlib import import_module
from typing import Any

from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.registry import SessionRegistry

_MCP_INSTALL_HINT = (
    'ACE MCP support is optional. Install it with '
    '`pip install "ace-framework[mcp]"` or `uv add "ace-framework[mcp]"`.'
)


def _load_mcp_server_runtime() -> tuple[type[Any], Any]:
    try:
        server_module = import_module("mcp.server")
        stdio_module = import_module("mcp.server.stdio")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "mcp":
            raise RuntimeError(_MCP_INSTALL_HINT) from exc
        raise
    return server_module.Server, stdio_module.stdio_server


def _load_register_tools():
    try:
        from ace_next.integrations.mcp.adapters import register_tools
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "mcp":
            raise RuntimeError(_MCP_INSTALL_HINT) from exc
        raise
    return register_tools


def create_server() -> Any:
    Server, _ = _load_mcp_server_runtime()
    register_tools = _load_register_tools()
    config = MCPServerConfig()

    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("ace_mcp_server")
    logger.info("Starting ACE MCP Server...")
    logger.info(f"Safe mode: {config.safe_mode}")
    logger.info(f"Default model: {config.default_model}")

    registry = SessionRegistry(config)
    handlers = MCPHandlers(registry, config)

    server = Server("ace-mcp-server")
    register_tools(server, handlers)

    return server


async def run_server() -> None:
    try:
        _, stdio_server = _load_mcp_server_runtime()
        server = create_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    except Exception as e:
        print(f"Failed to start ACE MCP Server: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI Entrypoint for ace-mcp."""
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
