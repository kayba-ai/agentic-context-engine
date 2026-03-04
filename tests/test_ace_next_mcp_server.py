import pytest
from unittest.mock import MagicMock

pytest.importorskip("mcp.server")
pytest.importorskip("mcp.types")

from ace_next.integrations.mcp.server import create_server
from mcp.server import Server
from mcp.types import ListToolsRequest

EXPECTED_TOOL_NAMES = {
    "ace.ask",
    "ace.learn.sample",
    "ace.learn.feedback",
    "ace.skillbook.get",
    "ace.skillbook.save",
    "ace.skillbook.load",
}


def test_create_server():
    server = create_server()
    assert isinstance(server, Server)
    assert server.name == "ace-mcp-server"


@pytest.mark.asyncio
async def test_tool_registration():
    """All 6 MVP tools must be registered (FR-002)."""
    server = create_server()

    handler = server.request_handlers.get(ListToolsRequest)
    assert handler is not None, "tools/list handler not registered"

    result = await handler(MagicMock())
    registered_names = {t.name for t in result.root.tools}
    assert registered_names == EXPECTED_TOOL_NAMES
