import anyio
import pytest

pytest.importorskip("mcp.server")
pytest.importorskip("mcp.types")

from ace.integrations.mcp.server import create_server
from mcp.client.session import ClientSession
from mcp.server import Server
from mcp.shared.memory import create_client_server_memory_streams

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

    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(
                    *server_streams, server.create_initialization_options()
                )
            )
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                result = await session.list_tools()
            tg.cancel_scope.cancel()

    registered_names = {t.name for t in result.tools}
    assert registered_names == EXPECTED_TOOL_NAMES
