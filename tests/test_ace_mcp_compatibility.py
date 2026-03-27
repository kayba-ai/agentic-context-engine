"""Focused compatibility tests for ACE's generic MCP server surface."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from ace.integrations.mcp.adapters import _MCP_INSTALL_HINT as ADAPTERS_INSTALL_HINT
from ace.integrations.mcp.adapters import _mcp_schema, register_tools
from ace.integrations.mcp.config import MCPServerConfig
from ace.integrations.mcp.errors import (
    ForbiddenInSafeModeError,
    SessionNotFoundError,
    ValidationError as ACEValidationError,
    map_error_to_mcp,
)
from ace.integrations.mcp.handlers import MCPHandlers
from ace.integrations.mcp.models import AskRequest, LearnSampleRequest
from ace.integrations.mcp.registry import SessionRegistry
from ace.integrations.mcp.server import _MCP_INSTALL_HINT as SERVER_INSTALL_HINT


_CLIENT_PATTERN = re.compile(
    r"(vs\s*code|vscode|visual\s*studio\s*code|cursor|windsurf)",
    re.IGNORECASE,
)


def _require_mcp():
    pytest.importorskip("mcp.server")
    pytest.importorskip("mcp.types")

    from mcp.server import Server
    from mcp.types import CallToolRequest, ListToolsRequest

    return Server, CallToolRequest, ListToolsRequest


def test_ask_request_schema_is_inlined():
    schema = _mcp_schema(AskRequest)
    schema_str = json.dumps(schema)

    assert "$ref" not in schema_str
    assert "$defs" not in schema_str
    assert "session_id" in schema.get("properties", {})
    assert "question" in schema.get("properties", {})


def test_nested_schema_is_inlined():
    schema = _mcp_schema(LearnSampleRequest)
    schema_str = json.dumps(schema)

    assert "$ref" not in schema_str
    assert "$defs" not in schema_str
    assert "samples" in schema.get("properties", {})


def test_install_hints_are_client_agnostic():
    assert not _CLIENT_PATTERN.search(SERVER_INSTALL_HINT)
    assert not _CLIENT_PATTERN.search(ADAPTERS_INSTALL_HINT)


def test_error_messages_are_client_agnostic():
    for err in (
        SessionNotFoundError("session-1"),
        ForbiddenInSafeModeError("ace.learn.sample"),
        ACEValidationError("prompt too long", details={"field": "question"}),
        RuntimeError("boom"),
    ):
        mapped = map_error_to_mcp(err)
        assert not _CLIENT_PATTERN.search(mapped["message"])


@pytest.fixture
def wired_server():
    Server, _, _ = _require_mcp()

    config = MCPServerConfig(safe_mode=False)
    registry = SessionRegistry(config)
    handlers = MCPHandlers(registry, config)
    server = Server("ace-mcp-server")
    register_tools(server, handlers)
    return server, registry


@pytest.mark.asyncio
async def test_published_tool_schemas_are_inlined(wired_server):
    server, _ = wired_server
    _, _, ListToolsRequest = _require_mcp()

    handler = server.request_handlers.get(ListToolsRequest)
    assert handler is not None

    result = await handler(MagicMock())
    for tool in result.root.tools:
        schema_str = json.dumps(tool.inputSchema)
        assert "$ref" not in schema_str
        assert "$defs" not in schema_str
        assert not _CLIENT_PATTERN.search(tool.description or "")


@pytest.mark.asyncio
async def test_call_tool_ace_ask_returns_json_payload(wired_server):
    server, _ = wired_server
    _, CallToolRequest, _ = _require_mcp()

    with patch("ace.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.ask.return_value = "The answer is 42."
        runner.skillbook.skills.return_value = []
        mock_runner_cls.from_model.return_value = runner

        handler = server.request_handlers.get(CallToolRequest)
        assert handler is not None

        req = MagicMock()
        req.params.name = "ace.ask"
        req.params.arguments = {
            "session_id": "generic-client-1",
            "question": "What is the meaning of life?",
        }

        result = await handler(req)
        assert not result.root.isError

        payload = json.loads(result.root.content[0].text)
        assert payload["answer"] == "The answer is 42."
        assert payload["session_id"] == "generic-client-1"


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_returns_structured_error(wired_server):
    server, _ = wired_server
    _, CallToolRequest, _ = _require_mcp()

    handler = server.request_handlers.get(CallToolRequest)
    assert handler is not None

    req = MagicMock()
    req.params.name = "nonexistent.tool"
    req.params.arguments = {}

    result = await handler(req)
    assert result.root.isError

    payload = json.loads(result.root.content[0].text)
    assert payload["code"] == "ACE_MCP_INTERNAL_ERROR"
    assert "Unknown tool" in payload["message"]


@pytest.mark.asyncio
async def test_session_ids_are_opaque_strings():
    config = MCPServerConfig()
    registry = SessionRegistry(config)

    with patch("ace.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        mock_runner_cls.from_model.side_effect = lambda *a, **kw: MagicMock()

        ids = [
            "simple-id",
            "uuid-550e8400-e29b-41d4-a716-446655440000",
            "cursor/project/session-1",
            "claude-code:workspace:12345",
        ]

        sessions = [await registry.get_or_create(session_id) for session_id in ids]

        assert [session.session_id for session in sessions] == ids
        assert len({id(session.runner) for session in sessions}) == len(ids)
