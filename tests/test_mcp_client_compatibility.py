"""Tests for ACE MCP server compatibility with non-VS Code clients.

Validates that the MCP server works correctly as a generic stdio-based
tool provider, without assumptions about any specific IDE or client.

Covers:
- Server creation and tool listing via generic MCP protocol
- Tool invocation through the standard call_tool interface
- Error messages are client-agnostic (no VS Code references)
- Schema compatibility (inlined $ref, no $defs consumers must resolve)
- Configuration via environment variables (client-independent)

Closes #56.
"""

import asyncio
import json
import re

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("mcp.server")
pytest.importorskip("mcp.types")

from mcp.server import Server
from mcp.types import ListToolsRequest, CallToolRequest

from ace_next.integrations.mcp.server import create_server
from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.registry import SessionRegistry
from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.adapters import register_tools, _mcp_schema
from ace_next.integrations.mcp.errors import (
    map_error_to_mcp,
    SessionNotFoundError,
    ForbiddenInSafeModeError,
    ValidationError as ACEValidationError,
)
from ace_next.integrations.mcp.models import (
    AskRequest,
    LearnSampleRequest,
    SkillbookGetRequest,
)


# ── Server identity and protocol ────────────────────────────────


class TestServerIdentity:
    """The server must present itself correctly to any MCP client."""

    def test_server_name_is_generic(self):
        """Server name should not reference any specific IDE."""
        server = create_server()
        assert server.name == "ace-mcp-server"
        assert "vscode" not in server.name.lower()
        assert "cursor" not in server.name.lower()

    def test_server_is_standard_mcp_server(self):
        """Server must be a standard mcp.server.Server instance."""
        server = create_server()
        assert isinstance(server, Server)


# ── Tool listing (tools/list) ───────────────────────────────────


EXPECTED_TOOL_NAMES = {
    "ace.ask",
    "ace.learn.sample",
    "ace.learn.feedback",
    "ace.skillbook.get",
    "ace.skillbook.save",
    "ace.skillbook.load",
}


class TestToolListing:
    """Any MCP client that calls tools/list must see all 6 tools."""

    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        server = create_server()
        handler = server.request_handlers.get(ListToolsRequest)
        assert handler is not None

        result = await handler(MagicMock())
        names = {t.name for t in result.root.tools}
        assert names == EXPECTED_TOOL_NAMES

    @pytest.mark.asyncio
    async def test_tool_descriptions_are_client_agnostic(self):
        """No tool description should mention a specific IDE."""
        server = create_server()
        handler = server.request_handlers.get(ListToolsRequest)
        result = await handler(MagicMock())

        ide_patterns = re.compile(
            r"(vs\s*code|vscode|visual\s*studio\s*code|cursor|windsurf)",
            re.IGNORECASE,
        )
        for tool in result.root.tools:
            assert not ide_patterns.search(tool.description or ""), (
                f"Tool '{tool.name}' description mentions a specific IDE: "
                f"{tool.description}"
            )

    @pytest.mark.asyncio
    async def test_tool_schemas_have_no_unresolved_refs(self):
        """Schemas must be self-contained (no $ref/$defs) for broad client compat."""
        server = create_server()
        handler = server.request_handlers.get(ListToolsRequest)
        result = await handler(MagicMock())

        for tool in result.root.tools:
            schema_str = json.dumps(tool.inputSchema)
            assert "$ref" not in schema_str, (
                f"Tool '{tool.name}' schema contains unresolved $ref. "
                "Some MCP clients (Claude Desktop, Cursor) cannot resolve these."
            )
            assert "$defs" not in schema_str, (
                f"Tool '{tool.name}' schema contains $defs. "
                "Schemas must be fully inlined for cross-client compatibility."
            )


# ── Schema inlining ─────────────────────────────────────────────


class TestSchemaInlining:
    """The _mcp_schema helper must produce fully inlined schemas."""

    def test_ask_request_schema_inlined(self):
        schema = _mcp_schema(AskRequest)
        schema_str = json.dumps(schema)
        assert "$ref" not in schema_str
        assert "$defs" not in schema_str

    def test_learn_sample_schema_inlined(self):
        """LearnSampleRequest has nested SampleItem -- must inline."""
        schema = _mcp_schema(LearnSampleRequest)
        schema_str = json.dumps(schema)
        assert "$ref" not in schema_str
        assert "$defs" not in schema_str

    def test_schemas_include_required_fields(self):
        schema = _mcp_schema(AskRequest)
        assert "session_id" in schema.get("properties", {})
        assert "question" in schema.get("properties", {})


# ── Tool invocation via generic call_tool ────────────────────────


class TestGenericToolInvocation:
    """Simulate how a generic MCP client calls tools via call_tool."""

    @pytest.fixture
    def wired_server(self):
        """Create a server with mocked ACELiteLLM backend."""
        config = MCPServerConfig(safe_mode=False)
        registry = SessionRegistry(config)
        handlers = MCPHandlers(registry, config)
        server = Server("ace-mcp-server")
        register_tools(server, handlers)
        return server, handlers, registry

    @pytest.mark.asyncio
    async def test_call_tool_ace_ask(self, wired_server):
        """A generic client sends a call_tool request for ace.ask."""
        server, handlers, registry = wired_server

        with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_cls:
            runner = MagicMock()
            runner.ask.return_value = "The answer is 42."
            runner.skillbook.skills.return_value = []
            mock_cls.from_model.return_value = runner

            # Simulate what any MCP client sends: name + arguments dict
            handler = server.request_handlers.get(CallToolRequest)
            assert handler is not None

            mock_req = MagicMock()
            mock_req.params.name = "ace.ask"
            mock_req.params.arguments = {
                "session_id": "generic-client-1",
                "question": "What is the meaning of life?",
            }

            result = await handler(mock_req)

            # Result must be valid JSON in a TextContent block
            assert not result.root.isError
            content = result.root.content[0].text
            parsed = json.loads(content)
            assert parsed["answer"] == "The answer is 42."
            assert parsed["session_id"] == "generic-client-1"

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool_returns_error(self, wired_server):
        """Unknown tool names return structured error, not crash."""
        server, _, _ = wired_server

        handler = server.request_handlers.get(CallToolRequest)
        mock_req = MagicMock()
        mock_req.params.name = "nonexistent.tool"
        mock_req.params.arguments = {}

        result = await handler(mock_req)
        assert result.root.isError
        error_data = json.loads(result.root.content[0].text)
        assert "code" in error_data
        assert error_data["code"] == "ACE_MCP_INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_call_tool_with_empty_arguments(self, wired_server):
        """Calling a tool with no arguments returns a validation error."""
        server, _, _ = wired_server

        handler = server.request_handlers.get(CallToolRequest)
        mock_req = MagicMock()
        mock_req.params.name = "ace.ask"
        mock_req.params.arguments = None  # Some clients send null

        result = await handler(mock_req)
        assert result.root.isError


# ── Error message client-agnosticism ─────────────────────────────


class TestErrorMessages:
    """All error messages must be free of IDE-specific language."""

    _IDE_PATTERN = re.compile(
        r"(vs\s*code|vscode|visual\s*studio\s*code|cursor|windsurf)",
        re.IGNORECASE,
    )

    def _assert_no_ide_ref(self, message: str, context: str):
        assert not self._IDE_PATTERN.search(message), (
            f"{context} contains IDE-specific reference: {message}"
        )

    def test_session_not_found_error(self):
        err = SessionNotFoundError("test-session")
        mapped = map_error_to_mcp(err)
        self._assert_no_ide_ref(mapped["message"], "SessionNotFoundError")

    def test_forbidden_in_safe_mode_error(self):
        err = ForbiddenInSafeModeError("ace.learn.sample")
        mapped = map_error_to_mcp(err)
        self._assert_no_ide_ref(mapped["message"], "ForbiddenInSafeModeError")

    def test_validation_error(self):
        err = ACEValidationError("prompt too long", details={"field": "question"})
        mapped = map_error_to_mcp(err)
        self._assert_no_ide_ref(mapped["message"], "ValidationError")

    def test_unknown_error_mapping(self):
        err = RuntimeError("something broke")
        mapped = map_error_to_mcp(err)
        self._assert_no_ide_ref(mapped["message"], "RuntimeError mapping")

    def test_install_hint_is_generic(self):
        """The install hint should not mention any specific IDE."""
        from ace_next.integrations.mcp.server import _MCP_INSTALL_HINT

        self._assert_no_ide_ref(_MCP_INSTALL_HINT, "MCP install hint")


# ── Configuration is IDE-independent ─────────────────────────────


class TestConfiguration:
    """Configuration must work via env vars, not IDE-specific settings files."""

    def test_config_uses_env_prefix(self):
        """All config comes from ACE_MCP_ env vars, not IDE settings files."""
        config = MCPServerConfig()
        # Verify the env prefix is set correctly
        assert config.model_config.get("env_prefix") == "ACE_MCP_"

    def test_config_defaults_are_reasonable(self):
        config = MCPServerConfig()
        assert config.default_model == "gpt-4o-mini"
        assert config.safe_mode is False
        assert config.session_ttl_seconds == 3600
        assert config.log_level == "INFO"

    def test_config_from_env(self, monkeypatch):
        """Config can be set entirely via environment variables."""
        monkeypatch.setenv("ACE_MCP_DEFAULT_MODEL", "anthropic/claude-haiku-3")
        monkeypatch.setenv("ACE_MCP_SAFE_MODE", "true")
        monkeypatch.setenv("ACE_MCP_LOG_LEVEL", "DEBUG")

        config = MCPServerConfig()
        assert config.default_model == "anthropic/claude-haiku-3"
        assert config.safe_mode is True
        assert config.log_level == "DEBUG"


# ── stdio transport compatibility ────────────────────────────────


class TestStdioTransport:
    """The server must use stdio transport, which is universal across clients."""

    def test_run_server_uses_stdio(self):
        """run_server should use stdio_server from the MCP SDK."""
        from ace_next.integrations.mcp.server import _load_mcp_server_runtime

        _, stdio_server = _load_mcp_server_runtime()
        # stdio_server should be a callable (context manager factory)
        assert callable(stdio_server)

    def test_main_entrypoint_exists(self):
        """The ace-mcp entrypoint must be importable."""
        from ace_next.integrations.mcp.server import main

        assert callable(main)


# ── Multi-session isolation (client-independent) ─────────────────


class TestMultiSessionIsolation:
    """Different clients (or sessions) must be isolated."""

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self):
        config = MCPServerConfig()
        registry = SessionRegistry(config)

        with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_cls:
            # Return a fresh mock for each from_model() call
            mock_cls.from_model.side_effect = lambda *a, **kw: MagicMock()

            s1 = await registry.get_or_create("claude-desktop-session")
            s2 = await registry.get_or_create("cursor-session")

            assert s1.session_id != s2.session_id
            assert s1.runner is not s2.runner

    @pytest.mark.asyncio
    async def test_session_ids_accept_any_string(self):
        """Clients may use different session ID formats."""
        config = MCPServerConfig()
        registry = SessionRegistry(config)

        with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_cls:
            mock_cls.from_model.return_value = MagicMock()

            # Various ID formats different clients might use
            ids = [
                "simple-id",
                "uuid-550e8400-e29b-41d4-a716-446655440000",
                "cursor/project/session-1",
                "claude-code:workspace:12345",
            ]
            sessions = []
            for sid in ids:
                s = await registry.get_or_create(sid)
                sessions.append(s)
                assert s.session_id == sid

            # All sessions are distinct
            assert len({id(s) for s in sessions}) == len(ids)
