from __future__ import annotations

import json
from importlib import import_module
from typing import Any

from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.models import (
    AskRequest,
    LearnFeedbackRequest,
    LearnSampleRequest,
    SkillbookGetRequest,
    SkillbookLoadRequest,
    SkillbookSaveRequest,
)
from ace_next.integrations.mcp.errors import map_error_to_mcp

_MCP_INSTALL_HINT = (
    "ACE MCP support is optional. Install it with "
    '`pip install "ace-framework[mcp]"` or `uv add "ace-framework[mcp]"`.'
)


def _load_mcp_types():
    try:
        return import_module("mcp.types")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "mcp":
            raise RuntimeError(_MCP_INSTALL_HINT) from exc
        raise


def _mcp_schema(model: Any) -> dict[str, Any]:
    """Return an MCP-friendly JSON schema for a Pydantic model.

    Some MCP clients (e.g. the Inspector) don't resolve ``$defs``/``$ref``
    correctly and reject valid input.  This helper inlines all ``$ref``
    pointers so the schema is self-contained.

    We keep ``additionalProperties: false`` in the published schema so
    clients know that extra fields will be rejected by validation.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs.get(ref_name, {}))
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(schema)


# ── Tool dispatch table ────────────────────────────────────────────

_TOOL_DISPATCH: dict[str, tuple[type, str]] = {
    "ace.ask": (AskRequest, "handle_ask"),
    "ace.learn.sample": (LearnSampleRequest, "handle_learn_sample"),
    "ace.learn.feedback": (LearnFeedbackRequest, "handle_learn_feedback"),
    "ace.skillbook.get": (SkillbookGetRequest, "handle_skillbook_get"),
    "ace.skillbook.save": (SkillbookSaveRequest, "handle_skillbook_save"),
    "ace.skillbook.load": (SkillbookLoadRequest, "handle_skillbook_load"),
}


def register_tools(server: Any, handlers: MCPHandlers) -> None:
    types = _load_mcp_types()

    @server.list_tools()
    async def handle_list_tools():
        return [
            types.Tool(
                name="ace.ask",
                description="Ask a question and get a response from ACE.",
                inputSchema=_mcp_schema(AskRequest),
            ),
            types.Tool(
                name="ace.learn.sample",
                description="Provide sample questions/answers for ACE to learn from.",
                inputSchema=_mcp_schema(LearnSampleRequest),
            ),
            types.Tool(
                name="ace.learn.feedback",
                description="Provide feedback on an ACE answer.",
                inputSchema=_mcp_schema(LearnFeedbackRequest),
            ),
            types.Tool(
                name="ace.skillbook.get",
                description="Get statistics and skills from the active skillbook.",
                inputSchema=_mcp_schema(SkillbookGetRequest),
            ),
            types.Tool(
                name="ace.skillbook.save",
                description="Save the active skillbook to disk.",
                inputSchema=_mcp_schema(SkillbookSaveRequest),
            ),
            types.Tool(
                name="ace.skillbook.load",
                description="Load a skillbook from disk into the session.",
                inputSchema=_mcp_schema(SkillbookLoadRequest),
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None):
        args = arguments or {}
        try:
            entry = _TOOL_DISPATCH.get(name)
            if entry is None:
                raise ValueError(f"Unknown tool: {name}")

            request_cls, handler_method = entry
            req = request_cls(**args)
            resp = await getattr(handlers, handler_method)(req)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=resp.model_dump_json())],
            )

        except Exception as e:
            mcp_err = map_error_to_mcp(e)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=json.dumps(mcp_err))],
            )
