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
    'ACE MCP support is optional. Install it with '
    '`pip install "ace-framework[mcp]"` or `uv add "ace-framework[mcp]"`.'
)


def _load_mcp_types():
    try:
        return import_module("mcp.types")
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "mcp":
            raise RuntimeError(_MCP_INSTALL_HINT) from exc
        raise


def register_tools(server: Any, handlers: MCPHandlers) -> None:
    types = _load_mcp_types()

    @server.list_tools()
    async def handle_list_tools():
        return [
            types.Tool(
                name="ace.ask",
                description="Ask a question and get a response from ACE.",
                inputSchema=AskRequest.model_json_schema(),
            ),
            types.Tool(
                name="ace.learn.sample",
                description="Provide sample questions/answers for ACE to learn from.",
                inputSchema=LearnSampleRequest.model_json_schema(),
            ),
            types.Tool(
                name="ace.learn.feedback",
                description="Provide feedback on an ACE answer.",
                inputSchema=LearnFeedbackRequest.model_json_schema(),
            ),
            types.Tool(
                name="ace.skillbook.get",
                description="Get statistics and skills from the active skillbook.",
                inputSchema=SkillbookGetRequest.model_json_schema(),
            ),
            types.Tool(
                name="ace.skillbook.save",
                description="Save the active skillbook to disk.",
                inputSchema=SkillbookSaveRequest.model_json_schema(),
            ),
            types.Tool(
                name="ace.skillbook.load",
                description="Load a skillbook from disk into the session.",
                inputSchema=SkillbookLoadRequest.model_json_schema(),
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None):
        args = arguments or {}
        try:
            if name == "ace.ask":
                req = AskRequest(**args)
                resp = await handlers.handle_ask(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]

            elif name == "ace.learn.sample":
                req = LearnSampleRequest(**args)
                resp = await handlers.handle_learn_sample(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]

            elif name == "ace.learn.feedback":
                req = LearnFeedbackRequest(**args)
                resp = await handlers.handle_learn_feedback(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]

            elif name == "ace.skillbook.get":
                req = SkillbookGetRequest(**args)
                resp = await handlers.handle_skillbook_get(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]

            elif name == "ace.skillbook.save":
                req = SkillbookSaveRequest(**args)
                resp = await handlers.handle_skillbook_save(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]

            elif name == "ace.skillbook.load":
                req = SkillbookLoadRequest(**args)
                resp = await handlers.handle_skillbook_load(req)
                return [types.TextContent(type="text", text=resp.model_dump_json())]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            mcp_err = map_error_to_mcp(e)
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=json.dumps(mcp_err))],
            )
