import asyncio
import os
import sys

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import get_default_environment
    from mcp.client.stdio import stdio_client
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    if (exc.name or "").split(".")[0] == "mcp":
        print(
            "This example requires the optional MCP extra. "
            'Install it with `pip install "ace-framework[mcp]"` or '
            '`uv add "ace-framework[mcp]"`.',
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise

try:
    from mcp.client.stdio import StdioServerParameters
except ImportError:  # pragma: no cover - fallback for SDK layout variants
    import mcp.client.stdio as mcp_stdio

    StdioServerParameters = mcp_stdio.StdioServerParameters


async def main():
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python examples/ace_next/mcp_client_demo.py <path_to_ace_mcp_cli>"
        )
        sys.exit(1)

    server_path = sys.argv[1]

    server_params = StdioServerParameters(
        command=server_path,  # usually "ace-mcp" or "uv"
        args=["run", "ace-mcp"] if "uv" in server_path else [],
        env={
            **get_default_environment(),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY", ""),
            "ACE_MCP_DEFAULT_MODEL": os.environ.get("ACE_MCP_DEFAULT_MODEL", ""),
        },
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("Connected to ACE MCP Server.")

            # List available tools
            tools = await session.list_tools()
            print("\\nAvailable Tools:")
            for tool in tools.tools:
                print(f" - {tool.name}")

            print("\\nTesting ace.ask...")
            ask_args = {"session_id": "demo-session-1", "question": "What is 2 + 2?"}

            result = await session.call_tool("ace.ask", ask_args)
            if result.isError:
                print(f"Error calling tool: {result.content}")
            else:
                for content in result.content:
                    print(f"Response: {content.text}")


if __name__ == "__main__":
    asyncio.run(main())
