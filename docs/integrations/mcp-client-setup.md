# ACE MCP Client Setup

The ACE MCP server runs over `stdio`, so any MCP client that can launch a
local command can connect to it.

This guide focuses on wiring `ace-mcp` into popular clients. For the full
tool reference, environment variables, and safety controls, see the
[MCP Server guide](mcp.md).

## Prerequisites

1. Install ACE with the MCP extra:

   ```bash
   pip install "ace-framework[mcp]"
   # or
   uv add "ace-framework[mcp]"
   ```

2. Set the model and provider credentials you want the server to use:

   ```bash
   export ACE_MCP_DEFAULT_MODEL="gpt-4o-mini"
   export OPENAI_API_KEY="sk-..."
   ```

3. Verify the server starts:

   ```bash
   ace-mcp
   ```

   It should log startup information to stderr and then wait for stdio input.

## Claude Code

Anthropic recommends managing Claude Code MCP servers with the `claude mcp`
commands. A user-scoped server can be added with:

```bash
claude mcp add-json -s user ace '{
  "type": "stdio",
  "command": "ace-mcp",
  "env": {
    "ACE_MCP_DEFAULT_MODEL": "gpt-4o-mini",
    "OPENAI_API_KEY": "sk-..."
  }
}'
```

Useful variants:

- `-s project` stores the server in `.mcp.json` for the current repo.
- `claude mcp list` shows configured servers.
- `claude mcp get ace` prints the saved config.

Once added, you can ask Claude Code to use ACE directly:

```text
Use ace.ask with session_id "repo-default" to summarize the conventions in this repo.
```

## Cursor

Cursor supports local stdio MCP servers. Add a server from the MCP settings UI
or your MCP config using this shape:

```json
{
  "mcpServers": {
    "ace": {
      "command": "ace-mcp",
      "env": {
        "ACE_MCP_DEFAULT_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

After saving, refresh MCP servers in Cursor and confirm the ACE tools appear.

## Windsurf

Windsurf exposes MCP configuration through **Windsurf Settings** >
**Cascade** > **MCP Servers**. Add a stdio server using the same command/env
shape:

```json
{
  "mcpServers": {
    "ace": {
      "command": "ace-mcp",
      "env": {
        "ACE_MCP_DEFAULT_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Restart the MCP connection if the tools do not appear immediately.

## Smoke Test with MCP Inspector

Before debugging a client-specific setup, verify the server generically with
the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector ace-mcp
```

If the server starts and the six ACE tools appear, the remaining work is
client configuration rather than ACE itself.

## Troubleshooting

### `ace-mcp` is not found

- Confirm the package was installed with the `mcp` extra.
- Run `which ace-mcp` (or the equivalent on your platform) and use the full
  path in the client config if needed.

### The client connects but no tools appear

- Start `ace-mcp` manually first to confirm it launches cleanly.
- Check stderr logs from the server.
- Set `ACE_MCP_LOG_LEVEL=DEBUG` for more verbose logging.

### Save/load should stay inside a safe directory

Set `ACE_MCP_SKILLBOOK_ROOT` to constrain `ace.skillbook.save` and
`ace.skillbook.load` to a specific directory.

## References

- [Anthropic: Claude Code MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [Anthropic: Claude Code settings and scopes](https://code.claude.com/docs/en/settings)
- [Cursor MCP docs](https://docs.cursor.com/advanced/model-context-protocol)
- [Windsurf MCP docs](https://docs.windsurf.com/en/windsurf/cascade/mcp)
