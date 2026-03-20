# ACE MCP Server -- Client Setup Guide

The ACE MCP server communicates over **stdio**, making it compatible with any MCP client. This guide covers setup for popular clients beyond VS Code: **Claude Code**, **Claude Desktop**, **Cursor**, and **Windsurf**.

For the full tool reference and configuration options, see the [MCP Integration Guide](mcp.md).

## Prerequisites

1. Install ACE with the MCP extra:

   ```bash
   pip install "ace-framework[mcp]"
   # or
   uv add "ace-framework[mcp]"
   ```

2. Set required environment variables:

   ```bash
   # Choose any LiteLLM-supported model
   export ACE_MCP_DEFAULT_MODEL="gpt-4o-mini"

   # Provider API key (depends on the model you chose)
   export OPENAI_API_KEY="sk-..."
   # or ANTHROPIC_API_KEY, AWS credentials for Bedrock, etc.
   ```

3. Verify the server starts:

   ```bash
   ace-mcp
   # Should print "Starting ACE MCP Server..." to stderr, then wait for stdio input.
   # Press Ctrl+C to stop.
   ```

## Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) connects to MCP servers defined in its settings file.

### Configuration

Add to `~/.claude.json` (global) or `<project>/.claude/settings.json` (project-specific):

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

If you installed with `uv` and `ace-mcp` is not on your PATH, use the full path:

```json
{
  "mcpServers": {
    "ace": {
      "command": "uv",
      "args": ["run", "ace-mcp"],
      "env": {
        "ACE_MCP_DEFAULT_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Usage

Once configured, ACE tools appear as MCP tools in Claude Code. You can invoke them by asking Claude to use them:

```
Use ace.ask to answer: "What design patterns should I use for this codebase?"
```

Or refer to them in your CLAUDE.md:

```markdown
When learning from feedback, use the ace.learn.feedback MCP tool.
```

## Claude Desktop

[Claude Desktop](https://claude.ai/download) reads MCP server config from its settings file.

### Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

With `uv`:

```json
{
  "mcpServers": {
    "ace": {
      "command": "uv",
      "args": ["run", "ace-mcp"],
      "env": {
        "ACE_MCP_DEFAULT_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Usage

Restart Claude Desktop after editing the config. ACE tools will appear in the tool list (hammer icon). Claude will use them automatically when relevant, or you can ask directly:

> Use the ace.ask tool with session_id "my-session" and question "How should I structure error handling?"

## Cursor

[Cursor](https://cursor.com) supports MCP servers via its settings.

### Configuration

Open **Cursor Settings** > **MCP** > **Add new MCP server**, or edit `~/.cursor/mcp.json`:

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

### Usage

After adding the server, ACE tools are available in Cursor's AI assistant. The tools work in both Chat and Composer modes.

## Windsurf

[Windsurf](https://codeium.com/windsurf) supports MCP servers through its configuration.

### Configuration

Edit `~/.codeium/windsurf/mcp_config.json`:

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

### Usage

Restart Windsurf after saving. ACE tools will be available to the Cascade AI assistant.

## Testing with the MCP Inspector

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a web-based tool for testing MCP servers interactively, independent of any IDE:

```bash
npx @modelcontextprotocol/inspector uv run ace-mcp
```

1. Set `ACE_MCP_DEFAULT_MODEL` in the Inspector's environment variables panel.
2. Click **Connect**.
3. Go to **Tools** to see all 6 ACE tools.
4. Click any tool, fill in the arguments, and click **Run** to test.

This is the recommended way to verify the server works before configuring a specific client.

## Troubleshooting

### Server does not start

```
RuntimeError: ACE MCP support is optional. Install it with
`pip install "ace-framework[mcp]"` or `uv add "ace-framework[mcp]"`.
```

The `mcp` extra is not installed. Run the install command shown in the error.

### Tools not appearing in client

- Verify the `command` path is correct. Run `which ace-mcp` to find the full path.
- Check the client's MCP logs for connection errors:
  - **Claude Code:** `~/.claude/logs/`
  - **Claude Desktop:** `~/Library/Logs/Claude/` (macOS)
  - **Cursor:** Developer Tools console (Help > Toggle Developer Tools)
- Try running the server manually (`ace-mcp`) to confirm it starts.

### Session or API errors

- Ensure the correct API key is set for your chosen model.
- Check stderr output for detailed error logs. Set `ACE_MCP_LOG_LEVEL=DEBUG` for verbose logging.
- Use the MCP Inspector to send test requests and inspect error responses.

### Model not found

The `ACE_MCP_DEFAULT_MODEL` value must be a valid [LiteLLM model identifier](https://docs.litellm.ai/docs/providers). Examples:

| Provider | Model ID |
|----------|----------|
| OpenAI | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| AWS Bedrock | `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Google | `gemini/gemini-2.0-flash` |
| Ollama | `ollama/llama3` |

## Environment Variables Reference

All settings use the `ACE_MCP_` prefix and work identically across all clients:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_MCP_DEFAULT_MODEL` | `gpt-4o-mini` | LiteLLM model identifier |
| `ACE_MCP_SAFE_MODE` | `false` | Block write tools (learn, save) |
| `ACE_MCP_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ACE_MCP_SESSION_TTL_SECONDS` | `3600` | Idle session timeout |
| `ACE_MCP_SKILLBOOK_ROOT` | unset | Restrict save/load paths |

See the [full configuration reference](mcp.md#configuration) for all options.
