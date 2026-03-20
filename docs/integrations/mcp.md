# ACE MCP Server

ACE (Agentic Context Engine) provides an optional MCP server that exposes ACE as a tool provider over the [Model Context Protocol](https://modelcontextprotocol.io/). Orchestration frameworks, IDEs (Cursor, Windsurf, Claude Code), and other MCP clients can connect to the server and use ACE skills at runtime.

The integration is fully opt-in. Installing ACE without the `mcp` extra does not pull in the MCP SDK, and attempting to start `ace-mcp` without that extra will fail with an install hint instead of breaking normal ACE imports.

> **Setting up a specific client?** See the [Client Setup Guide](mcp-client-setup.md) for Claude Code, Claude Desktop, Cursor, and Windsurf configuration examples.

## Installation

```bash
pip install "ace-framework[mcp]"
# or using uv:
uv add "ace-framework[mcp]"
```

## Running the Server

Start the server using the provided CLI entrypoint:

```bash
ace-mcp
```

By default it communicates over `stdio`, making it ready for integration as a local tool provider in any MCP-compatible client.

Example with a specific model:

```bash
ACE_MCP_DEFAULT_MODEL=bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 ace-mcp
```

## Configuration

All settings are read from environment variables with the `ACE_MCP_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_MCP_DEFAULT_MODEL` | `gpt-4o-mini` | LiteLLM model identifier used when creating new sessions. Any [LiteLLM-supported model](https://docs.litellm.ai/docs/providers) works (e.g. `bedrock/...`, `anthropic/...`, `openai/...`). |
| `ACE_MCP_SAFE_MODE` | `false` | When `true`, blocks `ace.learn.sample`, `ace.learn.feedback`, `ace.skillbook.save`, and `ace.skillbook.load`. Read-only tools (`ace.ask`, `ace.skillbook.get`) remain available. |
| `ACE_MCP_ALLOW_SAVE_LOAD` | `true` | When `false`, blocks `ace.skillbook.save` and `ace.skillbook.load` independently of safe mode. |
| `ACE_MCP_MAX_SAMPLES_PER_CALL` | `25` | Maximum number of samples accepted in a single `ace.learn.sample` call. |
| `ACE_MCP_MAX_PROMPT_CHARS` | `100000` | Maximum total characters across question + context fields. |
| `ACE_MCP_SESSION_TTL_SECONDS` | `3600` | Idle time (seconds) before a session is garbage-collected. |
| `ACE_MCP_LEARN_TIMEOUT_SECONDS` | `300` | Maximum seconds for a single `ace.learn.sample` or `ace.learn.feedback` call before returning `ACE_MCP_TIMEOUT`. |
| `ACE_MCP_SKILLBOOK_ROOT` | unset | If set, `ace.skillbook.save` and `ace.skillbook.load` reject paths outside this directory. |
| `ACE_MCP_LOG_LEVEL` | `INFO` | Server log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Logs go to stderr. |

## Tools

### `ace.ask`

Ask a question using the current skillbook. Does **not** mutate the skillbook.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier for state isolation. |
| `question` | string | yes | The question to answer. |
| `context` | string | no | Additional context for the question. |
| `session_config` | object | no | Override `model`, `temperature`, `max_tokens` for this session. |

Returns: `answer`, `skill_count`.

### `ace.learn.sample`

Provide sample question/answer pairs for ACE to learn from. Runs the full ACE pipeline (Agent, Evaluate, Reflect, Update, Apply).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier. |
| `samples` | array | yes | List of `{question, context?, ground_truth?, metadata?}` items (1–25). |
| `epochs` | int | no | Number of learning passes (default: 1, max: 20). |
| `session_config` | object | no | Override model settings. |

Returns: `processed`, `failed`, `skill_count_before`, `skill_count_after`, `new_skill_count`.

Blocked by: `ACE_MCP_SAFE_MODE=true`.

### `ace.learn.feedback`

Provide feedback on a previous answer. If a prior `ace.ask` exists for the session, learns directly from that interaction. Otherwise, builds a trace from the provided fields and learns from it.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier. |
| `question` | string | yes | The original question. |
| `answer` | string | yes | The answer being evaluated. |
| `feedback` | string | yes | Feedback about answer quality. |
| `context` | string | no | Original context. |
| `ground_truth` | string | no | The correct answer. |
| `session_config` | object | no | Override model settings. |

Returns: `learned` (always `true` on success — the learning path executed), `skill_count_before`, `skill_count_after`, `new_skill_count`.

Blocked by: `ACE_MCP_SAFE_MODE=true`.

### `ace.skillbook.get`

Retrieve skills and statistics from the active skillbook.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier. |
| `limit` | int | no | Max skills to return (default: 20, max: 200). |
| `include_invalid` | bool | no | Include invalidated skills (default: false). |

Returns: `stats`, `skills[]` (each with `id`, `content`, `topic`, `helpful`, `harmful`, `neutral`).

### `ace.skillbook.save`

Save the session's skillbook to a file on disk.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier. |
| `path` | string | yes | File path to save to. |

Returns: `path` (resolved absolute path), `saved_skill_count`.

Blocked by: `ACE_MCP_SAFE_MODE=true` (`ACE_MCP_FORBIDDEN_IN_SAFE_MODE`) or `ACE_MCP_ALLOW_SAVE_LOAD=false` (`ACE_MCP_SAVE_LOAD_DISABLED`). The path is resolved to a canonical absolute path before validation and file I/O. Must be under `ACE_MCP_SKILLBOOK_ROOT` if configured.

### `ace.skillbook.load`

Load a skillbook from disk into the session, replacing the current one.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier. |
| `path` | string | yes | File path to load from. |

Returns: `path` (resolved absolute path), `skill_count`.

Blocked by: `ACE_MCP_SAFE_MODE=true` (`ACE_MCP_FORBIDDEN_IN_SAFE_MODE`) or `ACE_MCP_ALLOW_SAVE_LOAD=false` (`ACE_MCP_SAVE_LOAD_DISABLED`). The path is resolved to a canonical absolute path before validation and file I/O. Must be under `ACE_MCP_SKILLBOOK_ROOT` if configured.

## Session Model

All state is isolated by `session_id`. Each session holds its own `ACELiteLLM` runner with an independent skillbook. Pass the same `session_id` across calls to persist context within server memory.

Sessions are garbage-collected after `ACE_MCP_SESSION_TTL_SECONDS` of inactivity. Per-session locks ensure concurrent requests to the same session are serialised.

Any tool that accepts `session_config` can override the model, temperature, and max_tokens for that session's runner on first creation. Once a session exists, `session_config` on subsequent calls is used only if it creates a new session.

## Architecture

The MCP server does **not** use custom pipeline steps. It is a thin async layer over `ACELiteLLM`:

```
MCP Client (stdio)
  → MCP SDK (handles JSON-RPC framing)
    → adapters.py (tool registration, schema generation, error mapping)
      → handlers.py (validation, session management, safety guards)
        → ACELiteLLM (sync runner — bridged via asyncio.to_thread)
          → Pipeline (internal — handles step execution, async_boundary, background learning)
```

The handlers use `asyncio.to_thread()` to call the sync `ACELiteLLM` methods from the async MCP event loop. This is the standard Python pattern for bridging async callers to sync APIs. The pipeline engine handles all internal async concerns (step-level `to_thread`, `async_boundary` for background learning) transparently.

## Testing with the MCP Inspector

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) provides a web UI for testing MCP servers interactively:

```bash
npx @modelcontextprotocol/inspector uv run ace-mcp
```

Set `ACE_MCP_DEFAULT_MODEL` in the Inspector's environment variables panel before connecting.

## File Layout

```
ace_next/integrations/mcp/
  __init__.py       ← Package marker
  server.py         ← Server creation and CLI entrypoint
  config.py         ← MCPServerConfig (pydantic-settings, env vars)
  registry.py       ← SessionRegistry (session lifecycle, TTL sweep)
  handlers.py       ← MCPHandlers (validation, safety, delegation to ACELiteLLM)
  adapters.py       ← MCP SDK glue (tool registration, schema inlining, error mapping)
  models.py         ← Pydantic request/response models for all six tools
  errors.py         ← ACEMCPError hierarchy and MCP error mapping
```
