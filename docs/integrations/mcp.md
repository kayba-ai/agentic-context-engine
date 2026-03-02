# ACE MCP Server

ACE (Agentic Context Engine) provides an optional MCP server to allow orchestration frameworks, IDEs (like Cursor, Windsurf, Claude Code), and other clients to use ACE as a tool provider.

The integration is fully opt-in. Installing ACE without the `mcp` extra does not pull in the MCP SDK, and attempting to start `ace-mcp` without that extra will fail with an install hint instead of breaking normal ACE imports.

## Installation

To enable the MCP server, install ACE with the `mcp` extra:

```bash
pip install "ace-framework[mcp]"
# or using uv:
uv add "ace-framework[mcp]"
```

## Running the Server

Start the server using the provided CLI command:

```bash
ace-mcp
```

By default it listens on standard out (`stdio`), making it ready for integration as a local tool provider. 

## Configuration

Set configuration via environment variables:

- `ACE_MCP_DEFAULT_MODEL` (default: `gpt-4o-mini`): AI model used when new sessions are created.
- `ACE_MCP_SAFE_MODE` (default: `false`): Disables mutating tools (learn, save, load) when set to `true`.
- `ACE_MCP_MAX_SAMPLES_PER_CALL` (default: `25`): Limit for learning samples in one call.
- `ACE_MCP_MAX_PROMPT_CHARS` (default: `100000`): Max limit for user prompt characters.
- `ACE_MCP_SESSION_TTL_SECONDS` (default: `3600`): Time-to-live for idle sessions in memory.
- `ACE_MCP_ALLOW_SAVE_LOAD` (default: `true`): If `false`, disables reading/writing skillbooks.
- `ACE_MCP_SKILLBOOK_ROOT` (default: unset): If set, save/load paths must stay under this directory.

## Tools Provided

- `ace.ask`: Ask a question and get a response from ACE.
- `ace.learn.sample`: Provide sample questions/answers for ACE to learn from.
- `ace.learn.feedback`: Provide feedback on an ACE answer.
- `ace.skillbook.get`: Get statistics and skills from the active skillbook.
- `ace.skillbook.save`: Save the active skillbook to disk.
- `ace.skillbook.load`: Replace the session skillbook with a skillbook loaded from disk.

Learning is explicit: `ace.ask` answers without mutating the skillbook, and learning only happens through `ace.learn.sample` or `ace.learn.feedback`.

All state is isolated by `session_id`. Pass the same `session_id` continuously to persist context within the server memory.
