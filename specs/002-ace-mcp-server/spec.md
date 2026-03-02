# Feature Specification: ACE MCP Server (Optional)

**Feature Branch**: `002-ace-mcp-server`  
**Created**: 2026-03-02  
**Status**: Draft  
**Input**: User description: "extend ACE to also work as an MCP server (optional but powerful)"

## Summary

Add an optional MCP server mode for ACE using the `ace_next` architecture. The server exposes ACE capabilities as MCP tools so external MCP clients (IDEs, copilots, orchestrators) can call ACE for inference and learning.

The feature is **opt-in** via optional dependency extras and a dedicated CLI entrypoint. Existing ACE APIs and integrations remain unchanged when MCP is not enabled.

## Scope

### In Scope (MVP)

- MCP server over stdio transport.
- Tool endpoints for ask/learn/skillbook operations.
- Per-session ACE runner lifecycle with in-memory state.
- Input/output validation with Pydantic models.
- Structured error mapping to MCP tool errors.
- Unit + integration tests for handler behavior and transport startup.
- Docs and runnable example client.

### Out of Scope (Post-MVP)

- HTTP/SSE MCP transport.
- Distributed or persistent multi-process session stores.
- AuthN/AuthZ and multi-tenant isolation beyond local process boundaries.
- Rich observability dashboards (basic logs only in MVP).

## Design Constraints

- Implement in `ace_next` integration layer, not in `pipeline/` or `ace_next/core/` for MVP.
- Preserve backward compatibility: no breaking changes to existing runners.
- Keep MCP dependency optional (no mandatory install-time dependency).
- Use existing ACE runner constructors (`from_model`, `from_roles`) and skillbook persistence APIs.

## User Scenarios & Testing

### User Story 1 — Ask Through MCP (P1)

A client calls `ace.ask` through MCP and gets an answer from a session-scoped ACE instance.

**Independent test**: start server, call `ace.ask`, assert valid answer payload and session state initialization.

### User Story 2 — Learn Through MCP (P1)

A client calls `ace.learn.sample` and `ace.learn.feedback`; ACE updates the skillbook for that session.

**Independent test**: call learning tools, then `ace.skillbook.get`, assert strategy count changed.

### User Story 3 — Persist/Restore Skillbook (P2)

A client saves skillbook to disk and later reloads it into the same/new session.

**Independent test**: `save` then `load` and assert stable strategy IDs/count.

### User Story 4 — Safe Optionality (P1)

Projects without MCP extras continue functioning exactly as before.

**Independent test**: import and use non-MCP modules without MCP installed.

## Functional Requirements

- **FR-001**: System MUST provide a CLI entrypoint to run ACE as an MCP server over stdio.
- **FR-002**: System MUST expose MCP tools: `ace.ask`, `ace.learn.sample`, `ace.learn.feedback`, `ace.skillbook.get`, `ace.skillbook.save`, `ace.skillbook.load`.
- **FR-003**: System MUST validate every tool request and response via typed schemas.
- **FR-004**: System MUST isolate state by `session_id`.
- **FR-005**: System MUST lazily initialize session runners with model/provider config.
- **FR-006**: System MUST support disabling mutating tools (`save/load` and learn tools) in safe mode.
- **FR-007**: System MUST map internal exceptions to stable MCP error codes/messages.
- **FR-008**: System MUST cap request sizes (max samples per call, max payload size) to prevent runaway execution.
- **FR-009**: System MUST remain fully optional via `mcp` extra dependency.
- **FR-010**: System MUST include handler-level success tests for each tool and server startup/registration smoke tests when the `mcp` extra is installed.

## Non-Functional Requirements

- **NFR-001**: Tool roundtrip latency for `ace.ask` should be < 2s p50 excluding model latency.
- **NFR-002**: Session operations must be thread-safe within a process.
- **NFR-003**: Server startup failures must emit actionable install/config error text.
- **NFR-004**: Existing test suite behavior remains unchanged when MCP extras are absent.

## Tool Contract Source of Truth

Canonical tool schemas are defined in: `specs/002-ace-mcp-server/contracts/tool-schemas.md`.

## Exact Module Tree (Implementation)

```text
ace_next/
  integrations/
    mcp/
      __init__.py
      server.py                 # create_server(), main(), startup wiring
      registry.py               # session registry + lifecycle (TTL, lazy init)
      config.py                 # MCPServerConfig, limits, safe-mode flags
      errors.py                 # domain errors + MCP error mapping
      models.py                 # Pydantic request/response schemas
      handlers.py               # tool handler implementations
      adapters.py               # maps handlers <-> MCP SDK tool registration

tests/
  test_ace_next_mcp_models.py      # schema validation and serialization
  test_ace_next_mcp_registry.py    # session lifecycle and locking
  test_ace_next_mcp_handlers.py    # tool behavior with mocked runner
  test_ace_next_mcp_server.py      # server startup + tool registration smoke

docs/
  integrations/
    mcp.md                      # install, run, tool catalog, examples

examples/
  ace_next/
    mcp_client_demo.py          # minimal MCP client invoking ace.ask
```

## Package and CLI Changes

- `pyproject.toml`
  - Add optional dependency group:
    - `mcp = ["mcp>=<pinned_version>"]` (or official Python MCP SDK package used by maintainers)
  - Add script entrypoint:
    - `ace-mcp = "ace_next.integrations.mcp.server:main"`

## Runtime Architecture

1. MCP server starts and registers tool definitions from `models.py`.
2. Incoming request is validated into typed request model.
3. `registry.py` returns/create session runner (`ACELiteLLM` by default).
4. `handlers.py` executes operation against runner.
5. Response serialized to typed output model.
6. Exceptions mapped through `errors.py` into stable MCP error responses.

## Session Model

- Key: `session_id: str`.
- Value: session object containing runner, creation time, last access time, lock.
- Concurrency: per-session lock around mutating operations.
- Expiry: configurable TTL cleanup (lazy sweep on access in MVP).

## Configuration Model

`MCPServerConfig` fields (MVP):

- `default_model: str = "gpt-4o-mini"`
- `safe_mode: bool = false`
- `max_samples_per_call: int = 25`
- `max_prompt_chars: int = 100_000`
- `session_ttl_seconds: int = 3600`
- `allow_save_load: bool = true`
- `skillbook_root: str | None = null`
- `log_level: str = "INFO"`

Environment variable mapping (MVP):

- `ACE_MCP_DEFAULT_MODEL`
- `ACE_MCP_SAFE_MODE`
- `ACE_MCP_MAX_SAMPLES_PER_CALL`
- `ACE_MCP_MAX_PROMPT_CHARS`
- `ACE_MCP_SESSION_TTL_SECONDS`
- `ACE_MCP_ALLOW_SAVE_LOAD`
- `ACE_MCP_SKILLBOOK_ROOT`

## Error Taxonomy

- `ACE_MCP_VALIDATION_ERROR`
- `ACE_MCP_SESSION_NOT_FOUND`
- `ACE_MCP_FORBIDDEN_IN_SAFE_MODE`
- `ACE_MCP_PROVIDER_ERROR`
- `ACE_MCP_TIMEOUT`
- `ACE_MCP_INTERNAL_ERROR`

Each error must include:

- `code` (stable string)
- `message` (human-readable)
- `details` (optional structured dict)

## Security & Safety

- Safe mode blocks mutating operations by policy.
- Path validation for save/load (reject paths outside `ACE_MCP_SKILLBOOK_ROOT` when set).
- Request size limits enforced before runner call.
- No secret values logged in payload dumps.

## Testing Plan

### Unit

- `models.py`: required fields, type coercion, limits.
- `registry.py`: session create/get/delete, TTL expiry, lock semantics.
- `handlers.py`: tool success paths + mapped failures.

### Integration

- Boot server with MCP SDK test harness when the `mcp` extra is installed.
- Validate tool registration/startup via smoke tests and cover tool success paths at the handler layer.
- Validate safe mode blocks expected tools.

### Regression

- Run existing `ace_next` tests ensuring no MCP dependency required unless explicitly installed.

## Delivery Plan (PR Sequence)

1. **PR-1**: skeleton package + config + models + tests.
2. **PR-2**: session registry + handlers for `ace.ask` and `ace.skillbook.get`.
3. **PR-3**: learning + save/load handlers, error mapping, limits.
4. **PR-4**: server bootstrap + CLI entrypoint + integration tests.
5. **PR-5**: docs + example client + changelog entry.

## Acceptance Criteria

- All implemented FR/NFR remain consistent with the optional dependency boundary.
- Tool schemas match contract doc exactly.
- `ace-mcp` starts successfully and serves all MVP tools.
- Existing non-MCP workflows are unaffected.

## Ready for Merge Checklist

- [x] Module tree implemented under `ace_next/integrations/mcp/`
- [x] Tool schemas implemented and contract-aligned (`ace.ask`, `ace.learn.sample`, `ace.learn.feedback`, `ace.skillbook.get/save/load`)
- [x] Optional dependency and CLI entrypoint wired (`mcp` extra, `ace-mcp`)
- [x] Safe mode policy enforced for mutating tools
- [x] Request-size limits enforced (`max_prompt_chars`, `max_samples_per_call`)
- [x] Optional root-bound path validation enforced for save/load (`ACE_MCP_SKILLBOOK_ROOT`)
- [x] MCP-focused tests passing (`test_ace_next_mcp_models/registry/handlers/server`)
- [x] Optional dependency boundary enforced with actionable install errors
- [x] Docs + example client updated (`docs/integrations/mcp.md`, `examples/ace_next/mcp_client_demo.py`)
- [x] Changelog updated for this feature
