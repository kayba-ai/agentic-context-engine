# Research: OpenClaw Integration

**Feature**: 001-openclaw-integration | **Date**: 2026-02-27

## R-001: OpenClaw JSONL Transcript Format

**Decision**: The parser must handle the nested OpenClaw JSONL event format, not flat role/content events.

**Rationale**: Inspecting the sample transcript (`b3db607f-...jsonl`) reveals the format is significantly different from what the current parser assumes:

- **Top-level fields**: `type`, `id`, `parentId`, `timestamp`, and optionally `message`
- **Event types**: `"session"`, `"message"`, `"thinking_level_change"`, `"custom"`
- **Message structure**: `event["message"]` contains `role`, `content` (array), `api`, `provider`, `model`, `usage`
- **Content array items**: Each has a `type` field — `"text"`, `"thinking"`, `"toolCall"`, `"toolResult"`
  - Text: `{"type": "text", "text": "..."}`
  - Thinking: `{"type": "thinking", "thinking": "...", "thinkingSignature": "reasoning_content"}`
  - Tool call: `{"type": "toolCall", "id": "...", "name": "...", "arguments": {...}}`
  - Tool result: `{"type": "toolResult", "toolCallId": "...", "content": [{"type": "text", "text": "..."}]}`

**Current parser bug**: `parse_session_jsonl()` reads `event.get("role")` and `event.get("content")` at the top level. The actual data has `event["message"]["role"]` and `event["message"]["content"]` (an array, not a string). The current parser will always produce empty traces from real OpenClaw sessions.

**Alternatives considered**: None — must match the actual format.

## R-002: Trace Dict Structure for TraceAnalyser

**Decision**: Keep the existing trace dict format `{question, reasoning, answer, skill_ids, feedback, ground_truth}` as TraceAnalyser accepts raw dicts placed on `ctx.trace`.

**Rationale**: `TraceAnalyser` uses raw traces — any dict type is accepted and placed on `ctx.trace`. The Reflector then reads `ctx.trace` (or falls back to `ctx.agent_output`). The trace dict keys should map cleanly to what Reflector's prompt template expects. Existing test patterns confirm this structure works.

**Alternatives considered**: Using `AgentOutput` dataclass — rejected because TraceAnalyser explicitly supports raw traces without wrapping.

## R-003: Thinking Content Handling

**Decision**: Preserve thinking content in full — no truncation or filtering.

**Rationale**: Thinking traces are integral to how the OpenClaw agent works. They contain the richest reasoning data and provide essential context for the ACE learning pipeline to understand the agent's decision-making process. Truncation would lose critical information that the Reflector needs to extract meaningful strategies.

**Clarification**: User explicitly confirmed thinking traces must be preserved in full (see spec Clarifications 2026-02-27).

**Alternatives rejected**:
- Head+tail truncation (500/200 chars) — rejected per user requirement; loses critical reasoning context.
- Filtering thinking blocks entirely — rejected; removes the most valuable data for learning.

## R-004: Tool Call/Result Pairing

**Decision**: Extract tool calls with their names and arguments from `"toolCall"` content items, and pair with results from subsequent `"toolResult"` events. Preserve all data in full — no truncation of arguments or results.

**Rationale**: Tool usage is a key signal for the Reflector. The JSONL format stores tool calls as content items within assistant messages, and tool results as separate content items (or messages) with matching `toolCallId`. Pairing them gives the Reflector a complete picture of what tools were used and what they returned.

**Clarification**: User explicitly confirmed tool arguments and results must be preserved in full (see spec Clarifications 2026-02-27).

**Alternatives rejected**: Ignoring tool calls — rejected because tool usage patterns are one of the most valuable things to learn from. Truncating tool results — rejected per user requirement.

## R-005: Session Discovery Path

**Decision**: Discover sessions from `~/.openclaw/agents/<agent_id>/sessions/*.jsonl` with the session directory structure matching the JSONL sample.

**Rationale**: The sample file's `"session"` event contains `"cwd": "/app"` and session metadata. OpenClaw stores sessions per-agent under the home directory. The `OPENCLAW_AGENT_ID` environment variable selects which agent's sessions to process.

**Alternatives considered**: Recursive glob for all agents — rejected for simplicity (Principle III). Users can run the script multiple times with different `OPENCLAW_AGENT_ID` values.

## R-006: Skillbook Persistence Format

**Decision**: Use the existing `Skillbook.save_to_file()` / `Skillbook.load_from_file()` JSON format. No custom serialization needed.

**Rationale**: The skillbook JSON format is well-defined with fields: `skills`, `sections`, `next_id`, `similarity_decisions`. Each skill has `id`, `section`, `content`, `justification`, `evidence`, `helpful/harmful/neutral`, timestamps, and `status`. The existing API handles all serialization.

**Alternatives considered**: None — reusing existing infrastructure per Principle III.

## R-007: AGENTS.md Sync Format

**Decision**: Use `wrap_skillbook_context()` from `ace_next.integrations` to format the skillbook content between HTML comment markers `<!-- ACE:SKILLBOOK:START/END -->`.

**Rationale**: `wrap_skillbook_context()` already produces a formatted string with skillbook strategies and usage instructions. The marker-based replacement pattern in `sync_to_agents_md()` is correct — preserves content outside markers, handles create/update cases.

**Alternatives considered**: Custom formatting — rejected because `wrap_skillbook_context()` already exists and produces the right format.

## R-008: Error Handling Strategy

**Decision**: Graceful degradation — skip individual malformed sessions, report errors, continue processing.

**Rationale**: Constitution Principle I (Ease of Use) requires that a single bad file doesn't crash the entire run. FR-010 explicitly requires skipping malformed files gracefully. The current try/except pattern in `parse_session_jsonl()` handles `JSONDecodeError` correctly.

**Alternatives considered**: Strict mode that fails on first error — rejected as it violates FR-010 and Principle I.

## R-009: Testing Strategy

**Decision**: Unit tests with MockLLMClient, using sample JSONL fixtures. No live LLM calls in tests.

**Rationale**: Existing test patterns in `tests/` use `MockLLMClient` that returns canned `ReflectorOutput` and `SkillManagerOutput`. This allows testing the full flow (parse → analyse → save → sync) without API costs or flakiness. The sample JSONL file can serve as a test fixture.

**Alternatives considered**: Integration tests with real LLM — useful but should be `@pytest.mark.slow` and optional.
