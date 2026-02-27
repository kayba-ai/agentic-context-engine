# Data Model: OpenClaw Integration

**Feature**: 001-openclaw-integration | **Date**: 2026-02-27

## Pipeline Steps

### LoadTracesStep (`ace_next/steps/load_traces.py`)

Generic step that reads a trace file from disk and puts raw content on `ctx.trace`.

| Attribute | Value |
|-----------|-------|
| `requires` | `frozenset({"sample"})` — `sample` is the file path (`str \| Path`) |
| `provides` | `frozenset({"trace"})` — raw file content (type depends on file format) |

**Behaviour**: Reads the file at `ctx.sample`, parses JSONL lines into `list[dict]`, places on `ctx.trace`. Skips unparseable lines gracefully.

### OpenClawToTraceStep (`ace_next/integrations/openclaw/to_trace.py`)

OpenClaw-specific step that converts raw JSONL events into a structured trace dict, preserving chronological order of queries, thinking, and tool uses.

| Attribute | Value |
|-----------|-------|
| `requires` | `frozenset({"trace"})` — raw `list[dict]` from LoadTracesStep |
| `provides` | `frozenset({"trace"})` — structured trace dict for ReflectStep |

**Behaviour**: Walks events in order, extracts message content items (text, thinking, toolCall, toolResult) preserving full content without truncation. Produces a trace dict with `{question, reasoning, answer, skill_ids, feedback, ground_truth}`. Transformation logic TBD (user will define separately).

## Entities

### SessionEvent

A single line from an OpenClaw JSONL transcript file.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | yes | Event type: `"session"`, `"message"`, `"thinking_level_change"`, `"custom"` |
| `id` | `str` | yes | Unique event identifier |
| `parentId` | `str \| None` | no | Parent event ID for threading |
| `timestamp` | `str` | yes | ISO 8601 timestamp |
| `message` | `MessagePayload \| None` | no | Present when `type == "message"` |
| `version` | `int \| None` | no | Present when `type == "session"` |
| `cwd` | `str \| None` | no | Working directory (session events only) |
| `thinkingLevel` | `str \| None` | no | Present when `type == "thinking_level_change"` |
| `data` | `dict \| None` | no | Present when `type == "custom"` |

### MessagePayload

The `message` field within a `SessionEvent` of type `"message"`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | `str` | yes | `"user"`, `"assistant"`, or `"toolResult"` |
| `content` | `list[ContentItem]` | yes | Array of content items |
| `api` | `str \| None` | no | API used (e.g., `"openai-completions"`) |
| `provider` | `str \| None` | no | Provider name (e.g., `"litellm"`) |
| `model` | `str \| None` | no | Model identifier |
| `usage` | `UsageInfo \| None` | no | Token/cost tracking |
| `stopReason` | `str \| None` | no | Why generation stopped |
| `timestamp` | `int \| None` | no | Unix timestamp (ms) |

### ContentItem

An individual content block within a message's content array.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | yes | `"text"`, `"thinking"`, `"toolCall"`, `"toolResult"` |
| `text` | `str \| None` | no | Present when `type == "text"` |
| `thinking` | `str \| None` | no | Present when `type == "thinking"` |
| `id` | `str \| None` | no | Tool call ID (toolCall) |
| `name` | `str \| None` | no | Tool name (toolCall) |
| `arguments` | `dict \| None` | no | Tool arguments (toolCall) |
| `toolCallId` | `str \| None` | no | Matching call ID (toolResult) |
| `content` | `list[dict] \| None` | no | Result content (toolResult) |

### Trace (dict)

The structured representation placed on `ctx.trace` by `OpenClawToTraceStep`. This is a plain dict matching the TraceAnalyser's raw trace interface.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `question` | `str` | yes | First user message in the session |
| `reasoning` | `str` | yes | Full chronological conversation: user messages, assistant responses, thinking (full), tool calls (full), tool results (full) |
| `answer` | `str` | yes | Last assistant text response |
| `skill_ids` | `list[str]` | yes | Always `[]` (no prior skills applied) |
| `feedback` | `str` | yes | Summary string (e.g., "Session completed with N tool calls") |
| `ground_truth` | `None` | yes | Always `None` (no ground truth for open-ended sessions) |

### Skillbook (existing)

Reused from `ace_next.core.skillbook.Skillbook`. No changes needed.

| Field | Type | Description |
|-------|------|-------------|
| `skills` | `dict[str, Skill]` | ID → Skill mapping |
| `sections` | `dict[str, list[str]]` | Section → skill IDs |
| `next_id` | `int` | Counter for new skill IDs |
| `similarity_decisions` | `dict` | Deduplication cache |

### Skill (existing)

Reused from `ace_next.core.skillbook.Skill`. No changes needed.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Format: `{section}-{5-digit-counter}` |
| `section` | `str` | Category/domain |
| `content` | `str` | Strategy description |
| `justification` | `str` | Why this strategy is valuable |
| `evidence` | `str` | Source session evidence |
| `helpful` | `int` | Positive vote count |
| `harmful` | `int` | Negative vote count |
| `neutral` | `int` | Neutral vote count |
| `created_at` | `str` | ISO 8601 timestamp |
| `updated_at` | `str` | ISO 8601 timestamp |
| `status` | `str` | `"active"` or `"invalid"` |

### ProcessedLog

Plain text file tracking which sessions have been processed.

| Aspect | Detail |
|--------|--------|
| **Path** | `~/.openclaw/ace_processed.txt` |
| **Format** | Newline-delimited session filenames (sorted) |
| **Example** | `b3db607f-7ae8-4089-b806-44800e961672.jsonl\nc4ef912a-...jsonl\n` |

## Relationships

```text
LoadTracesStep:
  ctx.sample (file path) → read JSONL → ctx.trace (list[dict] raw events)

OpenClawToTraceStep:
  ctx.trace (list[dict] raw events) → convert → ctx.trace (structured trace dict)

Pipeline composition:
  LoadTracesStep → OpenClawToTraceStep → ReflectStep → TagStep → UpdateStep → ApplyStep

SessionEvent (JSONL line)
  └── contains → MessagePayload
       └── contains → ContentItem[]
            ├── text → user messages / assistant responses (full)
            ├── thinking → reasoning content (full, no truncation)
            ├── toolCall → tool invocation data (full)
            └── toolResult → tool output data (full)

Skillbook → save_to_file() → ace_skillbook.json
Skillbook → wrap_skillbook_context() → sync_to_agents_md() → AGENTS.md
Session filenames → ProcessedLog (ace_processed.txt)
```

## Validation Rules

1. **LoadTracesStep**: Skips unparseable JSONL lines. Returns empty list for empty/missing files.
2. **SessionEvent**: Events with `type != "message"` are skipped by OpenClawToTraceStep.
3. **MessagePayload**: Messages must have `role` in `{"user", "assistant"}` and non-empty `content` array.
4. **ContentItem**: Unknown `type` values are silently skipped.
5. **Trace**: Must have non-empty `question` (at least one user message). Sessions with no user messages produce `None` from OpenClawToTraceStep.
6. **No truncation**: Thinking content, tool call arguments, and tool results are all preserved in full per clarifications.
