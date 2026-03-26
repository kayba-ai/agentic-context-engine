# Claude SDK Integration

The Claude SDK integration provides composable ACE steps for the
[Anthropic Python SDK](https://docs.anthropic.com/en/api/client-sdks). Use it
when you want direct Messages API access inside your own pipeline instead of a
prebuilt runner.

## Quick Start

```python
from ace import Pipeline, Reflector, SkillManager, Skillbook, learning_tail
from ace.integrations import ClaudeSDKExecuteStep, ClaudeSDKToTrace

skillbook = Skillbook()
pipe = Pipeline([
    ClaudeSDKExecuteStep(model="claude-sonnet-4-20250514"),
    ClaudeSDKToTrace(),
    *learning_tail(Reflector("gpt-4o-mini"), SkillManager("gpt-4o-mini"), skillbook),
])
```

## Installation

```bash
uv add ace-framework[claude-sdk]
```

For observability, also install and configure Logfire:

```bash
uv add ace-framework[logfire]
```

```python
from ace.observability import configure_logfire

configure_logfire()
```

## What It Provides

- `ClaudeSDKExecuteStep` — injects skillbook context, calls the Anthropic
  Messages API, and writes a validated `ClaudeSDKResult` to `ctx.trace`
- `ClaudeSDKToTrace` — converts `ClaudeSDKResult` into the standard ACE trace
  dict consumed by `ReflectStep`
- `ClaudeSDKResult` — Pydantic model with validated output, token usage,
  latency, tool calls, and raw response access
- `ToolCall` — Pydantic model for captured Claude tool invocations

## Parameters

### ClaudeSDKExecuteStep

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Claude model ID |
| `system_prompt` | `str \| None` | `None` | Base system prompt |
| `max_tokens` | `int` | `4096` | Maximum output tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `tools` | `list[dict] \| None` | `None` | Anthropic tool definitions |
| `api_key` | `str \| None` | `None` | Optional API key override |
| `base_url` | `str \| None` | `None` | Optional API base URL |
| `inject_skillbook` | `bool` | `True` | Prepend skillbook context to the system prompt |
| `client` | `Any` | `None` | Injected Anthropic client for testing or custom transport |

## Observability

When Logfire is configured, the step emits three layers of observability:

1. Step-level `logfire.span(...)` around `ClaudeSDKExecuteStep`
2. Structured `logfire.info(...)` and `logfire.error(...)` events with tokens,
   latency, stop reason, and tool counts
3. `logfire.instrument_anthropic(client)` auto-instrumentation for the
   underlying SDK calls

The result model also captures:

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `latency_seconds`
- `stop_reason`
- `tool_calls`

## Tool Use

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

execute = ClaudeSDKExecuteStep(
    model="claude-sonnet-4-20250514",
    tools=tools,
)
```

If Claude returns tool use blocks, they are captured on
`ClaudeSDKResult.tool_calls` as validated `ToolCall` models.

## Validation

`ClaudeSDKExecuteStep` validates its configuration with Pydantic before the
client is constructed. `ClaudeSDKResult` and `ToolCall` are also Pydantic
models, so invalid token counts, latency values, or malformed tool calls are
rejected early.

## What to Read Next

- [Integration Pattern](../guides/integration.md) — the shared
  INJECT/EXECUTE/LEARN design
- [Composing Pipelines](../guides/composing-pipelines.md) — mix SDK steps with
  other ACE steps
