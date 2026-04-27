# @kayba_ai/openclaw-tracing

OpenClaw plugin that captures every agent turn — user message, full LLM prompt and response (including thinking blocks), tool calls, final reply — and ships it as a structured Kayba trace.

Pairs with [@kayba_ai/tracing](../typescript/). Whatever signals a tool-level plugin (e.g. a trader plugin) is already emitting via `kayba.trace()` will land in the same Kayba folder and can be cross-referenced with these turn-level traces by `runId`.

## What you get

- One trace per agent turn, named `agent.turn`
- Child span `llm.call` capturing the full LLM input/output for the turn
- Span attributes: `openclaw.runId`, `openclaw.sessionId`, `openclaw.agentId`, `openclaw.channelId`, `openclaw.senderId`
- Token usage, stop reason, and the assistant's full content blocks (thinking + text + tool calls) on the `llm.call` span output
- Compatible with ACE — the captured shape contains everything `OpenClawToTraceStep` needs

## Install

```bash
openclaw plugins install @kayba_ai/openclaw-tracing
```

Then add the config block to `~/.openclaw/openclaw.json`. The `hooks.allowConversationAccess` flag is required because conversation-content hooks are gated for non-bundled plugins:

```json
{
  "plugins": {
    "allow": ["kayba-tracing"],
    "entries": {
      "kayba-tracing": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "apiKey": "kayba_ak_...",
          "folder": "main"
        }
      }
    }
  }
}
```

Restart the gateway. From this point on, every agent turn produces one trace at https://use.kayba.ai/traces/v2.

## Config

| Key | Type | Default | Notes |
|---|---|---|---|
| `apiKey` | string | (required) | From https://use.kayba.ai/settings/api-keys |
| `baseUrl` | string | `https://use.kayba.ai` | For self-hosted Kayba |
| `folder` | string | `null` | Dashboard folder for grouping |
| `captureSystemPrompt` | boolean | `true` | Include the (potentially large) system prompt on each `llm.call` span |
| `captureHistory` | boolean | `true` | Include `historyMessages` on each `llm.call` span (large for long sessions) |
| `maxAttributeBytes` | integer | `65536` | Per-attribute truncation cap |

## How it works

The plugin subscribes to OpenClaw's typed hooks via `api.on(...)`:

| Hook | Used for |
|---|---|
| `message_received` | open turn, capture user message + sender |
| `before_agent_start` | bind `runId` |
| `llm_input` | capture prompt, system prompt, history, model, provider |
| `llm_output` | capture response, assistant content blocks, usage |
| `agent_end` | finalize and emit the trace |

Race protection: `agent_end` and `llm_output` can fire in either order. The plugin defers finalize by `~250ms` after `agent_end` to absorb a late `llm_output`, and falls through immediately if both have already arrived.

Stale-turn safety: turn state older than 5 minutes is evicted. A turn that never reaches `agent_end` (crashed mid-loop) is dropped silently — the trader plugin's per-tool spans still land independently.

## Why a separate plugin and not part of `@kayba_ai/tracing`

OpenClaw's plugin loader requires a manifest (`openclaw.plugin.json` + `package.json` with `openclaw.extensions[]`) and must be installed via `openclaw plugins install`. Bundling that into the generic SDK would force the OpenClaw runtime as a dependency on every SDK consumer, including the trader plugin which uses the SDK directly without the OpenClaw plugin contract.

## License

MIT
