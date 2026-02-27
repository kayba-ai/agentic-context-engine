# OpenClaw Integration

The OpenClaw integration learns from [OpenClaw](https://docs.openclaw.ai) session transcripts and injects learned strategies back into the agent via `AGENTS.md`. Unlike other integrations, OpenClaw does not have a runner class — it uses a standalone script that composes pipeline steps with `TraceAnalyser`.

## Installation

```bash
uv sync  # core deps only — no extras needed
```

## Quick Start

```bash
# Set your LLM API key
export ANTHROPIC_API_KEY="your-key"

# Learn from all past sessions
uv run python examples/openclaw/learn_from_traces.py

# Preview what would be processed (no LLM calls, no file changes)
uv run python examples/openclaw/learn_from_traces.py --dry-run

# Reprocess everything (ignore what's already been learned)
uv run python examples/openclaw/learn_from_traces.py --reprocess
```

## How It Works

```
OpenClaw runs sessions  -->  JSONL transcripts on disk
                                     |
                        learn_from_traces.py (cron)
                                     |
                    LoadTracesStep -> OpenClawToTraceStep
                                     |
                        TraceAnalyser (Reflect -> Update -> Apply)
                                     |
                     +---------+-----+--------+
                     |                        |
              ace_skillbook.json        AGENTS.md (updated)
                                              |
                              OpenClaw loads on next session
```

1. **LOAD** — `LoadTracesStep` reads JSONL session files from `~/.openclaw/agents/<id>/sessions/`
2. **CONVERT** — `OpenClawToTraceStep` transforms raw events into ACE trace format (pass-through for now)
3. **LEARN** — `TraceAnalyser` runs the learning pipeline (Reflect, Tag, Update, Apply)
4. **SYNC** — Updated skillbook is written between `<!-- ACE:SKILLBOOK:START/END -->` markers in `AGENTS.md`

## Pipeline Steps

### LoadTracesStep

Generic JSONL file loader. Reads a file path from `ctx.sample`, parses each line as JSON, populates `ctx.trace`.

| Field | Value |
|-------|-------|
| Location | `ace_next/steps/load_traces.py` |
| Requires | `sample` |
| Provides | `trace` |
| Side effects | None (pure) |

### OpenClawToTraceStep

OpenClaw-specific trace converter. Currently a pass-through — transformation logic to be defined when the OpenClaw trace schema stabilises.

| Field | Value |
|-------|-------|
| Location | `ace_next/integrations/openclaw/to_trace.py` |
| Requires | `trace` |
| Provides | `trace` |
| Side effects | None (pure) |

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | API key for LLM provider |
| `ACE_MODEL` | `anthropic/claude-sonnet-4-20250514` | LLM for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `main` | OpenClaw agent to learn from |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw home directory |
| `OPENCLAW_WORKSPACE` | `~/.openclaw/workspace` | Workspace where AGENTS.md lives |

## Files Produced

| Path | Description |
|---|---|
| `~/.openclaw/ace_skillbook.json` | Persistent skillbook (survives across runs) |
| `~/.openclaw/ace_processed.txt` | Log of already-processed session filenames |
| `~/.openclaw/workspace/AGENTS.md` | Updated with learned strategies between marker comments |

## Cron Setup

```bash
*/30 * * * * cd /path/to/agentic-context-engine && uv run python examples/openclaw/learn_from_traces.py >> /tmp/ace-openclaw.log 2>&1
```

## What to Read Next

- [Integration Pattern](../guides/integration.md) — how the INJECT/EXECUTE/LEARN pattern works
- [The Skillbook](../concepts/skillbook.md) — how learned strategies are stored
- [ACE Design](../ACE_DESIGN.md) — architecture and step reference
