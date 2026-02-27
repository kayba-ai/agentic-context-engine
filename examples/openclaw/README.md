# OpenClaw + ACE Integration

Learn from [OpenClaw](https://docs.openclaw.ai) session transcripts and inject
learned strategies back into the agent via `AGENTS.md`.

## How it works

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

1. OpenClaw writes session transcripts to `~/.openclaw/agents/<id>/sessions/*.jsonl`
2. `LoadTracesStep` reads JSONL files into raw event lists
3. `OpenClawToTraceStep` converts events to structured traces (pass-through for now)
4. `TraceAnalyser` runs the ACE learning pipeline (Reflect -> Tag -> Update -> Apply)
5. The updated skillbook is saved and synced into `AGENTS.md` between marker comments

## Setup

```bash
# Install core dependencies
uv sync

# Set your LLM API key
export ANTHROPIC_API_KEY="your-key"
```

## Usage

```bash
# Learn from all past sessions
uv run python examples/openclaw/learn_from_traces.py

# Preview what would be processed (no LLM calls, no file changes)
uv run python examples/openclaw/learn_from_traces.py --dry-run

# Reprocess everything (ignore what's already been learned)
uv run python examples/openclaw/learn_from_traces.py --reprocess
```

### Cron

```bash
*/30 * * * * cd /path/to/agentic-context-engine && uv run python examples/openclaw/learn_from_traces.py >> /tmp/ace-openclaw.log 2>&1
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | API key for LLM provider |
| `ACE_MODEL` | `anthropic/claude-sonnet-4-20250514` | LLM for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `main` | OpenClaw agent to learn from |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw home directory |
| `OPENCLAW_WORKSPACE` | `~/.openclaw/workspace` | Workspace where AGENTS.md lives |

## Files produced

| Path | Description |
|---|---|
| `~/.openclaw/ace_skillbook.json` | Persistent skillbook (survives across runs) |
| `~/.openclaw/ace_processed.txt` | Log of already-processed session filenames |
| `~/.openclaw/workspace/AGENTS.md` | Updated with learned strategies between `<!-- ACE:SKILLBOOK:START/END -->` markers |

## Architecture

The integration uses two pipeline steps from the ACE framework:

- **`LoadTracesStep`** (`ace_next/steps/load_traces.py`) — Generic step that reads a JSONL file and places parsed events on `ctx.trace`
- **`OpenClawToTraceStep`** (`ace_next/integrations/openclaw/to_trace.py`) — OpenClaw-specific step that converts raw events to structured traces (currently a pass-through; transformation logic to be defined)

These steps can be composed with `learning_tail()` in a full pipeline, or used standalone as in this script.
