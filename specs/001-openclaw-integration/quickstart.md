# Quick Start: OpenClaw Integration

**Feature**: 001-openclaw-integration | **Date**: 2026-02-27

## Prerequisites

- Python 3.12+
- An OpenClaw agent that has completed at least one session
- An Anthropic API key (or any LiteLLM-supported provider)

## Setup (3 steps)

### 1. Install

```bash
git clone https://github.com/kayba-ai/agentic-context-engine.git
cd agentic-context-engine
uv sync
```

### 2. Configure

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Optional overrides:

```bash
export OPENCLAW_AGENT_ID="main"           # which agent to learn from
export OPENCLAW_HOME="~/.openclaw"        # OpenClaw home directory
export ACE_MODEL="anthropic/claude-sonnet-4-20250514"  # LLM model
```

### 3. Run

```bash
# Learn from all past sessions
uv run python examples/openclaw/learn_from_traces.py

# Preview what would be processed (no LLM calls, no file changes)
uv run python examples/openclaw/learn_from_traces.py --dry-run

# Reprocess everything (ignore what's already been learned)
uv run python examples/openclaw/learn_from_traces.py --reprocess
```

## What Happens

1. **Discovers** session transcripts from `~/.openclaw/agents/<id>/sessions/`
2. **Parses** JSONL files into structured traces (user messages, reasoning, tool calls)
3. **Learns** by running ACE's Reflect → Tag → Update → Apply pipeline
4. **Saves** strategies to `~/.openclaw/ace_skillbook.json`
5. **Syncs** strategies into `~/.openclaw/workspace/AGENTS.md`
6. Your OpenClaw agent reads the updated AGENTS.md on its next session

## Automate (optional)

Run every 30 minutes via cron:

```bash
crontab -e
# Add:
*/30 * * * * cd /path/to/agentic-context-engine && uv run python examples/openclaw/learn_from_traces.py >> /tmp/ace-openclaw.log 2>&1
```

## Verify

After running, check the output:

```bash
# View learned strategies
cat ~/.openclaw/ace_skillbook.json | python -m json.tool | head -50

# View what was injected into your agent
grep -A 20 "ACE:SKILLBOOK:START" ~/.openclaw/workspace/AGENTS.md
```
