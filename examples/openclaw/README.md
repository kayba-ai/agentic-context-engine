# OpenClaw + ACE Integration

Learn from [OpenClaw](https://docs.openclaw.ai) session transcripts and build a
self-improving skillbook of reusable strategies.

For the full setup guide, see [docs/integrations/openclaw.md](../../docs/integrations/openclaw.md).

## Quick Start (Docker — Recommended)

Extend your OpenClaw Docker image with ACE pre-installed. The agent runs
`ace-learn` at session start automatically.

```bash
# 1. Copy Dockerfile.ace into your OpenClaw directory
cp examples/openclaw/Dockerfile.ace /path/to/your/openclaw/

# 2. Build (from the OpenClaw directory)
docker build -t openclaw:base .
docker build -t openclaw:local --build-arg OPENCLAW_IMAGE=openclaw:base -f Dockerfile.ace .

# 3. Point OpenClaw at the new image (in your .env file)
#    OPENCLAW_IMAGE=openclaw:local

# 4. Pass your LLM API key in docker-compose.yml (see docs for all providers)
#    environment:
#      AWS_BEARER_TOKEN_BEDROCK: ${AWS_BEARER_TOKEN_BEDROCK}

# 5. Add auto-learning to AGENTS.md (see AGENTS.md.snippet)

# 6. Restart the gateway
docker compose down && docker compose up -d openclaw-gateway
```

### Verify

```bash
# Dry run — parses sessions without making LLM calls
docker run --rm -v ~/.openclaw:/home/node/.openclaw openclaw:local ace-learn --dry-run

# Full run
docker run --rm \
  -v ~/.openclaw:/home/node/.openclaw \
  -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
  openclaw:local ace-learn
```

## Quick Start (Host)

Run ACE on the host machine. Useful if you don't want to customize Docker.

```bash
# 1. Install
git clone https://github.com/Kayba-ai/agentic-context-engine.git
cd agentic-context-engine
uv sync

# 2. Set your LLM API key
export ANTHROPIC_API_KEY="your-key"

# 3. Dry run (no LLM calls, just parse sessions)
uv run python examples/openclaw/kayba-ace/learn_from_traces.py --dry-run

# 4. Learn from all new sessions
uv run python examples/openclaw/kayba-ace/learn_from_traces.py
```

## How It Works

```
OpenClaw sessions  -->  JSONL transcripts on disk
                                 |
                    ace-learn / learn_from_traces.py
                                 |
                LoadTracesStep --> OpenClawToTraceStep
                                 |
                    TraceAnalyser (Reflect -> Tag -> Update -> Apply)
                                 |
                 +---------------+----------------+
                 |                                |
          ace_skillbook.json              ace_skillbook.md
                                                  |
                               AGENTS.md tells agent to read skillbook
                                                  |
                               Agent loads strategies into context
```

1. OpenClaw writes session transcripts to `~/.openclaw/agents/<id>/sessions/*.jsonl`
2. `LoadTracesStep` reads JSONL files into raw event lists
3. `OpenClawToTraceStep` converts events to structured traces
4. `TraceAnalyser` runs the ACE learning pipeline (Reflect -> Tag -> Update -> Apply)
5. Updated skillbook is saved; the agent reads `ace_skillbook.md` into its context

## CLI Usage

```bash
# Learn from all new sessions (default agent: main)
ace-learn                              # Docker
uv run python examples/openclaw/kayba-ace/learn_from_traces.py  # Host

# Process specific trace files
ace-learn <trace.jsonl> [<trace2.jsonl> ...]

# Reprocess all sessions (ignore already-processed log)
ace-learn --reprocess

# Custom output directory
ace-learn --output ./out

# Enable Opik observability logging
ace-learn --opik

# Use a different agent ID
ace-learn --agent other-agent
```

## Files

| File | Description |
|---|---|
| `kayba-ace/` | Skill folder: `learn_from_traces.py`, `SKILL.md` (copied to OpenClaw workspace by `setup.py`) |
| `Dockerfile.ace` | Extends OpenClaw image with Python 3.12 + ACE |
| `ace-learn.sh` | Wrapper script (reference copy; Dockerfile inlines it) |
| `AGENTS.md.snippet` | Paste into your AGENTS.md for auto-learning |
| `setup.py` | Automated setup: copies skill folder, patches AGENTS.md |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ACE_MODEL` | `bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0` | LLM for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `main` | Agent ID for session discovery |
| `OPENCLAW_HOME` | `$HOME/.openclaw` | Used by `ace-learn` only; do not set as a gateway env var |
| `LITELLM_API_KEY` | - | API key (for non-Bedrock providers) |
| `SPH_LITELLM_KEY` | - | Alternative API key variable |
| `AWS_BEARER_TOKEN_BEDROCK` | - | AWS Bedrock bearer token |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |

## Outputs

| File | Format | Description |
|---|---|---|
| `ace_skillbook.json` | JSON | Full skillbook (machine-readable, persists across runs) |
| `ace_skillbook.md` | Markdown | Human-readable skillbook grouped by section |
| `ace_processed.txt` | Text | Tracks which sessions have already been processed |

## Automate with Cron (Host Only)

```bash
*/30 * * * * cd /path/to/agentic-context-engine && uv run python examples/openclaw/kayba-ace/learn_from_traces.py >> /tmp/ace-openclaw.log 2>&1
```
