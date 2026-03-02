# ACE — Learn from Traces

This skill ships `learn_from_traces.py`, a script that reads OpenClaw session
transcripts, feeds them through the ACE learning pipeline, and writes an
updated skillbook to disk.

## Usage

```bash
python learn_from_traces.py [OPTIONS] [FILES...]
```

The script auto-discovers new sessions from `~/.openclaw/agents/<agent>/sessions/`
and only processes files that haven't been processed before. Processed filenames
are tracked in `ace_processed.txt`.

## Options

| Flag | Description |
|---|---|
| `--dry-run` | Parse sessions but skip the learning step (no LLM calls) |
| `--reprocess` | Ignore the processed log and reprocess all sessions |
| `--agent ID` | OpenClaw agent ID (default: `$OPENCLAW_AGENT_ID` or `main`) |
| `--output DIR` | Output directory for skillbook files (default: script directory) |
| `--opik` | Enable Opik observability logging |

Pass one or more JSONL file paths as positional arguments to process specific
files instead of auto-discovering sessions.

## Examples

### Learn from all new sessions

```bash
python learn_from_traces.py
```

Discovers unprocessed sessions under `~/.openclaw/agents/main/sessions/`,
runs the learning pipeline, and writes the updated skillbook.

### Dry run (no LLM calls)

```bash
python learn_from_traces.py --dry-run
```

Parses and validates sessions without calling the LLM. Useful for checking
that session files are readable before committing to a full run.

### Process a specific trace file

```bash
python learn_from_traces.py ~/.openclaw/agents/main/sessions/f967d602.jsonl
```

Skips auto-discovery and processes only the given file. The processed log
is not updated when files are passed directly.

### Process multiple files

```bash
python learn_from_traces.py session1.jsonl session2.jsonl session3.jsonl
```

### Reprocess all sessions

```bash
python learn_from_traces.py --reprocess
```

Ignores `ace_processed.txt` and reprocesses every session file. Useful after
upgrading ACE or when you want to rebuild the skillbook from scratch.

### Use a different agent

```bash
python learn_from_traces.py --agent my-agent
```

Looks for sessions in `~/.openclaw/agents/my-agent/sessions/` instead of
the default `main`.

### Write output to a custom directory

```bash
python learn_from_traces.py --output /tmp/ace-out
```

Writes `ace_skillbook.json`, `ace_skillbook.md`, and `ace_processed.txt`
to `/tmp/ace-out/` instead of the script's directory.

### Enable Opik observability

```bash
python learn_from_traces.py --opik
```

Logs LLM calls and pipeline steps to [Opik](https://www.comet.com/opik)
for debugging and monitoring.

### Run from the ACE repo (host setup)

```bash
cd /path/to/agentic-context-engine
uv run python examples/openclaw/kayba-ace/learn_from_traces.py --dry-run
```

When running from a repo checkout, `uv run` ensures `ace_next` is importable.

### Run inside Docker

```bash
# Dry run
docker run --rm -v ~/.openclaw:/home/node/.openclaw openclaw:local ace-learn --dry-run

# Full run with API key
docker run --rm \
  -v ~/.openclaw:/home/node/.openclaw \
  -e AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
  openclaw:local ace-learn
```

Inside the extended Docker image, `ace-learn` is a wrapper that calls this
script with the correct paths.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ACE_MODEL` | `bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0` | LLM model for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `main` | Default agent ID (overridden by `--agent`) |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw home directory |

The script also loads `.env` files from `$OPENCLAW_HOME/.env` and `~/.env`.
Set your API key in one of these variables:

- `AWS_BEARER_TOKEN_BEDROCK`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `LITELLM_API_KEY`
- `SPH_LITELLM_KEY`

## Output Files

| File | Description |
|---|---|
| `ace_skillbook.json` | Machine-readable skillbook (persists across runs) |
| `ace_skillbook.md` | Human-readable skillbook loaded by the agent |
| `ace_processed.txt` | Tracks which sessions have been processed |

## Setup

Run the setup script from the ACE repo to copy this skill into your OpenClaw workspace:

```bash
python examples/openclaw/setup.py
```

Full guide: https://kayba-ai.github.io/agentic-context-engine/integrations/openclaw/
