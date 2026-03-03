# ClaudeRR Guide

## What is ClaudeRR

`ClaudeRRStep` uses the Claude Agent SDK to spawn a Claude Code subprocess as the reflector. It writes trace data to a temp directory and lets Claude analyze it with built-in tools (Read, Bash, Grep, etc.).

## Bedrock Setup

`ClaudeRRConfig.env` passes environment variables to the subprocess:

```python
claude_env = {"CLAUDE_CODE_USE_BEDROCK": "1"}
for key in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_REGION_NAME"):
    val = os.environ.get(key)
    if val:
        claude_env[key] = val
```

Strip `bedrock/` prefix from model name — Claude Code uses native model IDs:

```python
cc_model = args.model.removeprefix("bedrock/")
claude_env["ANTHROPIC_MODEL"] = cc_model
```

## Nesting Block

Claude Code blocks nested instances. Clear these env vars before launching:

```python
for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"):
    os.environ.pop(k, None)
```

Or from shell:

```bash
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS uv run ...
```

## Running from CLI

```bash
uv run python scripts/run_tau_benchmark.py \
  --replay-reflector-inputs .data/haiku_reflector_inputs_run1 \
  --model "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0" \
  --reflector-prompts v4 \
  --claude-rr
```

Add `--batch` for batch mode (combines all traces into one mega-context).

## Batch vs Sequential

| Mode | Flag | Behaviour | Output |
|------|------|-----------|--------|
| Sequential | (default) | Reflects on each trace individually | Domain-specific skills (58 from 30 traces in v4) |
| Batch | `--batch` | Concatenates all traces, reflects once | Fewer high-level skills (7-10) |

Sequential is the high-quality mode. Batch risks the reflector analysing the framing artifact instead of individual traces.

## Trace Format

Traces live in `.data/haiku_reflector_inputs_run{N}/` as `task_0.json` .. `task_N.json`:

```json
{
  "question": "...",
  "agent_output": {"final_answer": "...", "reasoning": "...", "skill_ids": []},
  "feedback": "PASS: ..." or "FAIL: ...",
  "ground_truth": "..."
}
```

The script loads them via `--replay-reflector-inputs <dir>`.

## Key Files

- `ace_next/rr/claude_rr.py` — `ClaudeRRStep` + `ClaudeRRConfig`
- `scripts/run_tau_benchmark.py` ~line 1448 — ClaudeRR setup in replay mode
