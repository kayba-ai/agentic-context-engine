# CLI Contract: learn_from_traces.py

**Feature**: 001-openclaw-integration | **Date**: 2026-02-27

## Command

```bash
uv run python examples/openclaw/learn_from_traces.py [OPTIONS]
```

## Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dry-run` | bool | `false` | Parse sessions and report findings without running the learning pipeline or modifying any files |
| `--reprocess` | bool | `false` | Ignore the processed log and reprocess all sessions from scratch |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | API key for LLM provider |
| `ACE_MODEL` | `"anthropic/claude-sonnet-4-20250514"` | LLM model for reflection and skill extraction |
| `OPENCLAW_AGENT_ID` | `"main"` | OpenClaw agent identifier |
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw home directory |
| `OPENCLAW_WORKSPACE` | `~/.openclaw/workspace` | Path to OpenClaw workspace (where AGENTS.md lives) |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success (including "nothing new to learn") |
| `1` | Error (missing sessions dir, API failure, corrupted skillbook) |

## Output Format

Console output with section headers:

```text
============================================================
  Discovering sessions
============================================================
  Sessions dir:  ~/.openclaw/agents/main/sessions
  Total sessions: 12
  Already processed: 10
  New to process: 2

============================================================
  Parsing sessions
============================================================
  + b3db607f-....jsonl: Hello world
  Parsed: 2, Skipped (empty): 0

============================================================
  Loading skillbook
============================================================
  Loaded 5 existing strategies from ~/.openclaw/ace_skillbook.json

============================================================
  Learning from 2 traces
============================================================
  Processed: 2/2
  New strategies: 1 (total: 6)

  Latest strategies:
    [session_mgmt-00006] Use greeting to establish session context...

============================================================
  Saving
============================================================
  Skillbook: ~/.openclaw/ace_skillbook.json
  Processed log: ~/.openclaw/ace_processed.txt

============================================================
  Syncing to OpenClaw
============================================================
  Synced 6 strategies to ~/.openclaw/workspace/AGENTS.md

============================================================
  Done
============================================================
```

## Files Produced

| File | Format | Description |
|------|--------|-------------|
| `~/.openclaw/ace_skillbook.json` | JSON | Persistent skillbook with all learned strategies |
| `~/.openclaw/ace_processed.txt` | Text | Newline-delimited list of processed session filenames |
| `~/.openclaw/workspace/AGENTS.md` | Markdown | Updated with strategies between `<!-- ACE:SKILLBOOK:START/END -->` markers |

## AGENTS.md Marker Contract

```markdown
<!-- ACE:SKILLBOOK:START -->
## Learned Strategies

These strategies were learned from your past sessions. Use relevant
ones to improve your responses. Cite strategy IDs (e.g.
[web-scraping-00001]) when you apply them.

{wrap_skillbook_context() output}
<!-- ACE:SKILLBOOK:END -->
```

- If markers exist: content between them is replaced
- If markers don't exist: section is appended to end of file
- Content outside markers is never modified
