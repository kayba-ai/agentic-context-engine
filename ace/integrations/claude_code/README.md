# ACE Claude Code Integration

Learn from Claude Code sessions. No API keys required - uses your existing Claude subscription.

## Overview

This integration enables ACE (Agentic Context Engineering) to learn from your Claude Code sessions. Run `ace-learn` after a session to extract strategies that improve future sessions.

**Key Features:**
- Zero setup - just run `ace-learn` after a session
- Learns directly into CLAUDE.md (auto-loaded by Claude Code)
- Uses Claude CLI (subscription auth, no API keys)

## Quick Start

```bash
# Install
pip install ace-framework

# After a Claude Code session, learn from it
ace-learn

# Check prerequisites
ace-learn doctor
```

## How It Works

```
Use Claude Code normally
        │
        ▼
Run `ace-learn` (or /ace-learn in Claude Code)
        │
        ▼
Reflector analyzes session transcript
        │
        ▼
SkillManager extracts strategies
        │
        ▼
CLAUDE.md updated with learned strategies
```

## Commands

| Command | Description |
|---------|-------------|
| `ace-learn` | Learn from latest transcript |
| `ace-learn --lines N` | Learn from last N lines only |
| `ace-learn doctor` | Verify prerequisites |
| `ace-learn insights` | Show learned strategies |
| `ace-learn remove <id>` | Remove a specific strategy |
| `ace-learn clear --confirm` | Clear all strategies |

## Storage

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Learned strategies (auto-read by Claude Code) |
| `.ace/skillbook.json` | Persistent skillbook (JSON) |

## Project Root Detection

ACE finds your project root by looking for these markers (in priority order):

| Marker | Description |
|--------|-------------|
| `.ace-root` | Explicit ACE root (for monorepos) |
| `.git` | Git repository |
| `pyproject.toml` | Python project |
| `package.json` | Node.js project |
| `Cargo.toml` | Rust project |
| `go.mod` | Go project |

**Monorepo Setup**: Create `.ace-root` at your monorepo root.

**No Project**: Falls back to home directory (`~/`).

## File Structure

```
ace/integrations/claude_code/
├── __init__.py      # Package exports
├── learner.py       # Main learner and CLI
├── cli_client.py    # Claude CLI wrapper (LLM client)
├── prompts.py       # Custom Reflector prompt for coding
└── README.md        # This file

<project>/
├── CLAUDE.md        # Skills injected here (ACE section)
└── .ace/
    └── skillbook.json   # Persistent skillbook
```

## Troubleshooting

Run `ace-learn doctor` to diagnose issues:

```
1. Claude CLI...     ✓ Found at: /usr/local/bin/claude
2. Transcript...     ✓ Latest: ~/.claude/projects/.../abc123.jsonl
3. Output...         ✓ Project: /path/to/project
```

**Common Issues:**

| Problem | Solution |
|---------|----------|
| No transcript found | Use Claude Code first |
| CLI not found | Install: `npm install -g @anthropic-ai/claude-code` |
