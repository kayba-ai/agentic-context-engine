# ACE Claude Code Integration

Learn from Claude Code sessions. No API keys required - uses your existing Claude subscription.

## Overview

This integration enables ACE (Agentic Context Engineering) to learn from your Claude Code sessions by reading Claude Code transcripts from `~/.claude/projects/`.

**Key Features:**
- Manual trigger: run `ace-learn` when you want to learn (not after every turn)
- Writes learned strategies into `CLAUDE.md` (auto-loaded by Claude Code)
- Persists a full skillbook to `.ace/skillbook.json`
- Uses the Claude Code CLI subscription (no API keys)
- Auto-patches Claude Code's `cli.js` system prompt for reliable, low-token `--print` learning runs

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
Run `ace-learn`
        │
        ▼
Read latest session transcript (~/.claude/projects/**/*.jsonl)
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
| `ace-learn setup` | Install `/ace-*` slash commands into Claude Code |
| `ace-learn doctor` | Verify prerequisites |
| `ace-learn insights` | Show learned strategies |
| `ace-learn remove <id>` | Remove a specific strategy |
| `ace-learn clear --confirm` | Clear all strategies |

## Storage

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Learned strategies (auto-read by Claude Code) |
| `.ace/skillbook.json` | Persistent skillbook (JSON) |
| `~/.ace/claude-learner/cli.js` | Patched Claude Code CLI (internal; created automatically when possible) |

## Claude CLI System Prompt Patching

For learning runs, ACE prefers a patched copy of Claude Code's `cli.js` with a minimal ACE-focused system prompt.
This avoids Claude Code's large default system prompt in `--print` mode and prevents tool-use attempts during learning.

- Written to: `~/.ace/claude-learner/cli.js` (does not modify your original installation)
- Auto-created on demand by `CLIClient` (no setup command)
- Falls back to the system `claude` binary if patching is unavailable

## Claude Code Slash Commands

ACE provides slash commands for use within Claude Code sessions.

### Installation

Run once after installing ACE:
```bash
ace-learn setup
```

This installs the following slash commands to `~/.claude/commands/`:

| Slash Command | Description |
|---------------|-------------|
| `/ace-learn` | Learn from current session |
| `/ace-learn-lines` | Learn from last N lines |
| `/ace-doctor` | Verify ACE setup |
| `/ace-insights` | Show learned strategies |
| `/ace-remove` | Remove a strategy |
| `/ace-clear` | Clear all strategies |

### Verification

Check if commands are installed:
```bash
ace-learn doctor
```

Look for the "Slash commands" section in the output.

## Project Root Detection

ACE finds your project root by looking for these markers (in priority order):

| Marker | Description |
|--------|-------------|
| `.ace-root` | Explicit ACE root (for monorepos) |
| `.git` | Git repository |
| `.hg` | Mercurial repository |
| `.svn` | Subversion repository |
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
├── prompt_patcher.py # Internal utility for patching Claude Code cli.js
├── prompts.py       # Custom Reflector prompt for coding
├── commands/        # Slash command templates (installed via `ace-learn setup`)
│   ├── ace-learn.md
│   ├── ace-learn-lines.md
│   ├── ace-doctor.md
│   ├── ace-insights.md
│   ├── ace-remove.md
│   └── ace-clear.md
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
| Patched CLI not created | Run `ace-learn doctor` to see why (Node missing, source cli.js not found, etc.) |
