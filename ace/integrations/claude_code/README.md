# ACE Claude Code Integration

Subscription-only learning from Claude Code sessions. No API keys required.

## Overview

This package enables ACE (Agentic Context Engineering) to learn from your Claude Code sessions automatically. It observes how you use Claude Code and extracts reusable strategies that improve future sessions.

**Key Features:**
- Native async hook - never slows down Claude Code
- Subscription-only via Claude CLI (no API keys needed)
- Per-project skill files in `.claude/skills/ace-learnings/`
- Global fallback for sessions outside projects

## Quick Start

### From PyPI (End Users)

```bash
pip install ace-framework
ace-learn setup
ace-learn doctor
```

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine

# Install in editable mode (installs ace-learn command globally)
pip install -e .

# Configure the Claude Code async hook
ace-learn setup

# Verify everything works
ace-learn doctor
```

## End-to-End Testing

After installation, verify ACE is working:

```bash
# 1. Check all prerequisites
ace-learn doctor

# 2. Open Claude Code in any project and have a conversation
#    (e.g., ask "list files in this directory")

# 3. Check if skills were generated
ls -la .claude/skills/ace-learnings/
cat .claude/skills/ace-learnings/SKILL.md
```

**Expected behavior:**
- Async hook triggers after each Claude Code response
- Learning runs in background (async: true)
- New strategies appear in `.claude/skills/ace-learnings/SKILL.md`

## Commands

### Inside Claude Code (Slash Commands)

These commands work directly in your Claude Code session:

| Command | Description |
|---------|-------------|
| `/ace-on` | Enable ACE learning (re-adds hook to settings) |
| `/ace-off` | Disable ACE learning (removes hook from settings) |
| `/ace-insights` | Show learned strategies from the skillbook |
| `/ace-clear` | Clear all learned strategies |
| `/ace-remove` | Remove a specific learned strategy |

### Auto-loaded Skill

The `ace-learnings` skill at `.claude/skills/ace-learnings/SKILL.md` is automatically loaded by Claude Code when present in a project. It provides learned coding patterns as context.

### Terminal Commands

```bash
ace-learn setup      # Configure Claude Code async hook
ace-learn doctor     # Verify prerequisites
ace-learn insights   # Show learned strategies
ace-learn disable    # Disable learning (removes hook)
ace-learn enable     # Re-enable learning (re-adds hook)
ace-learn patch      # Create patched CLI for token savings
ace-learn unpatch    # Remove patched CLI
```

## How It Works

1. **Async Hook**: When a Claude Code query completes, the Stop hook runs `ace-learn` asynchronously

2. **Learning**: ACE analyzes the session transcript and extracts reusable strategies via Claude CLI

3. **Skills**: Learned strategies are written to `.claude/skills/ace-learnings/SKILL.md` in your project

## Hook Configuration

The async hook is configured in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "ace-learn",
        "async": true,
        "timeout": 300
      }]
    }]
  }
}
```

## Project Root Detection

ACE determines where to store learned skills based on project markers:

| Priority | Marker | Description |
|----------|--------|-------------|
| 1 | `.ace-root` | Explicit ACE root (for monorepos) |
| 2 | `.git` | Git repository |
| 3 | `pyproject.toml` | Python project |
| 4 | `package.json` | Node.js project |
| 5 | `Cargo.toml` | Rust project |
| 6 | `go.mod` | Go project |

**Monorepo Setup**: Create a `.ace-root` file at your monorepo root to ensure ACE uses that location instead of nested package directories.

**Global Fallback**: Sessions outside any project store skills in `~/.claude/skills/ace-learnings-global/`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ACE_PROJECT_DIR` | Override project root detection |
| `ACE_CLAUDE_CLI_JS` | Path to patched CLI JS file |
| `ACE_CLAUDE_BIN` | Path to claude binary |
| `ACE_CLI_PATH` | Alternative path to claude CLI |

## File Structure

```
ace/integrations/claude_code/
├── __init__.py          # Package exports
├── hook.py              # Main learner and CLI
├── cli_client.py        # Claude CLI wrapper
├── prompt_patcher.py    # Optional CLI patcher
└── README.md            # This file
```

## Troubleshooting

Run `ace-learn doctor` to diagnose issues. Common problems:

1. **Claude CLI not found**: Install Claude Code CLI
2. **Hook not configured**: Run `ace-learn setup`
3. **No skills generated**: Ensure sessions are non-trivial (3+ tool calls)

## Architecture

```
Claude Code Session
        │
        ▼
    Stop Hook (async)    ← Native async, non-blocking
        │
        ▼
    ace-learn            ← Runs in background
        │
        ▼
    Claude CLI           ← Subscription-only LLM calls
        │
        ▼
    .claude/skills/      ← Learned strategies
```
