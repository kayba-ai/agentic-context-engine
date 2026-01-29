# ACE Claude Code Integration

Subscription-only learning from Claude Code sessions. No API keys required.

## Overview

This package enables ACE (Agentic Context Engineering) to learn from your Claude Code sessions. Sessions are captured automatically, and you trigger learning manually with `/ace-learn` when you want to extract insights.

**Key Features:**
- Instant capture hook (~10ms) - never slows down Claude Code
- Manual learning via `/ace-learn` - learn when you're ready
- Incremental learning - only processes new turns by default
- Custom Reflector prompt optimized for coding sessions
- Subscription-only via Claude CLI (no API keys needed)
- Per-project skill files in `.claude/skills/ace-learnings/`

## Quick Start

```bash
pip install ace-framework
ace-learn setup
ace-learn doctor
```

Then in Claude Code, type `/ace-learn` after a session to learn from it.

## How It Works

```
Session ends → Stop hook captures session pointer (instant)
                        │
                        ▼
              .ace-last-hook.json saved
                        │
User types /ace-learn ──┘
                        │
                        ▼
              ace-learn learn-last
                        │
                        ▼
              Reflector + SkillManager
                        │
                        ▼
              SKILL.md updated
```

1. **Capture** (automatic): When a session ends, the Stop hook saves a pointer to the transcript
2. **Learn** (manual): Type `/ace-learn` to analyze the session and extract strategies
3. **Apply**: Learned strategies are automatically loaded in future sessions

## Commands

### Inside Claude Code (Slash Commands)

| Command | Description |
|---------|-------------|
| `/ace-learn` | Learn from this session (incremental - only new turns) |
| `/ace-insights` | Show learned strategies from the skillbook |
| `/ace-clear` | Clear all learned strategies |
| `/ace-remove` | Remove a specific learned strategy |
| `/ace-on` | Enable capture hook |
| `/ace-off` | Disable capture hook |

**Tip:** For full session learning (e.g., after giving feedback), run `ace-learn learn-last --full` in terminal.

### Terminal Commands

```bash
ace-learn setup              # Configure capture hook (run once)
ace-learn learn-last         # Learn incrementally (only new turns)
ace-learn learn-last --full  # Learn from entire session
ace-learn doctor             # Verify prerequisites
ace-learn insights           # Show learned strategies
ace-learn clear --confirm    # Clear all insights
ace-learn capture            # (internal) Called by Stop hook
```

## Incremental Learning

By default, `/ace-learn` only processes turns since the last time you ran it:

```
First /ace-learn  → processes turns 1-50, saves state
... more conversation ...
Second /ace-learn → processes turns 49-75 (with 2-turn lookback for context)
```

Use `--full` to reprocess the entire session:
- After giving feedback you want ACE to learn from
- To re-analyze with fresh eyes
- When incremental says "No new turns"

## Hook Configuration

The capture hook in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "ace-learn capture",
        "async": true,
        "timeout": 10
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

**Monorepo Setup**: Create a `.ace-root` file at your monorepo root.

**Global Fallback**: Sessions outside any project use `~/.claude/skills/ace-learnings-global/`

## File Structure

```
ace/integrations/claude_code/
├── __init__.py          # Package exports
├── hook.py              # Main learner and CLI
├── cli_client.py        # Claude CLI wrapper
├── prompts.py           # Custom Reflector prompt for coding sessions
├── prompt_patcher.py    # Optional CLI patcher for token savings
└── README.md            # This file

<project>/.claude/skills/ace-learnings/
├── SKILL.md                  # Auto-loaded by Claude Code
├── skillbook.json            # Persistent skill store
├── .ace-last-hook.json       # Last captured session pointer
├── .ace-learning-state.json  # Incremental learning progress
└── categories/               # Detailed strategies by category
```

## Troubleshooting

Run `ace-learn doctor` to diagnose issues:

```
1. Claude CLI...        ✓ Found
2. CLI Resolution...    ✓ Patched CLI (token savings)
3. Hook configuration...✓ Capture mode enabled
4. Skill output...      ✓ Project detected
```

Common problems:
- **No captured session**: Use Claude Code first, then run `/ace-learn`
- **No new turns**: Use `ace-learn learn-last --full` to reprocess
- **CLI timeout**: Large sessions (100+ tool calls) may take longer
