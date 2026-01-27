# ACE Skill Improvement Integration

Retrospectively improve workflow playbook skill files based on Claude Code session transcripts.

## What This Does

After completing a Claude Code session, you can run `/skill-retro` to analyze what happened and automatically improve your skill files with new learnings. The tool extracts insights from the session and applies minimal, high-signal edits to your playbooks.

## Architecture

This integration uses ACE's standard 3-LLM-call pattern:

```
┌─────────────────┐
│   Transcript    │  Claude Code session (JSONL)
│   (input)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reflector     │  LLM Call 1: Extract learnings
│                 │  → extracted_learnings, key_insight, skill_tags
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SkillManager   │  LLM Call 2: Convert to operations
│                 │  → ADD, UPDATE, TAG, REMOVE operations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Publisher     │  LLM Call 3: Apply to playbook
│                 │  → Minimal edits, quality filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Updated Skill  │  With backup + atomic write
│   (output)      │
└─────────────────┘
```

All LLM calls use your **Claude Code subscription** via `CLIClient` - no API keys required.

## Installation

Installed automatically with ACE:

```bash
cd agentic-context-engine
pip install -e .
```

Verify:
```bash
ace-skill-improve --help
```

## Usage

### Option 1: Slash Command (Recommended)

In Claude Code, type:

```
/skill-retro
```

Claude Code will guide you through:
1. Finding the transcript file
2. Running the improvement (dry-run first)
3. Reviewing the diff
4. Applying changes if approved

### Option 2: CLI Directly

```bash
# Step 1: Find your transcript
ls -lt ~/.claude/projects/*/*.jsonl | head -5

# Step 2: Run dry-run (safe, shows diff only)
ace-skill-improve run \
  --skill /path/to/SKILL.md \
  --transcript ~/.claude/projects/xxx/session.jsonl

# Step 3: If changes look good, apply them
ace-skill-improve run \
  --skill /path/to/SKILL.md \
  --transcript ~/.claude/projects/xxx/session.jsonl \
  --apply
```

### Option 3: Python API

```python
from ace.integrations.claude_code_skill_improvement import (
    SkillImprover,
    TranscriptParser,
    render_full_chat,
)

# View a transcript
parser = TranscriptParser()
transcript = parser.parse("~/.claude/projects/.../session.jsonl")
print(render_full_chat(transcript))

# Run improvement
improver = SkillImprover()
result = improver.improve(
    skill_path="./SKILL.md",
    transcript_path="~/.claude/projects/.../session.jsonl",
    dry_run=True,
)

print(result.diff)
if result.has_changes:
    print(f"Accepted: {len(result.publish_output.accepted)}")
    print(f"Rejected: {len(result.publish_output.rejected)}")
```

## CLI Reference

### `ace-skill-improve run`

Run the improvement process on a transcript.

```
ace-skill-improve run --skill <path> --transcript <path> [--dry-run|--apply] [-v]

Options:
  --skill, -s       Path to the skill file to improve (required)
  --transcript, -t  Path to the transcript JSONL file (required)
  --dry-run         Preview changes without writing (default)
  --apply           Apply changes (creates backup first)
  --verbose, -v     Enable verbose logging
```

**Output sections:**
- **DIFF**: Unified diff showing exact changes
- **ACCEPTED CHANGES**: Learnings that will be added, with reasons
- **REJECTED LEARNINGS**: Learnings filtered out, with reasons

### `ace-skill-improve setup`

Create the `/skill-retro` slash command.

```
ace-skill-improve setup --skill <path>

Options:
  --skill, -s       Path to the skill file (embedded in slash command)
```

Creates `~/.claude/commands/skill-retro.md` with instructions for Claude Code.

### `ace-skill-improve show`

View a parsed transcript.

```
ace-skill-improve show --transcript <path> [--summary]

Options:
  --transcript, -t  Path to the transcript JSONL file (required)
  --summary         Show JSON summary only (turns, tool calls, success rate)
```

## Quality Constraints

The Publisher step enforces quality to prevent skill file bloat:

| Constraint | Default | Purpose |
|------------|---------|---------|
| Max lines added | 20 | Prevent large changes |
| Max diff percent | 30% | Reject rewrites |
| Frontmatter preservation | Required | Keep YAML valid |
| Deduplication | Semantic | No redundant strategies |
| Generality filter | On | Reject session-specific quirks |

If changes exceed thresholds, they're rejected with an explanation.

## File Locations

| File | Location |
|------|----------|
| Transcripts | `~/.claude/projects/<project-hash>/<session-id>.jsonl` |
| Project skills | `<project>/.claude/skills/ace-learnings/SKILL.md` |
| Global skills | `~/.claude/skills/ace-learnings-global/SKILL.md` |
| Slash commands | `~/.claude/commands/skill-retro.md` |
| Backups | `<skill-dir>/SKILL.<timestamp>.backup.md` |

## How It Differs from ace-learn

| Feature | ace-learn | ace-skill-improve |
|---------|-----------|-------------------|
| Trigger | Automatic (hook/daemon) | Manual (slash command/CLI) |
| Output | skillbook.json → generated SKILL.md | Direct edits to existing SKILL.md |
| Target | ACE skillbook (structured) | Any workflow playbook |
| LLM Calls | 2 (Reflector + SkillManager) | 3 (+ Publisher) |
| Use Case | Learn general coding patterns | Improve specific workflow playbooks |

## Module Structure

```
ace/integrations/claude_code_skill_improvement/
├── __init__.py      # Public API exports
├── cli.py           # CLI entrypoint + slash command creation
├── improver.py      # SkillImprover orchestrator (3 LLM calls)
├── models.py        # Pydantic schemas (PublishOutput, etc.)
├── transcript.py    # TranscriptParser + render_full_chat
└── README.md        # This file
```

## Exported API

```python
from ace.integrations.claude_code_skill_improvement import (
    # Main class
    SkillImprover,

    # Models
    PublishOutput,
    AcceptedChange,
    RejectedLearning,
    SkillImprovementResult,

    # Transcript utilities
    TranscriptParser,
    ParsedTranscript,
    Turn,
    ToolCall,
    render_full_chat,
    get_transcript_summary,
    TranscriptTooLargeError,

    # Setup
    create_slash_commands,
)
```

## Troubleshooting

### "Transcript too large"

The transcript exceeds size limits (default 10MB). This is a v1 limitation - the tool fails fast on oversized transcripts rather than summarizing.

**Workaround:** Use a shorter session or wait for v2 with digest support.

### "Diff exceeds threshold"

The Publisher rejected all changes because they would modify >30% of the file.

**This is intentional** - it prevents the tool from rewriting your skill file. Review the transcript manually for specific insights.

### "No changes to make"

The session didn't produce learnings that pass the quality filters. This can happen if:
- The session was too short
- Learnings were too session-specific
- Learnings duplicated existing strategies

### CLI not found

```bash
# Reinstall
cd agentic-context-engine
pip install -e .

# Verify
which ace-skill-improve
```

### Slash command not working

```bash
# Recreate it
ace-skill-improve setup --skill /path/to/your/SKILL.md

# Or via ace-learn setup (creates all ACE slash commands)
ace-learn setup
```

## Example Session

```bash
$ ace-skill-improve run \
    --skill ~/.claude/skills/my-workflow/SKILL.md \
    --transcript ~/.claude/projects/abc123/session.jsonl

============================================================
DIFF
============================================================
--- a/SKILL.md
+++ b/SKILL.md
@@ -15,6 +15,8 @@
 ## Strategies

 - Always read files before editing
+- Run tests after making changes to verify correctness
+- Use atomic commits with descriptive messages

============================================================
ACCEPTED CHANGES
============================================================

  + Run tests after making changes
    Reason: Session showed test failures caught bugs early
    Evidence: Steps 12-15 showed test-fix-test cycle

  + Use atomic commits
    Reason: Generalizable best practice observed
    Evidence: User requested commit after each logical change

============================================================
REJECTED LEARNINGS
============================================================

  - Check for Python 3.11 compatibility
    Reason: Session-specific (project uses 3.11)

============================================================
DRY RUN - No changes written
To apply changes: ace-skill-improve run --skill ... --transcript ... --apply
```

## Requirements

- Python 3.11+
- Claude Code CLI installed and authenticated
- ACE framework (`pip install -e .`)

No API keys required - uses Claude Code subscription.
