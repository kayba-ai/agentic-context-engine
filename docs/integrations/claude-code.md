# Claude Code Integration

The `ClaudeCode` runner wraps the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) with ACE learning. The agent runs coding tasks in your project directory and learns strategies from each execution — improving code generation, debugging, and project-specific patterns over time.

## Quick Start

```python
from ace_next import ClaudeCode

runner = ClaudeCode.from_model(working_dir="./my_project")

results = runner.run("Add unit tests for utils.py")
runner.save("coding_expert.json")
```

## Installation

```bash
pip install ace-framework[claude-code]
```

## Prerequisites

- Claude Code CLI installed and authenticated
- A project directory with source code

## Parameters

### from_model()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | `str` | `None` | Path to the project directory |
| `ace_model` | `str` | `"gpt-4o-mini"` | Model for Reflector + SkillManager |
| `ace_max_tokens` | `int` | `2048` | Max tokens for ACE LLM |
| `ace_llm` | `LLMClientLike` | `None` | Pre-built LLM for ACE roles |

### from_roles()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reflector` | `ReflectorLike` | — | Reflector instance |
| `skill_manager` | `SkillManagerLike` | — | SkillManager instance |
| `working_dir` | `str` | `None` | Project directory |
| `timeout` | `int` | `600` | Execution timeout (seconds) |
| `model` | `str` | `None` | Claude model override |
| `allowed_tools` | `list[str]` | `None` | Allowed Claude Code tools |
| `skillbook_path` | `str` | `None` | Load saved skillbook |
| `dedup_config` | `DeduplicationConfig` | `None` | Deduplication config |
| `checkpoint_dir` | `str` | `None` | Checkpoint directory |

## Methods

```python
results = runner.run(tasks, epochs=1)       # Run with learning
runner.save("path.json")                    # Save skillbook
runner.wait_for_background()                # Wait for async learning
runner.get_strategies()                     # View learned strategies
```

## How It Works

1. **INJECT** — Skillbook strategies are written to `CLAUDE.md` at the project root
2. **EXECUTE** — Claude Code CLI runs the task in the project directory
3. **Extract trace** — ACE reads the Claude Code execution transcript
4. **LEARN** — Reflector analyzes the trace, SkillManager updates the skillbook

## Pipeline Skill

The **kayba-pipeline** skill gives Claude Code a 7-stage evaluation and improvement pipeline that can be triggered directly from the chat. Install it with:

```bash
kayba setup
```

This copies the skill into `.claude/skills/kayba-pipeline/`. The pipeline stages are:

1. **Analyze traces** — extract patterns from agent execution transcripts
2. **Compute metrics** — score traces against quality dimensions
3. **Build rubric** — generate a structured evaluation rubric
4. **Plan fixes** — propose concrete improvements
5. **HITL review** — optional human-in-the-loop approval gate
6. **Apply fixes** — execute the approved changes
7. **Verify** — confirm fixes pass the rubric

Trigger the pipeline by saying **"run the pipeline"** or **"kayba pipeline"** in Claude Code with a traces folder in your project.

To skip skill installation: `kayba setup --no-skills`. See [Hosted API](hosted-api.md#agent-setup) for all `kayba setup` options.

## How It Works (continued)

The agent learns project-specific patterns like:

- Code style and conventions
- Common debugging approaches
- Test patterns and frameworks used
- Module structure and dependencies

## Running Multiple Tasks

```python
results = runner.run([
    "Add unit tests for utils.py",
    "Fix the bug in the login handler",
    "Refactor the database module to use connection pooling",
])
```

## Resuming from a Saved Skillbook

```python
runner = ClaudeCode.from_model(
    working_dir="./my_project",
    skillbook_path="coding_expert.json",
)
```

## What to Read Next

- [Integration Pattern](../guides/integration.md) — how the INJECT/EXECUTE/LEARN pattern works
- [The Skillbook](../concepts/skillbook.md) — how learned strategies are stored
