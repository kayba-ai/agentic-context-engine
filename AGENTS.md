# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Guidelines

### Design Document Maintenance
Before working on code in `ace/` or `ace_next/`, read `docs/ACE_DESIGN.md` to understand the current architecture.
Before working on code in `pipeline/` or `ace_next/core/`, read `docs/PIPELINE_DESIGN.md` to understand the pipeline engine.

When changes to code alter architecture, add new modules, change public APIs, rename concepts, or modify execution flow described in these documents, update the respective design doc to reflect the changes:
- `docs/ACE_DESIGN.md` — core ACE architecture: roles, skillbook, adaptation loops, insight levels, integration patterns
- `docs/PIPELINE_DESIGN.md` — pipeline engine: steps, StepProtocol, Pipeline, SubRunner, RR pipeline

### Project Structure
- `ace/` — core library (skillbook, roles, adapters, integrations, observability)
- `ace_next/` — pipeline-based rewrite built on top of `pipeline/` (see `docs/ACE_DESIGN.md`)
- `pipeline/` — generic pipeline engine that `ace_next` is built on (see `docs/PIPELINE_DESIGN.md`)
- `tests/` — unit/integration tests (pytest + unittest)
- `examples/` — runnable demos grouped by integration
- `benchmarks/`, `scripts/` — research/evaluation tooling (not shipped to PyPI)
- `docs/` — guides and reference material

### Commands
- `uv sync` — install all dependencies
- `uv run pytest` — run tests (coverage enforced `--cov-fail-under=25`)
- `uv run pytest -m unit` / `-m integration` / `-m slow` — run by marker
- `uv run black ace/ tests/ examples/` — format code
- `uv run mypy ace/` — type check

### Coding Style
- PEP 8 with Black formatting (line length 88)
- Type hints and docstrings for public APIs
- Python 3.12 target
- Test files: `tests/test_*.py`; functions: `test_*`; classes: `Test*`

### Testing
- Pytest is the primary runner
- Add tests for new features; include regression tests for bug fixes

### Commits
- Conventional Commits: `feat(scope): subject`, `fix(scope): subject`
- PRs should include description, test results, and relevant docs updates

### ACE Roles (quick reference)

| Role | Responsibility | Key Class |
|------|---------------|-----------|
| **Agent** | Executes tasks using skillbook strategies | `Agent` |
| **Reflector** | Analyzes execution results | `Reflector` |
| **SkillManager** | Updates the skillbook with new strategies | `SkillManager` |

### Integration Agents

| Agent | Framework | Use Case |
|-------|-----------|----------|
| `ACELiteLLM` | LiteLLM (100+ providers) | Simple self-improving agent |
| `ACELangChain` | LangChain | Wrap chains/agents with learning |
| `ACEAgent` | browser-use | Browser automation with learning |
| `ACEClaudeCode` | Claude Code CLI | Coding tasks with learning |

## ACE Learned Strategies

<!-- ACE:START - Do not edit manually -->
skills[6	]{id	section	content	helpful	harmful	neutral}:
  claude_code_transcripts-00001	claude_code_transcripts	Filter 'progress' and 'queue-operation' entry types from transcripts	1	0	0
  cli_debugging-00002	cli_debugging	Log subprocess stdout/stderr before retrying failed CLI commands	2	0	0
  cli_input_limits-00003	cli_input_limits	Use --lines flag to limit transcript size for CLI prompt limits	2	0	0
  transcript_compression-00004	transcript_compression	"Return minimal entries with only {type, content} fields, discarding all metadata"	2	0	0
  transcript_compression-00005	transcript_compression	"Use head+tail truncation for tool results: 500 chars start, 200 chars end"	1	0	0
  transcript_compression-00006	transcript_compression	Filter 'thinking' blocks from nested content arrays, not just entry types	1	0	0
<!-- ACE:END -->
