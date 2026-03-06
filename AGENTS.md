# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Guidelines

### Core Code Protection
**Do NOT modify core modules (`ace/`, `ace_next/core/`, `pipeline/`) without explicit user approval.** Before proposing any change to these directories:
1. Read the relevant design docs (`docs/ACE_DESIGN.md`, `docs/PIPELINE_DESIGN.md`) thoroughly.
2. Evaluate whether the change is truly required or if it can be achieved outside the core (e.g., in an integration, step, or example).
3. Clearly explain the proposed change and its justification to the user **before** making any edits.
4. Wait for the user to explicitly accept before proceeding.

### Documentation Maintenance
Before working on code in `ace/` or `ace_next/`, read `docs/ACE_DESIGN.md` to understand the current architecture.
Before working on code in `pipeline/` or `ace_next/core/`, read `docs/PIPELINE_DESIGN.md` to understand the pipeline engine.

**Docs MUST be kept in sync with code.** Any change that alters a public API, renames a concept, adds/removes a module, or changes execution flow **requires** a corresponding update to the relevant docs. Do not merge code changes that make the documentation inaccurate.

Key design docs:
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
  - `docs/getting-started/` — installation, setup, quick-start
  - `docs/concepts/` — core concepts: roles, skillbook, insight levels
  - `docs/guides/` — in-depth guides: full pipeline, integration, testing, prompts
  - `docs/integrations/` — per-integration docs (LiteLLM, LangChain, browser-use, Claude Code, Opik)
  - `docs/pipeline/` — pipeline engine docs: core concepts, custom steps, branching, error handling
  - `docs/api/` — API reference
  - `docs/ACE_DESIGN.md` — architecture design doc (keep in sync with code)
  - `docs/PIPELINE_DESIGN.md` — pipeline engine design doc (keep in sync with code)

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