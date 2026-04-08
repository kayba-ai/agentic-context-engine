# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Guidelines

### Pipeline-First Development (MANDATORY)
**All new functionality MUST be implemented as pipeline Steps composed via the Pipeline engine.** Do NOT write standalone scripts, ad-hoc loops, or inline logic that bypasses the pipeline. Before writing any code:

1. Read `docs/design/PIPELINE_DESIGN.md` to understand the Step → Pipeline → Branch model.
2. Implement logic as a `Step` class with `requires`/`provides` declarations and a `__call__(self, ctx) -> ctx` method.
3. Compose steps using `Pipeline().then(...)` and `.branch(...)` — never manual for-loops or direct function chaining.
4. Use `StepContext.replace()` for immutable context updates — never mutate context directly.
5. Put integration-specific data in `metadata`, not new context fields, unless the field is shared across multiple pipelines.

**Anti-patterns to reject:**
- Writing a function that calls multiple steps manually instead of composing them in a Pipeline
- Inline reflection/evaluation logic instead of creating a ReflectStep or EvaluateStep
- Ad-hoc `ThreadPoolExecutor` usage instead of `async_boundary` and `max_workers` on steps
- Standalone scripts that duplicate pipeline functionality without using the pipeline engine
- Bypassing `requires`/`provides` contracts by accessing context fields not declared in `requires`

If a task seems like it cannot fit the pipeline model, explain why to the user before proceeding — do not silently circumvent it.

### Core Code Protection
**Do NOT modify core modules (`ace/core/`, `pipeline/`) without explicit user approval.** Before proposing any change to these directories:
1. Read the relevant design docs (`docs/design/ACE_ARCHITECTURE.md`, `docs/design/PIPELINE_DESIGN.md`) thoroughly.
2. Evaluate whether the change is truly required or if it can be achieved outside the core (e.g., in an integration, step, or example).
3. Clearly explain the proposed change and its justification to the user **before** making any edits.
4. Wait for the user to explicitly accept before proceeding.

### Documentation Maintenance
Before working on code in `ace/`, read `docs/design/ACE_ARCHITECTURE.md` to understand the current architecture.
Before working on code in `pipeline/` or `ace/core/`, read `docs/design/PIPELINE_DESIGN.md` to understand the pipeline engine.

**Docs MUST be kept in sync with code.** Any change that alters a public API, renames a concept, adds/removes a module, or changes execution flow **requires** a corresponding update to the relevant docs. Do not merge code changes that make the documentation inaccurate.

Key design docs:
- `docs/design/ACE_ARCHITECTURE.md` — ACE architecture: layers, core concepts, roles, steps, runners, integrations
- `docs/design/ACE_REFERENCE.md` — ACE code reference: full implementations, API signatures, usage examples
- `docs/design/ACE_DECISIONS.md` — design decisions and rejected alternatives (ACE, pipeline, migration)
- `docs/design/PIPELINE_DESIGN.md` — pipeline engine: steps, StepProtocol, Pipeline, Branch, concurrency
- If you need to work with collected traces from Logfire, read `agent-guides/logfire.md`

### Project Structure
- `ace/` — core library: roles (PydanticAI-backed), skillbook, steps, runners, providers, RR, integrations, observability
- `pipeline/` — generic pipeline engine that `ace` is built on (see `docs/design/PIPELINE_DESIGN.md`)
- `ace-eval/` — evaluation framework (submodule, separate repo)
- `tests/` — unit/integration tests (pytest)
- `examples/` — runnable demos grouped by integration
- `agent-guides/` — internal development guides for LLM agents; not part of the public docs site
- `docs/` — guides and reference material
  - `docs/design/ACE_ARCHITECTURE.md` — architecture and concepts (keep in sync with code)
  - `docs/design/ACE_REFERENCE.md` — code reference and examples (keep in sync with code)
  - `docs/design/ACE_DECISIONS.md` — design decisions and rejected alternatives
  - `docs/design/PIPELINE_DESIGN.md` — pipeline engine design doc (keep in sync with code)

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
- Do NOT add `Co-Authored-By` trailers to commit messages
- PRs should include description, test results, and relevant docs updates

### ACE Roles (quick reference)

| Role | Responsibility | Key Class |
|------|---------------|-----------|
| **Agent** | Executes tasks using skillbook strategies | `Agent` |
| **Reflector** | Analyzes execution results | `Reflector` |
| **SkillManager** | Updates the skillbook with new strategies | `SkillManager` |

### Integration Runners

| Runner | Framework | Use Case |
|--------|-----------|----------|
| `ACELiteLLM` | LiteLLM (100+ providers) | Simple self-improving agent |
| `ACELangChain` | LangChain | Wrap chains/agents with learning |
| `ACEBrowserUse` | browser-use | Browser automation with learning |
| `ACEClaudeCode` | Claude Code CLI | Coding tasks with learning |

NEVER USE FALLBACKS OR IMPLEMENT THINGS I NEVER ASKED FOR. 
IF IT'S STRAIGHFORWARD, IMPLEMENT IT STRAIGHFORWARD.