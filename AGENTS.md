# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Repository Guidelines

### Pipeline-First Development (MANDATORY)
**All new functionality MUST be implemented as pipeline Steps composed via the Pipeline engine.** Do NOT write standalone scripts, ad-hoc loops, or inline logic that bypasses the pipeline. Before writing any code:

1. Read `docs/PIPELINE_DESIGN.md` to understand the Step -> Pipeline -> Branch model.
2. Implement logic as a `Step` class with `requires`/`provides` declarations and a `__call__(self, ctx) -> ctx` method.
3. Compose steps using `Pipeline().then(...)` and `.branch(...)` - never manual for-loops or direct function chaining.
4. Use `StepContext.replace()` for immutable context updates - never mutate context directly.
5. Put integration-specific data in `metadata`, not new context fields, unless the field is shared across multiple pipelines.

**Anti-patterns to reject:**
- Writing a function that calls multiple steps manually instead of composing them in a Pipeline
- Inline reflection/evaluation logic instead of creating a `ReflectStep` or `EvaluateStep`
- Ad-hoc `ThreadPoolExecutor` usage instead of `async_boundary` and `max_workers` on steps
- Standalone scripts that duplicate pipeline functionality without using the pipeline engine
- Bypassing `requires`/`provides` contracts by accessing context fields not declared in `requires`

If a task seems like it cannot fit the pipeline model, explain why to the user before proceeding - do not silently circumvent it.

### Core Code Protection
**Do NOT modify core modules (`ace/`, `ace/core/`, `pipeline/`) without explicit user approval.** Before proposing any change to these directories:
1. Read the relevant design docs (`docs/ACE_DESIGN.md`, `docs/PIPELINE_DESIGN.md`) thoroughly.
2. Evaluate whether the change is truly required or if it can be achieved outside the core (for example, in an integration, step, or example).
3. Clearly explain the proposed change and its justification to the user before making any edits.
4. Wait for the user to explicitly accept before proceeding.

### Documentation Maintenance
Before working on code in `ace/`, read `docs/ACE_DESIGN.md` to understand the current architecture.
Before working on code in `pipeline/`, read `docs/PIPELINE_DESIGN.md` to understand the pipeline engine.
Before working on code in `ace/rr/`, read `docs/RR_DESIGN.md` to understand the recursive reflection design.
Before working on code in `ace/cli/`, read `docs/CLI_DESIGN.md` to understand the CLI architecture.

**Docs MUST be kept in sync with code.** Any change that alters a public API, renames a concept, adds or removes a module, or changes execution flow requires a corresponding update to the relevant docs. Do not merge code changes that make the documentation inaccurate.

Key design docs:
- `docs/ACE_DESIGN.md` - core ACE architecture: roles, runners, skillbook, adaptation loops, integrations, and public API
- `docs/PIPELINE_DESIGN.md` - pipeline engine: steps, `StepProtocol`, `Pipeline`, branching, execution, and `SubRunner`
- `docs/RR_DESIGN.md` - recursive reflection design in `ace/rr/`
- `docs/CLI_DESIGN.md` - CLI architecture, lazy imports, and command design
- `docs/PYDANTIC_AI_MIGRATION.md` - migration context for the PydanticAI-based architecture
- If you need to work with collected traces from Logfire, read `agent-guides/logfire.md`

### Project Structure
- `ace/` - main package: core data types, role implementations, steps, runners, integrations, providers, recursive reflection, and observability
- `pipeline/` - generic pipeline engine used by ACE
- `ace-eval/` - evaluation framework submodule / separate repo workspace
- `tests/` - pytest-based test suite, including pipeline engine and RR coverage
- `examples/` - runnable demos for ACE, integrations, and pipeline composition
- `benchmarks/` - benchmark loaders and task definitions
- `scripts/` - helper scripts and research tooling
- `agent-guides/` - internal development guides for LLM agents; not part of the public docs site
- `docs/` - guides and reference material
  - `docs/getting-started/` - installation, setup, and quick start
  - `docs/concepts/` - core concepts such as roles, skillbook, updates, and insight levels
  - `docs/guides/` - in-depth guides for full pipelines, composition, prompts, integration, and testing
  - `docs/integrations/` - per-integration docs for LiteLLM, browser-use, LangChain, Claude Code, Claude SDK, MCP, OpenClaw, hosted API, and Opik
  - `docs/pipeline/` - pipeline engine guides and API reference
  - `docs/api/` - package API index
  - `docs/ACE_DESIGN.md`, `docs/PIPELINE_DESIGN.md`, `docs/RR_DESIGN.md`, `docs/CLI_DESIGN.md` - architecture references

### Commands
- `uv sync` - install dependencies
- `uv run pytest` - run tests with coverage on `ace` and `pipeline` (`--cov-fail-under=25`)
- `uv run pytest -m unit` - run unit tests
- `uv run pytest -m integration` - run integration tests
- `uv run pytest -m slow` - run slow tests
- `uv run pytest -m requires_api` - run tests that need live API credentials
- `uv run black ace/ pipeline/ tests/ examples/` - format code
- `uv run mypy ace/` - type check the main package

### Coding Style
- PEP 8 with Black formatting (line length 88)
- Type hints and docstrings for public APIs
- Python 3.12 target
- Test files: `tests/test_*.py`; functions: `test_*`; classes: `Test*`

### Testing
- Pytest is the primary runner
- Some tests use `unittest`-style classes but still run under pytest
- Use the existing markers: `unit`, `integration`, `slow`, and `requires_api`
- Add tests for new features and regression tests for bug fixes

### Commits
- Conventional Commits: `feat(scope): subject`, `fix(scope): subject`
- PRs should include description, test results, and relevant docs updates

### ACE Roles (quick reference)

| Role | Responsibility | Key Class |
|------|---------------|-----------|
| **Agent** | Executes tasks using skillbook strategies | `Agent` |
| **Reflector** | Analyzes execution results | `Reflector` |
| **SkillManager** | Updates the skillbook with new strategies | `SkillManager` |

### Public Runners

| Runner | Framework | Use Case |
|--------|-----------|----------|
| `ACELiteLLM` | LiteLLM | Batteries-included self-improving agent with `.ask()`, `.learn()`, and trace learning helpers |
| `ACE` | Core ACE runner | Full learning loop over `Sample` + `TaskEnvironment` |
| `TraceAnalyser` | Offline traces | Learn from recorded traces without re-running tasks |
| `BrowserUse` | browser-use | Browser automation with learning |
| `LangChain` | LangChain | Wrap chains, agents, or graphs with learning |
| `ClaudeCode` | Claude Code CLI | Coding tasks with learning |

The Anthropic SDK integration lives in `ace/integrations/claude_sdk.py` and is step-based rather than a public runner class.
