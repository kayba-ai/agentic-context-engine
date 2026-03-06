<!--
  Sync Impact Report
  ==================
  Version change: 1.0.0 → 1.1.0
  Principles added:
    - IV. Clean & Modular Code
  Principles unchanged:
    - I. Ease of Use First
    - II. Practical Value
    - III. Simplicity
  Sections unchanged:
    - Development Standards
    - Quality Gates
    - Governance
  Removed sections: None
  Templates validated:
    ✅ .specify/templates/plan-template.md — Constitution Check section compatible
    ✅ .specify/templates/spec-template.md — No conflicts
    ✅ .specify/templates/tasks-template.md — No conflicts
    ✅ .specify/templates/checklist-template.md — No conflicts
  Follow-up TODOs: None
-->

# ACE Framework Constitution

## Core Principles

### I. Ease of Use First

Every public API, integration, and workflow MUST prioritize developer
experience above all else.

- New users MUST be able to install and run a working example in under
  5 minutes with no more than 3 lines of code.
- Sensible defaults MUST be provided for every configuration option.
  Users MUST NOT be required to understand internals to get started.
- Breaking changes to the public API MUST follow deprecation warnings
  for at least one minor release before removal.
- Documentation MUST include a copy-pasteable quick start for every
  integration (LiteLLM, LangChain, browser-use, Claude Code).

### II. Practical Value

Every feature MUST solve a real, demonstrable problem for users
building AI agents.

- Features MUST NOT be added speculatively. Each addition MUST have a
  concrete use case tied to agent improvement or developer workflow.
- Performance claims MUST be backed by reproducible benchmarks or
  examples. No unsubstantiated marketing language in docs or code.
- Integration wrappers MUST add measurable value (learning, skillbook
  evolution) beyond what the wrapped framework already provides.

### III. Simplicity

Prefer the simplest solution that works. Complexity MUST be justified.

- YAGNI: Do not build for hypothetical future requirements. Three
  similar lines of code are better than a premature abstraction.
- New abstractions MUST be used in at least two places before
  extraction into a shared utility.
- Dependencies MUST be kept minimal. Optional extras (observability,
  LangChain, transformers) stay optional — the core install MUST
  remain lightweight.

### IV. Clean & Modular Code

All code MUST be clean, modular, and extensible.

- Modules MUST have a single, clear responsibility. Each file MUST
  do one thing well and expose a well-defined interface.
- Public APIs MUST be designed for extension without modification.
  New integrations, LLM providers, and adapters MUST be addable
  without changing existing code (open/closed principle).
- Internal boundaries MUST be respected: core library (`ace/`),
  integrations (`ace/integrations/`), LLM providers
  (`ace/llm_providers/`), and observability (`ace/observability/`)
  MUST NOT have circular dependencies.
- Functions and classes MUST be small enough to understand at a
  glance. If a function requires scrolling, it MUST be decomposed.

## Development Standards

- **Language**: Python 3.12 with type hints on all public APIs.
- **Formatting**: Black (line length 88). All code MUST pass
  `black --check` before merge.
- **Testing**: pytest with coverage enforcement (`--cov-fail-under=25`).
  New features MUST include tests. Bug fixes MUST include regression
  tests.
- **Distribution**: PyPI package `ace-framework`. Core install MUST NOT
  exceed ~150MB. Heavy dependencies belong in optional extras.
- **Commit style**: Conventional Commits (`feat(scope): subject`).

## Quality Gates

- All PRs MUST pass CI (formatting, type checks, test suite) before
  merge.
- Public API changes MUST update relevant documentation (README,
  docstrings, quick start guides).
- Benchmark results MUST NOT regress without explicit justification in
  the PR description.
- Skillbook format changes MUST maintain backward compatibility with
  existing saved skillbooks or provide a migration path.

## Governance

This constitution is the highest-authority document for the ACE
Framework project. All design decisions, PRs, and code reviews MUST
verify compliance with these principles.

- **Amendments**: Any change to this constitution MUST be documented
  with a version bump, rationale, and updated `LAST_AMENDED_DATE`.
- **Versioning**: MAJOR for principle removals or redefinitions, MINOR
  for new principles or material expansions, PATCH for clarifications.
- **Compliance**: Use `CLAUDE.md` for runtime development guidance.
  This constitution defines the non-negotiable rules that `CLAUDE.md`
  guidance MUST NOT contradict.
- **Review**: Constitution compliance SHOULD be checked at the start
  of each feature planning cycle (`/speckit.plan` Constitution Check).

**Version**: 1.1.0 | **Ratified**: 2026-02-25 | **Last Amended**: 2026-02-25
