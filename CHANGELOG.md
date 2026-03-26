# Changelog

All notable changes to ACE Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-03-26

### Added
- **PydanticAI migration** — ACE roles (Agent, Reflector, SkillManager) rebuilt on PydanticAI agents with structured output, replacing the legacy role system
- **Kayba CLI** — full hosted API client with trace upload/management, interactive run, insights, prompts, batch processing, materialization, and integration commands (`kayba` entry point)
- **XML skill rendering** — per-task retrieval for benchmark evaluations
- **Bedrock API key support** — AWS Bedrock provider authentication

## [0.8.8] - 2026-03-17

### Added
- **Pipeline hooks & cancellation** — `PipelineHook` protocol and `CancellationToken` for observing and controlling pipeline execution
- **Kayba pipeline skills for Claude Code** — 7-stage dynamic evaluation pipeline that generates custom benchmarks tailored to your agent's domain. Instead of static test suites, the skills analyze your API, build domain-aware metrics and rubrics, create action plans, and run human-in-the-loop validation — all as composable Claude Code skills
- **`kayba setup` command** — one command to install the full evaluation skill pipeline into your `.claude/skills/` directory, ready to use inside Claude Code out of the box

### Docs
- Documented `kayba setup` skills installation

### Try it free
**7-day free trial** — Try the full Kayba evaluation pipeline on our hosted solution with zero setup. Sign up at [kayba.ai](https://kayba.ai) and run `kayba setup` to start building dynamic evals for your agents today.

## [0.8.7] - 2026-03-17

### Added
- **Improved Opik trace naming** — traces now display the question text (first 80 chars) instead of generic names like "ace_pipeline" or "rr_reflect"
- **Thread ID support for Opik** — `OpikStep` and `RROpikStep` accept an optional `thread_id` parameter for grouping related traces

## [0.8.5] - 2026-03-04

### Added
- **Self-contained RR module** (`ace/rr/`) — sandbox, subagent, trace_context, config, code_extraction, message_trimming extracted from `ace/reflector/` into a standalone package
- **v5.6 prompt promoted as default** — new prompt evolution (v4 → v5.1–v5.6) for the `ace` RR pipeline
- **`build_steps()` API** — all runners gain a `build_steps()` classmethod for pipeline customization
- **Shared `CallBudget`** — single budget instance shared across RR pipeline steps
- **ACE MCP server (optional)** — stdio MCP server in `ace.integrations.mcp` with tools: `ace.ask`, `ace.learn.sample`, `ace.learn.feedback`, `ace.skillbook.get`, `ace.skillbook.save`, `ace.skillbook.load`
- **Session-scoped state management** — in-memory `session_id` registry with TTL cleanup and per-session async locking
- **MCP packaging + CLI** — optional `mcp` extra and `ace-mcp` entrypoint
- **MCP docs and demo client** — integration guide and stdio client example
- **Composing pipelines guide** — new `docs/guides/composing-pipelines.md`
- **RR examples** — `rr_demo.py`, `rr_opik_demo.py`, `compose_custom_pipeline.py`

### Changed
- **RR backward-compat shims** — original `ace/reflector/` files now re-export from `ace.rr` (no duplication)
- **`RRStep` dual protocol** — implements both `StepProtocol` and `ReflectorLike`
- **Sandbox hardening** — hardened `getattr` in sandbox execution environment
- **Opik made opt-in** — moved `opik` from hard dependency to `observability` extra
- **Safety controls** — runtime request limits (`max_prompt_chars`, `max_samples_per_call`) and optional root-bound path enforcement for save/load via `ACE_MCP_SKILLBOOK_ROOT`
- **Schema-driven validation** — MCP request/response models aligned to `specs/002-ace-mcp-server/contracts/tool-schemas.md`
- **`learn_from_feedback` routed through pipeline** — feedback learning now uses the pipeline engine

### Testing
- Added MCP test suite: models, registry, handlers, and server registration/startup smoke tests
- Added optional-dependency boundary checks for the MCP integration
- RR steps at 94%, sandbox at 92%, runner at 74%, MCP models at 100%

## [0.8.4] - 2026-02-27

### Added
- **OpenClaw integration** — learn from OpenClaw session transcripts (JSONL) via new `OpenClawToTraceStep` and `LoadTracesStep` pipeline steps (#86)
- **ExportSkillbookMarkdownStep** — export skillbook to markdown file
- OpenClaw example script and integration docs

## [0.8.3] - 2026-02-21

### Added
- **Pipeline engine** — generic pipeline framework with branching, async boundaries, and parallel execution (#78)
- **Trace passthrough** — `_build_traces()` helper and raw trace data passed to RecursiveReflector sandbox

## [0.8.2] - 2026-02-18

### Added
- **RecursiveReflector None-response guard** — gracefully handles empty/None LLM responses (e.g. from Gemini) with retry prompt instead of crashing
- **`LiteLLMClient.complete_messages()`** — native multi-turn completion that preserves structured message lists

## [0.8.1] - 2026-02-18

### Added
- **Insight source tracing** — `InsightSource` dataclass tracks skill provenance (epoch, sample, trace refs, error identification, learning text)
- **Sample.id** promoted to first-class field with UUID auto-generation
- **Skillbook query API** — `source_map()`, `source_summary()`, `source_filter()` for skill lineage
- Insight sources wired through `OfflineACE`, `OnlineACE`, and async learning pipelines
- `UpdateOperation.learning_index` for linking operations to reflector learnings
- Bedrock e2e example (`examples/litellm/bedrock_insight_source_test.py`)
- `docs/INSIGHT_SOURCES.md` guide

## [0.8.0] - 2026-02-17

### Added
- **Recursive reflector** with sandboxed code execution for validation
- **TAU-bench integration** with config-driven YAML profiles, prompt sweep, capture/replay, and label support
- **v3 prompt templates** for agent, reflector, and skill manager roles
- **Trace context module** exposing agent system prompt and execution context to reflector

### Fixed
- Opik cloud mode support when `OPIK_API_KEY` is set
- Bedrock/SageMaker API key lookup skipped for managed providers
- Reflector trace quality improvements (user messages, turn separators)

### Changed
- v3 prompts set as default prompt version
- Reflector now includes agent system prompt in trace context

## [0.7.3] - 2026-02-04

### Added
- ACE learning for Claude Code via `/ace-learn` (transcript-based learning and skillbook updates).
- CLI patching to minimize Claude Code system prompt overhead for learning runs.

### Fixed
- Claude Code transcript parsing for feedback and last-prompt extraction edge cases.

### Changed
- Unified agent guidance into `AGENTS.md` with `CLAUDE.md` symlink.

## [0.7.0] - 2025-12-04

### ⚠️ Breaking Changes
- **Complete terminology rename** - Playbook → Skillbook, Bullet → Skill
  - `Playbook` → `Skillbook`
  - `Bullet` → `Skill`
  - `Generator` → `Agent`
  - `Curator` → `SkillManager`
  - `OfflineAdapter` → `OfflineACE`
  - `OnlineAdapter` → `OnlineACE`
  - `DeltaOperation` → `UpdateOperation`
  - `DeltaBatch` → `UpdateBatch`
  - **Migration**: Update imports and method calls to use new names
  - **JSON files**: Change `"bullets"` key to `"skills"` in saved skillbooks

### Added
- **Deduplication consolidation_operations field** - SkillManagerOutput now properly captures consolidation operations from LLM responses

### Fixed
- **Deduplication not working** - Added `consolidation_operations` field to SkillManagerOutput Pydantic model. Previously, Instructor was silently dropping these operations.

## [0.5.0] - 2025-11-20

### ⚠️ Breaking Changes
- **Playbook format changed to TOON (Token-Oriented Object Notation)**
  - `Playbook.as_prompt()` now returns TOON format instead of markdown
  - **Reason**: 16-62% token savings for improved scalability and reduced inference costs
  - **Migration**: No action needed if using playbook with Generator/Curator/Reflector
  - **Debugging**: Use `playbook._as_markdown_debug()` or `str(playbook)` for human-readable output
  - **Details**: Uses tab delimiters and excludes internal metadata (created_at, updated_at)

### Added
- **ACELiteLLM integration** - Simple conversational agent with automatic learning
- **ACELangChain integration** - Wrap LangChain Runnables with ACE learning
- **Custom integration pattern** - Wrap ANY agentic system with ACE learning
  - Base utilities in `ace/integrations/base.py` with `wrap_playbook_context()` helper
  - Complete working example in `examples/custom_integration_example.py`
  - Integration Pattern: Inject playbook → Execute agent → Learn from results
- **Integration exports** - Import ACEAgent, ACELiteLLM, ACELangChain from `ace` package root
- **TOON compression for playbooks** - 16-62% token reduction vs markdown
- **Citation-based tracking** - Strategies cited inline as `[section-00001]`, auto-extracted from reasoning
- **Enhanced browser traces** - Full execution logs (2200+ chars) passed to Reflector
- **Test coverage** - Improved from 28% to 70% (241 tests total)

### Changed
- **Renamed SimpleAgent → ACELiteLLM** - Clearer naming for conversational agent integration
- `Playbook.__str__()` returns markdown (TOON reserved for LLM consumption via `as_prompt()`)

### Fixed
- **Browser-use trace integration** - Reflector now receives complete execution traces
  - Fixed initial query duplication (task appeared in both question and reasoning)
  - Fixed missing trace data (reasoning field now contains 2200+ chars vs 154 chars)
  - Fixed screenshot attribute bug causing AttributeError on step.state.screenshot
  - Fixed invalid bullet ID filtering - hallucinated/malformed citations now filtered out
  - Added comprehensive regression tests to catch these issues
  - Impact: Reflector can now properly analyze browser agent's thought process
  - Test coverage improved: 69% → 79% for browser_use.py
- Prompt v2.1 test assertions updated to match current format
- All 206 tests now pass (was 189)

## [0.4.0] - 2025-10-26

### Added
- **Production Observability** with Opik integration
  - Enterprise-grade monitoring and tracing
  - Automatic token usage and cost tracking for all LLM calls
  - Real-time cost monitoring via Opik dashboard
  - Graceful degradation when Opik is not installed
- **Browser Automation Demos** showing ACE vs baseline performance
  - Domain checker demo with learning capabilities
  - Form filler demo with adaptive strategies
  - Side-by-side comparison of baseline vs ACE-enhanced automation
- Support for UV package manager (10-100x faster than pip)
  - Added uv.lock for reproducible builds
  - UV-specific installation and development instructions
- Improved documentation structure with multiple guides
  - QUICK_START.md for 5-minute quickstart
  - API_REFERENCE.md for complete API documentation
  - PROMPT_ENGINEERING.md for advanced techniques
  - SETUP_GUIDE.md for development setup
  - TESTING_GUIDE.md for testing procedures
- Optional dependency groups for modular installation
  - `observability` for Opik integration
  - `demos` for browser automation examples
  - `langchain` for LangChain support
  - `transformers` for local model support
  - `dev` for development tools
  - `all` for all features combined

### Changed
- **Replaced explainability module with observability**
  - Removed empty ace/explainability directory
  - Migrated to production-grade Opik monitoring
  - Updated all documentation to reflect this change
- Improved Python version requirements consistency (3.12 everywhere)
- Enhanced README with clearer examples and installation options
- Reorganized examples directory for better discoverability
- Updated CLAUDE.md with comprehensive codebase guidance

### Fixed
- Package configuration in pyproject.toml
- Documentation references to non-existent explainability module
- Python version inconsistencies across documentation files

### Removed
- Empty ace/explainability module (replaced by observability)
- Outdated references to explainability features in documentation

## [0.3.0] - 2025-10-16

### Added
- **Experimental v2 Prompts** with state-of-the-art prompt engineering
  - Confidence scoring at bullet and answer levels
  - Domain-specific variants for math and code generation
  - Hierarchical structure with identity headers and metadata
  - Concrete examples and anti-patterns for better guidance
  - PromptManager for version control and A/B testing
- Comprehensive prompt engineering documentation (`docs/PROMPT_ENGINEERING.md`)
- Advanced examples demonstrating v2 prompts (`examples/advanced_prompts_v2.py`)
- Comparison script for v1 vs v2 prompts (`examples/compare_v1_v2_prompts.py`)
- Playbook persistence with `save_to_file()` and `load_from_file()` methods
- Example demonstrating playbook save/load functionality (`examples/playbook_persistence.py`)
- py.typed file for PEP 561 type hint support
- Mermaid flowchart visualization in README showing ACE learning loop

### Changed
- Enhanced docstrings with comprehensive examples throughout codebase
- Improved README with v2 prompts section and visual diagrams
- Updated formatting to comply with Black code style

### Fixed
- README incorrectly referenced non-existent docs/ directory
- Test badge URL in README (test.yml → tests.yml)
- Code formatting issues detected by GitHub Actions

## [0.2.0] - 2025-10-15

### Added
- LangChain integration via `LangChainLiteLLMClient` for advanced workflows
- Router support for load balancing across multiple model deployments
- Comprehensive example for LangChain usage (`examples/langchain_example.py`)
- Optional installation group: `pip install ace-framework[langchain]`
- PyPI badges and Quick Links section in README
- CHANGELOG.md for version tracking

### Fixed
- Parameter filtering in LiteLLM and LangChain clients (refinement_round, max_refinement_rounds)
- GitHub Actions workflow using deprecated artifact actions v3 → v4

### Changed
- Improved README with better structure and badges
- Updated .gitignore to exclude build artifacts and development files

### Removed
- Unnecessary development files from repository

## [0.1.1] - 2025-10-15

### Fixed
- GitHub Actions workflow for PyPI publishing
- Updated artifact upload/download actions from v3 to v4

## [0.1.0] - 2025-10-15

### Added
- Initial release of ACE Framework
- Core ACE implementation based on paper (arXiv:2510.04618)
- Three-role architecture: Generator, Reflector, and Curator
- Playbook system for storing and evolving strategies
- LiteLLM integration supporting 100+ LLM providers
- Offline and Online adaptation modes
- Async and streaming support
- Example scripts for quick start
- Comprehensive test suite
- PyPI packaging and GitHub Actions CI/CD

### Features
- Self-improving agents that learn from experience
- Delta operations for incremental playbook updates
- Support for OpenAI, Anthropic, Google, and more via LiteLLM
- Type hints and modern Python practices
- MIT licensed for open source use

[0.9.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.9...v0.9.0
[0.8.8]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.7...v0.8.8
[0.8.7]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.6...v0.8.7
[0.8.5]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.4...v0.8.5
[0.8.4]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.7.0...v0.7.3
[0.7.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kayba-ai/agentic-context-engine/releases/tag/v0.1.0
