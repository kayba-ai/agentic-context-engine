# Changelog

All notable changes to ACE Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Improved Python version requirements consistency (3.11+ everywhere)
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

[0.7.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Kayba-ai/agentic-context-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Kayba-ai/agentic-context-engine/releases/tag/v0.1.0