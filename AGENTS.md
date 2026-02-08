# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Lint Commands

```bash
uv sync                                    # Install all dependencies
uv run pytest                              # Run all tests (coverage enforced >=25%)
uv run pytest tests/test_roles.py          # Run a single test file
uv run pytest -k "test_reflect"            # Run tests matching a pattern
uv run pytest -m unit                      # Run only unit tests (markers: unit, integration, slow)
uv run black ace/ tests/ examples/         # Format code
uv run black --check ace/ tests/ examples/ # Check formatting
uv run mypy ace/                           # Type check
```

## Repository Overview

ACE (Agentic Context Engine) enables AI agents to learn from execution feedback through in-context learning. No fine-tuning required. The framework maintains a **Skillbook** — a living document of strategies that evolves with each task.

**Three roles share the same LLM instance:**
- **Agent** — produces answers using the current skillbook
- **Reflector** — analyzes execution traces (simple single-pass or recursive code-based)
- **SkillManager** — emits incremental update operations (ADD, UPDATE, TAG, REMOVE) to the skillbook

## Architecture: Two Usage Patterns

**Full ACE Pipeline** (new agents): Sample → Agent → Environment → Reflector → SkillManager → Skillbook
- Use `OfflineACE` for multi-epoch training, `OnlineACE` for single-pass streaming
- See `ace/adaptation.py`

**Integration Pattern** (wrapping existing agents like browser-use, LangChain):
1. INJECT skillbook context into external agent
2. EXECUTE with external framework (no ACE Agent involved)
3. LEARN from results via Reflector + SkillManager
- See `ace/integrations/base.py` for the pattern

## Key Module Map

| Module | Purpose |
|--------|---------|
| `ace/roles.py` | Agent, Reflector, SkillManager implementations |
| `ace/adaptation.py` | OfflineACE/OnlineACE orchestration loops, async learning pipeline |
| `ace/skillbook.py` | Skill storage with TOON format (token-efficient serialization) |
| `ace/updates.py` | UpdateOperation/UpdateBatch for incremental skillbook changes |
| `ace/reflector/` | **Recursive Reflector** — REPL-based trace analysis (feature branch focus) |
| `ace/integrations/` | Wrappers: ACELiteLLM, ACEAgent, ACELangChain, ACEClaudeCode |
| `ace/llm_providers/` | LiteLLMClient (100+ providers), LangChainClient, InstructorClient |
| `ace/deduplication/` | Skill similarity detection and consolidation |
| `ace/observability/` | Opik integration for tracing and cost tracking |
| `ace/prompts_v2_1.py` | **Recommended** production prompts (+17% over v1) |

## Recursive Reflector (`ace/reflector/`)

The key feature on this branch. Instead of single-pass LLM reflection, the LLM generates Python code to explore traces in a sandboxed REPL loop (3-20 iterations).

- `recursive.py` — `RecursiveReflector` orchestrates the code execution loop
- `sandbox.py` — `TraceSandbox` provides restricted exec() with safe builtins (no file/network)
- `trace_context.py` — `TraceContext`/`TraceStep` with factory methods for different trace formats
- `subagent.py` — `SubAgentLLM` wraps ask_llm() with a shared `CallBudget` (max 30 calls)
- `config.py` — `RecursiveConfig` tuning (max_iterations=20, timeout=30s, max_context_chars=50K)

**Safety gates:** FINAL() rejected on iteration 0 (force exploration); FINAL() rejected after code errors (prevent hallucination); message trimming with semantic scoring.

**Mode selection** in `Reflector._select_mode()`: SIMPLE for small traces, RECURSIVE for complex multi-step traces, AUTO to let the reflector decide.

## Skillbook Format

- `skillbook.as_prompt()` → TOON format (for LLM consumption, 16-62% token savings)
- `str(skillbook)` → Markdown format (for human debugging)
- Skills have helpful/harmful/neutral counters for effectiveness tracking

## Testing Patterns

- `tests/conftest.py` defines `MockLLMClient` with `complete_structured()` to prevent Instructor auto-wrapping
- Fixtures: `empty_skillbook`, `sample_skillbook`, `sample_question`, `agent_valid_json`
- Markers: `unit`, `integration`, `slow`, `requires_api`
- Tests follow pattern: `tests/test_*.py`, functions `test_*`, classes `Test*`

## Coding Conventions

- PEP 8 with Black (line length 88), Python 3.12
- Type hints and docstrings for public APIs
- Conventional Commits: `feat(scope): subject`, `fix(scope): subject`
- Branch naming: `<type>/<description>` (e.g., `feature/recursive-reflector`)
- Instructor auto-wrapping: all roles detect `complete_structured()` on the LLM — if missing, Instructor wraps automatically

## Agent-Specific Notes

- `ace-learn` writes learned guidance to `CLAUDE.md` and `.ace/skillbook.json` at the project root
- Avoid editing generated skillbook files by hand

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
