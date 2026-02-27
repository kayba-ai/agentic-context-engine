# Implementation Plan: OpenClaw Integration

**Branch**: `001-openclaw-integration` | **Date**: 2026-02-27 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-openclaw-integration/spec.md`

## Summary

Integrate ACE with OpenClaw to automatically learn from session transcripts (JSONL) and sync strategies back into the agent's workspace (AGENTS.md). Uses the existing `TraceAnalyser` pipeline to run Reflect → Tag → Update → Apply on parsed transcripts, with incremental processing and dry-run support. The implementation adds two new pipeline steps — a generic `LoadTracesStep` in `ace_next/steps/` and an OpenClaw-specific `OpenClawToTraceStep` in `ace_next/integrations/openclaw/` — composed with the learning tail in an example script (`examples/openclaw/learn_from_traces.py`).

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: ace_next (Skillbook, Reflector, SkillManager, TraceAnalyser, LiteLLMClient, wrap_skillbook_context), pydantic >=2.0.0, litellm >=1.78.0
**Storage**: JSON file (skillbook at `~/.openclaw/ace_skillbook.json`), plain text (processed log at `~/.openclaw/ace_processed.txt`), JSONL (OpenClaw session transcripts)
**Testing**: pytest with pytest-cov (coverage enforced `--cov-fail-under=25`), MockLLMClient pattern from existing tests
**Target Platform**: Linux/macOS (local development machines where OpenClaw runs)
**Project Type**: Example/integration script (shipped in `examples/openclaw/`, not in the core library)
**Performance Goals**: Incremental runs (no new sessions) complete in <5 seconds without LLM calls (SC-003); handle 500+ sessions in a single run (SC-005)
**Constraints**: No new core dependencies; uses only existing ACE pipeline components; environment variable configuration
**Scale/Scope**: Single CLI script + tests; targets individual developer workstations with 1-500+ session transcripts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Ease of Use First** | PASS | SC-001 requires <5 min setup with <=3 config steps. Single script entry point. Config via environment variables with sensible defaults. README with copy-pasteable examples. |
| **II. Practical Value** | PASS | Solves a concrete problem: extracting strategies from real OpenClaw sessions. Adds learning/skillbook evolution on top of OpenClaw (measurable value per constitution). |
| **III. Simplicity** | PASS | Single script in `examples/`, no new abstractions. Reuses existing TraceAnalyser, Skillbook, Reflector, SkillManager. No new dependencies. Plain-text processed log (not a DB). |
| **IV. Clean & Modular Code** | PASS | Parsing, learning, syncing, and tracking are separate functions. Uses existing ACE module boundaries. No circular dependencies introduced. |

**Gate Result**: PASS — No violations. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-openclaw-integration/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI contract)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
ace_next/steps/
└── load_traces.py           # LoadTracesStep — generic file→ctx.trace loader

ace_next/integrations/openclaw/
├── __init__.py              # Exports OpenClawToTraceStep
└── to_trace.py              # OpenClawToTraceStep — JSONL events→trace dict

examples/openclaw/
├── learn_from_traces.py     # Main entry point (composes steps + learning tail)
├── README.md                # Integration documentation (existing)
└── *.jsonl                  # Sample session transcripts

tests/
├── test_load_traces_step.py # Unit tests for LoadTracesStep
└── test_openclaw.py         # Unit tests for OpenClawToTraceStep, end-to-end

docs/integrations/
└── openclaw.md              # Integration guide (new)
```

**Structure Decision**: Two new pipeline steps following existing patterns. `LoadTracesStep` is generic (reads files, puts raw data on `ctx.trace`) and lives in `ace_next/steps/`. `OpenClawToTraceStep` is integration-specific (converts OpenClaw JSONL to trace dict) and lives in `ace_next/integrations/openclaw/`. The example script composes these steps with `learning_tail()`. No changes to existing core classes.

## Constitution Re-Check (Post-Design)

| Principle | Status | Post-Design Evidence |
|-----------|--------|---------------------|
| **I. Ease of Use First** | PASS | quickstart.md confirms 3-step setup. CLI contract shows clear flags and output. |
| **II. Practical Value** | PASS | R-001 confirmed real JSONL format parsing. Thinking content (R-003) and tool calls (R-004) provide rich learning signal. |
| **III. Simplicity** | PASS | No new entities beyond what spec defined. Reuses all existing ACE APIs. Plain dict traces, no new dataclasses. |
| **IV. Clean & Modular Code** | PASS | Data model shows clean separation: parsing (JSONL → Trace), learning (TraceAnalyser), persistence (Skillbook), sync (AGENTS.md). Each is a distinct function. |

**Post-Design Gate Result**: PASS — No violations introduced during design.

## Complexity Tracking

> No constitution violations — this section is intentionally empty.
