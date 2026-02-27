# Tasks: OpenClaw Integration

**Input**: Design documents from `/specs/001-openclaw-integration/`
**Prerequisites**: plan.md, spec.md, data-model.md, research.md, contracts/cli.md, quickstart.md

**Tests**: Included — plan.md specifies test files and CLAUDE.md requires tests for new features (R-009).

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Create package structure for the new OpenClaw integration

- [X] T001 Create `ace_next/integrations/openclaw/` package directory with `__init__.py`

---

## Phase 2: Foundational (Pipeline Steps)

**Purpose**: Implement the two new pipeline steps that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 [P] Implement `LoadTracesStep` in `ace_next/steps/load_traces.py` — generic step that reads JSONL file at `ctx.sample`, parses lines into `list[dict]`, places on `ctx.trace`; `requires={"sample"}`, `provides={"trace"}`; skip unparseable lines gracefully per FR-010 and data-model.md validation rules
- [X] T003 [P] Implement `OpenClawToTraceStep` (pass-through) in `ace_next/integrations/openclaw/to_trace.py` — `requires={"trace"}`, `provides={"trace"}`; for now returns `ctx` unchanged (transformation logic deferred per user decision); follow existing ToTrace step pattern from `ClaudeCodeToTrace`/`BrowserToTrace`
- [X] T004 Export `OpenClawToTraceStep` from `ace_next/integrations/openclaw/__init__.py` and add to `ace_next/integrations/__init__.py` exports; export `LoadTracesStep` from `ace_next/steps/__init__.py`

**Checkpoint**: Pipeline steps ready — user story implementation can now begin

---

## Phase 3: User Story 1 — One-Off Learning from Past Sessions (Priority: P1) MVP

**Goal**: Developer runs a single command and ACE discovers session transcripts, parses them, runs the learning pipeline, and saves strategies to a persistent skillbook file.

**Independent Test**: Provide sample JSONL files, run the script, verify strategies extracted and saved to `ace_skillbook.json`.

**Acceptance**: FR-001, FR-002, FR-003, FR-004, FR-010, FR-011, FR-012, FR-013

### Implementation for User Story 1

- [X] T005 [US1] Rewrite `examples/openclaw/learn_from_traces.py`: remove broken `parse_session_jsonl()` function; implement `discover_sessions()` that globs `~/.openclaw/agents/<agent_id>/sessions/*.jsonl` using env vars `OPENCLAW_HOME`, `OPENCLAW_AGENT_ID` per R-005 and contracts/cli.md; report clear error if directory missing per edge cases
- [X] T006 [US1] Implement pipeline composition in `examples/openclaw/learn_from_traces.py`: for each discovered session, run `LoadTracesStep → OpenClawToTraceStep → learning_tail()` using `TraceAnalyser.from_roles()` with `LiteLLMClient`, `Reflector`, `SkillManager`; configure model via `ACE_MODEL` env var per FR-012
- [X] T007 [US1] Implement skillbook load/save in `examples/openclaw/learn_from_traces.py`: load from `~/.openclaw/ace_skillbook.json` via `Skillbook.load_from_file()` (create new if missing); save after learning via `Skillbook.save_to_file()` per FR-004 and R-006
- [X] T008 [US1] Implement summary reporting in `examples/openclaw/learn_from_traces.py` per CLI contract output format: sessions discovered, already processed, new to process, parsed/skipped counts, strategies before/after/new, latest strategy preview per FR-011
- [X] T009 [US1] Implement CLI entry point with `argparse` in `examples/openclaw/learn_from_traces.py`: `--dry-run` and `--reprocess` flags (wired in later phases), error handling for missing API key and corrupted skillbook per edge cases, `if __name__ == "__main__"` block per FR-013

**Checkpoint**: One-off learning fully functional — can discover, parse, learn, and save strategies

---

## Phase 4: User Story 2 — Strategy Sync to Agent Workspace (Priority: P2)

**Goal**: Learned strategies are injected into the OpenClaw agent's `AGENTS.md` between marker boundaries so the agent reads them on next session.

**Independent Test**: After learning, verify `AGENTS.md` contains strategies between `<!-- ACE:SKILLBOOK:START/END -->` markers with existing content preserved.

**Acceptance**: FR-005, FR-006

### Implementation for User Story 2

- [X] T010 [US2] Implement `sync_to_agents_md()` in `examples/openclaw/learn_from_traces.py`: use `wrap_skillbook_context()` to format strategies; write between `<!-- ACE:SKILLBOOK:START -->` / `<!-- ACE:SKILLBOOK:END -->` markers per contracts/cli.md AGENTS.md Marker Contract; replace if markers exist, append if not; preserve content outside markers per FR-006; create file if missing; use `OPENCLAW_WORKSPACE` env var
- [X] T011 [US2] Integrate sync into `main()` flow in `examples/openclaw/learn_from_traces.py`: call `sync_to_agents_md()` after skillbook save; skip sync in dry-run mode; report sync path in summary output

**Checkpoint**: Full learn-and-sync cycle works — strategies extracted and injected into workspace

---

## Phase 5: User Story 3 — Incremental Processing (Priority: P3)

**Goal**: Only new sessions are processed on subsequent runs; `--reprocess` overrides to reprocess all.

**Independent Test**: Run twice — second run skips already-processed sessions. Run with `--reprocess` — all sessions reprocessed.

**Acceptance**: FR-007, FR-008

### Implementation for User Story 3

- [X] T012 [US3] Implement processed log read/write in `examples/openclaw/learn_from_traces.py`: read/write `~/.openclaw/ace_processed.txt` as newline-delimited sorted session filenames per data-model.md ProcessedLog; filter `discover_sessions()` output to exclude already-processed files per FR-007
- [X] T013 [US3] Wire `--reprocess` flag in `examples/openclaw/learn_from_traces.py`: when set, ignore processed log and process all discovered sessions per FR-008; update processed log after successful processing regardless of flag

**Checkpoint**: Incremental processing works — repeat runs skip processed sessions, `--reprocess` overrides

---

## Phase 6: User Story 4 — Dry Run Preview (Priority: P4)

**Goal**: `--dry-run` parses sessions and reports findings without running the learning pipeline or modifying any files.

**Independent Test**: Run with `--dry-run`, verify no skillbook/workspace/processed-log files created or modified.

**Acceptance**: FR-009

### Implementation for User Story 4

- [X] T014 [US4] Wire `--dry-run` flag in `examples/openclaw/learn_from_traces.py`: when set, discover and parse sessions, display summary of extracted data (session count, trace previews), but skip `TraceAnalyser.run()`, skip `Skillbook.save_to_file()`, skip `sync_to_agents_md()`, skip processed log write per FR-009

**Checkpoint**: All 4 user stories independently functional

---

## Phase 7: Polish & Testing

**Purpose**: Tests, documentation, and cross-cutting validation

- [X] T015 [P] Unit tests for `LoadTracesStep` in `tests/test_load_traces_step.py`: test JSONL parsing, empty file, missing file, unparseable lines skipped, valid multi-line JSONL; use sample JSONL fixture from `examples/openclaw/b3db607f-*.jsonl`
- [X] T016 [P] Unit tests for `OpenClawToTraceStep` in `tests/test_openclaw.py`: test pass-through behavior, verify requires/provides contract, verify step returns context unchanged; use `MockLLMClient` pattern from existing tests per R-009
- [X] T017 End-to-end test in `tests/test_openclaw.py`: test full pipeline `LoadTracesStep → OpenClawToTraceStep → learning_tail()` with `MockReflector` and `MockSkillManager`; verify skillbook receives new strategies; use sample JSONL fixture
- [X] T018 Update `examples/openclaw/README.md` with current usage matching quickstart.md and contracts/cli.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — T002/T003 need the package directory from T001
- **US1 (Phase 3)**: Depends on Phase 2 — needs LoadTracesStep and OpenClawToTraceStep
- **US2 (Phase 4)**: Depends on US1 — sync needs a working learning flow to produce strategies
- **US3 (Phase 5)**: Depends on US1 — incremental processing adds to the base learning flow
- **US4 (Phase 6)**: Depends on US1 — dry-run modifies the base learning flow
- **Polish (Phase 7)**: Depends on Phase 2 (tests for steps) and US4 (all features complete for e2e test)

### User Story Dependencies

- **US1 (P1)**: Requires Foundational (Phase 2) — core learning flow
- **US2 (P2)**: Requires US1 — needs working skillbook to sync
- **US3 (P3)**: Requires US1 — adds filtering on top of discovery
- **US4 (P4)**: Requires US1 — adds early-exit branch to main flow
- **US3 and US4 are independent of each other** — can be implemented in either order after US1

### Within Each Phase

- T002 and T003 are parallel (different files)
- T005 → T006 → T007 → T008 → T009 are sequential (same file, building up)
- T010 → T011 are sequential (same file)
- T012 → T013 are sequential (same file)
- T015 and T016 are parallel (different test files)

### Parallel Opportunities

- **Phase 2**: T002 (LoadTracesStep) and T003 (OpenClawToTraceStep) — different files
- **Phase 7**: T015 (test_load_traces_step.py) and T016 (test_openclaw.py) — different files
- **Cross-phase**: T015 can start as soon as T002 completes; T016 can start as soon as T003 completes

---

## Parallel Example: Foundational Phase

```bash
# Launch both pipeline steps in parallel (different files):
Task: "Implement LoadTracesStep in ace_next/steps/load_traces.py"
Task: "Implement OpenClawToTraceStep in ace_next/integrations/openclaw/to_trace.py"
```

## Parallel Example: Testing Phase

```bash
# Launch both test files in parallel:
Task: "Unit tests for LoadTracesStep in tests/test_load_traces_step.py"
Task: "Unit tests for OpenClawToTraceStep in tests/test_openclaw.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002–T004)
3. Complete Phase 3: User Story 1 (T005–T009)
4. **STOP and VALIDATE**: Run script against sample JSONL, verify strategies saved
5. If working → continue to US2–US4

### Incremental Delivery

1. Setup + Foundational → Pipeline steps ready
2. Add US1 → Test with sample JSONL → Core learning works (MVP!)
3. Add US2 → Verify AGENTS.md updated → Full loop closed
4. Add US3 → Run twice, verify incremental → Production-ready
5. Add US4 → Verify dry-run → Developer-friendly
6. Polish → Tests + docs → Ship-ready

---

## Notes

- T003 (OpenClawToTraceStep) is a **pass-through** for now — transformation logic deferred per user decision
- The existing `learn_from_traces.py` (345 lines) will be **rewritten** starting at T005, not patched incrementally
- Sample JSONL fixture: `examples/openclaw/b3db607f-7ae8-4089-b806-44800e961672.jsonl`
- MockLLMClient, MockReflector, MockSkillManager patterns from `tests/conftest.py` and `tests/test_ace_next_steps.py`
- All env var defaults per contracts/cli.md: `OPENCLAW_HOME=~/.openclaw`, `OPENCLAW_AGENT_ID=main`, `ACE_MODEL=anthropic/claude-sonnet-4-20250514`
