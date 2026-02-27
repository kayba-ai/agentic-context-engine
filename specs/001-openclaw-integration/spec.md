# Feature Specification: OpenClaw Integration

**Feature Branch**: `001-openclaw-integration`
**Created**: 2026-02-27
**Status**: Draft
**Input**: User description: "Integrate ACE with OpenClaw to automatically learn from session transcripts and sync strategies back into the agent's workspace"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One-Off Learning from Past Sessions (Priority: P1)

A developer has been running an OpenClaw agent that accumulates session transcripts over time. They install the ACE framework and want to immediately extract useful strategies from all existing sessions. They run a single command, and ACE parses the transcripts, identifies patterns and lessons, and produces a set of learned strategies. The developer can inspect what was learned before deciding to use it.

**Why this priority**: This is the core value proposition — extracting actionable strategies from historical agent sessions. Without this, no other feature matters. It delivers immediate value from day one with zero ongoing configuration.

**Independent Test**: Can be fully tested by providing sample session transcript files and verifying that strategies are extracted and saved to a persistent skillbook file.

**Acceptance Scenarios**:

1. **Given** an OpenClaw agent has completed at least one session with transcript files on disk, **When** the developer runs the learning process, **Then** the system discovers all session files, parses them into trace data, runs the learning pipeline, and saves learned strategies to a persistent skillbook file.
2. **Given** the learning process has completed, **When** the developer inspects the output, **Then** they see a summary of how many sessions were processed, how many strategies were extracted, and a preview of the latest strategies.
3. **Given** some session files contain no usable content (empty or malformed), **When** the learning process encounters them, **Then** they are skipped with a count of skipped sessions reported, and processing continues for remaining files.

---

### User Story 2 - Strategy Sync to Agent Workspace (Priority: P2)

After learning from sessions, the developer wants the learned strategies injected back into the OpenClaw agent's workspace so the agent can use them on its next run. The system writes strategies into a workspace file that OpenClaw reads at session start, formatted between clearly marked boundaries so other content in the file is preserved.

**Why this priority**: Learning without application has no value. This closes the feedback loop — strategies extracted from past sessions directly improve future sessions. It's the second half of the core value proposition.

**Independent Test**: Can be fully tested by running learning on sample sessions and verifying that the workspace file is created/updated with formatted strategies between marker boundaries, with any existing content outside the markers preserved.

**Acceptance Scenarios**:

1. **Given** the skillbook contains learned strategies, **When** the sync process runs, **Then** the strategies are written into the agent's workspace file between clearly defined marker comments.
2. **Given** the workspace file already contains content outside the marker comments, **When** the sync process runs, **Then** all existing content outside the markers is preserved unchanged.
3. **Given** the workspace file does not yet exist, **When** the sync process runs, **Then** the file is created with the strategies section.
4. **Given** the sync process has previously written strategies and new strategies have been learned, **When** the sync runs again, **Then** the marker section is replaced with the updated strategies.

---

### User Story 3 - Incremental Processing of New Sessions (Priority: P3)

A developer runs the OpenClaw agent regularly, generating new sessions over time. They want the learning process to only process new sessions each time it runs, avoiding redundant reprocessing of sessions already learned from. They also want the option to reprocess everything if needed (e.g., after resetting the skillbook).

**Why this priority**: For ongoing use, incremental processing prevents wasted computation and LLM API calls. Without it, every run would reprocess the entire history, which is slow and expensive. The reprocess escape hatch ensures developers are never stuck.

**Independent Test**: Can be fully tested by running the learning process twice — the second run should skip previously processed sessions and only process new ones. Running with a reprocess flag should process all sessions again.

**Acceptance Scenarios**:

1. **Given** the learning process has previously run and processed 5 sessions, **When** it runs again with 2 new sessions available, **Then** only the 2 new sessions are processed.
2. **Given** the processed session log tracks which sessions have been handled, **When** the developer requests a full reprocess, **Then** all sessions are processed regardless of the log.
3. **Given** the processed session log does not exist (first run), **When** the learning process runs, **Then** all available sessions are treated as new.

---

### User Story 4 - Dry Run Preview (Priority: P4)

A developer wants to see what sessions would be processed and what data would be extracted without actually running the learning pipeline or spending LLM API credits. They run the process in a preview mode that parses sessions and reports what it found, but does not call the learning pipeline or modify any files.

**Why this priority**: Developers need to verify their setup and understand what data is available before committing to an LLM-powered learning run. This reduces waste and builds confidence in the integration.

**Independent Test**: Can be fully tested by running in preview mode with sample sessions and verifying that session parsing results are displayed but no skillbook or workspace files are created or modified.

**Acceptance Scenarios**:

1. **Given** new session files exist, **When** the developer runs in preview mode, **Then** the system reports how many sessions were found, parses them, and displays a summary of extracted data without calling the learning pipeline.
2. **Given** the developer runs in preview mode, **When** the process completes, **Then** no skillbook file, workspace file, or processed log is created or modified.

---

### Edge Cases

- What happens when the session transcript directory does not exist (OpenClaw not installed or agent never run)? The system reports a clear error message indicating the expected directory and suggests verifying the installation.
- What happens when all discovered sessions have already been processed? The system reports "nothing new to learn from" and exits cleanly without errors.
- What happens when the LLM API key is missing or invalid? The system fails with a clear error before attempting any API calls, without corrupting existing skillbook or workspace files.
- What happens when the session transcript format changes between OpenClaw versions? The parser handles missing or unexpected fields gracefully, skipping unparseable sessions and reporting them as skipped.
- What happens when the skillbook file is corrupted or invalid? The system reports the issue and offers the option to start fresh with a new skillbook rather than crashing.
- What happens when the workspace file has been manually edited and the marker comments were removed? The system appends a new marker section to the end of the file rather than silently overwriting content.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST discover session transcript files from the configured OpenClaw agent's session directory.
- **FR-002**: System MUST parse session transcripts into structured trace data containing the original user request, agent reasoning (including full thinking traces without truncation), final answer, and tool usage information.
- **FR-003**: System MUST run the ACE learning pipeline (reflect, tag, update, apply) on parsed traces to extract strategies.
- **FR-004**: System MUST persist learned strategies to a skillbook file that survives across runs.
- **FR-005**: System MUST sync learned strategies into the OpenClaw agent's workspace file between clearly defined marker boundaries.
- **FR-006**: System MUST preserve all existing content in the workspace file outside of the marker boundaries during sync.
- **FR-007**: System MUST track which sessions have already been processed to enable incremental learning.
- **FR-008**: System MUST support a reprocess mode that ignores the processed log and reprocesses all sessions.
- **FR-009**: System MUST support a preview mode that parses sessions and reports findings without running the learning pipeline or modifying files.
- **FR-010**: System MUST skip malformed or empty session files gracefully and continue processing remaining files.
- **FR-011**: System MUST report a summary after each run including: sessions discovered, sessions processed, sessions skipped, strategies before and after, and new strategies added.
- **FR-012**: System MUST allow configuration of the OpenClaw home directory, agent identifier, workspace path, and LLM model through environment variables or explicit parameters.
- **FR-013**: System MUST be runnable as a standalone command for one-off use and be schedulable for recurring automated runs (e.g., via cron or task scheduler).

### Key Entities

- **Session Transcript**: A record of a single OpenClaw agent session containing the sequence of user messages, agent responses, and tool invocations. Stored as a file on disk in the OpenClaw sessions directory.
- **Trace**: A structured representation of a session transcript containing the original question, agent reasoning chain (including full, untruncated thinking blocks), final answer, tool call summary, and optional feedback. Used as input to the ACE learning pipeline.
- **Skillbook**: A persistent collection of learned strategies. Each strategy has an identifier, topic section, content description, and effectiveness scores (helpful/harmful/neutral). Survives across runs and grows over time.
- **Strategy**: A single learned lesson within the skillbook. Describes a specific technique, pattern, or approach that the agent found useful (or harmful). Includes evidence and justification from the session that produced it.
- **Workspace File**: The file in the OpenClaw agent's workspace where learned strategies are injected for the agent to read on its next session. Contains a marked section that is updated by the sync process while preserving all other content.
- **Processed Log**: A record of which session files have already been analyzed, enabling incremental processing. Prevents redundant reprocessing of historical sessions.

## Clarifications

### Session 2026-02-27

- Q: Should thinking traces be truncated or filtered from parsed traces? → A: No. Thinking traces must be preserved in full — they are integral to how the agent works and provide essential reasoning context for the learning pipeline.
- Q: Should tool call arguments and results be preserved in full or truncated? → A: Preserve in full. No truncation for tool arguments or tool results.
- Q: Where should trace loading and conversion logic live? → A: A generic `LoadTracesStep` in `ace_next/steps/` loads raw file contents onto `ctx.trace`. An OpenClaw-specific `OpenClawToTraceStep` in `ace_next/integrations/openclaw/` converts raw JSONL events into the structured trace dict, preserving chronological order of queries, thinking, and tool uses. The transformation logic will be defined separately.

## Assumptions

- OpenClaw stores session transcripts as individual files (one per session) in a predictable directory structure under the OpenClaw home directory.
- OpenClaw reads the workspace file (AGENTS.md) at the start of each session, making it the appropriate injection point for learned strategies.
- The LLM API key is provided through standard environment variable configuration.
- The default LLM model for reflection and skill extraction follows the project's standard configuration pattern.
- Session transcripts contain structured entries with role (user/assistant) and content fields, with optional tool invocation entries.
- A single run processes all new sessions in one batch — there is no need for streaming or real-time processing of in-progress sessions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer can go from installation to first learned strategies in under 5 minutes with no more than 3 configuration steps.
- **SC-002**: The system processes 100 session transcripts and produces at least 1 learned strategy per 10 sessions on average.
- **SC-003**: Incremental runs (no new sessions) complete in under 5 seconds without making any LLM calls.
- **SC-004**: After learning, the OpenClaw agent's workspace file contains all current strategies in a format the agent can read and apply on its next session.
- **SC-005**: The system handles 500+ historical sessions without failures or data loss in a single run.
- **SC-006**: 90% of developers can complete the full setup-learn-sync cycle on their first attempt following the documentation.
- **SC-007**: Strategies learned from OpenClaw sessions are cited by the agent in at least 20% of subsequent sessions where a relevant strategy exists.
