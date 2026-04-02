# RR Orchestration: Manager-Style Scaling for the Recursive Reflector

> Design document for scaling batch RR without stuffing all raw traces into one reasoning context or silently degrading outputs.

**Status:** Proposed  
**Depends on:** `docs/design/RR_DESIGN.md`  
**Scope:** `ace/rr/` only

---

## Goal

Make batch RR behave like a manager supervising teams of workers:

- The orchestrator sees the overall task and decides what to do next.
- Worker sessions handle focused subsets of traces.
- The orchestrator can inspect completed work, evaluate quality, and reassign follow-up work.
- Final outputs are accepted only when coverage and output contracts are satisfied.

This design keeps the public `RRStep` contract unchanged:

- `requires = {"trace", "skillbook"}`
- `provides = {"reflections"}`

Single-trace RR remains unchanged.

---

## Why Current Batch RR Does Not Scale

The current RR batch path runs one main agent session over the entire batch. That creates three problems on large inputs:

1. The main prompt grows with batch size.
   The current prompt builder emits a preview row for every batch item and a large data summary. The full raw traces still live in the sandbox, but the initial user prompt becomes heavy before any reasoning starts.

2. One session is doing too many jobs at once.
   A single RR session must discover structure, choose what matters, fan out sub-analysis, compare patterns, and produce per-item results for the whole batch.

3. Existing batch fallback behavior is not acceptable for orchestration.
   Current RR can duplicate a generic reflection when per-item results are missing. That is tolerable as a legacy fallback, but it is explicitly disallowed in orchestration mode.

The orchestration design fixes this by moving deep analysis into explicit worker RR sessions while keeping the orchestrator responsible for planning, evaluation, and final acceptance.

---

## Design Principles

1. No hard strategy cutoff.
   There is no batch-size threshold that forces orchestration. Any batch session uses an orchestrator-capable RR path. The orchestrator decides whether to analyze directly or delegate based on trace count, trace size, heterogeneity, and observed complexity.

2. No silent fallbacks.
   Missing coverage, invalid worker output, overlaps, or malformed per-item results are surfaced explicitly. The batch run fails loudly if the orchestrator does not resolve them.

3. Explicit worker assignments.
   Every worker is launched with a stated goal and success criteria, not just a list of indices.

4. Manager-style oversight, not streaming telemetry.
   The orchestrator does not need live token-by-token visibility into workers. It does need structured visibility into what it asked for, what came back, how much it cost, and whether it was usable.

5. Keep changes inside `ace/rr/`.
   No `pipeline/`, `ace/core/`, or `ace/steps/` changes are required.

---

## High-Level Model

For any batch trace:

1. Start one orchestrator RR session.
2. Give it a slim batch prompt plus full sandbox access.
3. Let it choose between:
   - direct analysis in the current session, or
   - spawning worker RR sessions for subsets of traces
4. Let it collect completed worker results.
5. Let it inspect and evaluate those results with existing RR tools.
6. Let it spawn follow-up work if needed.
7. Validate final coverage and output contracts.
8. Merge per-item reflections back into original item order.

The orchestrator acts like a manager:

- it knows the overall goal
- it assigns work
- it reviews outcomes
- it decides what to do next

---

## Session Topology

```text
RRStep._run_batch_reflections()
│
└── Orchestrator RR session
    │  Agent: self._orchestrator_agent
    │  Tools: execute_code, analyze, batch_analyze,
    │         spawn_analysis, collect_results
    │
    │  decide: analyze directly or delegate
    │  execute_code(...)                       -> inspect shape / build helpers
    │  spawn_analysis(... goal=..., criteria=...)
    │  spawn_analysis(... goal=..., criteria=...)
    │  collect_results()
    │  execute_code(cluster_results[...])      -> inspect outputs
    │  analyze(... context="review cluster X")
    │  spawn_analysis(... retry / narrower scope)
    │  collect_results()
    │  validate coverage and quality
    │  produce final ReflectorOutput
    │
    ├── Worker RR session: cluster A
    │   Agent: self._worker_agent
    │   Tools: execute_code, analyze?, batch_analyze?
    │   Input: sub-batch + inherited helpers + explicit assignment
    │   Output: ReflectorOutput with per-item results
    │
    ├── Worker RR session: cluster B
    │   ...
    │
    └── Worker RR session: retry cluster
        ...
```

The orchestrator's own final `ReflectorOutput` is still useful for metadata and cross-cluster synthesis, but per-item reflections come from worker sessions when delegation is used.

---

## Agent Instances

Create dedicated agents in `RRStep.__init__`:

| Instance | Purpose | Tool visibility |
|---|---|---|
| `self._single_agent` | Existing single-trace RR path | `execute_code`, `analyze`, `batch_analyze` |
| `self._orchestrator_agent` | All batch sessions | `execute_code`, `analyze`, `batch_analyze`, `spawn_analysis`, `collect_results` |
| `self._worker_agent` | Spawned worker RR sessions | `execute_code`, optional `analyze`, optional `batch_analyze`, never orchestration tools |
| `self._sub_agent` | Existing text sub-agent | `execute_code` only |

Important constraint:

- If worker sub-agents are disabled, `analyze` and `batch_analyze` must be omitted from the worker tool list via `prepare=` or a dedicated no-subagent worker agent.
- Returning stub strings like `"(analyze unavailable)"` is not acceptable in worker mode. Tools that are not usable must not be visible.

### Construction strategy

Single-trace RR does not need batch-only agents.

To keep the common path clean, `self._orchestrator_agent` and `self._worker_agent` should be created lazily on first batch use rather than eagerly in every `RRStep.__init__`.

This is not a behavioral requirement, but it is the preferred construction strategy for v1:

- single-trace runs only pay for the existing RR setup
- batch-only behavior is initialized when it is actually needed
- code intent is clearer: these are batch orchestration agents, not universal RR agents

---

## Orchestrator Prompt and Context Control

### No threshold gating

Batch mode always uses the orchestrator-capable RR path. The orchestrator decides whether delegation is necessary.

That decision should be based on:

- number of batch items
- total serialized trace size
- total message count
- evidence of heterogeneity across traces
- complexity of the evaluation criteria or embedded policy

### Slim batch manager prompt

The orchestrator should not reuse the current full batch prompt unchanged. A dedicated slim batch prompt is required.

The slim prompt includes:

- total batch item count
- total serialized size
- total message count
- capped preview rows
- pass/fail summary when detectable
- capped exemplar item IDs
- count of precomputed survey groups
- note that full data remains available in `execute_code`

The slim prompt must avoid:

- one preview row per item
- long item summary lists
- large inline survey-item dumps

Workers can keep using the existing prompt builder because their sub-batches are much smaller.

### Manager guidance

The orchestrator prompt should tell the model to decide early between:

- direct analysis in the current session
- launching worker assignments

It should explicitly say that:

- delegation is optional
- the decision should consider trace count and trace size
- the orchestrator is responsible for validating results before finalizing

### Parallel local extraction inside `execute_code`

The orchestrator and workers should also be taught to use sandbox-level local parallelism for pure extraction work.

Specifically:

- use `parallel_map(...)` inside a single `execute_code(...)` call for feature extraction, counting, shape probes, or compact summary generation across many traces
- do not use `parallel_map(...)` for semantic judgment or final synthesis
- prefer one well-structured extraction pass over many small sequential code calls

This avoids wasting tool iterations on serial local work while keeping actual reasoning in RR sessions and sub-agents.

---

## Worker Assignments

### Tool: `spawn_analysis`

```python
spawn_analysis(
    cluster_name: str,
    trace_indices: list[int],
    goal: str,
    success_criteria: str = "",
) -> str
```

This launches a worker RR session for a subset of traces.

### Assignment contract

Each worker assignment is explicit:

- `cluster_name`: stable identifier
- `trace_indices`: original indices in the parent batch
- `goal`: what the worker should figure out
- `success_criteria`: what counts as a usable result

Examples:

- `"Find the root cause shared by these failed traces and produce per-item learnings."`
- `"Compare these policy-violation traces against the successful ones and verify the agent's claims."`

### Behavior

`spawn_analysis` does the following:

1. Validate the assignment.
   - indices must be in range
   - indices must be unique within the assignment
   - overlapping active assignments are rejected
   - duplicate cluster names are rejected unless explicitly marked as a retry/replacement in a later extension

2. Slice the batch trace while preserving its original container shape.

3. Build a worker session through the same `_run_reflection_session(...)` lifecycle used by top-level RR.

4. Propagate registered helper definitions into the worker sandbox.

5. Submit the worker session to a bounded executor.

6. Record the assignment immediately in orchestration state with status `queued` or `running`.

7. Return a short status line to the orchestrator.

### Batch slicing

Add a private helper:

```python
def _slice_batch_trace(self, traces: Any, indices: list[int]) -> Any:
    """Slice a batch trace to a subset of items, preserving container shape."""
```

This preserves:

- raw `list[...]` shape
- dicts with `"items"`
- dicts with `"tasks"`
- legacy combined `"steps"` batches
- batch-level metadata outside the item list

### Worker budget

Workers should receive an explicit budget computed by the runtime, not by a hard batch cutoff.

The budget heuristic should consider:

- assigned trace count
- serialized size of the assigned sub-batch
- optional safety caps from config

The orchestrator decides whether to delegate. The runtime decides a safe budget for the delegated task.

---

## Result Collection and Visibility

### Tool: `collect_results`

```python
collect_results() -> str
```

This blocks until all currently pending worker assignments finish, then records their outputs in structured form.

This tool needs a bounded wait so orchestration cannot block indefinitely on a slow worker.

### Behavior

For each completed worker assignment:

1. Capture the full worker `ReflectorOutput`.
2. Validate that per-item results exist and match the assigned trace count.
3. Store the result as a structured record in `cluster_results`.
4. Expose `cluster_results` in the orchestrator sandbox for later inspection.

The orchestrator can then use:

- `execute_code` to inspect `cluster_results`
- `analyze` to critique or summarize worker outputs
- additional `spawn_analysis(...)` calls to rerun or narrow the work

### Timeout semantics

The timeout used by `collect_results()` is a collection timeout, not guaranteed thread cancellation.

If a worker does not finish within the configured collection window:

- mark the assignment as `status="failed"` or `status="timed_out"`
- record the issue explicitly in `issues`
- do not silently accept partial coverage

This timeout bounds orchestration waiting behavior. It does not by itself guarantee that an underlying provider call is forcibly terminated, so provider-level request timeouts should still be configured separately where available.

### `cluster_results` schema

Each entry should preserve both the assignment and the returned result:

```python
cluster_results[cluster_name] = {
    "assignment": {
        "trace_indices": [2, 7, 15],
        "goal": "Compare these policy-failure traces to successful ones",
        "success_criteria": "Return per-item reflections with verified evidence",
    },
    "status": "completed",  # queued | running | completed | failed | invalid
    "issues": [],           # explicit validation or runtime issues
    "worker_output": ReflectorOutput(...),
    "per_item_reflections": (ReflectorOutput, ...),
    "usage": {
        "requests": 12,
        "input_tokens": 4000,
        "output_tokens": 1800,
        "total_tokens": 5800,
    },
    "rr_trace": {
        "total_iterations": 3,
        "subagent_calls": [...],
        "timed_out": False,
    },
}
```

For failures:

```python
cluster_results[cluster_name] = {
    "assignment": {...},
    "status": "failed",
    "issues": ["UsageLimitExceeded"],
    "worker_output": None,
    "per_item_reflections": (),
    "usage": {...},
    "rr_trace": {...},
}
```

### Manager-style oversight

This gives the orchestrator what it needs:

- what it asked for
- what came back
- whether the result matched the requested shape
- how much it cost
- what needs follow-up

That is sufficient for manager-style behavior without live streaming visibility.

### Orchestrator's own `ReflectorOutput`

When workers are used, the orchestrator still ends by producing a `ReflectorOutput`, because that is how the RR agent session terminates.

For orchestrated runs, that output is not the source of per-item reflections. Its role is:

- cross-cluster synthesis
- batch-level metadata
- observability / debugging

It should be preserved in orchestration metadata, for example under `raw["orchestration_summary"]` or an equivalent field, but final per-item merge must come from validated worker outputs.

If the orchestrator chooses direct analysis and does not spawn workers, then its own output remains the primary batch output and must include valid `raw["items"]`.

---

## No Silent Fallbacks

Orchestration mode forbids hidden repair paths.

These behaviors are explicitly disallowed:

- duplicating one generic reflection across a worker's assigned traces
- creating empty placeholder reflections for missing traces
- silently ignoring failed worker assignments
- silently resolving overlaps with last-write-wins semantics
- silently accepting malformed `raw["items"]`

Instead:

1. Invalid worker output is recorded as `status="invalid"` with explicit `issues`.
2. Missing coverage remains missing.
3. Overlapping assignments are rejected or surfaced explicitly.
4. Final merge is blocked until the orchestrator resolves all issues.

If the session ends with any of the following unresolved:

- missing trace coverage
- overlapping accepted coverage
- failed worker assignments
- invalid worker output shape

then `_run_batch_reflections()` must fail loudly with a diagnostic error. It must not fabricate usable-looking reflections.

---

## Final Validation and Merge

### Validation rules

Before returning from batch RR:

1. Every original trace index must be covered exactly once by a completed, valid worker result, or by the orchestrator's own direct batch output if no workers were used.
2. Every completed worker result must have `len(per_item_reflections) == len(assignment.trace_indices)`.
3. Every per-item reflection must correspond to the worker's assigned indices in order.

### Merge behavior

Only after validation passes:

```python
def _merge_cluster_results(
    self,
    deps: RRDeps,
    batch_items: list[Any],
) -> tuple[ReflectorOutput, ...]:
    """Merge validated cluster results into original item order."""
```

This method:

- preallocates a result array of length `len(batch_items)`
- writes each validated per-item reflection into its original position
- raises if any slot is missing or duplicated

There is no skipped-item placeholder path.

### Direct-analysis path

If the orchestrator chooses not to spawn workers, the existing batch RR output path remains valid:

- the orchestrator produces `raw["items"]`
- `_split_batch_reflection(...)` is used
- output shape must still be validated

---

## Concurrency Model

### Bounded worker pool

Worker RR sessions are launched through a `ThreadPoolExecutor` with:

```python
max_workers = self.config.max_cluster_workers
```

This gives the desired queue semantics:

- up to `max_cluster_workers` workers run in parallel
- additional assignments wait in the executor queue
- queued work starts automatically when a running worker finishes

That is the intended runtime behavior:

- the orchestrator can queue more work than can run immediately
- the runtime enforces the parallelism cap
- finished slots are reused automatically without extra orchestration logic

### Recommended manager behavior

The orchestrator should generally work in waves:

1. spawn a bounded set of assignments
2. collect results
3. inspect and evaluate them
4. decide whether to respawn, split, or finish

The design permits queueing more than one wave at once, but that is not the recommended prompting strategy.

### Worker sub-agents

Worker concurrency should start conservative:

- `worker_enable_subagent = False` by default, or
- `worker_subagent_max_parallel` should be much lower than the orchestrator's own `subagent_max_parallel`

This keeps cost and provider rate-limit pressure predictable.

### Local extraction parallelism

Separate from worker-session parallelism, each sandbox already supports `parallel_map(...)` for local Python work.

The orchestration design should expose conservative controls for that path too, so extraction-heavy code can run in parallel without creating many extra RR tool iterations.

Recommended runtime knobs:

- `local_parallel_max_concurrency`
- `local_parallel_timeout`

These values should be passed into `TraceSandbox(...)` when orchestrator and worker sandboxes are created.

### Pool lifecycle

- The pool is created lazily on first `spawn_analysis(...)`.
- The pool stays alive across multiple spawn/collect cycles in the same orchestrator session.
- The pool is shut down in `_run_reflection_session(..., is_orchestrator=True)` cleanup.
- Cleanup harvests any completed worker results and records unresolved work explicitly.

---

## Factored Session Helper

Add a shared private helper in `ace/rr/runner.py`:

```python
def _run_reflection_session(
    self,
    *,
    traces: Any,
    skillbook: Any,
    agent: PydanticAgent[RRDeps, ReflectorOutput],
    max_llm_calls: int,
    config: RecursiveConfig,
    is_orchestrator: bool = False,
    assignment: dict[str, Any] | None = None,
    inherited_helpers: dict[str, dict[str, str]] | None = None,
) -> tuple[ReflectorOutput, RRDeps]:
```

This helper owns the full RR session lifecycle:

1. build traces dict if needed
2. create sandbox
3. inject inherited helpers
4. build `RRDeps`
5. build the correct prompt
6. run the agent with usage limits
7. enrich output metadata
8. clean up worker pool if this is the orchestrator session

The key requirement is reuse:

- top-level batch RR
- direct orchestrator analysis
- worker RR sessions

must all go through the same session runner so behavior stays consistent.

Per-run state is returned in `deps` and never stored on `self`.

---

## Config Changes

Remove the threshold field from the earlier draft.

Add or retain the following orchestration-focused settings:

```python
@dataclass
class RecursiveConfig:
    # Existing RR fields...

    orchestrator_max_llm_calls: int = 50
    max_cluster_workers: int = 5
    worker_collect_timeout: float = 120.0
    worker_enable_subagent: bool = False
    worker_subagent_max_parallel: int = 2
    local_parallel_max_concurrency: int = 8
    local_parallel_timeout: float | None = 30.0
```

Notes:

- There is no `orchestration_threshold`.
- Strategy is chosen by the orchestrator, not by a fixed cutoff.
- Worker budgets are estimated from assignment size and serialized sub-batch size inside runtime code, not by a threshold gate.
- `worker_collect_timeout` bounds how long orchestration waits for a worker result; it is not a guarantee of hard cancellation.
- local extraction parallelism is configured separately from worker RR session parallelism.

If additional knobs are needed later, they should control budgets and concurrency, not whether orchestration is allowed.

---

## Prompt Changes

Add a dedicated orchestration prompt section in `ace/rr/prompts.py`.

It should tell the orchestrator:

- you are managing a batch analysis task
- decide whether to work directly or delegate
- create explicit worker assignments with clear goals
- collect and evaluate worker outputs before finalizing
- do not rely on missing or malformed outputs
- unresolved coverage or invalid results must be fixed before finishing

The prompt should also state clearly:

- workers inherit helper definitions
- workers cannot spawn further worker sessions
- final acceptance depends on validated per-item coverage
- use `parallel_map(...)` for extraction-only work when scanning many traces locally

---

## Changes to `RRDeps`

Extend `RRDeps` with orchestration state:

```python
@dataclass
class RRDeps:
    sandbox: TraceSandbox
    trace_data: dict[str, Any]
    skillbook_text: str
    config: RecursiveConfig
    iteration: int = 0
    sub_agent: PydanticAgent[SubAgentDeps, str] | None = None
    sub_agent_history: list[dict[str, Any]] = field(default_factory=list)

    # Orchestration state
    is_orchestrator: bool = False
    pending_clusters: dict[str, dict[str, Any]] = field(default_factory=dict)
    cluster_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    cluster_pool: ThreadPoolExecutor | None = None
```

Only orchestrator sessions populate these fields.

---

## File Changes

| File | Action | What changes |
|---|---|---|
| `ace/rr/config.py` | Edit | Remove threshold-gating idea, add orchestration concurrency settings |
| `ace/rr/agent.py` | Edit | Extend `RRDeps`, register `spawn_analysis` and `collect_results` on orchestrator agent only |
| `ace/rr/runner.py` | Edit | Factor `_run_reflection_session`, add slim batch prompt path, add `_slice_batch_trace`, strict validation, strict merge, bounded worker pool |
| `ace/rr/prompts.py` | Edit | Add manager-style orchestration prompt section |
| `docs/design/RR_DESIGN.md` | Edit | Reference this doc for manager-style batch RR |
| `tests/test_rr_pipeline/test_orchestration.py` | New | Add orchestration tests |

**Not touched:** `pipeline/`, `ace/core/`, `ace/steps/`

---

## Testing

### Unit tests

- batch sessions always use the orchestrator-capable RR path
- single-trace sessions keep the existing RR path
- batch-only agents are created lazily on first batch use
- `spawn_analysis` rejects out-of-range indices
- `spawn_analysis` rejects overlapping active assignments
- `_slice_batch_trace` preserves original batch container shape
- worker assignments preserve helper definitions
- queued worker assignments start only when pool capacity is available
- `collect_results` marks over-time workers explicitly instead of blocking indefinitely
- `collect_results` records full assignment and worker output
- malformed worker output is marked `invalid`, never repaired silently
- missing coverage causes final validation failure
- overlapping accepted coverage causes final validation failure
- orchestrator output is preserved as orchestration metadata, not merged as per-item output
- `parallel_map(...)` can be used for extraction-only work without extra RR tool iterations
- per-run orchestration state is not stored on `self`

### Integration tests

- small batch: orchestrator chooses direct analysis and returns valid per-item output
- medium batch: orchestrator spawns workers, reviews results, and merges valid coverage
- retry flow: orchestrator inspects a weak result and respawns a narrower assignment

### Stress tests

- large batch with bounded worker pool produces complete validated coverage
- no worker-pool leakage after orchestrator session ends
- concurrent pipeline runs do not interfere with each other

---

## What This Does Not Change

- `RRStep` remains a single pipeline step
- `UpdateStep` remains serialized
- `SkillManager` still receives `tuple[ReflectorOutput, ...]`
- single-trace RR behavior remains unchanged
- the pipeline engine remains unchanged

---

## Future Work

If orchestration becomes the dominant production mode, a map-reduce step pattern may eventually be cleaner at the pipeline level. For now, keeping the design inside `ace/rr/` is the right first step:

- it preserves the current public runner API
- it validates the manager/worker pattern with real workloads
- it avoids premature changes to `pipeline/`

Known v1 limitation:

- accepted coverage is exact-once only, so the same trace cannot be deliberately assigned to multiple worker angles at the same time

That is the right default for a strict first version, but a later design may want explicit multi-angle assignments with a second-stage reconciliation step.
