# Pipeline Implementation TODO

> **Strategy**: Build the new architecture incrementally in two new top-level packages
> (`pipeline/` and `ace2/`). The existing `ace/` folder is left **untouched** throughout.
> Once everything works and tests pass, `ace2/` can be renamed to `ace/`.

Two top-level packages with a strict one-way dependency:

```
pipeline/    Pure execution engine. No ACE concepts. Zero upward dependencies.
ace2/        ACE domain logic. Imports from pipeline/, nothing else imports ace2/.
```

```
pipeline/                        ace2/
  __init__.py                      steps/
  context.py    <──────────────      __init__.py
  protocol.py   <──────────────      agent.py
  errors.py     <──────────────      evaluate.py
  pipeline.py   <──────────────      reflect.py
  branch.py     <──────────────      update.py
                                   pipelines/
                                     __init__.py
                                     factories.py
                                     offline.py
                                     online.py
                                   integrations/
                                     browser_use/
                                     langchain/
                                     claude_code/
                                   skillbook.py   (copy/adapt from ace/)
                                   roles.py       (copy/adapt from ace/)
                                   ...
```

Work strictly bottom-up — `pipeline/` first, then `ace2/steps/`, then `ace2/pipelines/`, then integrations.

---

## Part 1 — `pipeline/` (generic engine)

No imports from `ace2/`. No knowledge of agents, skillbooks, or environments.

### 1.1 `pipeline/errors.py`

- [ ] `PipelineOrderError(Exception)` — step requires a field no earlier step provides
- [ ] `PipelineConfigError(Exception)` — invalid pipeline wiring (multiple `async_boundary`, boundary inside Branch)
- [ ] `BranchError(Exception)` — carries `list[BaseException]` of all branch failures

### 1.2 `pipeline/context.py`

- [ ] `StepContext` as `@dataclass(frozen=True)`
- [ ] Named fields for cross-pipeline concepts: `sample`, `skillbook`, `environment`, `epoch`, `total_epochs`, `step_index`, `total_steps`, `recent_reflections`, `agent_output`, `environment_result`, `reflection`, `skill_manager_output`
- [ ] `metadata: MappingProxyType` — `__post_init__` coerces plain dict; integration-specific data always goes here
- [ ] `replace(**changes) -> StepContext` — thin wrapper over `dataclasses.replace`

### 1.3 `pipeline/protocol.py`

- [ ] Import `collections.abc.Set as AbstractSet`
- [ ] `@runtime_checkable class StepProtocol(Protocol)`:
  - `requires: AbstractSet[str]`
  - `provides: AbstractSet[str]`
  - `__call__(ctx: StepContext) -> StepContext`
- [ ] `@dataclass class SampleResult`:
  - `sample: Any`
  - `output: StepContext | None`
  - `error: Exception | None`
  - `failed_at: str | None` — step class name that raised
  - `cause: Exception | None = None` — inner exception when `failed_at == "Branch"`

### 1.4 `pipeline/branch.py`

- [ ] `MergeStrategy` enum: `RAISE_ON_CONFLICT`, `LAST_WRITE_WINS`, `NAMESPACED`
- [ ] Built-in merge function for each strategy
- [ ] `Branch(*pipelines, merge=MergeStrategy.RAISE_ON_CONFLICT)` satisfying `StepProtocol`:
  - `requires`/`provides` inferred from union of all child pipelines
  - `__call__` sync: `ThreadPoolExecutor` fan-out, collect all, merge or raise `BranchError`
  - `__call__` async: `asyncio.gather(..., return_exceptions=True)`, merge or raise `BranchError`
  - All branches always run to completion before raising (never cancel on first failure)
  - `failed_at = "Branch"`, `cause` = inner exception on failure

### 1.5 `pipeline/pipeline.py`

- [ ] `Pipeline` is a **concrete class** (not abstract), satisfies `StepProtocol`
- [ ] `__init__(steps=[])` — normalizes `requires`/`provides` to `frozenset` at construction
- [ ] `_infer_contracts(steps) -> (frozenset, frozenset)` static method
- [ ] `_validate_steps(steps)` — raises `PipelineOrderError` for unsatisfied deps; raises `PipelineConfigError` for:
  - More than one `async_boundary = True` step in the same pipeline
  - Any `async_boundary = True` step inside a `Branch` child
  - Emits a warning (not error) when `async_boundary` is declared on a `Pipeline` used as a nested step (boundary is ignored)
- [ ] `then(step) -> Pipeline` — fluent builder, returns `self`
- [ ] `branch(*pipelines, merge=...) -> Pipeline` — appends a `Branch`, returns `self`
- [ ] `__call__(ctx) -> StepContext` — sequential runner satisfying `StepProtocol` for nesting:
  - Detects `asyncio.iscoroutinefunction(step.__call__)` — awaits async steps, wraps sync with `asyncio.to_thread`
- [ ] `run(samples, workers=1) -> list[SampleResult]` — foreground fan-out via `ThreadPoolExecutor`:
  - Splits at `async_boundary`: foreground steps in calling thread, background steps in per-step-class pool
  - Every sample produces a `SampleResult` (success or failure — nothing dropped silently)
- [ ] `run_async(samples, workers=1) -> list[SampleResult]` — async entry point via `asyncio.gather`
- [ ] `wait_for_background(timeout: float | None = None)` — drains background pools; raises `TimeoutError` on timeout
- [ ] Background pool is per step class (class-level `ThreadPoolExecutor`), lazy, shared across pipeline instances
- [ ] `max_workers` class attribute on a step configures its pool (default: 1)
- [ ] `workers` on `run()` is a separate foreground pool — the two do not interact

### 1.6 `pipeline/__init__.py`

- [ ] Re-export: `Pipeline`, `Branch`, `MergeStrategy`, `BranchError`, `StepProtocol`, `StepContext`, `SampleResult`, `PipelineOrderError`, `PipelineConfigError`

---

## Part 2 — `ace2/steps/` (ACE-specific steps)

Imports from `pipeline/`. Knows about ACE roles (`Agent`, `Reflector`, `SkillManager`).

- [ ] Create `ace2/steps/` package
- [ ] `agent.py` → `AgentStep(agent: Agent)`
  - `requires = {"sample", "skillbook", "recent_reflections"}`, `provides = {"agent_output"}`
  - Returns `ctx.replace(agent_output=...)`
- [ ] `evaluate.py` → `EvaluateStep()`
  - `requires = {"sample", "agent_output", "environment"}`, `provides = {"environment_result"}`
  - Returns `ctx.replace(environment_result=...)`
- [ ] `reflect.py` → `ReflectStep(reflector, max_refinement_rounds, reflection_window)`
  - `requires = {"sample", "agent_output", "environment_result", "skillbook"}`, `provides = {"reflection"}`
  - Returns `ctx.replace(reflection=..., recent_reflections=...)` — no mutation
  - `async_boundary = True`, `max_workers = 3`
- [ ] `update.py` → `UpdateStep(skill_manager: SkillManager)`
  - `requires = {"reflection", "skillbook", "sample", "environment_result"}`, `provides = {"skill_manager_output"}`
  - Returns `ctx.replace(skill_manager_output=...)` — skillbook updated in-place (shared object)
  - `max_workers = 1`
- [ ] `ace2/steps/__init__.py` re-exports all four

---

## Part 3 — `ace2/pipelines/` (factory + runners)

### `ace2/pipelines/factories.py`

- [ ] `ace_pipeline(agent, reflector, skill_manager, **opts) -> Pipeline`
  - `Pipeline().then(AgentStep).then(EvaluateStep).then(ReflectStep).then(UpdateStep)`
  - Accepts `max_refinement_rounds`, `reflection_window`, passes them to step constructors

### `ace2/pipelines/offline.py`

- [ ] `OfflineACE(pipeline: Pipeline, skillbook: Skillbook)` — holds pipeline by composition
- [ ] `run(samples, environment, epochs=1, checkpoint_interval=None, checkpoint_dir=None, wait_for_background=True) -> list[SampleResult]`
  - Epoch loop; checkpointing via `skillbook.save_to_file()`
  - Calls `pipeline.wait_for_background()` — raises `TimeoutError` on timeout
- [ ] `OfflineACE.from_roles(agent, reflector, sm, **opts)` convenience constructor

### `ace2/pipelines/online.py`

- [ ] `OnlineACE(pipeline: Pipeline, skillbook: Skillbook)`
- [ ] `run(samples: Iterable, environment, wait_for_background=True) -> list[SampleResult]`
  - Single-pass stream loop
- [ ] `OnlineACE.from_roles(agent, reflector, sm, **opts)` convenience constructor

### `ace2/pipelines/__init__.py`

- [ ] Re-exports: `ace_pipeline`, `OfflineACE`, `OnlineACE`

---

## Part 4 — `ace2/integrations/` (per-framework pipeline factories)

Each integration is a subpackage with its own steps and a factory function.

### `ace2/integrations/browser_use/`

- [ ] `steps/execute.py` → `BrowserExecuteStep` (async `__call__`, wraps browser-use agent)
- [ ] `pipeline.py` → `browser_pipeline(agent, reflector, sm) -> Pipeline`
  - `Pipeline().then(BrowserExecuteStep).then(ReflectStep).then(UpdateStep)`

### `ace2/integrations/langchain/`

- [ ] `steps/execute.py` → `LangChainExecuteStep`
- [ ] `pipeline.py` → `langchain_pipeline(chain, reflector, sm) -> Pipeline`

### `ace2/integrations/claude_code/`

- [ ] `steps/execute.py` → `ClaudeCodeExecuteStep`
- [ ] `steps/persist.py` → `PersistStep`
- [ ] `pipeline.py` → `claude_code_pipeline(reflector, sm) -> Pipeline`

---

## Tests

- [ ] `tests/test_pipeline_engine.py` — unit tests for `Pipeline`, `Branch`, `StepContext`, `SampleResult`, error classes. No ACE imports — use dummy steps.
- [ ] `tests/test_ace2_steps.py` — unit tests for each ACE step in isolation (mock `Agent`, `Reflector`, `SkillManager`)
- [ ] `tests/test_ace2_pipelines.py` — integration tests for `OfflineACE` and `OnlineACE` with `DummyLLMClient`

---

## Part 5 — Cut-over (after all tests pass)

Once Parts 1–4 are done and green:

- [ ] Rename `ace2/` → `ace/` (or swap via git)
- [ ] Update all imports across codebase and examples to new paths
- [ ] Archive or remove old `ace/pipeline/` directory

---

## Build order

```
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6    pipeline/ (no deps)
2                                        ace2/steps/ (depends on Part 1)
3                                        ace2/pipelines/ (depends on Parts 1–2)
4                                        ace2/integrations/ (depends on Parts 1–2)
tests                                    written alongside each part
5                                        cut-over (after all tests pass)
```
