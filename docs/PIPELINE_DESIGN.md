# Pipeline Architecture Design

Design decisions for the generalized pipeline system.

---

## Core Primitives

Everything in the framework composes from three primitives:

```
Sequential:  A → B → C
Branch:      A → (B ∥ C) → D    (fork + implicit join)
Pipeline:    a step that is itself a pipeline (nesting / reuse)
```

---

## Step

A `Step` is the smallest unit of work. It receives a `StepContext`, does one focused thing, and returns the context.

```python
class MyStep:
    requires = {"agent_output"}   # fields it reads
    provides = {"reflection"}     # fields it writes

    def __call__(self, ctx: StepContext) -> StepContext:
        ...
        return ctx
```

Rules:
- Always synchronous within its own execution
- Must declare `requires` and `provides` — the pipeline validates ordering at construction time
- Steps declare their own parallelism constraints (see below)

### Step protocol

For static type checking, the framework exposes a `typing.Protocol`:

```python
from collections.abc import Set as AbstractSet
from typing import Protocol, runtime_checkable

@runtime_checkable
class StepProtocol(Protocol):
    requires: AbstractSet[str]
    provides: AbstractSet[str]

    def __call__(self, ctx: StepContext) -> StepContext: ...
```

`AbstractSet[str]` accepts both `set` and `frozenset` — steps declare plain set literals; the pipeline normalizes them to `frozenset` at construction time before doing any contract validation. `Pipeline` and `Branch` both satisfy this protocol, so they can be nested wherever a `Step` is expected without extra annotation. `@runtime_checkable` lets the pipeline validator use `isinstance(step, StepProtocol)` at construction time to give a clear error if a step is missing required attributes, rather than failing at call time.

### StepContext — immutability contract

`StepContext` is a frozen dataclass. Steps never mutate the incoming context — they return a new one via `.replace()`:

```python
from types import MappingProxyType

@dataclass(frozen=True)
class StepContext:
    sample: Any
    agent_output: str | None = None
    reflection: str | None = None
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        # Ensures mutation is a hard runtime error even if caller passes a plain dict
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))

    def replace(self, **changes) -> "StepContext":
        return dataclasses.replace(self, **changes)
```

Updating metadata follows the same immutable pattern as any other field:

```python
return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "key": value}))
```

Steps follow this pattern:

```python
def __call__(self, ctx: StepContext) -> StepContext:
    result = self.agent.run(ctx.sample)
    return ctx.replace(agent_output=result)
```

`frozen=True` makes mutation a hard error at runtime rather than a subtle bug. It also makes `Branch` safe by default — since `StepContext` is immutable, all branches can receive the same object without risk; no deep copy is needed.

**Field naming rule:** Named fields (`agent_output`, `reflection`) are reserved for concepts shared across all ACE pipelines. Integration-specific data always goes in `metadata`. This prevents the base class from accumulating fields over time as integrations are added.

---

## Pipeline

A `Pipeline` is an ordered list of steps that runs sequentially for a single input. It also satisfies the `Step` protocol, so it can be embedded inside another pipeline.

```python
pipe = Pipeline([
    AgentStep(),
    EvaluateStep(),
    ReflectStep(),
    UpdateStep(),
])
```

**Fluent builder API (preferred):**

```python
pipe = (
    Pipeline()
    .then(AgentStep())
    .then(EvaluateStep())
    .then(ReflectStep())
    .then(UpdateStep())
)
```

**Fan-out across contexts:**

```python
pipe.run(contexts, workers=4)   # same pipeline, N contexts in parallel
```

### Inner pipeline as a fan-out step

A `Pipeline`-as-`Step` receives one context and must return one context — but nothing prevents it from internally expanding to multiple sub-inputs. This is the **map-reduce step** pattern:

```python
class MultiSearchStep:
    """Generates N queries from one context, runs them in parallel, merges."""
    def __call__(self, ctx: StepContext) -> StepContext:
        queries = generate_queries(ctx.sample)                          # 1 → N
        sub_ctxs = [StepContext(sample=q) for q in queries]
        sub_pipe = Pipeline().then(FetchStep())
        results = sub_pipe.run(sub_ctxs, workers=len(queries))         # parallel
        return ctx.replace(agent_output=merge(results))                 # N → 1
```

`sub_pipe.run()` is a top-level runner call, so `async_boundary` and `workers` on its inner steps fire normally. From the outer pipeline's perspective, `MultiSearchStep` is a black box that takes one context and returns one context — the fan-out is an internal implementation detail.

### requires/provides for nested pipelines

When a `Pipeline` is used as a `Step` inside another pipeline, its `requires` and `provides` are computed automatically at construction time from its inner steps — no manual annotation needed.

```python
class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        self.requires, self.provides = self._infer_contracts(steps)

    @staticmethod
    def _infer_contracts(steps):
        provided_so_far = set()
        external_requires = set()
        for step in steps:
            external_requires |= step.requires - provided_so_far
            provided_so_far |= step.provides
        return frozenset(external_requires), frozenset(provided_so_far)
```

- `requires` = everything the pipeline needs from the outside (what its first steps need that no earlier inner step provides)
- `provides` = union of everything any inner step writes

The outer pipeline validates against these aggregated values at construction time, so nesting never breaks the contract.

**Deliberate constraint:** `_infer_contracts` assumes all `Branch` children always run. It has no concept of conditional branches where only some children execute. If one branch provided a field that a later step required but other branches did not, static validation would pass while the pipeline could fail at runtime. Conditional branching — where a branch may or may not run depending on context — is out of scope; all branches in a `Branch` are always executed.

---

## Branch

A `Branch` is a step that runs multiple pipelines in parallel and joins before returning. It is just a `Step` — no special pipeline mode needed.

```python
pipe = (
    Pipeline()
    .then(AgentStep())
    .then(EvaluateStep())
    .branch(
        Pipeline().then(ReflectStep()),
        Pipeline().then(LogStep()),
    )
    .then(UpdateStep())   # only runs after both branches complete
)
```

`wait` is implicit — any step after a `Branch` waits for all branches to finish.

### Context merging

Each branch receives the same context reference. Since `StepContext` is frozen, no copy is needed — branches cannot mutate what they receive. When all branches complete, their output contexts are merged back into one before the next step runs.

The merge function receives the list of output contexts and returns a single context:

```python
Branch(
    Pipeline().then(ReflectStep()),
    Pipeline().then(LogStep()),
    merge=lambda ctxs: dataclasses.replace(
        ctxs[0],
        metadata={**ctxs[0].metadata, **ctxs[1].metadata}
    )
)
```

**Built-in merge strategies:**

| Strategy | Behaviour |
|---|---|
| `raise_on_conflict` | raises if two branches write the same field — safe default, no silent data loss |
| `last_write_wins` | last branch's value wins on conflict — simple but lossy |
| `namespaced` | branches write to `ctx.metadata["branch_0"]` etc., no conflict possible |
| custom `merge=fn` | `fn(ctxs: list[StepContext]) -> StepContext` — full control |

The actual default when no `merge=` argument is passed is `raise_on_conflict`. The constructor signature makes this explicit:

```python
def __init__(self, *pipelines, merge=MergeStrategy.RAISE_ON_CONFLICT):
    ...
```

In practice, branches that write disjoint fields (e.g. Reflect writes `reflection`, Log writes `metadata["log"]`) never conflict and the merge is a no-op — `raise_on_conflict` passes through without raising.

---

## Async Behavior

"Async" means three different things in this framework, operating at different levels. It is important to keep them separate — they solve different problems.

| Type | Level | Problem it solves |
|---|---|---|
| Async step | single step | don't block the thread during I/O |
| `async_boundary` | across samples | start the next sample before the current one finishes |
| Branch parallelism | within one sample | run independent work simultaneously on the same data |

---

### 1. Async steps — non-blocking I/O

**Problem:** A step makes a network call (LLM API, HTTP, subprocess). It should not block the thread while waiting for a response.

**Solution:** Define the step as a coroutine. The pipeline detects this automatically and awaits it. Sync steps get wrapped with `asyncio.to_thread()` so they are safe in an async context too.

```python
# Sync step — no changes needed
class AgentStep:
    def __call__(self, ctx: StepContext) -> StepContext: ...

# Async step — native coroutine, awaited by the pipeline
class BrowserExecuteStep:
    async def __call__(self, ctx: StepContext) -> StepContext: ...
```

```python
# Pipeline runner — handles both transparently
for step in self.steps:
    if asyncio.iscoroutinefunction(step.__call__):
        ctx = await step(ctx)
    else:
        ctx = await asyncio.to_thread(step, ctx)
```

Pipeline entry points: `pipe.run(contexts)` for sync callers, `await pipe.run_async(contexts)` for async callers (e.g. inside browser-use).

This type is about **not blocking**. Nothing runs in parallel — the pipeline is still sequential, it just yields the thread during waits.

---

### 2. async_boundary — pipeline across samples

**Problem:** Reflect and Update are slow (LLM calls). If we wait for them before starting the next sample, throughput is poor. We want to fire them off and immediately move to sample N+1.

**Solution:** A step declares `async_boundary = True`. Everything from that step onwards runs in a background executor. The pipeline loop does not wait — it moves straight to the next sample.

```python
class ReflectStep:
    async_boundary = True   # hand off to background from here
    max_workers = 3         # up to 3 reflections running in parallel

class UpdateStep:
    max_workers = 1         # must serialize — writes to shared skillbook
```

```
sample 1:  [Agent] [Evaluate] ──fire──► [Reflect] [Update]  (background)
sample 2:  [Agent] [Evaluate] ──fire──► [Reflect] [Update]  (background)
sample 3:  [Agent] [Evaluate] ...
                              ↑
                        async_boundary
```

This type is about **throughput**. Multiple samples are in-flight simultaneously, at different stages of the pipeline. The caller only waits for steps before the boundary.

Note: `max_workers` controls how many background instances of a step run concurrently. Steps that write shared state (like `UpdateStep`) must use `max_workers = 1` to avoid races.

**Background pool is per step class, shared across pipeline instances.** `ReflectStep.max_workers = 3` means a single pool of 3 threads for all `ReflectStep` instances. This avoids pool proliferation and makes `max_workers` a straightforward capacity knob independent of how many pipelines are running.

**Pool lifecycle:** The `ThreadPoolExecutor` for each step class is created lazily at first use (not at class definition or pipeline construction) and persists for the process lifetime. Callers that need explicit cleanup can call `StepClass._executor.shutdown(wait=True)`. If two users of the same step class need different concurrency limits (e.g. different LLM backends behind the same step type), they should subclass rather than share the class attribute.

**Boundary rules:**
- The **first** step with `async_boundary = True` is the handoff point. Only one boundary per pipeline.
- If multiple steps in the same pipeline declare `async_boundary = True`, the pipeline raises `PipelineConfigError` at construction time. A duplicate boundary is almost always a copy-paste mistake, not a deliberate choice.
- `async_boundary` inside a `Branch` child pipeline raises `PipelineConfigError` at construction time. Branch children always block until joined; detaching mid-branch is incoherent and there is no valid interpretation.
- `async_boundary` inside a `Pipeline`-as-`Step` raises a **warning** at construction time (not an error). When a pipeline is used as a step inside another pipeline, there is no "next sample" to move to — the outer pipeline is blocked waiting for the inner one to return a context. The boundary is ignored and the inner pipeline runs fully synchronously. The warning surfaces this declared intent being ignored so callers can investigate. The same pipeline definition works both as a top-level runner (where `async_boundary` fires) and as a nested step (where it warns and is ignored) — no reconfiguration needed.

---

### 3. Branch parallelism — concurrent work on the same sample

**Problem:** Two independent steps could run at the same time on the same sample (e.g. reflect and log), but a linear pipeline forces them to be sequential.

**Solution:** `Branch` forks the context, runs each sub-pipeline in parallel, then joins before the next step. In sync mode it uses `ThreadPoolExecutor`; in async mode it uses `asyncio.gather()`.

```python
pipe = (
    Pipeline()
    .then(EvaluateStep())
    .branch(
        Pipeline().then(ReflectStep()),   # runs in parallel
        Pipeline().then(LogStep()),       # runs in parallel
    )
    .then(UpdateStep())   # waits for both branches
)
```

```python
# Branch internals (async mode)
async def __call__(self, ctx: StepContext) -> StepContext:
    results = await asyncio.gather(
        *[p(ctx) for p in self.pipelines],
        return_exceptions=True,   # all branches run to completion even if one fails
    )
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise BranchError(failures)   # caller sees all branch failures, not just the first
    return self.merge(results)
```

`return_exceptions=True` is required for consistent error handling: without it, the first branch failure cancels all remaining branches and the `SampleResult` would silently drop their work. With it, all branches complete and the runner captures the full failure set.

This type is about **latency within a single sample**. Nothing moves to the next sample — the pipeline waits for the join before continuing.

---

### Rule of thumb

| Question | Answer |
|---|---|
| Does the step wait on I/O? | `async def __call__` |
| Do I want to process more samples while previous ones are still learning? | `async_boundary` on the step where the handoff happens |
| Can two steps on the same sample run simultaneously? | `Branch` |
| Do I want N samples going through the pipeline at the same time? | `workers=N` on `run()` |

Each mechanism is independent. They compose freely — you can have async steps inside branches, behind an `async_boundary`, run with multiple workers.

---

## Concurrency Model

Parallelism is declared on the **step**, not the pipeline. The pipeline executor reads these at runtime:

```python
class ReflectStep:
    async_boundary = True   # hand off to background threads from here
    max_workers = 3         # up to 3 running in parallel

class UpdateStep:
    max_workers = 1         # must serialize (writes to shared skillbook)
```

**Fan-out (same step, different samples):**
Controlled by `max_workers` on the step. Each step class has a single shared `ThreadPoolExecutor` — `ReflectStep.max_workers = 3` means one pool of 3 threads regardless of how many pipeline instances are running.

**Pipeline split (pipelining across samples):**
`async_boundary = True` on a step tells the runner to hand off everything from that step onwards to background threads, freeing the caller to start the next sample immediately.

```
sample 1:  [AgentStep] [EvaluateStep] ──► [ReflectStep] [UpdateStep]
sample 2:  [AgentStep] [EvaluateStep] ──► ...             (background)
                                      ↑
                               async_boundary
```

This replaces the hardcoded `steps[:2]` / `steps[2:]` split that existed in the old `AsyncLearningPipeline`.

### workers vs max_workers — independent pools

These two knobs control different thread pools and do not interact:

| Knob | Pool | Controls |
|---|---|---|
| `pipe.run(contexts, workers=N)` | foreground pool | how many contexts run through pre-boundary steps simultaneously |
| `step.max_workers = K` | background pool per step class | how many instances of that step run in the background simultaneously |

A sample leaves the foreground pool when it crosses the `async_boundary` point and enters the background step's pool. With `workers=4` and `ReflectStep.max_workers=3`, you can have 4 samples in Agent/Evaluate and 3 reflections running concurrently — two separate pools, no multiplication.

Mental model: `workers` controls throughput *into* the pipeline; `max_workers` controls throughput *through* each slow background step.

**LLM rate limits:** `workers` and `max_workers` are independent pools, but total concurrent outbound LLM calls = foreground calls + background calls. With `workers=4` and `ReflectStep.max_workers=3`, up to 7 LLM requests may be in-flight simultaneously. Account for this when configuring per-provider rate limits.

---

## Error Handling

Failure semantics differ depending on which side of the `async_boundary` a step is on.

**Foreground steps** (before the boundary): the runner catches exceptions per sample and records them in a `SampleResult`. The pipeline then moves to the next sample.

```python
# Pipeline runner (foreground loop)
for ctx in contexts:
    try:
        for step in self.foreground_steps:
            ctx = step(ctx)
        self._submit_to_background(ctx)
        results.append(SampleResult(sample=ctx.sample, output=ctx, error=None, failed_at=None))
    except Exception as e:
        results.append(SampleResult(sample=ctx.sample, output=None, error=e, failed_at=type(step).__name__))
```

**Background steps** (after the boundary): the caller has already moved on, so exceptions cannot propagate. Background failures are captured and attached to the `SampleResult` — nothing is dropped silently.

```python
@dataclass
class SampleResult:
    sample: Any
    output: StepContext | None     # None if a step failed
    error: Exception | None        # set if any step failed
    failed_at: str | None          # name of the step class that failed
    cause: Exception | None = None # for BranchError: the inner step exception
```

Every sample produces a result — either successful with `output` set, or failed with `error` and `failed_at` set. After `run()` completes (or after `wait_for_learning()`), callers can inspect results for failures.

When a `Branch` step fails, `failed_at` is `"Branch"` and `error` is a `BranchError`. `cause` carries the inner exception from the failing branch so callers can see which inner step actually failed, not just the outer wrapper.

Retry logic is the responsibility of individual steps, not the pipeline.

**Shutdown:** `wait_for_learning(timeout=N)` raises `TimeoutError` if background steps have not drained within `N` seconds. Individual step implementations are responsible for their own per-call timeouts (e.g. LLM API call timeouts).

---

## Integrations as Pipelines

Each external framework integration (browser-use, LangChain, Claude Code) is its own `Pipeline` subclass with integration-specific steps. It is **not** embedded as a step inside `ACEPipeline`.

```
ace/integrations/
  browser_use/
    pipeline.py          ← BrowserPipeline
    steps/
      execute.py         ← BrowserExecuteStep
  langchain/
    pipeline.py          ← LangChainPipeline
    steps/
      execute.py         ← LangChainExecuteStep
  claude_code/
    pipeline.py          ← ClaudeCodePipeline
    steps/
      execute.py         ← ClaudeCodeExecuteStep
      persist.py         ← PersistStep
```

Each integration pipeline replaces `AgentStep + EvaluateStep` with its own execute step, then reuses the shared `ReflectStep` and `UpdateStep`:

```python
BrowserPipeline:
  [BrowserExecuteStep, ReflectStep, UpdateStep]

LangChainPipeline:
  [LangChainExecuteStep, ReflectStep, UpdateStep]

ClaudeCodePipeline:
  [ClaudeCodeExecuteStep, ReflectStep, UpdateStep, PersistStep]
```

---

## Generic Steps Folder

`ace/pipeline/steps/` contains only steps that are reusable across any pipeline — one file per class:

```
ace/pipeline/steps/
  __init__.py
  agent.py         ← AgentStep
  evaluate.py      ← EvaluateStep
  reflect.py       ← ReflectStep
  update.py        ← UpdateStep
```

Integration-specific steps live next to their pipeline, not here.

---

## Summary Table

| Concept | Unit | Threading | Communication |
|---|---|---|---|
| `Step` | single unit of work | always sync | via `StepContext` |
| `Pipeline` | ordered step list for one input | `workers=N` across inputs | via `StepContext` |
| `Branch` | parallel pipeline list | always parallel internally | copy + merge of `StepContext` |
| `Pipeline` as a `Step` | reuse / nesting | inherits parent context | via `StepContext` |

---

## What Was Rejected and Why

**`PipelineProcess` (external wrapper):**
Adding a separate class to wrap pipelines with executor/queue machinery was considered. Rejected — it adds an indirection layer without benefit for this project's use case. Concurrency is declared on steps instead.

**Special async pipeline subclass:**
Having an `AsyncPipeline` type was considered. Rejected — it mixes sequential logic with concurrency concerns in the same class. The `async_boundary` marker on steps is data-driven and doesn't require subclassing.

**Full DAG executor (auto-inferred parallelism):**
The `requires`/`provides` graph already contains enough information to infer which steps can run in parallel. Deferred — `Branch` covers the explicit fork/join case; automatic DAG inference can be added later if needed.

**Alternative `requires`/`provides` declaration styles:**
Four alternatives to plain set class attributes were considered:

- `__init_subclass__` keyword args (`class MyStep(Step, requires={"agent_output"})`): moves the declaration to the class header but requires inheriting from a base `Step` class, eliminating the structural Protocol advantage — any object with the right attributes is a step without needing to inherit anything.
- `ClassVar` annotations (`requires: ClassVar[frozenset[str]] = ...`): more type-checker friendly but adds verbosity with no semantic change.
- Function decorator wrapping `__call__`: removes class boilerplate for stateless steps but introduces two styles (decorated functions vs classes with collaborators like `self.reflector`), inconsistency not worth the reduction.
- Decomposed signature / Hamilton-style (steps receive named fields as parameters instead of `StepContext`): elegant zero-annotation contracts — `requires` and `provides` are inferred from function signature at zero cost. Rejected because it loses explicit ordering control (order is inferred from data dependencies, not declared; independent steps have undefined order), collapses the two-tier `StepContext`/`metadata` structure into a flat dict (integration-specific data collides with shared fields), and makes side-effect steps with no consumed output impossible to anchor in the sequence.

Plain set class attributes with pipeline normalization to `frozenset` at construction time is the right balance: explicit, readable, no inheritance required, and the ordering and context model stay intact.

---

## External Libraries Considered

This pattern is known as **Pipes and Filters**. Several open source libraries implement variants of it. None were adopted — reasons below.

**[Kedro](https://kedro.org/)** — closest to the `requires`/`provides` model. Nodes declare explicit named inputs and outputs; pipelines are composable. The gap: requires a "data catalog" abstraction for named datasets, has no `async_boundary` concept, and is oriented toward ML/ETL rather than agentic loops. Fighting the data catalog to pass a `StepContext` would cost more than writing the primitives cleanly.

**[Hamilton](https://github.com/dagworks-inc/hamilton)** — lightest-weight equivalent. Functions declare inputs as parameters and outputs as return types; the framework infers the DAG. No server, no UI. The gap: no built-in async boundary, no fork/join `Branch`, no per-step `max_workers`. Gets contract validation for free but requires building all concurrency from scratch anyway.

**[Pypeln](https://github.com/cgarciae/pypeln)** — designed for exactly the "process N samples through concurrent stages" problem. Has sync, thread, and async modes. The gap: no typed contracts, no `Branch`, no nested pipelines. Gets the `async_boundary`-style throughput but not the structural guarantees.

**[Dagster](https://dagster.io/)** — closest overall feature set. Ops (≈ Steps) with typed inputs/outputs, jobs (≈ Pipelines), graph-based branching. The gap: it is a platform, not a library. Brings a scheduler, UI, asset catalog, and significant operational overhead. Too heavy to embed inside ACE.

**Conclusion:** The specific combination of `async_boundary`, per-step `max_workers`, `Pipeline`-as-`Step` nesting, and `SampleResult` error wrapping is not provided by any of the above out of the box. Adapting any of them would cost as much as writing the ~300-line core cleanly.

**What is borrowed rather than written:** `concurrent.futures.ThreadPoolExecutor` for the background step pools, and `asyncio.gather` (or `anyio` task groups) for `Branch` internals. These are well-tested primitives that are not reinvented.
