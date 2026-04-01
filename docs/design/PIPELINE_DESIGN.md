# Pipeline Architecture Design

Design decisions for the generalized pipeline system. Trying to keep is as generic as possible.
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
    provides = {"reflections"}    # fields it writes

    def __call__(self, ctx: StepContext) -> StepContext:
        ...
        return ctx
```

Rules:
- Always synchronous within its own execution
- Must declare `requires` and `provides` — the pipeline validates ordering at construction time
- Steps declare their own parallelism constraints (see below)

### Step protocol

For static type checking, the framework exposes a generic `typing.Protocol`:

```python
from typing import Protocol, TypeVar, runtime_checkable

Ctx = TypeVar("Ctx", bound=StepContext)

@runtime_checkable
class StepProtocol(Protocol[Ctx]):
    requires: frozenset[str]
    provides: frozenset[str]

    def __call__(self, ctx: Ctx) -> Ctx: ...
```

`StepProtocol` is generic over the context type. The base `StepProtocol` (or `StepProtocol[StepContext]`) is satisfied by `Pipeline` and `Branch`, so they can be nested wherever a step is expected. Domain-specific steps use the parameterized form — e.g. `StepProtocol[ACEStepContext]` — so that mypy validates the `__call__` signature against the concrete context subclass without needing `# type: ignore` comments.

`@runtime_checkable` lets the pipeline validator use `isinstance(step, StepProtocol)` at construction time to give a clear error if a step is missing required attributes, rather than failing at call time. The type parameter is erased at runtime, so `isinstance` checks work the same as with a non-generic protocol.

### StepContext — immutability contract

`StepContext` is a frozen dataclass. Steps never mutate the incoming context — they return a new one via `.replace()`.

The pipeline engine defines a minimal base with only two fields:

```python
from types import MappingProxyType

@dataclass(frozen=True)
class StepContext:
    sample: Any
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        # Ensures mutation is a hard runtime error even if caller passes a plain dict
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))

    def replace(self, **changes) -> "StepContext":
        return dataclasses.replace(self, **changes)
```

The engine never reads anything beyond `sample` and `metadata`. All domain-specific fields are added by subclassing.

#### Subclassing for domain fields

Consuming applications subclass `StepContext` to add named fields for concepts shared across their pipelines:

```python
@dataclass(frozen=True)
class ACEContext(StepContext):
    # Shared across all ACE pipelines
    skillbook: Skillbook | None = None
    environment: TaskEnvironment | None = None

    # Produced by steps (None until the providing step runs)
    agent_output: AgentOutput | None = None
    environment_result: EnvironmentResult | None = None
    reflections: tuple[ReflectorOutput, ...] = ()
    skill_manager_output: UpdateBatch | None = None

    # Runner bookkeeping
    epoch: int = 1
    total_epochs: int = 1
    step_index: int = 0
    total_steps: int = 0
```

The `requires`/`provides` validation works on attribute names (strings) — it checks that the field exists on the context object at runtime, so it is subclass-agnostic. A step that declares `requires = {"skillbook"}` works whether the context is `ACEContext` or any other subclass that has a `skillbook` attribute.

Data that is specific to a single integration or step goes in `metadata` to prevent field accumulation on the subclass. For example, `metadata["browser_history"]` for browser-use or `metadata["transcript_path"]` for Claude Code.

#### Immutable update patterns

Updating metadata follows the same immutable pattern as any other field:

```python
return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "key": value}))
```

Steps follow this pattern:

```python
def __call__(self, ctx: StepContext) -> StepContext:
    result = do_work(ctx.sample)
    return ctx.replace(result=result)
```

`frozen=True` makes mutation a hard error at runtime rather than a subtle bug. It also makes `Branch` safe by default — since `StepContext` is immutable, all branches can receive the same object without risk; no deep copy is needed.

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

**Shutdown:** `wait_for_background(timeout=N)` raises `TimeoutError` if background steps have not drained within `N` seconds. Individual step implementations are responsible for their own per-call timeouts (e.g. LLM API call timeouts).

**Monitoring:** `background_stats()` returns a `dict` with `active` and `completed` counts for background threads. Thread-safe — can be called from any thread while the pipeline is running. This is the public API for monitoring background progress; callers should not access `_bg_lock` or `_bg_threads` directly.

**Foreground progress:** `run()` and `run_async()` accept an optional `on_sample_done` callback (`Callable[[SampleResult], None] | None`). It fires once per context after foreground steps complete (or fail), before background steps start. The callback must not block the event loop — lightweight operations like `tqdm.update()` are fine. Defaults to `None` (no-op). This is the foreground-side complement to `background_stats()`.

---

## Pipeline Hooks

Hooks let external code observe pipeline execution without modifying data flow. They solve a different problem than steps: steps transform data (`StepContext` in, `StepContext` out), hooks observe transitions (step started, step finished).

The motivating use case is hosted/web deployments that need operational concerns — progress streaming, metrics, logging, billing — wired into the pipeline without modifying the step chain or the pipeline engine for each new concern.

### Separation of concerns

The pipeline has three distinct concerns, each with its own mechanism:

| Concern | Mechanism | Who owns it |
|---|---|---|
| Data flow | Steps (`requires`/`provides`, `__call__`) | Step author |
| Observation | Hooks (`before_step`/`after_step`) | Deployment environment |
| Lifecycle control | `cancel_token` (see Cancellation below) | Caller |

Steps own data. Hooks observe execution. Cancellation controls lifecycle. These three never overlap — a hook cannot modify context, and cancellation is not a hook.

### Hook protocol

```python
@runtime_checkable
class PipelineHook(Protocol):
    def before_step(self, step_name: str, ctx: StepContext) -> None: ...
    def after_step(self, step_name: str, ctx: StepContext) -> None: ...
```

**Design constraints:**

- **`-> None`, not `-> StepContext`** — hooks observe, they do not transform. Context flow stays exclusively in the step chain via `requires`/`provides`. This eliminates the "second communication channel" problem — hooks cannot inject data that a later step silently depends on.
- **`step_name: str`**, not the step object — hooks know what ran, but cannot call, inspect, or mutate the step instance. This prevents hooks from becoming an implicit dependency of step behavior.
- **Non-blocking** — hooks must not block the event loop. They are in the hot path between steps. Heavy work (HTTP POST, disk write) should be dispatched to a background task or queue, not done inline. Same constraint as `on_sample_done`.
- **No ordering guarantees between hooks** — hooks in the list are called sequentially in insertion order, but a hook must not depend on side effects of another hook. If ordering matters, combine them into one hook.
- **Exception isolation** — if a hook raises, the pipeline logs the error and continues. A broken metrics hook must not kill the pipeline. Hook exceptions are never surfaced in `SampleResult`.

### Pipeline integration

Hooks are set at construction time — they are structural, like steps. A pipeline's observation behavior is fixed for its lifetime.

```python
class Pipeline:
    def __init__(self, steps=None, hooks=None):
        self._hooks = list(hooks or [])
        ...
```

The step execution loop calls hooks around each foreground step:

```python
for step in foreground_steps:
    step_name = type(step).__name__
    for hook in self._hooks:
        hook.before_step(step_name, ctx)
    ctx = await step(ctx)
    for hook in self._hooks:
        hook.after_step(step_name, ctx)
```

Hooks fire for **foreground steps only**. Background steps (after `async_boundary`) do not trigger hooks — the caller has already moved on, and hook callbacks from background threads would violate the non-blocking contract. Background observability is handled via `background_stats()`.

### Branch and nesting behavior

- **Branch:** hooks fire once for the `Branch` step as a whole (`step_name = "Branch"`), not for each inner step of each child pipeline. Branch children are an internal implementation detail — hooks observe the outer pipeline's step sequence only. This keeps hook output predictable regardless of how many branches exist or how deep they nest.
- **Nested Pipeline-as-Step:** same rule. The outer pipeline fires hooks for the nested pipeline step (`step_name = "MySubPipeline"`), not for its inner steps. If the nested pipeline has its own hooks, those fire independently within its own execution.

### Example: progress streaming for a web app

```python
class ProgressHook:
    """Pushes step events to an async queue for SSE streaming."""

    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def before_step(self, step_name: str, ctx: StepContext) -> None:
        self._queue.put_nowait({"type": "step_started", "step": step_name})

    def after_step(self, step_name: str, ctx: StepContext) -> None:
        self._queue.put_nowait({"type": "step_done", "step": step_name})
```

```python
# Web endpoint wiring (not part of pipeline/)
queue = asyncio.Queue()
pipe = Pipeline(steps, hooks=[ProgressHook(queue)])
asyncio.create_task(pipe.run_async(contexts))
# SSE endpoint reads from queue
```

The hook implementation lives in the hosted deployment code, not in `pipeline/`. The pipeline engine provides the protocol and the call sites — nothing more.

---

## Cancellation

`cancel_token` lets a caller stop a running pipeline between steps. The motivating use case is a web app where the user clicks "Stop" and the server needs to halt processing without waiting for the remaining steps or samples to complete.

### CancellationToken

```python
class CancellationToken:
    """Thread-safe cancellation signal."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation. Thread-safe, idempotent."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()
```

`threading.Event` rather than `asyncio.Event` because the token must be cancellable from any thread — a web endpoint handler, a background task, a signal handler. The pipeline checks it synchronously between steps, so no async machinery is needed.

### Pipeline integration

`cancel_token` is passed per-invocation on `run()` and `run_async()`, not on `__init__`. A token is scoped to a single execution — each web request creates a fresh token. The pipeline object stays reusable across runs.

```python
pipe.run(contexts, cancel_token=token)
await pipe.run_async(contexts, cancel_token=token)
```

The runner checks the token at two points:

1. **Before each foreground step** — if cancelled, the current sample gets `error=PipelineCancelled()` and `failed_at` set to the step that would have run next.
2. **Before each new sample** — if cancelled, remaining samples are not started. Samples already in-flight (via `workers > 1`) complete their current step but are cancelled before the next one.

```python
for step in foreground_steps:
    if cancel_token is not None and cancel_token.is_cancelled:
        result.error = PipelineCancelled()
        result.failed_at = type(step).__name__
        return result
    ctx = await step(ctx)
```

### Contextvar bridge — making the token visible inside steps

The pipeline checks the token between steps.  Code *inside* a step (e.g. an LLM client making a streaming API call) may also want to check it — but steps, roles, and LLM clients do not receive the token as a parameter.

The pipeline bridges this gap with a `contextvars.ContextVar`.  Before running foreground steps, `run_async()` sets the current cancel token in the contextvar:

```python
from contextvars import ContextVar

cancel_token_var: ContextVar[CancellationToken | None] = ContextVar(
    "cancel_token_var", default=None
)
```

```python
# Inside Pipeline.run_async()
_reset = cancel_token_var.set(cancel_token)
try:
    # ... process samples, run steps
finally:
    cancel_token_var.reset(_reset)
```

Any code in the call stack — a step, a role, an LLM client — can read the token without any signature changes:

```python
# Inside LLM client code or any code inside a step — no parameter changes
token = cancel_token_var.get(None)
if token is not None and token.is_cancelled:
    raise PipelineCancelled("Cancelled during LLM call")
```

`asyncio.to_thread()` (used by the pipeline for sync steps) automatically copies context variables to the worker thread, so the token is visible in sync steps too.

**Why a contextvar and not a parameter:**  The call chain from pipeline to LLM client crosses four layers (pipeline → step → role → client).  Threading a parameter through every layer would require changing every method signature in between — steps and roles that have no business knowing about cancellation.  A contextvar is the standard Python mechanism for request-scoped data that crosses layers without explicit plumbing.

### What cancellation does NOT do

- **It does not interrupt a running step by default.** Cancellation is checked *between* steps by the pipeline.  Code inside a step can opt in to intra-step cancellation by reading `cancel_token_var` (see above) — but this is a step/client-level concern, not a pipeline-level one.
- **It does not cancel background steps.** Background work (after `async_boundary`) runs in separate threads and is not interrupted. `wait_for_background()` still works normally. If you need to cancel background work, shut down the step-class executors directly.
- **It does not affect hooks.** Hooks still fire for the step that was executing when cancellation was detected — `after_step` is called, then the cancellation check runs before the *next* step.

### PipelineCancelled

```python
class PipelineCancelled(Exception):
    """Raised (internally) when a cancel_token is triggered between steps.

    Surfaces in ``SampleResult.error`` — never propagated to the caller
    of ``run()`` / ``run_async()``.  Callers check for this type to
    distinguish cancellation from step failures.
    """
```

`PipelineCancelled` follows the same error-handling pattern as step exceptions: it is caught per-sample and recorded in `SampleResult`, not propagated. The runner continues to the next sample (which will also be cancelled if the token is still set). This means `run()` always returns a complete list of `SampleResult` — some successful, some failed, some cancelled.

### Example: web app cancel endpoint

```python
# Start a run
token = CancellationToken()
active_runs[run_id] = token
task = asyncio.create_task(pipe.run_async(contexts, cancel_token=token))

# Cancel endpoint
@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    active_runs[run_id].cancel()
    return {"status": "cancelling"}
```

---

## Summary Table

| Concept | Unit | Threading | Communication |
|---|---|---|---|
| `Step` | single unit of work | always sync | via `StepContext` |
| `Pipeline` | ordered step list for one input | `workers=N` across inputs | via `StepContext` |
| `Branch` | parallel pipeline list | always parallel internally | copy + merge of `StepContext` |
| `Pipeline` as a `Step` | reuse / nesting | inherits parent context | via `StepContext` |
| `PipelineHook` | observation point | runs in caller thread | `-> None` (read-only) |
| `CancellationToken` | lifecycle signal | thread-safe (`threading.Event`) | checked between steps |

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

**Alternative hook/cancellation designs:**
Three alternatives to the observation-only `PipelineHook` + separate `cancel_token` design were considered:

- Context-modifying hooks (`before_step` returns `StepContext`): hooks could transform context between steps — powerful but creates a second data-flow channel invisible to `requires`/`provides` validation. A hook could inject a field that a later step silently depends on, and the pipeline validator would not catch the dependency. Rejected to preserve the invariant that all data flow goes through the step chain.
- Cancellation as a hook (`CancellationHook` that raises in `before_step`): keeps everything in one mechanism, but mixes observation and control. If hooks are supposed to be safe to fail (exception isolation), a cancellation hook that *must* propagate its exception breaks that contract. Rejected — cancellation is a lifecycle concern, not an observation concern, so it gets its own parameter.
- Cancellation via `metadata` on `StepContext`: put a `CancellationToken` in `metadata` and have each step check it. Follows "behavior on the step" but couples every step to a cancellation concept, and steps that forget to check it silently ignore cancellation. Rejected — cancellation should be guaranteed by the pipeline, not opt-in per step.
- Additional `run_async` callback parameters (no hook protocol): add `on_step_done` and `cancel_token` as parameters on `run()`/`run_async()`, following the `on_sample_done` precedent. Minimal and consistent, but each new operational concern (metrics, billing, auth context) requires adding another parameter to the pipeline's public API, which accumulates over time. The hook protocol pays a small upfront design cost to avoid this parameter growth.

---

## External Libraries Considered

This pattern is known as **Pipes and Filters**. Several open source libraries implement variants of it. None were adopted — reasons below.

**[Kedro](https://kedro.org/)** — closest to the `requires`/`provides` model. Nodes declare explicit named inputs and outputs; pipelines are composable. The gap: requires a "data catalog" abstraction for named datasets, has no `async_boundary` concept, and is oriented toward ML/ETL rather than agentic loops. Fighting the data catalog to pass a `StepContext` would cost more than writing the primitives cleanly.

**[Hamilton](https://github.com/dagworks-inc/hamilton)** — lightest-weight equivalent. Functions declare inputs as parameters and outputs as return types; the framework infers the DAG. No server, no UI. The gap: no built-in async boundary, no fork/join `Branch`, no per-step `max_workers`. Gets contract validation for free but requires building all concurrency from scratch anyway.

**[Pypeln](https://github.com/cgarciae/pypeln)** — designed for exactly the "process N samples through concurrent stages" problem. Has sync, thread, and async modes. The gap: no typed contracts, no `Branch`, no nested pipelines. Gets the `async_boundary`-style throughput but not the structural guarantees.

**[Dagster](https://dagster.io/)** — closest overall feature set. Ops (≈ Steps) with typed inputs/outputs, jobs (≈ Pipelines), graph-based branching. The gap: it is a platform, not a library. Brings a scheduler, UI, asset catalog, and significant operational overhead. Too heavy to embed inside ACE.

**Conclusion:** The specific combination of `async_boundary`, per-step `max_workers`, `Pipeline`-as-`Step` nesting, and `SampleResult` error wrapping is not provided by any of the above out of the box. Adapting any of them would cost as much as writing the ~300-line core cleanly.

**What is borrowed rather than written:** `concurrent.futures.ThreadPoolExecutor` for the background step pools, and `asyncio.gather` (or `anyio` task groups) for `Branch` internals.
