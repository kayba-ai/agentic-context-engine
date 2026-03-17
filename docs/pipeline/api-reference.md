# API Reference

Complete reference for all public classes, methods, and enums in the pipeline engine.

---

## `pipeline.context`

### `StepContext`

Frozen dataclass passed from step to step. The pipeline engine only reads `sample` and `metadata` — domain-specific fields are added by subclassing.

```python
@dataclass(frozen=True)
class StepContext:
    sample: Any = None
    metadata: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `replace` | `(**changes: Any) -> StepContext` | Return a new context with the given fields replaced. Uses `dataclasses.replace` internally. |

**Behavior:**

- `metadata` is auto-coerced from `dict` to `MappingProxyType` in `__post_init__`
- Subclasses inherit `.replace()` — it works on all fields including subclass-defined ones

---

## `pipeline.protocol`

### `StepProtocol`

Structural protocol that every step (and Pipeline/Branch) must satisfy.

```python
@runtime_checkable
class StepProtocol(Protocol):
    requires: AbstractSet[str]
    provides: AbstractSet[str]

    def __call__(self, ctx: StepContext) -> StepContext: ...
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `requires` | `AbstractSet[str]` | Metadata keys the step reads |
| `provides` | `AbstractSet[str]` | Metadata keys the step writes |
| `__call__` | `(StepContext) -> StepContext` | Execute the step |

**Notes:**

- `AbstractSet[str]` accepts both `set` and `frozenset`
- `@runtime_checkable` enables `isinstance(step, StepProtocol)` checks

---

### `SampleResult`

Outcome for one sample after the pipeline has run.

```python
@dataclass
class SampleResult:
    sample: Any
    output: StepContext | None
    error: Exception | None
    failed_at: str | None
    cause: Exception | None = None
```

| Field | Type | Description |
|-------|------|-------------|
| `sample` | `Any` | The original input sample |
| `output` | `StepContext \| None` | Final context (`None` if any step failed) |
| `error` | `Exception \| None` | The exception (`None` if succeeded) |
| `failed_at` | `str \| None` | Class name of the step that raised (`None` if succeeded) |
| `cause` | `Exception \| None` | Inner exception for `BranchError` failures (default `None`) |

**Notes:**

- Mutable — background threads update it in-place when background steps complete
- For background steps, `output`/`error` may be `None` until `wait_for_background()` completes

---

## `pipeline.pipeline`

### `Pipeline`

Ordered sequence of steps. Satisfies `StepProtocol` — can be nested inside other pipelines.

#### Constructor

```python
Pipeline(steps: list | None = None, hooks: list[PipelineHook] | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `list \| None` | `None` | Optional initial list of steps |
| `hooks` | `list[PipelineHook] \| None` | `None` | Observation-only hooks fired around each foreground step |

Validates step ordering and infers contracts at construction time.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `requires` | `frozenset[str]` | Fields the pipeline needs from external context (auto-inferred) |
| `provides` | `frozenset[str]` | Fields the pipeline writes (auto-inferred, union of all steps) |

#### Methods

##### `then`

```python
def then(self, step: object) -> Pipeline
```

Append a step and return `self` for chaining. Validates ordering immediately.

| Parameter | Type | Description |
|-----------|------|-------------|
| `step` | `object` | Any object satisfying `StepProtocol` |

**Returns:** `self` (for method chaining)

**Raises:** `PipelineOrderError` if the step requires a field produced by a later step

---

##### `branch`

```python
def branch(
    self,
    *pipelines: object,
    merge: MergeStrategy | Callable = MergeStrategy.RAISE_ON_CONFLICT,
) -> Pipeline
```

Append a `Branch` step and return `self` for chaining. Shorthand for `.then(Branch(*pipelines, merge=merge))`.

**Returns:** `self` (for method chaining)

---

##### `run`

```python
def run(
    self,
    contexts: Iterable[StepContext],
    workers: int = 1,
    on_sample_done: Callable[[SampleResult], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> list[SampleResult]
```

Process contexts through the pipeline (sync entry point).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | `Iterable[StepContext]` | — | Input contexts to process |
| `workers` | `int` | `1` | Max concurrent samples in foreground steps |
| `on_sample_done` | `Callable \| None` | `None` | Callback after each sample's foreground steps complete (or fail). Must not block. |
| `cancel_token` | `CancellationToken \| None` | `None` | Cancellation signal. Checked before each step and each new sample. Pass a fresh token per invocation. |

**Returns:** `list[SampleResult]` — one result per input context

**Notes:** Calls `asyncio.run(self.run_async(...))` internally. For background steps, call `wait_for_background()` after this returns. When `cancel_token` is provided, also sets `cancel_token_var` so code inside steps (e.g. LLM clients) can read it.

---

##### `run_async`

```python
async def run_async(
    self,
    contexts: Iterable[StepContext],
    workers: int = 1,
    on_sample_done: Callable[[SampleResult], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> list[SampleResult]
```

Async entry point. Use `await pipe.run_async(contexts)` from coroutine contexts.

Same parameters and return type as `run()`.

---

##### `__call__`

```python
def __call__(self, ctx: StepContext) -> StepContext
```

Run all steps sequentially on a single context. Used when the pipeline is nested as a step inside another pipeline.

**Notes:** `async_boundary` markers are ignored in this mode — all steps run to completion.

---

##### `wait_for_background`

```python
def wait_for_background(self, timeout: float | None = None) -> None
```

Block until all background tasks complete.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float \| None` | `None` | Max seconds to wait. `None` = wait indefinitely. |

**Raises:** `TimeoutError` if timeout elapses before completion

---

##### `background_stats`

```python
def background_stats(self) -> dict[str, int]
```

Return a snapshot of background task progress. Thread-safe.

**Returns:** `{"active": int, "completed": int}`

---

## `pipeline.branch`

### `MergeStrategy`

Enum of built-in merge strategies for `Branch` outputs.

```python
class MergeStrategy(Enum):
    RAISE_ON_CONFLICT = "raise_on_conflict"
    LAST_WRITE_WINS = "last_write_wins"
    NAMESPACED = "namespaced"
```

| Value | Behavior |
|-------|----------|
| `RAISE_ON_CONFLICT` | Raises `ValueError` if two branches write different values to the same named field. Metadata merges with last-writer-wins. |
| `LAST_WRITE_WINS` | Last branch's value wins for every conflicting field. |
| `NAMESPACED` | Each branch's output stored at `metadata["branch_N"]`. No conflict possible. |

---

### `Branch`

Runs multiple pipelines in parallel, then merges their outputs. Satisfies `StepProtocol`.

#### Constructor

```python
Branch(
    *pipelines: object,
    merge: MergeStrategy | Callable = MergeStrategy.RAISE_ON_CONFLICT,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*pipelines` | `object` | — | Child pipelines to run in parallel (at least one required) |
| `merge` | `MergeStrategy \| Callable` | `RAISE_ON_CONFLICT` | Merge strategy or custom `fn(list[StepContext]) -> StepContext` |

**Raises:** `ValueError` if no pipelines are provided

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `requires` | `frozenset[str]` | Union of all children's requires |
| `provides` | `frozenset[str]` | Union of all children's provides |
| `pipelines` | `list` | The child pipelines |

#### Methods

##### `__call__`

```python
def __call__(self, ctx: StepContext) -> StepContext
```

Sync fan-out via `ThreadPoolExecutor`. All branches run to completion before any failure is raised.

**Raises:** `BranchError` if any branch fails

---

##### `__call_async__`

```python
async def __call_async__(self, ctx: StepContext) -> StepContext
```

Async fan-out via `asyncio.gather`. Sync children are wrapped with `asyncio.to_thread`.

**Raises:** `BranchError` if any branch fails

---

## `pipeline.protocol`  — Hooks

### `PipelineHook`

Observation-only protocol fired around each foreground step. Hooks cannot modify context — both methods return `None`.

```python
@runtime_checkable
class PipelineHook(Protocol):
    def before_step(self, step_name: str, ctx: StepContext) -> None: ...
    def after_step(self, step_name: str, ctx: StepContext) -> None: ...
```

| Method | Parameters | Description |
|--------|-----------|-------------|
| `before_step` | `step_name: str, ctx: StepContext` | Called before each foreground step executes |
| `after_step` | `step_name: str, ctx: StepContext` | Called after each foreground step completes |

**Notes:**

- `step_name` is `type(step).__name__` — hooks know what ran but cannot inspect or mutate the step instance
- Hooks fire for foreground steps only — background steps (after `async_boundary`) do not trigger hooks
- If a hook raises, the pipeline logs the error and continues — a broken hook never kills the pipeline
- For `Branch` steps, hooks fire once for `"Branch"` as a whole, not for inner steps

---

## `pipeline.errors`

### `CancellationToken`

Thread-safe cancellation signal. Create a fresh token per `run()` invocation.

```python
class CancellationToken:
    def cancel(self) -> None: ...

    @property
    def is_cancelled(self) -> bool: ...
```

| Method / Property | Description |
|-------------------|-------------|
| `cancel()` | Signal cancellation. Thread-safe, idempotent. |
| `is_cancelled` | `True` after `cancel()` has been called. |

---

### `cancel_token_var`

`ContextVar` set by `Pipeline.run_async()` so code inside steps (e.g. LLM clients) can read the current cancel token without parameter changes.

```python
cancel_token_var: ContextVar[CancellationToken | None]  # default: None
```

**Notes:**

- Set before steps run, reset after `run_async()` completes
- `asyncio.to_thread()` copies contextvars automatically — visible in sync steps too
- Read with `cancel_token_var.get(None)` — returns `None` when no pipeline is running

---

### `PipelineCancelled`

```python
class PipelineCancelled(Exception): ...
```

A `cancel_token` was triggered. Surfaces in `SampleResult.error` — never propagated to the caller of `run()`. Callers check `isinstance(result.error, PipelineCancelled)` to distinguish cancellation from step failures.

---

### `PipelineOrderError`

```python
class PipelineOrderError(Exception): ...
```

A step requires a field that no earlier step provides (but a later step does). Raised at **construction time**.

---

### `PipelineConfigError`

```python
class PipelineConfigError(Exception): ...
```

Invalid pipeline wiring. Raised at **construction time**. Examples:

- More than one `async_boundary = True` step in the same pipeline
- An `async_boundary = True` step inside a `Branch` child

---

### `BranchError`

```python
class BranchError(Exception):
    failures: list[BaseException]
```

One or more branch pipelines failed. All branches always run to completion before this is raised. Raised at **runtime**.

| Attribute | Type | Description |
|-----------|------|-------------|
| `failures` | `list[BaseException]` | One exception per failed branch |

---

## Step class attributes

Optional attributes a step class can declare to control pipeline behavior:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `requires` | `set[str] \| frozenset[str]` | *(required)* | Metadata keys the step reads |
| `provides` | `set[str] \| frozenset[str]` | *(required)* | Metadata keys the step writes |
| `async_boundary` | `bool` | `False` | Marks the foreground/background split point |
| `max_workers` | `int` | `1` | Max concurrent background threads for this step class |
