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
Pipeline(steps: list | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `list \| None` | `None` | Optional initial list of steps |

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
) -> list[SampleResult]
```

Process contexts through the pipeline (sync entry point).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contexts` | `Iterable[StepContext]` | — | Input contexts to process |
| `workers` | `int` | `1` | Max concurrent samples in foreground steps |

**Returns:** `list[SampleResult]` — one result per input context

**Notes:** Calls `asyncio.run(self.run_async(...))` internally. For background steps, call `wait_for_background()` after this returns.

---

##### `run_async`

```python
async def run_async(
    self,
    contexts: Iterable[StepContext],
    workers: int = 1,
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

## `pipeline.errors`

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
