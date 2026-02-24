# Error Handling

The pipeline engine guarantees that every sample produces a `SampleResult` — nothing is dropped silently. One failing sample never blocks others. Retry logic is the responsibility of individual steps, not the pipeline.

---

## SampleResult

Every sample that enters `run()` produces exactly one `SampleResult`:

```python
@dataclass
class SampleResult:
    sample: Any                      # the original input
    output: StepContext | None       # final context (None if failed)
    error: Exception | None          # the exception (None if succeeded)
    failed_at: str | None            # step class name where error occurred
    cause: Exception | None = None   # inner exception for BranchError
```

| Field | On success | On failure |
|-------|-----------|------------|
| `sample` | original input | original input |
| `output` | final `StepContext` | `None` |
| `error` | `None` | the exception |
| `failed_at` | `None` | class name of the failing step (e.g. `"Tokenize"`) |
| `cause` | `None` | inner exception when `failed_at == "Branch"` |

!!! note "Background steps"
    For steps after an `async_boundary`, `output` and `error` may still be `None` when `run()` returns. Call `pipe.wait_for_background()` to block until all background work completes and results are finalized.

---

## Construction-time errors

These are caught **before any data flows** — they surface immediately when you build the pipeline.

### PipelineOrderError

Raised when a step requires a field that is produced by a **later** step in the pipeline:

```python
from pipeline import Pipeline
from pipeline.errors import PipelineOrderError

class Uppercase:
    requires = frozenset({"tokens"})
    provides = frozenset({"upper_tokens"})
    def __call__(self, ctx): ...

class Tokenize:
    requires = frozenset()
    provides = frozenset({"tokens"})
    def __call__(self, ctx): ...

try:
    Pipeline().then(Uppercase()).then(Tokenize())
except PipelineOrderError as e:
    print(e)
    # Uppercase requires {"tokens"} but it is provided by a later step
```

!!! tip
    `PipelineOrderError` is always a bug — reorder your steps. Fields not produced by **any** step in the pipeline are treated as external inputs and do not trigger this error.

### PipelineConfigError

Raised for invalid pipeline wiring:

**Multiple async boundaries:**

```python
from pipeline.errors import PipelineConfigError

class StepA:
    requires = frozenset()
    provides = frozenset({"a"})
    async_boundary = True
    def __call__(self, ctx): ...

class StepB:
    requires = frozenset({"a"})
    provides = frozenset({"b"})
    async_boundary = True
    def __call__(self, ctx): ...

try:
    Pipeline().then(StepA()).then(StepB())
except PipelineConfigError:
    print("Only one async_boundary per pipeline is allowed")
```

**Async boundary inside a Branch child:**

```python
try:
    Pipeline().branch(
        Pipeline().then(StepA()),  # async_boundary = True inside branch
    )
except PipelineConfigError:
    print("async_boundary inside Branch children is not allowed")
```

---

## Runtime errors

### Foreground failures

When a step before the `async_boundary` (or in a pipeline with no boundary) raises an exception, the pipeline catches it per-sample and records it in the `SampleResult`:

```python
class Boom:
    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx):
        raise RuntimeError(f"Failed on {ctx.sample!r}")

pipe = Pipeline().then(Tokenize()).then(Boom())

results = pipe.run([
    StepContext(sample="good"),
    StepContext(sample="also good"),
])

for r in results:
    if r.error:
        print(f"Sample '{r.sample}' failed at {r.failed_at}: {r.error}")
    else:
        print(f"Sample '{r.sample}' succeeded")
```

```
Sample 'good' failed at Boom: Failed on 'good'
Sample 'also good' failed at Boom: Failed on 'also good'
```

Each sample is processed independently — one failure does not prevent others from running.

### Background failures

When a step **after** the `async_boundary` raises, the caller has already moved on. The exception is captured and attached to the `SampleResult` in-place:

```python
pipe = Pipeline().then(Tokenize()).then(BrokenBackgroundStep())

results = pipe.run(samples)
# results returned immediately — background still running

pipe.wait_for_background(timeout=10.0)

# Now check for background failures
for r in results:
    if r.error:
        print(f"Background failure at {r.failed_at}: {r.error}")
```

### BranchError

When one or more branch pipelines fail, a `BranchError` is raised with the full list of failures:

```python
from pipeline.errors import BranchError

results = pipe.run(contexts)

for r in results:
    if isinstance(r.error, BranchError):
        print(f"{len(r.error.failures)} branch(es) failed:")
        for f in r.error.failures:
            print(f"  {type(f).__name__}: {f}")
    elif r.error:
        print(f"Step failure at {r.failed_at}: {r.error}")
```

All branches run to completion before `BranchError` is raised — no branch is cancelled when another fails. The `SampleResult.cause` field carries the inner exception from the failing branch.

---

## Inspecting results

The standard pattern after `run()`:

```python
results = pipe.run(contexts)
pipe.wait_for_background()  # if using async_boundary

succeeded = [r for r in results if r.error is None]
failed = [r for r in results if r.error is not None]

print(f"{len(succeeded)} succeeded, {len(failed)} failed")

for r in failed:
    print(f"  Sample: {r.sample}")
    print(f"  Failed at: {r.failed_at}")
    print(f"  Error: {r.error}")
```

---

## Background monitoring

### `wait_for_background()`

Blocks until all background tasks complete:

```python
# Wait indefinitely
pipe.wait_for_background()

# Wait with timeout — raises TimeoutError if not done
pipe.wait_for_background(timeout=30.0)
```

Completed threads are removed from the tracking list after this call.

### `background_stats()`

Returns a snapshot of background task progress. Thread-safe — can be called from any thread while the pipeline is running:

```python
stats = pipe.background_stats()
print(stats)
# {'active': 2, 'completed': 8}
```
