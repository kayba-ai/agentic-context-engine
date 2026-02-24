# Building Custom Steps

This guide covers everything you need to create your own pipeline steps — from the minimal contract to advanced patterns like dependency injection, async execution, and testing.

---

## The step contract

Any Python object with `requires`, `provides`, and `__call__` is a valid step. No base class needed.

```python
from types import MappingProxyType
from pipeline import StepContext


class MyStep:
    requires = frozenset({"input_field"})
    provides = frozenset({"output_field"})

    def __call__(self, ctx: StepContext) -> StepContext:
        result = process(ctx.metadata["input_field"])
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "output_field": result})
        )
```

Rules:

- `requires` and `provides` can be `set` or `frozenset` — the pipeline normalizes to `frozenset`
- `__call__` receives a `StepContext` and must return a `StepContext`
- Never mutate the incoming context — always use `.replace()`

---

## Sync vs async steps

=== "Sync"

    ```python
    class ComputeStep:
        requires = frozenset({"data"})
        provides = frozenset({"result"})

        def __call__(self, ctx: StepContext) -> StepContext:
            result = expensive_computation(ctx.metadata["data"])
            return ctx.replace(
                metadata=MappingProxyType({**ctx.metadata, "result": result})
            )
    ```

=== "Async"

    ```python
    class FetchStep:
        requires = frozenset({"url"})
        provides = frozenset({"response"})

        async def __call__(self, ctx: StepContext) -> StepContext:
            async with aiohttp.ClientSession() as session:
                resp = await session.get(ctx.metadata["url"])
                data = await resp.json()
            return ctx.replace(
                metadata=MappingProxyType({**ctx.metadata, "response": data})
            )
    ```

Use async steps for I/O-bound work (HTTP requests, API calls, file I/O). The pipeline detects and handles both transparently.

---

## Dependency injection

Steps that need external collaborators receive them via `__init__`. The `__call__` method stays stateless — it only uses `self.*` for injected dependencies and `ctx` for data.

```python
class ScoringStep:
    requires = frozenset({"predictions"})
    provides = frozenset({"scores"})

    def __init__(self, scorer, threshold: float = 0.5):
        self.scorer = scorer
        self.threshold = threshold

    def __call__(self, ctx: StepContext) -> StepContext:
        raw_scores = self.scorer.evaluate(ctx.metadata["predictions"])
        filtered = {k: v for k, v in raw_scores.items() if v >= self.threshold}
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "scores": filtered})
        )
```

This makes testing easy — inject mocks:

```python
pipe = Pipeline().then(ScoringStep(scorer=mock_scorer, threshold=0.8))
```

---

## Declaring concurrency

Two optional class attributes control how a step participates in concurrent execution:

### `async_boundary`

Marks the foreground/background split point. Everything from this step onward runs in a background thread:

```python
class AnalyzeStep:
    requires = frozenset({"data"})
    provides = frozenset({"analysis"})
    async_boundary = True  # background from here

    def __call__(self, ctx: StepContext) -> StepContext: ...
```

See [Execution Model — Async Boundary](execution.md#async-boundary-fire-and-forget-background) for details.

### `max_workers`

Controls the per-step-class thread pool size for background execution:

```python
class ParallelAnalyzeStep:
    requires = frozenset({"data"})
    provides = frozenset({"analysis"})
    async_boundary = True
    max_workers = 4  # up to 4 concurrent analyses

    def __call__(self, ctx: StepContext) -> StepContext: ...
```

Default is `max_workers = 1` (serialized).

!!! warning
    Steps that write shared state (e.g. updating an external database or accumulating results into a shared object) must use `max_workers = 1` to avoid race conditions.

---

## Subclassing StepContext

When `metadata` becomes unwieldy, subclass `StepContext` to add named fields:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MLContext(StepContext):
    predictions: list | None = None
    scores: dict | None = None
    report: str | None = None
```

Steps write to named fields using `.replace()`:

```python
class PredictStep:
    requires = frozenset()
    provides = frozenset({"predictions"})

    def __init__(self, model):
        self.model = model

    def __call__(self, ctx: MLContext) -> MLContext:
        preds = self.model.predict(ctx.sample)
        return ctx.replace(predictions=preds)
```

!!! tip "When to subclass"
    - **Named fields**: Data shared across multiple steps that benefits from type checking
    - **Metadata**: Step-specific or integration-specific transient data (e.g. `metadata["cache_key"]`)

    The `requires`/`provides` validation works on attribute names, so it's subclass-agnostic. A step declaring `requires = {"predictions"}` works with any context subclass that has a `predictions` attribute.

---

## Testing steps

### Unit test — step in isolation

```python
from types import MappingProxyType
from pipeline import StepContext


def test_tokenize_splits_words():
    step = Tokenize()
    ctx = StepContext(sample="hello world")

    result = step(ctx)

    assert result.metadata["tokens"] == ["hello", "world"]
    assert result.metadata["word_count"] == 2


def test_uppercase_transforms_tokens():
    step = Uppercase()
    ctx = StepContext(
        metadata=MappingProxyType({"tokens": ["hello", "world"]})
    )

    result = step(ctx)

    assert result.metadata["upper_tokens"] == ["HELLO", "WORLD"]
```

### Protocol compliance

```python
from pipeline import StepProtocol


def test_step_satisfies_protocol():
    step = Tokenize()
    assert isinstance(step, StepProtocol)
    assert hasattr(step, "requires")
    assert hasattr(step, "provides")
    assert callable(step)
```

### Pipeline integration test

```python
from pipeline import Pipeline, StepContext


def test_full_pipeline():
    pipe = Pipeline().then(Tokenize()).then(Uppercase())
    results = pipe.run([StepContext(sample="hello world")])

    assert len(results) == 1
    assert results[0].error is None
    assert results[0].output.metadata["upper_tokens"] == ["HELLO", "WORLD"]
```

---

## Common patterns

### Map-reduce step

A step that internally fans out to multiple sub-inputs:

```python
class MultiSearchStep:
    requires = frozenset()
    provides = frozenset({"search_results"})

    def __call__(self, ctx: StepContext) -> StepContext:
        queries = generate_queries(ctx.sample)                   # 1 → N
        sub_ctxs = [StepContext(sample=q) for q in queries]
        sub_pipe = Pipeline().then(FetchStep())
        results = sub_pipe.run(sub_ctxs, workers=len(queries))  # parallel
        merged = merge_results(results)                          # N → 1
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "search_results": merged})
        )
```

From the outer pipeline's perspective, this is a black box that takes one context and returns one.

### Logging / observability step

A pass-through step that logs without modifying data:

```python
class LogStep:
    requires = frozenset()
    provides = frozenset()

    def __init__(self, logger):
        self.logger = logger

    def __call__(self, ctx: StepContext) -> StepContext:
        self.logger.info(f"Processing sample: {ctx.sample}")
        self.logger.debug(f"Metadata keys: {list(ctx.metadata.keys())}")
        return ctx  # pass through unchanged
```

### Retry wrapper

A step that wraps another step with retry logic:

```python
import time


class RetryStep:
    def __init__(self, inner, max_retries: int = 3, delay: float = 1.0):
        self.inner = inner
        self.max_retries = max_retries
        self.delay = delay
        self.requires = inner.requires
        self.provides = inner.provides

    def __call__(self, ctx: StepContext) -> StepContext:
        for attempt in range(self.max_retries):
            try:
                return self.inner(ctx)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.delay * (attempt + 1))
```

Usage:

```python
pipe = Pipeline().then(RetryStep(FlakyAPIStep(), max_retries=3))
```
