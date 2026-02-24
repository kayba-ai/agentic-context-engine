# Core Concepts

The pipeline engine is built on four foundational concepts: the **Step protocol**, the **StepContext**, the **contract system**, and the **Pipeline** compositor. Understanding these gives you the mental model for everything else.

---

## StepProtocol

A Step is any Python object that satisfies the `StepProtocol` — a structural (duck-typed) interface. No base class is required.

```python
from collections.abc import Set as AbstractSet
from typing import Protocol, runtime_checkable

@runtime_checkable
class StepProtocol(Protocol):
    requires: AbstractSet[str]   # metadata keys this step reads
    provides: AbstractSet[str]   # metadata keys this step writes

    def __call__(self, ctx: StepContext) -> StepContext: ...
```

Any object with `requires`, `provides`, and a `__call__` method is a valid step:

```python
class Tokenize:
    requires = frozenset()                     # no dependencies
    provides = frozenset({"tokens", "word_count"})

    def __call__(self, ctx: StepContext) -> StepContext:
        tokens = str(ctx.sample).split()
        return ctx.replace(
            metadata=MappingProxyType({
                **ctx.metadata,
                "tokens": tokens,
                "word_count": len(tokens),
            })
        )
```

Key details:

- **`AbstractSet[str]`** accepts both `set` and `frozenset`. Steps can use plain set literals — the pipeline normalizes them to `frozenset` at construction time.
- **`@runtime_checkable`** lets the pipeline use `isinstance(step, StepProtocol)` at construction time to catch missing attributes early, rather than failing at call time.
- **`Pipeline` and `Branch`** both satisfy this protocol, so they can be nested wherever a step is expected.

---

## StepContext

`StepContext` is the data carrier passed from step to step. It is a **frozen dataclass** — steps never mutate the incoming context.

```python
from dataclasses import dataclass, field
from types import MappingProxyType

@dataclass(frozen=True)
class StepContext:
    sample: Any = None
    metadata: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )

    def replace(self, **changes) -> "StepContext":
        return dataclasses.replace(self, **changes)
```

The engine only reads `sample` and `metadata`. All domain-specific fields are added by subclassing.

### The `.replace()` pattern

Steps create new contexts — they never mutate the incoming one:

```python
def __call__(self, ctx: StepContext) -> StepContext:
    result = process(ctx.sample)
    return ctx.replace(
        metadata=MappingProxyType({**ctx.metadata, "result": result})
    )
```

This is the only way to "modify" a context. `frozen=True` makes mutation a hard error at runtime rather than a subtle bug.

### Metadata auto-coercion

If a caller passes a plain `dict` as metadata, `StepContext.__post_init__` automatically wraps it in `MappingProxyType`, ensuring mutation is always a runtime error:

```python
# Both of these produce identical immutable metadata:
ctx = StepContext(sample="hello", metadata={"key": "value"})
ctx = StepContext(sample="hello", metadata=MappingProxyType({"key": "value"}))
```

### Subclassing for domain fields

Applications subclass `StepContext` to add named fields for concepts shared across their pipelines:

```python
@dataclass(frozen=True)
class MLContext(StepContext):
    # Shared configuration
    model_config: dict | None = None

    # Produced by steps (None until the providing step runs)
    predictions: list | None = None
    scores: dict | None = None
    report: str | None = None
```

Use **named fields** for data shared across multiple steps in the pipeline. Use **`metadata`** for integration-specific or step-specific transient data that doesn't warrant a dedicated field.

!!! tip "When to use which"
    - Named field: `predictions`, `scores` — shared by multiple steps, type-checkable
    - Metadata: `metadata["debug_log"]`, `metadata["cache_key"]` — step-specific, doesn't pollute the class

### Why immutability?

- **Branch safety** — All branches receive the same frozen context. No deep copy is needed since no branch can mutate what it receives.
- **Thread safety** — Steps running concurrently (via `workers` or `Branch`) can safely share context objects.
- **Debugging** — Each step returns a new context, creating a clear trace of data transformations.

---

## Contracts: requires and provides

Every step declares:

- **`requires`** — the set of field names it reads from the context
- **`provides`** — the set of field names it writes to the context

The pipeline validates these at construction time.

### How validation works

When you build a pipeline with `.then()`, the engine checks step ordering immediately:

```python
pipe = (
    Pipeline()
    .then(Tokenize())      # provides: {"tokens", "word_count"}
    .then(Uppercase())     # requires: {"tokens"} ✓ — Tokenize provides it
)
```

If a step requires a field that a **later** step provides, the pipeline raises `PipelineOrderError`:

```python
# This raises PipelineOrderError at construction time:
pipe = Pipeline().then(Uppercase()).then(Tokenize())
# ↑ Uppercase requires "tokens", but Tokenize (which provides it) comes after
```

### External inputs vs internal dependencies

Fields not produced by any step in the pipeline are treated as **external inputs** — they must be present in the initial `StepContext` passed to `run()`. These do not trigger ordering errors:

```python
class ScoreStep:
    requires = frozenset({"predictions"})  # external input
    provides = frozenset({"scores"})

# No error — "predictions" is expected to come from the initial context
pipe = Pipeline().then(ScoreStep())
```

### Contract inference for nested pipelines

When a `Pipeline` is used as a step inside another pipeline, its `requires` and `provides` are computed automatically from its inner steps:

```python
inner = Pipeline().then(Tokenize()).then(Uppercase())

# Inferred automatically:
# inner.requires = frozenset()              — Tokenize needs nothing external
# inner.provides = frozenset({"tokens", "word_count", "upper_tokens"})

outer = Pipeline().then(inner).then(Summarize())
# Summarize's requirements validated against inner.provides
```

The inference algorithm:

1. Walk steps in order, tracking what has been provided so far
2. `requires` = fields needed by steps that no earlier step provides (external dependencies)
3. `provides` = union of all fields any step writes

!!! warning "All Branch children always run"
    The contract system assumes all `Branch` children execute. There is no concept of conditional branches where only some children run — all branches always run. If a branch provides a field that a later step requires, validation passes; if that branch were to not run, the pipeline would fail at runtime.

---

## Pipeline

A `Pipeline` is an ordered list of steps that runs sequentially for a single input. It satisfies the `StepProtocol`, so it can be nested inside other pipelines.

### Building a pipeline

Two equivalent forms:

=== "Fluent builder (preferred)"

    ```python
    pipe = (
        Pipeline()
        .then(Tokenize())
        .then(Uppercase())
        .then(Summarize())
    )
    ```

=== "Constructor list"

    ```python
    pipe = Pipeline([
        Tokenize(),
        Uppercase(),
        Summarize(),
    ])
    ```

Both validate step ordering at construction time. The fluent builder validates **after each `.then()` call**, giving precise error messages about which step caused the violation.

### The `.branch()` shorthand

Instead of manually creating a `Branch`, use the fluent shorthand:

```python
pipe = (
    Pipeline()
    .then(Tokenize())
    .branch(
        Pipeline().then(Uppercase()),
        Pipeline().then(Reverse()),
        merge=MergeStrategy.RAISE_ON_CONFLICT,
    )
    .then(Summarize())
)
```

This is equivalent to `.then(Branch(...))`.

### Nesting

A pipeline used as a step is a black box — the outer pipeline sees only its aggregated `requires` and `provides`:

```python
preprocessing = Pipeline().then(Tokenize()).then(Uppercase())
postprocessing = Pipeline().then(Summarize()).then(FormatStep())

full = Pipeline().then(preprocessing).then(postprocessing)
```

!!! note "Inner pipeline as a fan-out step"
    A step receives one context and must return one context — but nothing prevents it from internally expanding to multiple sub-inputs:

    ```python
    class MultiSearchStep:
        requires = frozenset()
        provides = frozenset({"search_results"})

        def __call__(self, ctx: StepContext) -> StepContext:
            queries = generate_queries(ctx.sample)
            sub_ctxs = [StepContext(sample=q) for q in queries]
            sub_pipe = Pipeline().then(FetchStep())
            results = sub_pipe.run(sub_ctxs, workers=len(queries))
            merged = merge_results(results)
            return ctx.replace(
                metadata=MappingProxyType({**ctx.metadata, "search_results": merged})
            )
    ```

    From the outer pipeline's perspective, `MultiSearchStep` is a single step. The fan-out is an internal implementation detail.
