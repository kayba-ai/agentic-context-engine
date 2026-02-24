# Quick Start

Build and run your first pipeline in under 30 lines.

---

## Define two steps

Every step needs three things: `requires`, `provides`, and a `__call__` method.

```python
from types import MappingProxyType
from pipeline import Pipeline, StepContext


class Tokenize:
    """Split text into tokens and count words."""
    requires = frozenset()
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


class Uppercase:
    """Convert tokens to uppercase."""
    requires = frozenset({"tokens"})
    provides = frozenset({"upper_tokens"})

    def __call__(self, ctx: StepContext) -> StepContext:
        upper = [t.upper() for t in ctx.metadata["tokens"]]
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "upper_tokens": upper})
        )
```

---

## Build and run

Chain steps with `.then()` and run with a list of contexts:

```python
pipe = Pipeline().then(Tokenize()).then(Uppercase())

results = pipe.run([
    StepContext(sample="hello world"),
    StepContext(sample="pipeline engine demo"),
])
```

The pipeline validates ordering at construction time — if `Uppercase` came before `Tokenize`, you'd get a `PipelineOrderError` immediately, not at runtime.

---

## Inspect results

Every sample produces exactly one `SampleResult`:

```python
for r in results:
    if r.error:
        print(f"Failed at {r.failed_at}: {r.error}")
    else:
        print(f"Sample: {r.sample}")
        print(f"Tokens: {r.output.metadata['upper_tokens']}")
        print(f"Count:  {r.output.metadata['word_count']}")
```

```
Sample: hello world
Tokens: ['HELLO', 'WORLD']
Count:  2

Sample: pipeline engine demo
Tokens: ['PIPELINE', 'ENGINE', 'DEMO']
Count:  3
```

---

## Add parallelism with Branch

Run independent steps simultaneously with `Branch`:

```python
from pipeline import MergeStrategy


class Reverse:
    """Reverse each token."""
    requires = frozenset({"tokens"})
    provides = frozenset({"reversed_tokens"})

    def __call__(self, ctx: StepContext) -> StepContext:
        rev = [t[::-1] for t in ctx.metadata["tokens"]]
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "reversed_tokens": rev})
        )


pipe = (
    Pipeline()
    .then(Tokenize())
    .branch(
        Pipeline().then(Uppercase()),    # runs in parallel
        Pipeline().then(Reverse()),      # runs in parallel
        merge=MergeStrategy.RAISE_ON_CONFLICT,
    )
)

results = pipe.run([StepContext(sample="fork join")])

meta = results[0].output.metadata
print(meta["upper_tokens"])     # ['FORK', 'JOIN']
print(meta["reversed_tokens"])  # ['krof', 'nioj']
```

Both branches write to different fields (`upper_tokens` vs `reversed_tokens`), so `RAISE_ON_CONFLICT` passes through without raising.

---

## Fire-and-forget with async_boundary

Some steps are slow and don't need to block the caller. Mark a step with `async_boundary = True` to hand everything from that point onward to a background thread — `run()` returns immediately after the foreground steps.

```python
import time


class SlowScore:
    """Expensive scoring that runs in the background."""
    requires = frozenset({"tokens"})
    provides = frozenset({"score"})
    async_boundary = True   # everything from here runs in background
    max_workers = 3         # up to 3 background threads

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.5)  # simulate slow work
        score = ctx.metadata["word_count"] * 10
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "score": score})
        )


pipe = Pipeline().then(Tokenize()).then(SlowScore())

# Returns immediately — only Tokenize runs in the foreground
results = pipe.run([
    StepContext(sample="hello world"),
    StepContext(sample="background processing demo"),
])

# Background scoring still running...
print(pipe.background_stats())  # {'active': 2, 'completed': 0}

# Block until background work finishes
pipe.wait_for_background(timeout=10.0)

# Now results are fully populated
for r in results:
    print(f"{r.sample}: score={r.output.metadata['score']}")
```

```
hello world: score=20
background processing demo: score=30
```

See [Execution Model](execution.md) for the full concurrency model — `workers` vs `max_workers`, async steps, and boundary rules.

---

## Try it interactively

All the examples on this page (and more) are available as a runnable Jupyter notebook:

[:material-notebook: Open the Pipeline Demo Notebook](https://github.com/kayba-ai/agentic-context-engine/blob/main/examples/pipeline_ex/pipeline_demo.ipynb){ .md-button }

---

## Next steps

- [**Core Concepts**](core-concepts.md) — Understand the contract system and how validation works
- [**Execution Model**](execution.md) — Learn about async steps, `async_boundary`, and workers
- [**Branching & Parallelism**](branching.md) — Deep dive into merge strategies and error handling
- [**Building Custom Steps**](custom-steps.md) — Dependency injection, testing, and common patterns
