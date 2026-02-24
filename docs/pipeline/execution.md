# Execution Model

"Async" means three different things in this framework. They operate at different levels and solve different problems. Keeping them separate is key to understanding the concurrency model.

---

## Three types of concurrency

| Type | Level | Problem it solves |
|------|-------|-------------------|
| **Async steps** | single step | Don't block the thread during I/O |
| **`async_boundary`** | across samples | Start the next sample before the current one finishes |
| **Branch parallelism** | within one sample | Run independent work simultaneously on the same data |

Each mechanism is independent. They compose freely — you can have async steps inside branches, behind an `async_boundary`, run with multiple workers.

---

## Entry points: `run()` and `run_async()`

=== "Sync"

    ```python
    # For regular (non-async) callers
    results = pipe.run(contexts, workers=4)
    ```

=== "Async"

    ```python
    # For async callers (e.g. inside an async framework)
    results = await pipe.run_async(contexts, workers=4)
    ```

`run()` is a thin wrapper that calls `asyncio.run(self.run_async(...))`. Both accept the same parameters and return `list[SampleResult]`.

---

## Workers: sample-level parallelism

The `workers` parameter on `run()` / `run_async()` controls how many samples are processed through foreground steps simultaneously:

```python
import time
from pipeline import Pipeline, StepContext


class SlowStep:
    requires = frozenset()
    provides = frozenset({"result"})

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.1)  # Simulate expensive work
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "result": "done"})
        )


pipe = Pipeline().then(SlowStep())
samples = [StepContext(sample=f"s{i}") for i in range(6)]

# Sequential: 6 × 0.1s ≈ 0.6s
results = pipe.run(samples, workers=1)

# Parallel: 0.1s (all 6 run at once)
results = pipe.run(samples, workers=6)
```

Under the hood, `workers` creates an `asyncio.Semaphore` — at most N samples flow through the foreground steps at any given time.

---

## Async steps — non-blocking I/O

A step that makes network calls (HTTP requests, API calls, subprocess) can be defined as a coroutine to avoid blocking the thread:

=== "Sync step"

    ```python
    class FetchStep:
        requires = frozenset()
        provides = frozenset({"response"})

        def __call__(self, ctx: StepContext) -> StepContext:
            response = requests.get(ctx.sample)  # blocks the thread
            return ctx.replace(
                metadata=MappingProxyType({**ctx.metadata, "response": response})
            )
    ```

=== "Async step"

    ```python
    class FetchStep:
        requires = frozenset()
        provides = frozenset({"response"})

        async def __call__(self, ctx: StepContext) -> StepContext:
            async with aiohttp.ClientSession() as session:
                response = await session.get(ctx.sample)  # yields the thread
            return ctx.replace(
                metadata=MappingProxyType({**ctx.metadata, "response": response})
            )
    ```

The pipeline detects async steps automatically via `asyncio.iscoroutinefunction` and awaits them. Sync steps are wrapped with `asyncio.to_thread()` so they're safe in an async context too.

!!! note
    Async steps are about **not blocking the thread**, not about parallelism. The pipeline is still sequential — it just yields the thread during I/O waits.

---

## Async boundary — fire-and-forget background

**Problem:** Some steps are slow (e.g. LLM calls for analysis). Waiting for them before starting the next sample hurts throughput.

**Solution:** A step declares `async_boundary = True`. Everything from that step onward runs in a background thread. The pipeline loop moves to the next sample immediately.

```python
class SlowScoreStep:
    requires = frozenset({"tokens"})
    provides = frozenset({"score"})
    async_boundary = True    # hand off to background from here
    max_workers = 3          # up to 3 scoring threads in parallel

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.5)  # Expensive scoring
        score = len(ctx.metadata["tokens"]) * 10
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "score": score})
        )
```

```mermaid
graph LR
    A1[Tokenize] --> B1[Uppercase] -->|async_boundary| C1[SlowScore]

    style A1 fill:#6366f1,stroke:#4f46e5,color:#fff
    style B1 fill:#6366f1,stroke:#4f46e5,color:#fff
    style C1 fill:#3b82f6,stroke:#2563eb,color:#fff
```

> **Indigo** = foreground (returns immediately) · **Blue** = background (fire-and-forget)

Multiple samples flow through this simultaneously — sample 2 starts its foreground steps while sample 1's background steps are still running.

### Using the boundary

```python
pipe = Pipeline().then(Tokenize()).then(Uppercase()).then(SlowScoreStep())

# run() returns immediately after foreground steps (Tokenize + Uppercase)
results = pipe.run(samples, workers=4)

# Background scoring continues — results not yet populated
print(pipe.background_stats())
# {'active': 3, 'completed': 1}

# Block until all background work finishes
pipe.wait_for_background(timeout=30.0)

# Now all SampleResult.output fields are fully populated
for r in results:
    print(r.output.metadata["score"])
```

### Background pool model

Each step **class** has a single shared `ThreadPoolExecutor`:

- `SlowScoreStep.max_workers = 3` means one pool of 3 threads for all `SlowScoreStep` instances, regardless of how many pipelines are running
- The pool is created lazily at first use and persists for the process lifetime
- If two users need different concurrency limits for the same step type, they should subclass

!!! warning "Boundary rules"
    - **One boundary per pipeline.** If multiple steps declare `async_boundary = True`, the pipeline raises `PipelineConfigError` at construction time.
    - **No boundary inside Branch children.** A boundary inside a branch child raises `PipelineConfigError`. Branch children always block until joined — detaching mid-branch is incoherent.
    - **Nested pipeline boundary is ignored.** When a pipeline is used as a step inside another pipeline, `async_boundary` is warned and ignored — there is no "next sample" to move to from the outer pipeline's perspective.

---

## `workers` vs `max_workers` — independent pools

These two knobs control different thread pools and do not interact:

| Knob | Pool | Controls |
|------|------|----------|
| `pipe.run(contexts, workers=N)` | foreground pool | How many samples run through pre-boundary steps simultaneously |
| `step.max_workers = K` | background pool (per step class) | How many instances of that step run in the background simultaneously |

A sample leaves the foreground pool when it crosses the `async_boundary` and enters the background step's pool.

**Mental model:** `workers` controls throughput *into* the pipeline; `max_workers` controls throughput *through* each background step.

!!! warning "Rate limits"
    `workers` and `max_workers` are independent pools, but total concurrent outbound calls = foreground calls + background calls. With `workers=4` and `max_workers=3`, up to 7 requests may be in-flight simultaneously. Account for this when configuring per-provider rate limits.

---

## Rule of thumb

| Question | Answer |
|----------|--------|
| Does the step wait on I/O? | `async def __call__` |
| Do I want to process more samples while previous ones are still in background steps? | `async_boundary = True` on the handoff step |
| Can two steps on the same sample run simultaneously? | [`Branch`](branching.md) |
| Do I want N samples going through the pipeline at the same time? | `workers=N` on `run()` |
