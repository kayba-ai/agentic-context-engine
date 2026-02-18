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
    requires = frozenset({"agent_output"})   # fields it reads
    provides = frozenset({"reflection"})     # fields it writes

    def __call__(self, ctx: StepContext) -> StepContext:
        ...
        return ctx
```

Rules:
- Always synchronous within its own execution
- Must declare `requires` and `provides` — the pipeline validates ordering at construction time
- Steps declare their own parallelism constraints (see below)

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

**Fan-out across samples:**

```python
pipe.run(samples, workers=4)   # same pipeline, N samples in parallel
```

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

Each branch receives a deep copy of the context at the fork point. When all branches complete, their output contexts are merged back into one before the next step runs.

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

The recommended default is `raise_on_conflict`. In practice, branches that write disjoint fields (e.g. Reflect writes `reflection`, Log writes `metadata["log"]`) never conflict and no merge function is needed.

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

Pipeline entry points: `pipe.run(samples)` for sync contexts, `await pipe.run_async(samples)` for async contexts (e.g. inside browser-use).

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
    results = await asyncio.gather(*[p(ctx) for p in self.pipelines])
    return self.merge(results)
```

This type is about **latency within a single sample**. Nothing moves to the next sample — the pipeline waits for the join before continuing.

---

### Rule of thumb

| Question | Answer |
|---|---|
| Does the step wait on I/O? | `async def __call__` |
| Do I want to process more samples while previous ones are still learning? | `async_boundary` on the step where the handoff happens |
| Can two steps on the same sample run simultaneously? | `Branch` |
| Do I want N samples going through the pipeline at the same time? | `workers=N` on `run()` |

Each mechanism is independent. They compose freely — you can have async steps inside branches, behind an async_boundary, run with multiple workers.

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
Controlled by `max_workers` on the step. The pipeline submits the step to a thread pool with that many workers.

**Pipeline split (pipelining across samples):**
`async_boundary = True` on a step tells the runner to hand off everything from that step onwards to background threads, freeing the caller to start the next sample immediately.

```
sample 1:  [AgentStep] [EvaluateStep] ──► [ReflectStep] [UpdateStep]
sample 2:  [AgentStep] [EvaluateStep] ──► ...             (background)
                                      ↑
                               async_boundary
```

This replaces the hardcoded `steps[:2]` / `steps[2:]` split that existed in the old `AsyncLearningPipeline`.

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
