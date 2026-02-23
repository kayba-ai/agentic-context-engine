# ACE Architecture Design

Specification for rewriting the legacy `ace/` module to use the pipeline engine.

---

## Goals

1. Replace the monolithic `ACEBase` / `OfflineACE` / `OnlineACE` in `adaptation.py` with pipeline-based classes.
2. Rename to match what each class actually does:
   - **TraceAnalyser**: takes pre-recorded traces, outputs a skillbook. Replaces the concept of "offline" learning.
   - **ACE**: the live adaptive pipeline. Replaces the concept of "online" learning. Also supports multi-epoch batch runs.
3. Clean OOP: shared base class, composition over inheritance, pluggable steps.
4. Unify the integration pattern — external frameworks produce raw trace objects (any type), TraceAnalyser passes them to the Reflector as-is.
5. Maximise step granularity — each step does one thing so concerns are separated and each step is independently testable.

---

## Naming Changes

| Legacy | New | What it does |
|---|---|---|
| `OfflineACE` | `TraceAnalyser` | Analyse pre-recorded traces → evolve a skillbook |
| `OnlineACE` | `ACE` | Live execution → feedback → learning loop |
| `ACEBase` | `ACERunner` | Shared runner infrastructure (composition, not inheritance from Pipeline) |
| `ACEStepResult` | Removed — use `SampleResult` from the pipeline engine | Unified result type |

---

## Core Types

### Sample (unchanged)

The existing `Sample` dataclass stays as-is. ACE uses it.

```python
@dataclass
class Sample:
    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: dict = field(default_factory=dict)
    id: str | None = None
```

### ACESample — protocol for step access

Steps access `ctx.sample.question` uniformly. A `Protocol` makes this duck typing explicit and type-safe:

```python
class ACESample(Protocol):
    """Minimal interface that Sample satisfies."""

    @property
    def question(self) -> str: ...

    @property
    def context(self) -> str: ...

    @property
    def ground_truth(self) -> str | None: ...

    @property
    def metadata(self) -> dict: ...
```

`ACEStepContext.sample` is typed as `ACESample`. `Sample` satisfies it structurally. Mypy validates both sides: producers must provide the attributes, consumers can rely on them.

### SkillbookView — read-only projection

The `Skillbook` is mutable — steps add, tag, and remove skills. Putting it directly on a `frozen=True` context would allow mutation through the reference (`ctx.skillbook.tag_skill(...)` succeeds even though `ctx.skillbook = other` fails). That breaks the immutability guarantee.

`SkillbookView` solves this. It wraps a `Skillbook` and exposes only read methods. Write methods don't exist on the class — calling them raises `AttributeError` at runtime and a type error at check time.

```python
class SkillbookView:
    """Read-only projection of a Skillbook. Safe on a frozen context."""

    __slots__ = ("_sb",)

    def __init__(self, skillbook: Skillbook) -> None:
        self._sb = skillbook

    def as_prompt(self) -> str:
        return self._sb.as_prompt()

    def get_skill(self, skill_id: str) -> Skill | None:
        return self._sb.get_skill(skill_id)

    def __len__(self) -> int:
        return len(self._sb)

    def __iter__(self):
        return iter(self._sb)

    def __repr__(self) -> str:
        return f"SkillbookView({len(self._sb)} skills)"
```

**Enforcement:**
- **Type checker** — mypy/pyright flags `ctx.skillbook.add_skill(...)` because `SkillbookView` has no such method.
- **Runtime** — `AttributeError` if someone calls a write method anyway.
- **Convention** — the underlying `_sb` is underscore-prefixed. Accessing it is a deliberate violation, not an accident.

Steps that only **read** the skillbook (AgentStep, ReflectStep, UpdateStep, ObservabilityStep) access `ctx.skillbook` — the view. Steps that **write** the skillbook (TagStep, ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor injection and use `self.skillbook`.

### ACEStepContext

Subclass of the pipeline engine's `StepContext`. Carries all step-to-step data for the ACE pipeline. The pipeline engine only knows about `sample` and `metadata`; all ACE-specific fields live here.

```python
@dataclass(frozen=True)
class ACEStepContext(StepContext):
    """Immutable context for the ACE pipeline.

    The skillbook field is a SkillbookView (read-only). Steps that need to
    write to the skillbook receive the real Skillbook via constructor injection.
    """

    sample: ACESample | None = None
    skillbook: SkillbookView | None = None
    environment: TaskEnvironment | None = None
    trace: object | None = None
    agent_output: AgentOutput | None = None
    environment_result: EnvironmentResult | None = None
    reflection: ReflectorOutput | None = None
    skill_manager_output: UpdateBatch | None = None
    epoch: int = 1
    total_epochs: int = 1
    step_index: int = 0
    total_steps: int | None = None
    global_sample_index: int = 0
```

The `trace` field holds the raw execution record from any external system — a browser-use `AgentHistoryList`, a LangChain result dict, a Claude Code transcript, or any arbitrary Python object. It has no enforced schema. The Reflector receives the raw trace and is responsible for making sense of it — this gives maximum flexibility for analysis without constraining trace format. Extraction helpers can be added later as an optional layer if needed.

**What goes on the context vs what gets injected:**

| | On the context | Injected via constructor |
|---|---|---|
| **Nature** | Step-to-step data + read-only dependencies | Mutable shared state |
| **Lifetime** | Per-sample (born in `_build_context`, dies after pipeline) | Per-runner (created once, shared across samples) |
| **Immutable?** | Yes — frozen fields, read-only views | No — mutable by design |
| **Examples** | `agent_output`, `reflection`, `skillbook` (view) | `skillbook` (real), `dedup_config` |
| **Validated by engine?** | Yes — `requires`/`provides` | No — runtime error if missing |

---

## Class Hierarchy

```
ACERunner (shared infrastructure: epoch loop, delegates to Pipeline.run())
├── TraceAnalyser       — [Reflect → Tag → Update → Apply]; input = any trace object
├── ACE                 — [Agent → Evaluate → Reflect → Tag → Update → Apply]; input = Sample + Environment
├── BrowserACE          — [BrowserExecute → Reflect → Tag → Update → Apply]; input = tasks
├── LangChainACE        — [LangChainExecute → Reflect → Tag → Update → Apply]; input = chain inputs
└── ClaudeCodeACE       — [ClaudeCodeExecute → Reflect → Tag → Update → Apply → Persist]; input = tasks
```

All compose a `Pipeline` rather than extending it. The pipeline is an implementation detail, not part of the public interface. Each subclass only overrides `run()` (public signature) and `_build_context()` (input mapping).

---

## ACERunner — shared base

Encapsulates everything that TraceAnalyser, ACE, and integration runners have in common. The runner's only job is the epoch loop and Iterable validation. Per-sample iteration, error handling, background execution, and checkpoints are all delegated to `Pipeline.run()`.

Subclasses only override `run()` (public signature) and `_build_context()` (input mapping).

```python
class ACERunner:
    """Shared runner infrastructure for all ACE runners."""

    def __init__(
        self,
        pipeline: Pipeline,
        skillbook: Skillbook,
    ) -> None:
        self.pipeline = pipeline
        self.skillbook = skillbook

    def save(self, path: str) -> None:
        """Save the current skillbook to disk."""
        self.skillbook.save_to_file(path)

    def wait_for_background(self, timeout: float | None = None) -> None:
        """Block until all background learning tasks complete.

        Delegates to Pipeline.wait_for_background(). Call this after run(wait=False)
        before saving the skillbook or reading final results.
        """
        self.pipeline.wait_for_background(timeout)

    @property
    def learning_stats(self) -> dict:
        """Return background learning progress.

        Useful after run(wait=False) to monitor learning without blocking.
        Delegates to Pipeline.background_stats() to avoid reaching into
        pipeline internals.
        """
        return self.pipeline.background_stats()
```

### Responsibilities

| Concern | Owner |
|---|---|
| Epoch loop + Iterable validation | `ACERunner._run()` |
| Per-sample iteration + error isolation | `Pipeline.run()` |
| Foreground/background split | `Pipeline.run()` (via `async_boundary`) |
| Concurrent workers | `Pipeline.run(workers=N)` |
| Checkpoints | `CheckpointStep` (in the pipeline, configured at construction) |
| Background drain | `ACERunner.wait_for_background()` → `Pipeline.wait_for_background()` |
| Background monitoring | `ACERunner.learning_stats` |
| Skillbook I/O | `save(path)` on the runner (delegates to `skillbook.save_to_file()`) |

Each sample is independent — no state persists across samples. The skillbook is the only cross-sample coupling: read-only steps see it via `ctx.skillbook` (a `SkillbookView`), write steps mutate it via `self.skillbook` (the real `Skillbook`, injected at construction).

**Eventual consistency:** `SkillbookView` is a thin delegation wrapper, not a snapshot — it reads from the live `Skillbook` at call time. When background learning is active (`async_boundary` on `ReflectStep`), concurrent samples may observe partially-updated skillbook state. For example, Sample 2's `ReflectStep` might read the skillbook mid-mutation by Sample 1's `ApplyStep`. This is by design: steps see a best-effort view of the current skillbook rather than a point-in-time snapshot. The trade-off is acceptable because (1) the Reflector and SkillManager use the skillbook as LLM prompt context, where a few missing or extra skills have negligible impact on output quality, (2) serialising all skillbook reads would eliminate the concurrency benefit of `max_workers > 1` on `ReflectStep`, and (3) write steps (`TagStep`, `ApplyStep`) already run with `max_workers = 1`, so writes are serialised — only reads interleave with writes. If stricter isolation is ever needed, `SkillbookView` can be changed to snapshot on construction (deep copy) without altering step code.

### Generic run loop

Every subclass delegates to `_run()`. The only thing that varies per subclass is (1) the public `run()` signature and (2) the `_build_context()` method that maps input items to `ACEStepContext`.

```python
def _run(
    self,
    items: Sequence | Iterable,
    *,
    epochs: int,
    wait: bool = True,
    **kwargs,
) -> list[SampleResult]:
    if epochs > 1 and not isinstance(items, Sequence):
        raise ValueError("Multi-epoch requires a Sequence, not a consumed Iterable.")

    results: list[SampleResult] = []
    n = len(items) if isinstance(items, Sequence) else None

    for epoch in range(1, epochs + 1):
        contexts = [
            self._build_context(item, epoch=epoch, total_epochs=epochs,
                                index=idx, total=n,
                                global_sample_index=(epoch - 1) * n + idx if n is not None else idx,
                                **kwargs)
            for idx, item in enumerate(items, start=1)
        ]
        epoch_results = self.pipeline.run(contexts)
        results.extend(epoch_results)

    if wait:
        self.pipeline.wait_for_background()
    return results
```

**`wait` parameter:** When `wait=True` (default), `_run()` blocks until all background learning completes before returning — results are fully populated. When `wait=False`, `_run()` returns immediately after the foreground steps finish. Background learning continues asynchronously. Use `wait_for_background()` to drain later, or `learning_stats` to monitor progress.

The runner builds fully-initialized `ACEStepContext` objects (epoch counters, pre-filled outputs for traces, etc.) and hands them to `Pipeline.run(contexts)`. Construction IS initialization — from that point on contexts are frozen and the pipeline processes what it receives without wrapping or guessing. `Pipeline.run()` handles iteration, error isolation, foreground/background split, and concurrent workers. The runner only owns the epoch loop.

---

## TraceAnalyser

Analyses pre-recorded traces without executing an agent. Runs the learning tail only. Accepts raw trace objects of any type — the raw trace is placed directly on `ctx.trace` and the Reflector is responsible for making sense of it.

### When to use

- You have execution logs from an external system (browser-use, LangChain, custom agent, human sessions).
- You want to build or refine a skillbook from historical data.
- You want to re-analyse the same data multiple times (multi-epoch) to extract deeper patterns.

### Pipeline

```
[ReflectStep] → [TagStep] → [UpdateStep] → [ApplyStep]
```

No AgentStep, no EvaluateStep. The trace already contains the agent's output and the evaluation feedback.

### Context building

TraceAnalyser places the raw trace directly on `ctx.trace`. No extraction, no conversion — the Reflector receives the trace as-is and has full freedom to analyze it however it sees fit.

```python
def _build_context(self, raw_trace, *, epoch, total_epochs, index, total, global_sample_index) -> ACEStepContext:
    return ACEStepContext(
        skillbook=SkillbookView(self.skillbook),
        trace=raw_trace,                         # raw object, no enforced schema
        epoch=epoch,
        total_epochs=total_epochs,
        step_index=index,
        total_steps=total,
        global_sample_index=global_sample_index,
    )
```

The `skillbook` field is a `SkillbookView` — read-only steps access it from the context. Write steps (TagStep, ApplyStep) receive the real `Skillbook` via constructor injection. Each sample is independent — no state carries over from previous samples.

### Interface

```python
class TraceAnalyser(ACERunner):
    """Analyse pre-recorded traces to build a skillbook."""

    @classmethod
    def from_client(cls, client, *, skillbook=None, **kwargs) -> "TraceAnalyser": ...

    @classmethod
    def from_roles(cls, *, reflector, skill_manager, skillbook=None, **kwargs) -> "TraceAnalyser": ...

    def run(
        self,
        traces: Sequence[Any],
        epochs: int = 1,
    ) -> list[SampleResult]: ...
```

Note: no `environment` parameter, no converter. The raw trace goes straight onto the context. The Reflector is responsible for making sense of it — this gives maximum flexibility for analysis without constraining trace format. Extraction into `agent_output`/`environment_result` can be added later as an optional step if needed. No checkpoint parameters — checkpoints are configured at construction time via the factory methods.

### Multi-epoch semantics

Each epoch re-processes all traces with the current (evolving) skillbook. Early epochs extract obvious patterns; later epochs refine and consolidate.

```
Epoch 1:  trace₁ → trace₂ → ... → traceₙ   (skillbook grows)
Epoch 2:  trace₁ → trace₂ → ... → traceₙ   (skillbook refines)
Epoch 3:  trace₁ → trace₂ → ... → traceₙ   (diminishing returns)
```

Each sample is independent. The only thing that evolves across samples (and epochs) is the skillbook itself — visible as a read-only `SkillbookView` on the context, mutated by write steps via the real `Skillbook` (constructor-injected).

### run() — delegates to _run()

```python
def run(self, traces, epochs=1, *, wait=True):
    return self._run(traces, epochs=epochs, wait=wait)
```

No epoch loop, no per-sample iteration — `_run()` handles all of that.

---

## ACE

The full live adaptive pipeline. An agent executes, the environment evaluates, the reflector analyses, the skill manager updates.

### When to use

- You are building a new agent from scratch.
- You have a `TaskEnvironment` that can evaluate outputs.
- You want closed-loop learning where the agent improves in real time.

### Pipeline

```
[AgentStep] → [EvaluateStep] → [ReflectStep] → [TagStep] → [UpdateStep] → [ApplyStep]
```

### Context building

```python
def _build_context(self, sample, *, epoch, total_epochs, index, total, global_sample_index, environment, **_) -> ACEStepContext:
    return ACEStepContext(
        sample=sample,
        skillbook=SkillbookView(self.skillbook),
        environment=environment,
        epoch=epoch,
        total_epochs=total_epochs,
        step_index=index,
        total_steps=total,
        global_sample_index=global_sample_index,
    )
```

Each sample is independent. The `skillbook` field is a `SkillbookView` (read-only). Write steps receive the real `Skillbook` via constructor injection. The `environment` is forwarded from `run()` → `_run(**kwargs)` → `_build_context()` — no instance state modified.

Note: `environment` is on the context here (unlike TraceAnalyser) because `EvaluateStep` needs it and doesn't own the environment — it's per-run, not per-step. The environment is an immutable evaluator, so putting it on the frozen context is safe.

### Interface

```python
class ACE(ACERunner):
    """Live adaptive pipeline: Agent → Evaluate → Reflect → Tag → Update → Apply."""

    @classmethod
    def from_client(cls, client, *, skillbook=None, **kwargs) -> "ACE": ...

    @classmethod
    def from_roles(cls, *, agent, reflector, skill_manager, skillbook=None, **kwargs) -> "ACE": ...

    def run(
        self,
        samples: Sequence[Sample] | Iterable[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]: ...
```

### Single-pass vs multi-epoch

A single class handles both use cases. `epochs=1` gives single-pass behaviour. `epochs > 1` gives multi-epoch batch training.

```python
# Single pass (was OnlineACE)
results = ace.run(samples, environment, epochs=1)

# Multi-epoch batch (was OfflineACE)
results = ace.run(training_set, environment, epochs=3)

# Fire-and-forget — agent results returned fast, learning continues in background
results = ace.run(samples, environment, wait=False)
```

When `samples` is an `Iterable` (not `Sequence`), `epochs` must be `1` — you cannot replay a consumed iterable. `_run()` raises `ValueError` if `epochs > 1` and `samples` is not a `Sequence`. Note: `_run()` materializes the full iterable into a list of contexts before passing them to `Pipeline.run()`. This is a deliberate simplification — see Potential Improvements.

### run() — delegates to _run()

```python
def run(self, samples, environment, epochs=1, *, wait=True):
    return self._run(samples, epochs=epochs, wait=wait, environment=environment)
```

The public `run()` forwards `environment` through `_run(**kwargs)` to `_build_context()`. No instance state is modified — the runner stays reentrant.

---

## Factory Methods

Both TraceAnalyser and ACE provide the same two factory patterns:

### `from_client` — one LLM, zero boilerplate

```python
# TraceAnalyser: creates Reflector + SkillManager from the client
analyser = TraceAnalyser.from_client(LiteLLMClient(model="gpt-4o-mini"))

# ACE: creates Agent + Reflector + SkillManager from the client
ace = ACE.from_client(LiteLLMClient(model="gpt-4o-mini"))
```

### `from_roles` — full control

```python
# TraceAnalyser: bring your own roles
analyser = TraceAnalyser.from_roles(
    reflector=Reflector(llm, prompt_template=custom_prompt),
    skill_manager=SkillManager(llm),
    skillbook=existing_skillbook,
)

# ACE: bring your own roles
ace = ACE.from_roles(
    agent=Agent(llm),
    reflector=Reflector(llm),
    skill_manager=SkillManager(llm),
    skillbook=existing_skillbook,
    dedup_config=DeduplicationConfig(similarity_threshold=0.85),
)
```

### Common parameters on `from_roles`

| Parameter | Default | Description |
|---|---|---|
| `skillbook` | `Skillbook()` | Starting skillbook (empty if not provided) |
| `max_refinement_rounds` | `1` | Reflector iteration depth |
| `dedup_config` | `None` | Appends a `DeduplicateStep` to the pipeline |
| `dedup_interval` | `10` | Deduplication frequency (samples between runs) |
| `checkpoint_dir` | `None` | Appends a `CheckpointStep` to the pipeline |
| `checkpoint_interval` | `10` | Checkpoint frequency (samples between saves) |

Checkpoint and deduplication are configured at construction time. The factory conditionally appends the corresponding steps to the pipeline tail:

```python
@classmethod
def from_roles(cls, *, reflector, skill_manager, skillbook=None,
               dedup_config=None, dedup_interval=10,
               checkpoint_dir=None, checkpoint_interval=10, **kwargs):
    skillbook = skillbook or Skillbook()
    steps = learning_tail(
        reflector, skill_manager, skillbook,
        dedup_config=dedup_config, dedup_interval=dedup_interval,
        checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
        **kwargs,
    )
    return cls(pipeline=Pipeline(steps), skillbook=skillbook)
```

Read-only steps (ReflectStep, UpdateStep) access the skillbook via `ctx.skillbook` (a `SkillbookView`). Write steps (TagStep, ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor.

---

## Steps

Reusable step implementations live in `ace/steps/`. Each is a single class in a single file. All satisfy the `StepProtocol` from the pipeline engine. Each step does exactly one thing.

**Design principle: steps are stateless.** A step's `__call__` is a pure function of its constructor arguments and the incoming `ACEStepContext`. No internal counters, no accumulated state between invocations. If a step needs run-scoped information (like a global sample index for interval logic), the runner computes it and places it on the context. This keeps steps predictable across multiple `run()` calls — behaviour depends only on what's in the context, not on invocation history.

### Step Summary

| Step | Requires (context) | Injected (constructor) | Provides | Side effects | `max_workers` |
|---|---|---|---|---|---|
| **AgentStep** | `sample`, `skillbook` | `agent` | `agent_output` | None | default (1) |
| **EvaluateStep** | `sample`, `agent_output`, `environment` | — | `environment_result` | None | default (1) |
| **ReflectStep** | `sample`, `agent_output`, `environment_result`, `skillbook` | `reflector` | `reflection` | None (pure) | 3; `async_boundary = True` |
| **TagStep** | `reflection` | `skillbook` (real) | — | Tags skills on skillbook | 1 |
| **UpdateStep** | `reflection`, `sample`, `environment_result`, `agent_output`, `skillbook` | `skill_manager` | `skill_manager_output` | None (pure) | 1 |
| **ApplyStep** | `skill_manager_output` | `skillbook` (real) | — | Applies update batch to skillbook | 1 |
| **DeduplicateStep** | `global_sample_index` | `dedup_config`, `skillbook` (real) | — | Consolidates similar skills | 1 |
| **CheckpointStep** | `global_sample_index` | `skillbook` (real) | — | Saves skillbook to disk | 1 |
| **ObservabilityStep** | `sample`, `agent_output`, `environment_result`, `reflection`, `skill_manager_output`, `skillbook` | — | — | Logs metrics to Opik | 1 |

**Requires vs Injected:** `Requires` lists context fields read by the step — validated by the pipeline engine at construction time. The `skillbook` field on the context is a `SkillbookView` (read-only). Steps that **write** to the skillbook (TagStep, ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor injection — marked as "(real)" in the table. These injected dependencies are not tracked by `requires`/`provides`.

Steps with `provides = —` are pure side-effect steps (`provides = frozenset()`). They mutate shared state (skillbook) or write to external systems (disk, Opik) but add no new fields to the context.

### AgentStep

```python
class AgentStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output"})

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        agent_output = self.agent.generate(
            question=ctx.sample.question,
            context=ctx.sample.context,
            skillbook=ctx.skillbook,       # SkillbookView (read-only)
            sample=ctx.sample,
        )
        return ctx.replace(agent_output=agent_output)
```

Reads the skillbook via `ctx.skillbook` (a `SkillbookView`). No constructor injection needed — read-only access is sufficient.

### EvaluateStep

```python
class EvaluateStep:
    requires = frozenset({"sample", "agent_output", "environment"})
    provides = frozenset({"environment_result"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        environment_result = ctx.environment.evaluate(
            sample=ctx.sample,
            agent_output=ctx.agent_output,
        )
        return ctx.replace(environment_result=environment_result)
```

Stateless. Reads the environment from the context (it's an immutable evaluator, safe on a frozen dataclass). No skillbook access needed.

### ReflectStep

```python
class ReflectStep:
    requires = frozenset({"sample", "agent_output", "environment_result", "skillbook"})
    provides = frozenset({"reflection"})

    async_boundary = True
    max_workers = 3

    def __init__(self, reflector: Reflector, *, max_refinement_rounds=1) -> None:
        self.reflector = reflector
        self.max_refinement_rounds = max_refinement_rounds

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        reflection = self.reflector.reflect(
            ...,
            skillbook=ctx.skillbook,       # SkillbookView (read-only)
        )
        return ctx.replace(reflection=reflection)
```

Pure — produces a reflection object, no side effects. Reads `ctx.skillbook` (a `SkillbookView`) for LLM prompt context. Declares `async_boundary = True` — everything from here onward runs in a background thread pool. This lets AgentStep + EvaluateStep return fast while learning continues.

### TagStep

```python
class TagStep:
    requires = frozenset({"reflection"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, skillbook: Skillbook) -> None:
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        for tag in ctx.reflection.skill_tags:
            try:
                self.skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                logger.warning("TagStep: skill_id %r not found, skipping tag %r", tag.id, tag.tag)
        return ctx
```

Side-effect step — tags skills on `self.skillbook` (the real `Skillbook`, injected via constructor — not the `SkillbookView` on the context). `max_workers = 1` serialises skillbook writes. Hallucinated skill IDs from the Reflector are logged at `WARNING` level rather than silently swallowed — this provides a diagnostic signal without aborting the pipeline.

Separated from ReflectStep so that:
- ReflectStep is a pure function (LLM call → reflection object) and can be tested without a skillbook.
- TagStep can be tested with a mock reflection without an LLM.

### UpdateStep

```python
class UpdateStep:
    requires = frozenset({"reflection", "sample", "environment_result", "agent_output", "skillbook"})
    provides = frozenset({"skill_manager_output"})

    max_workers = 1

    def __init__(self, skill_manager: SkillManager) -> None:
        self.skill_manager = skill_manager

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        output = self.skill_manager.update(
            ...,
            skillbook=ctx.skillbook,       # SkillbookView (read-only)
        )
        # Attach insight-source provenance metadata
        for op in output.operations:
            op.metadata["insight_source"] = build_insight_source(ctx)
        return ctx.replace(skill_manager_output=output)
```

Pure — generates update operations via an LLM call (reading `ctx.skillbook` for context) and attaches provenance metadata, but does not mutate the skillbook. `max_workers = 1` because the skill manager reads the current skillbook state and concurrent calls would see stale data.

### ApplyStep

```python
class ApplyStep:
    requires = frozenset({"skill_manager_output"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, skillbook: Skillbook) -> None:
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        self.skillbook.apply_update(ctx.skill_manager_output)
        return ctx
```

Side-effect step — applies the update batch to `self.skillbook` (the real `Skillbook`, injected via constructor). Separated from UpdateStep so that:
- UpdateStep can be tested without mutating a skillbook (check that correct operations are generated).
- ApplyStep can be tested with a mock update batch (check that operations are applied correctly).

### DeduplicateStep

```python
class DeduplicateStep:
    requires = frozenset({"global_sample_index"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, config: DeduplicationConfig, skillbook: Skillbook, *, interval: int = 10) -> None:
        self.manager = DeduplicationManager(config)
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx
        self.manager.deduplicate(self.skillbook)
        return ctx
```

Optional side-effect step — consolidates similar skills in `self.skillbook` (injected). Appended to the pipeline by factory methods when `dedup_config` is provided. Stateless — uses `ctx.global_sample_index` (computed by the runner) with a configurable `interval` (default 10) to skip most invocations. Deduplication involves O(n²) similarity comparisons across all skills, so running it on every sample would be expensive as the skillbook grows.

### CheckpointStep

Optional tail step that periodically saves the skillbook to disk. Stateless — derives the checkpoint decision from context fields.

```python
class CheckpointStep:
    requires = frozenset({"global_sample_index"})
    provides = frozenset()

    def __init__(self, directory: str | Path, skillbook: Skillbook, *, interval: int = 10) -> None:
        self.directory = Path(directory)
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx

        self.directory.mkdir(parents=True, exist_ok=True)
        self.skillbook.save_to_file(str(self.directory / f"checkpoint_{ctx.global_sample_index}.json"))
        self.skillbook.save_to_file(str(self.directory / "latest.json"))
        return ctx
```

Key points:
- **Stateless.** Uses `ctx.global_sample_index` (computed by the runner) for interval logic. No internal counter, no reset needed.
- **`provides` is empty** — it only writes to disk, does not modify the context.
- **Skillbook via constructor** — saves `self.skillbook`, not a context field.
- **Placement:** Appended after ApplyStep by the factory when `checkpoint_dir` is provided. When `async_boundary` is set, checkpoints happen in the background tail.
- **`max_workers` not set** — inherits default of 1 from the pipeline engine, which is correct (disk writes should be serialised).

### ObservabilityStep

```python
class ObservabilityStep:
    requires = frozenset({"sample", "agent_output", "environment_result", "reflection", "skill_manager_output", "skillbook"})
    provides = frozenset()

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        # Log metrics to Opik, including skillbook size from ctx.skillbook
        skill_count = len(ctx.skillbook)   # SkillbookView (read-only)
        ...
        return ctx
```

Optional side-effect step — logs metrics to Opik. Reads `ctx.skillbook` (a `SkillbookView`) for metrics (skill count, etc.). No constructor needed — fully stateless. Appended to the pipeline when Opik is installed.

---

## Integration Pattern

External frameworks (browser-use, LangChain, Claude Code) integrate by providing **execute steps** that compose into an `ACERunner`. No separate pipeline classes per integration — just steps that plug into the standard runner infrastructure.

### Core idea: integrations are sub-pipelines

An integration provides the "execute" part. The "learn" part is always the same. The runner composes them:

```
Standard ACE:      [Agent → Evaluate]      → [Reflect → Tag → Update → Apply]
                    ╰── execute (built-in) ╯   ╰──────── learn (shared) ──────╯

Browser-use:       [BrowserExecute]         → [Reflect → Tag → Update → Apply]
                    ╰── execute (integration)╯  ╰──────── learn (shared) ──────╯

LangChain:         [LangChainExecute]       → [Reflect → Tag → Update → Apply]

Claude Code:       [ClaudeCodeExecute]      → [Reflect → Tag → Update → Apply → Persist]
```

Each integration provides one or more execute steps. These steps `provide` at minimum `{"agent_output", "environment_result"}` — the same fields that `AgentStep + EvaluateStep` produce in the standard ACE pipeline. This is the contract that `ReflectStep` requires.

### Execute step contract

Every integration execute step must satisfy:

```python
class SomeExecuteStep:
    requires = frozenset({"sample", "skillbook"})                     # minimum
    provides = frozenset({"agent_output", "environment_result"})      # minimum

    def __init__(self, framework_client) -> None:
        self.framework_client = framework_client

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        # Run framework using ctx.skillbook (SkillbookView), evaluate result
        ...
        return ctx.replace(agent_output=..., environment_result=...)
```

Example — browser-use:

```python
class BrowserExecuteStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output", "environment_result"})

    def __init__(self, browser_agent) -> None:
        self.browser_agent = browser_agent

    async def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        # Inject skillbook context into the browser agent
        skillbook_context = ctx.skillbook.as_prompt()  # SkillbookView (read-only)

        # Execute with framework
        history = await self.browser_agent.run(
            task=ctx.sample.question,
            additional_context=skillbook_context,
        )

        # Convert to ACE types
        agent_output = AgentOutput(
            reasoning=history.model_thoughts(),
            final_answer=history.final_result() or "No result",
            skill_ids=[],
            raw={"steps": len(history.history)},
        )
        env_result = EnvironmentResult(
            feedback=f"Completed in {len(history.history)} steps" if history.is_done() else "Failed",
            ground_truth=None,
        )
        return ctx.replace(agent_output=agent_output, environment_result=env_result)
```

### Composing into a runner

Integrations compose their execute step(s) with the shared learning tail and pass the result to `ACERunner`:

```python
class BrowserACE(ACERunner):
    """Browser-use integration runner."""

    @classmethod
    def from_client(cls, browser_agent, ace_client, *, skillbook=None, **kwargs):
        skillbook = skillbook or Skillbook()
        steps = [
            BrowserExecuteStep(browser_agent),
            *learning_tail(Reflector(ace_client), SkillManager(ace_client), skillbook, **kwargs),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(self, tasks, *, epochs=1, wait=True, **kwargs):
        samples = [Sample(question=t) if isinstance(t, str) else t for t in tasks]
        return self._run(samples, epochs=epochs, wait=wait)

    def _build_context(self, sample, *, epoch, total_epochs, index, total, global_sample_index):
        return ACEStepContext(
            sample=sample,
            skillbook=SkillbookView(self.skillbook),
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
            global_sample_index=global_sample_index,
        )
```

The pattern is the same for every integration: subclass `ACERunner`, provide a factory that wires the execute step(s) + learning tail, override `run()` with the integration-specific signature, and implement `_build_context()`.

### `learning_tail()` — reusable learning steps

Every integration assembles the same `[Reflect → Tag → Update → Apply]` suffix with optional dedup and checkpoint steps. Rather than duplicating this wiring in every factory method, `learning_tail()` returns the standard step list:

```python
# ace/steps/__init__.py

def learning_tail(
    reflector: Reflector,
    skill_manager: SkillManager,
    skillbook: Skillbook,
    *,
    dedup_config: DeduplicationConfig | None = None,
    dedup_interval: int = 10,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 10,
) -> list[Step]:
    """Return the standard ACE learning steps.

    Use this when building custom integrations that provide their own
    execute step(s) but want the standard learning pipeline.
    """
    steps: list[Step] = [
        ReflectStep(reflector),
        TagStep(skillbook),
        UpdateStep(skill_manager),
        ApplyStep(skillbook),
    ]
    if dedup_config:
        steps.append(DeduplicateStep(dedup_config, skillbook, interval=dedup_interval))
    if checkpoint_dir:
        steps.append(CheckpointStep(checkpoint_dir, skillbook, interval=checkpoint_interval))
    return steps
```

Integration factories become shorter and less error-prone:

```python
class BrowserACE(ACERunner):
    @classmethod
    def from_client(cls, browser_agent, ace_client, *, skillbook=None, **kwargs):
        skillbook = skillbook or Skillbook()
        steps = [
            BrowserExecuteStep(browser_agent),
            *learning_tail(Reflector(ace_client), SkillManager(ace_client), skillbook, **kwargs),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)
```

Power users building fully custom pipelines can also use it:

```python
from ace.steps import learning_tail

skillbook = Skillbook.load_from_file("expert.json")
steps = [
    MyCustomExecuteStep(my_agent),
    MyValidationStep(),  # custom step before learning
    *learning_tail(reflector, skill_manager, skillbook, dedup_config=dedup),
]
runner = ACERunner(Pipeline(steps), skillbook)
```

### TraceAnalyser — batch learning from recorded executions

Integrations also support offline learning. When an integration records execution history (browser-use AgentHistory, LangChain intermediate_steps, Claude Code transcripts), it feeds the raw objects directly to TraceAnalyser:

```python
# Record browser executions
histories = [await agent.run(task) for task in tasks]

# Feed raw histories directly — Reflector analyses them as-is
analyser = TraceAnalyser.from_client(llm_client)
analyser.run(histories, epochs=2)
analyser.save("browser_expert.json")
```

### Live vs offline

| | Integration Runner | TraceAnalyser |
|---|---|---|
| When | Live execution | Post-hoc analysis |
| Agent | Framework runs it | Already ran |
| Feedback | Generated live | Baked into trace |
| Use case | Production deployment | Historical batch learning, debugging |

Both update the same skillbook. A common workflow: TraceAnalyser builds an initial skillbook from historical data, then an integration runner refines it during live deployment.

---

## High-Level Integration Wrappers

The user-facing wrappers (ACELiteLLM, ACEAgent, ACELangChain) remain as convenience classes. Internally, they lazy-init and cache the appropriate runner. Since runners are reentrant (no per-call instance state), caching is safe.

```python
class ACELiteLLM:
    def __init__(self, llm, skillbook, ...):
        self.agent = Agent(llm, ...)
        self.reflector = Reflector(llm, ...)
        self.skill_manager = SkillManager(llm, ...)
        self.skillbook = skillbook
        self._ace: ACE | None = None
        self._analyser: TraceAnalyser | None = None

    def _get_ace(self) -> ACE:
        if self._ace is None:
            self._ace = ACE.from_roles(
                agent=self.agent,
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                skillbook=self.skillbook,
            )
        return self._ace

    def _get_analyser(self) -> TraceAnalyser:
        if self._analyser is None:
            self._analyser = TraceAnalyser.from_roles(
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                skillbook=self.skillbook,
            )
        return self._analyser

    def ask(self, question) -> str:
        """Use current skillbook to answer."""
        ...

    def learn(self, samples, environment, epochs=1, wait=True):
        """Delegate to ACE."""
        return self._get_ace().run(samples, environment, epochs=epochs, wait=wait)

    def learn_from_traces(self, traces, epochs=1, wait=True):
        """Delegate to TraceAnalyser."""
        return self._get_analyser().run(traces, epochs=epochs, wait=wait)


class ACEAgent:
    """Browser-use convenience wrapper. Delegates to BrowserACE."""

    def __init__(self, browser_agent, ace_client, **kwargs):
        self.runner = BrowserACE.from_client(
            browser_agent, ace_client, **kwargs
        )

    async def run(self, task):
        results = self.runner.run([task])
        return results[0]

    @property
    def skillbook(self):
        return self.runner.skillbook
```

Each wrapper is a thin facade over an `ACERunner` subclass. The runner owns the pipeline, the skillbook, and the epoch loop.

---

## Directory Structure

```
ace/
  steps/                    ← generic steps (one file per class)
    __init__.py
    agent.py                ← AgentStep
    evaluate.py             ← EvaluateStep
    reflect.py              ← ReflectStep
    tag.py                  ← TagStep
    update.py               ← UpdateStep
    apply.py                ← ApplyStep
    deduplicate.py          ← DeduplicateStep
    checkpoint.py           ← CheckpointStep
    observability.py        ← ObservabilityStep
  trace_analyser.py         ← TraceAnalyser class
  ace.py                    ← ACE class
  runner.py                 ← ACERunner base class
  integrations/
    browser_use/
      runner.py             ← BrowserACE (ACERunner subclass)
      steps/
        execute.py          ← BrowserExecuteStep
    langchain/
      runner.py             ← LangChainACE (ACERunner subclass)
      steps/
        execute.py          ← LangChainExecuteStep
    claude_code/
      runner.py             ← ClaudeCodeACE (ACERunner subclass)
      steps/
        execute.py          ← ClaudeCodeExecuteStep
        persist.py          ← PersistStep
    litellm.py              ← ACELiteLLM (high-level wrapper)
    base.py                 ← wrap_skillbook_context (unchanged)
```

Each integration provides: (1) an execute step and (2) an `ACERunner` subclass that composes the step with the learning tail. For offline analysis, raw trace objects are passed directly to TraceAnalyser. No separate pipeline classes.

### What moves where

| Old location | New location | Notes |
|---|---|---|
| `ace/adaptation.py` | Deleted | Replaced by `runner.py`, `ace.py`, `trace_analyser.py` |
| `ace/async_learning.py` | Deleted | Replaced by pipeline engine `async_boundary` |
| `ace/environments.py` | Unchanged | `Sample`, `EnvironmentResult`, `TaskEnvironment`, `SimpleEnvironment` stay |
| `ace2/` | Deleted | Superseded by this design |
| New | `ace/steps/tag.py` | TagStep (split from ReflectStep) |
| New | `ace/steps/apply.py` | ApplyStep (split from UpdateStep) |
| New | `ace/steps/deduplicate.py` | DeduplicateStep (extracted from SkillManager) |
| New | `ace/steps/checkpoint.py` | CheckpointStep |
| New | `ace/steps/observability.py` | ObservabilityStep |
| New | `ace/trace_analyser.py` | TraceAnalyser class |
| New | `ace/runner.py` | ACERunner base class |
| `ace/integrations/browser_use.py` | `ace/integrations/browser_use/` | Split into runner + steps |
| `ace/integrations/langchain.py` | `ace/integrations/langchain/` | Split into runner + steps |
| `ace/integrations/claude_code.py` | `ace/integrations/claude_code/` | Split into runner + steps |

---

## Async Behaviour

Both TraceAnalyser and ACE inherit async capabilities from the pipeline engine. No custom async machinery is needed.

### ReflectStep as async boundary

`ReflectStep.async_boundary = True` means: when the pipeline processes a sample, everything before ReflectStep (Agent, Evaluate) runs in the foreground, and everything from ReflectStep onwards (Tag, Update, Apply, Deduplicate, Checkpoint) runs in a background thread pool.

```
sample 1:  [AgentStep] [EvaluateStep] ──fire──► [ReflectStep] [TagStep] [UpdateStep] [ApplyStep]  (background)
sample 2:  [AgentStep] [EvaluateStep] ──fire──► [ReflectStep] [TagStep] [UpdateStep] [ApplyStep]  (background)
                                       ↑
                                 async_boundary
```

For TraceAnalyser, there is no AgentStep or EvaluateStep in the foreground. The boundary still applies — context building is foreground, the learning tail is background:

```
trace 1:  [build_context] ──fire──► [ReflectStep] [TagStep] [UpdateStep] [ApplyStep]  (background)
trace 2:  [build_context] ──fire──► [ReflectStep] [TagStep] [UpdateStep] [ApplyStep]  (background)
```

### Controlling concurrency

| Knob | Where | Effect |
|---|---|---|
| `ReflectStep.max_workers = 3` | Step class attribute | Up to 3 reflections run in parallel |
| `TagStep.max_workers = 1` | Step class attribute | Serialises skill tagging |
| `UpdateStep.max_workers = 1` | Step class attribute | Serialises skill manager LLM calls |
| `ApplyStep.max_workers = 1` | Step class attribute | Serialises skillbook writes |
| `wait_for_background(timeout)` | Runner method | Blocks until background threads drain |

No custom `AsyncLearningPipeline` class, no manual thread management, no `asyncio.create_task` for background learning. The pipeline engine handles all of it.

---

## Error Handling

Follows the pipeline engine's error model without additions.

**Per-sample isolation:** A failing sample does not abort the run. The pipeline catches the exception, records it in `SampleResult.error` and `SampleResult.failed_at`, and continues to the next sample.

**Background failures:** Captured and attached to `SampleResult` by the pipeline engine. The runner calls `wait_for_background()` at the end to ensure all results are complete.

**No retry logic in the runner.** Retries are the responsibility of individual steps (e.g., LLM call retries via `tenacity` in the role classes).

---

## Usage Examples

### TraceAnalyser — learn from browser-use history

```python
from ace import TraceAnalyser
from ace.llm_providers import LiteLLMClient

# Raw traces — plain dicts, no enforced schema
traces = [
    {
        "task": "Find the cheapest flight to Tokyo",
        "output": "$450 on ANA, departing March 15",
        "feedback": "Correct price found in 8 steps",
        "reasoning": "Step 1: Navigate to Google Flights...",
    },
    {
        "task": "Book a hotel in Shibuya",
        "output": "Failed: could not find checkout button",
        "feedback": "Task failed after 15 steps — checkout button was behind a cookie modal",
        "reasoning": "Step 1: Navigate to Booking.com...",
    },
]

# Analyse — raw traces go directly to the Reflector via ctx.trace
analyser = TraceAnalyser.from_client(
    LiteLLMClient(model="gpt-4o-mini"),
)
results = analyser.run(traces, epochs=2)
analyser.save("travel_agent.json")
```

### ACE — live Q&A training

```python
from ace import ACE, Sample, SimpleEnvironment
from ace.llm_providers import LiteLLMClient

samples = [
    Sample(question="Capital of France?", ground_truth="Paris"),
    Sample(question="Largest ocean?", ground_truth="Pacific"),
]

ace = ACE.from_client(LiteLLMClient(model="gpt-4o-mini"))
results = ace.run(samples, SimpleEnvironment(), epochs=3)
ace.save("geography.json")
```

### ACE — single-pass with iterable

```python
# Any Iterable works with epochs=1 (consumed once, not replayed)
samples = load_samples_from_csv("eval_set.csv")  # returns a list or generator

ace = ACE.from_client(LiteLLMClient(model="gpt-4o-mini"))
results = ace.run(samples, environment, epochs=1)
```

### ACE — with checkpoints and deduplication

```python
ace = ACE.from_roles(
    agent=Agent(llm),
    reflector=Reflector(llm),
    skill_manager=SkillManager(llm),
    dedup_config=DeduplicationConfig(similarity_threshold=0.85),
    checkpoint_dir="./checkpoints",
    checkpoint_interval=10,
)
# Pipeline: Agent → Evaluate → Reflect → Tag → Update → Apply → Deduplicate → Checkpoint
results = ace.run(samples, environment, epochs=3)
```

### Integration — browser-use runner

```python
from ace.integrations.browser_use import BrowserACE
from browser_use import Agent as BrowserAgent

browser_agent = BrowserAgent(llm=ChatOpenAI(model="gpt-4o"))
runner = BrowserACE.from_client(
    browser_agent,
    ace_client=LiteLLMClient(model="gpt-4o-mini"),
)

# Live execution + learning
results = runner.run(["Find top HN post", "Check weather in Tokyo"])
runner.save("browser_expert.json")
```

### Fire-and-forget — get results while learning continues

```python
ace = ACE.from_client(LiteLLMClient(model="gpt-4o-mini"))

# wait=False: returns after foreground steps (Agent + Evaluate)
# Background learning (Reflect → Tag → Update → Apply) continues
results = ace.run(samples, environment, epochs=1, wait=False)

# Use agent outputs immediately
for r in results:
    print(r.output.agent_output.final_answer)

# Check learning progress
print(ace.learning_stats)
# {"active": 3, "completed": 12}

# Block when you need the skillbook finalised
ace.wait_for_background(timeout=60.0)
ace.save("learned.json")
```

### Mixed workflow — batch then live

```python
# Phase 1: build skillbook from historical traces
analyser = TraceAnalyser.from_client(llm, skillbook=Skillbook())
analyser.run(historical_traces, epochs=3)
skillbook = analyser.skillbook

# Phase 2: deploy with live learning
ace = ACE.from_client(llm, skillbook=skillbook)
ace.run(live_samples, environment, epochs=1)
ace.save("production.json")
```

---

## Potential Improvements

Issues acknowledged but deferred from this version of the spec.

**Streaming / lazy iteration:**
`_run()` eagerly materializes the full iterable into a list of `ACEStepContext` objects before passing them to `Pipeline.run()`. For a large generator with `epochs=1`, the entire input gets buffered into memory. True streaming would require the pipeline to accept an iterator and process items one-at-a-time (e.g., `for ctx in contexts: pipeline.run_one(ctx)`), or an async iterator pattern with `asyncio.as_completed`. This is a deliberate simplification — batch materialization keeps the epoch loop and error handling straightforward. Revisit if memory pressure from large single-pass runs becomes a real problem.

**Builder API for custom pipelines:**
The current API offers two extremes: factory methods (`from_client`, `from_roles`) that hide the pipeline entirely, and manual `Pipeline([...])` construction that requires understanding `ACEStepContext`, `SkillbookView`, step contracts, and `ACERunner` subclassing. Users who want to insert a custom step between Reflect and Update, or swap the execute head while keeping the learning tail, fall into a gap where neither approach serves them well.

A builder API would bridge this gap:

```python
from ace import ACEBuilder

# Start from a preset, customise from there
ace = (
    ACEBuilder(model="gpt-4o-mini")
    .execute(MyCustomExecuteStep(my_agent))     # replace the execute head
    .add_step(MyValidationStep(), after="reflect")  # insert custom step
    .deduplicate(similarity_threshold=0.85)
    .checkpoint("./checkpoints", interval=10)
    .build()
)
results = ace.run(samples, environment)

# Or build from the standard ACE preset and tweak
ace = (
    ACEBuilder.from_preset("ace", model="gpt-4o-mini")
    .add_step(MyLoggingStep(), after="apply")
    .build()
)
```

The builder would handle `SkillbookView` wiring, step ordering validation, and `ACERunner` construction internally. Users compose by name ("reflect", "apply") rather than by importing step classes.

**When to implement:** When there is evidence of users building custom pipelines with `learning_tail()` and hitting friction with the manual wiring. The `learning_tail()` helper (see Integration Pattern section) covers the most common customisation — custom execute step + standard learning — without a builder. A builder adds value when users need fine-grained insertion points (between existing steps) or want to compose from presets without understanding the step internals.

**Risk:** A builder that mirrors the step list adds a second way to construct pipelines, which means two things to document, test, and keep in sync. It can also hide the `requires`/`provides` contracts — when a validation step is inserted at the wrong position, the error comes from the pipeline engine (field missing) rather than the builder (wrong position name), making debugging indirect. Mitigate by having the builder validate the final step chain at `build()` time and surfacing clear errors.

**Skillbook rollback and versioning:**
Currently the skillbook is mutated in place with no way to undo a bad update. If the LLM hallucinates a harmful skill or a batch degrades overall quality, the only recovery is restoring from a checkpoint file. A lightweight versioning mechanism — e.g., snapshotting skillbook state at epoch boundaries or before each `ApplyStep`, with a `rollback(to_version)` method — would enable automatic revert when a validation metric degrades, A/B comparison between skillbook versions, and safer experimentation with aggressive learning rates. This could live as a `VersionedSkillbook` wrapper or as an optional `SnapshotStep` inserted before `ApplyStep`. Deferred because the current checkpoint-to-disk approach covers the most common recovery scenario (resume after crash), and in-memory versioning adds memory overhead proportional to skillbook size times number of snapshots.

---

## What Was Rejected and Why

**Runner extends Pipeline:**
Making TraceAnalyser and ACE subclasses of `Pipeline` was considered. Rejected — the runner is not a pipeline. It owns the epoch loop. Composition (`self.pipeline`) keeps responsibilities separate.

**Cross-sample state (reflection window):**
A rolling window of recent reflections that persists across samples was considered, with variants: on the runner, on `StepContext`, on step instances, via a shared mediator object. All rejected — each sample should be independent. The only cross-sample coupling is the skillbook itself, which evolves as samples are processed. Adding a reflection window complicates the model (reset between epochs, eventual consistency with background steps, ordering issues with concurrent workers) for marginal benefit. Note: `StepContext.recent_reflections` from the old implementation should be removed.

**Separate Online and Offline classes:**
Keeping two runner classes for single-pass and multi-epoch was considered. Rejected — the only difference is `epochs=1` vs `epochs > 1`, which is a parameter, not a class distinction. ACE handles both. TraceAnalyser is a separate class because its input type is fundamentally different (raw traces vs `Sample + Environment`), not because of epoch count.

**Structured Trace dataclass:**
A `@dataclass Trace` with typed fields (`task`, `output`, `feedback`, `reasoning`, etc.) was considered. Rejected — it imposes a schema on trace data that doesn't match reality. External frameworks produce wildly different trace shapes (browser-use `AgentHistoryList`, LangChain result dicts, Claude Code transcripts). Forcing them through a common dataclass means either losing information (fields that don't map) or adding catch-all `metadata: dict` buckets that defeat the purpose of typing. Instead, `ctx.trace` is `object | None` — the raw trace as-is. The Reflector receives it directly and is responsible for making sense of it. This gives maximum flexibility for analysis without constraining trace format. Extraction helpers (converter functions, typed intermediate representations) can be layered on later if needed.

**Steps that accept both traces and samples:**
Making ReflectStep and UpdateStep polymorphic over input type was considered. Rejected — steps always receive `StepContext` with the same named fields. The runner (via `_build_context`) is responsible for building the context correctly. Steps do not need to know whether the data came from a raw trace or from live execution.

**Observability in the runner:**
Keeping observability logic in `ACERunner._track_observability_data()` was considered. Rejected — it mixes concerns. A dedicated `ObservabilityStep` is independently testable, optional, and composable.

**Custom AsyncLearningPipeline:**
The legacy `ace/async_learning.py` implements a manual thread pool with reflector and skill manager queues. Rejected — the pipeline engine's `async_boundary` and `max_workers` provide the same functionality with less code and consistent semantics.

**Per-integration pipeline classes:**
Having each integration define its own pipeline class was considered. Rejected — every integration pipeline has the same learning tail; only the execute step differs. Separate pipeline classes duplicate the learning tail wiring and the runner infrastructure. Instead, integrations provide execute steps that compose into an `ACERunner` subclass, reusing the shared `_run()` loop and epoch logic.

**Checkpoints in the runner:**
Having the runner own checkpoint logic (via `run()` parameters) was considered. Rejected — a `CheckpointStep` at the end of the pipeline tail keeps checkpointing within the pipeline formalism. Checkpoint configuration belongs at construction time (factory methods), not at call time (`run()`).

**Mutable Skillbook directly on the context:**
Storing the real `Skillbook` as a field on `ACEStepContext` was the initial design. Rejected — `StepContext` is frozen, but `Skillbook` is a mutable object. Placing it on the context creates the illusion of immutability while allowing any step to mutate shared state through the reference. Instead, the context carries a `SkillbookView` (read-only projection) that exposes only read methods (`as_prompt()`, `get_skill()`, `__len__`). Write methods don't exist on the view — calling them raises `AttributeError` at runtime and a type error at check time. Steps that need to write (TagStep, ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor injection. This gives us both: pipeline engine validation (skillbook is in `requires`/`provides`) and true immutability enforcement on the context.

**Combined Reflect+Tag and Update+Apply steps:**
Keeping ReflectStep as both reflection and tagging, and UpdateStep as both generation and application was considered. Rejected — each combination mixes a pure function (LLM call producing output) with a side effect (skillbook mutation). Splitting them means pure steps can be tested without a skillbook, side-effect steps can be tested without an LLM, and concerns are cleanly separated.
