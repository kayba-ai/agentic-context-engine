# ACE Architecture Design

Specification for rewriting the legacy `ace/` module to use the pipeline engine.

---

## Implementation Status

Implemented in `ace_next/` (parallel to `ace/` for easy rollback). The package is fully self-contained — all types are copied locally, zero imports from `ace/`.

| Component | Status | Location |
|---|---|---|
| Core types (`ACEStepContext`, `SkillbookView`, `ACESample`) | Done | `ace_next/context.py` |
| Data types (`Skill`, `Skillbook`, `UpdateBatch`, outputs) | Done | `ace_next/skill.py`, `skillbook.py`, `updates.py`, `outputs.py` |
| Environments (`Sample`, `TaskEnvironment`, etc.) | Done | `ace_next/environments.py` |
| Protocols (`AgentLike`, `ReflectorLike`, etc.) | Done | `ace_next/protocols/` |
| Steps (all 10) | Done | `ace_next/steps/` |
| `learning_tail()` helper | Done | `ace_next/steps/__init__.py` |
| `ACERunner` base class | Done | `ace_next/runners/base.py` |
| `TraceAnalyser` | Done | `ace_next/runners/trace_analyser.py` |
| `ACE` runner | Done | `ace_next/runners/ace.py` |
| Implementations (`Agent`, `Reflector`, `SkillManager`) | Done | `ace_next/implementations/` |
| Deduplication (`DeduplicationManager`, `SimilarityDetector`) | Done | `ace_next/deduplication/` |
| Integration steps (`BrowserExecuteStep`, `LangChainExecuteStep`, `ClaudeCodeExecuteStep`) | Done | `ace_next/integrations/` |
| Integration runners (`BrowserUse`, `LangChain`, `ClaudeCode`) | Done | `ace_next/runners/` |

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

    def skills(self, include_invalid: bool = False) -> list[Skill]:
        return self._sb.skills(include_invalid=include_invalid)

    def stats(self) -> dict[str, object]:
        return self._sb.stats()

    def __len__(self) -> int:
        return len(self._sb.skills())

    def __iter__(self):
        return iter(self._sb.skills())

    def __repr__(self) -> str:
        return f"SkillbookView({len(self)} skills)"
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
    trace: object | None = None
    agent_output: AgentOutput | None = None
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
| **Examples** | `agent_output`, `reflection`, `skillbook` (view) | `skillbook` (real), `environment`, `dedup_manager` |
| **Validated by engine?** | Yes — `requires`/`provides` | No — runtime error if missing |

---

## Protocols

Steps depend on protocols, not concrete classes. Each protocol defines the minimal interface a step needs. Concrete implementations satisfy them structurally — no inheritance required.

All protocols live in `ace_next/protocols/` (one file per protocol, re-exported from `__init__.py`).

| Protocol | Method | Used by | Satisfied by |
|---|---|---|---|
| `AgentLike` | `generate(question, context, skillbook, reflection, **kwargs) → AgentOutput` | `AgentStep` | `Agent` |
| `ReflectorLike` | `reflect(question, agent_output, skillbook, ground_truth, feedback, **kwargs) → ReflectorOutput` | `ReflectStep` | `Reflector` |
| `SkillManagerLike` | `update_skills(reflection, skillbook, question_context, progress, **kwargs) → SkillManagerOutput` | `UpdateStep` | `SkillManager` |
| `DeduplicationManagerLike` | `get_similarity_report(skillbook) → str \| None` | `DeduplicateStep` | `DeduplicationManager` |
| `LLMClientLike` | `complete(prompt, **kwargs) → Any` + `complete_structured(prompt, response_model, **kwargs) → T` | `Agent`, `Reflector`, `SkillManager` | Any LLM client with both methods |

### LLMClientLike

The implementations (`Agent`, `Reflector`, `SkillManager`) all depend on `LLMClientLike` — a protocol requiring two methods:

```python
@runtime_checkable
class LLMClientLike(Protocol):
    def complete(self, prompt: str, **kwargs: Any) -> Any: ...
    def complete_structured(self, prompt: str, response_model: type[T], **kwargs: Any) -> T: ...
```

`complete_structured` returns a validated Pydantic model instance. This is the key capability — implementations call `llm.complete_structured(prompt, AgentOutput)` and get back a typed, validated object. Any LLM client that provides both methods satisfies the protocol: `LiteLLMClient` wrapped with Instructor, a custom OpenAI wrapper, or a mock for testing.

**Design decision:** The old `ace/roles.py` auto-wrapped LLM clients with Instructor if `complete_structured` was missing. In `ace_next`, this auto-wrapping is removed — callers must pass a pre-wrapped client. This eliminates the hidden dependency on `ace.llm_providers.instructor_client` and makes the requirement explicit.

### Why protocols, not ABC

Protocols use structural typing (duck typing checked by mypy). A class satisfies a protocol if it has the right methods — no `class Agent(AgentLike)` inheritance needed. This means:
- Users can pass any object with a matching method, not just subclasses.
- Mocks satisfy protocols without ceremony.
- Steps are decoupled from implementations at the type level, not just by convention.

---

## Class Hierarchy

```
ACERunner (shared infrastructure: epoch loop, delegates to Pipeline.run())
├── TraceAnalyser       — [Reflect → Tag → Update → Apply]; input = any trace object
├── ACE                 — [Agent → Evaluate → Reflect → Tag → Update → Apply]; input = Sample + Environment
├── BrowserUse          — [BrowserExecute → BrowserToTrace → Reflect → Tag → Update → Apply]; input = task strings
├── LangChain           — [LangChainExecute → LangChainToTrace → Reflect → Tag → Update → Apply]; input = chain inputs
└── ClaudeCode          — [ClaudeCodeExecute → ClaudeCodeToTrace → Reflect → Tag → Update → Apply]; input = task strings
```

All compose a `Pipeline` rather than extending it. The pipeline is an implementation detail, not part of the public interface. Each subclass only overrides `run()` (public signature) and `_build_context()` (input mapping).

Each integration runner uses two steps before the learning tail: (1) an **execute step** that produces an integration-specific result type (e.g. `BrowserResult`), and (2) a **ToTrace step** that converts that result into the standardised trace dict the learning tail expects. This separation keeps framework-specific logic in the execute step and trace formatting in the converter — each is independently testable.

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

Note: no `environment` parameter, no converter. The raw trace goes straight onto the context. The Reflector is responsible for making sense of it — this gives maximum flexibility for analysis without constraining trace format. Extraction into structured fields can be added later as an optional step if needed. No checkpoint parameters — checkpoints are configured at construction time via the factory methods.

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

The full live adaptive pipeline. An agent executes, the reflector analyses, the skill manager updates. Optionally evaluates against a `TaskEnvironment` for feedback-driven learning.

### When to use

- You are building a new agent from scratch.
- You want closed-loop learning where the agent improves in real time.
- Optionally: you have a `TaskEnvironment` that can evaluate outputs (provides richer feedback for the Reflector).

### Pipeline

```
[AgentStep] → [EvaluateStep] → [ReflectStep] → [TagStep] → [UpdateStep] → [ApplyStep]
```

### Context building

```python
def _build_context(self, sample, *, epoch, total_epochs, index, total, global_sample_index, **_) -> ACEStepContext:
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

Each sample is independent. The `skillbook` field is a `SkillbookView` (read-only). Write steps receive the real `Skillbook` via constructor injection. The environment (if any) is injected into `EvaluateStep` at construction time — it does not appear on the context.

### Interface

```python
class ACE(ACERunner):
    """Live adaptive pipeline: Agent → Evaluate → Reflect → Tag → Update → Apply."""

    @classmethod
    def from_client(cls, client, *, environment=None, skillbook=None, **kwargs) -> "ACE": ...

    @classmethod
    def from_roles(cls, *, agent, reflector, skill_manager, environment=None, skillbook=None, **kwargs) -> "ACE": ...

    def run(
        self,
        samples: Sequence[Sample] | Iterable[Sample],
        epochs: int = 1,
        *,
        wait: bool = True,
    ) -> list[SampleResult]: ...
```

The `environment` is optional and provided at construction time, not at `run()` time. When provided, `EvaluateStep` uses it to generate feedback that enriches the trace. When omitted, the trace still contains the agent's output, question, context, and ground truth — the Reflector can learn from ground-truth comparison or from the agent's reasoning alone.

### Single-pass vs multi-epoch

A single class handles both use cases. `epochs=1` gives single-pass behaviour. `epochs > 1` gives multi-epoch batch training.

```python
# Single pass (was OnlineACE)
results = ace.run(samples, epochs=1)

# Multi-epoch batch (was OfflineACE)
results = ace.run(training_set, epochs=3)

# Fire-and-forget — agent results returned fast, learning continues in background
results = ace.run(samples, wait=False)
```

When `samples` is an `Iterable` (not `Sequence`), `epochs` must be `1` — you cannot replay a consumed iterable. `_run()` raises `ValueError` if `epochs > 1` and `samples` is not a `Sequence`. Note: `_run()` materializes the full iterable into a list of contexts before passing them to `Pipeline.run()`. This is a deliberate simplification — see Potential Improvements.

### run() — delegates to _run()

```python
def run(self, samples, epochs=1, *, wait=True):
    return self._run(samples, epochs=epochs, wait=wait)
```

No instance state is modified — the runner stays reentrant.

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
    dedup_manager=dedup_manager(similarity_threshold=0.85),
)
```

### Common parameters on `from_roles`

| Parameter | Default | Description |
|---|---|---|
| `skillbook` | `Skillbook()` | Starting skillbook (empty if not provided) |
| `max_refinement_rounds` | `1` | Reflector iteration depth |
| `dedup_manager` | `None` | Appends a `DeduplicateStep` to the pipeline |
| `dedup_interval` | `10` | Deduplication frequency (samples between runs) |
| `checkpoint_dir` | `None` | Appends a `CheckpointStep` to the pipeline |
| `checkpoint_interval` | `10` | Checkpoint frequency (samples between saves) |

Checkpoint and deduplication are configured at construction time. The factory conditionally appends the corresponding steps to the pipeline tail. Both classes follow the same pattern — ACE prepends its execute steps:

```python
# TraceAnalyser — learning tail only
@classmethod
def from_roles(cls, *, reflector, skill_manager, skillbook=None,
               dedup_manager=None, dedup_interval=10,
               checkpoint_dir=None, checkpoint_interval=10, **kwargs):
    skillbook = skillbook or Skillbook()
    steps = learning_tail(
        reflector, skill_manager, skillbook,
        dedup_manager=dedup_manager, dedup_interval=dedup_interval,
        checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
        **kwargs,
    )
    return cls(pipeline=Pipeline(steps), skillbook=skillbook)

# ACE — execute head + learning tail
@classmethod
def from_roles(cls, *, agent, reflector, skill_manager, environment=None,
               skillbook=None, dedup_manager=None, dedup_interval=10,
               checkpoint_dir=None, checkpoint_interval=10, **kwargs):
    skillbook = skillbook or Skillbook()
    steps = [
        AgentStep(agent),
        EvaluateStep(environment),
        *learning_tail(
            reflector, skill_manager, skillbook,
            dedup_manager=dedup_manager, dedup_interval=dedup_interval,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
            **kwargs,
        ),
    ]
    return cls(pipeline=Pipeline(steps), skillbook=skillbook)
```

Read-only steps (ReflectStep, UpdateStep) access the skillbook via `ctx.skillbook` (a `SkillbookView`). Write steps (TagStep, ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor.

---

## Steps

Reusable step implementations live in `ace_next/steps/`. Each is a single class in a single file. All satisfy the `StepProtocol` from the pipeline engine. Each step does exactly one thing.

**Design principle: steps are stateless.** A step's `__call__` is a pure function of its constructor arguments and the incoming `ACEStepContext`. No internal counters, no accumulated state between invocations. If a step needs run-scoped information (like a global sample index for interval logic), the runner computes it and places it on the context. This keeps steps predictable across multiple `run()` calls — behaviour depends only on what's in the context, not on invocation history.

### Step Summary

| Step | Requires (context) | Injected (constructor) | Provides | Side effects | `max_workers` |
|---|---|---|---|---|---|
| **AgentStep** | `sample`, `skillbook` | `agent` | `agent_output` | None | default (1) |
| **EvaluateStep** | `sample`, `agent_output` | `environment` (optional) | `trace` | None | default (1) |
| **ReflectStep** | `trace`, `skillbook` | `reflector` | `reflection` | None (pure) | 3; `async_boundary = True` |
| **TagStep** | `reflection` | `skillbook` (real) | — | Tags skills on skillbook | 1 |
| **UpdateStep** | `reflection`, `skillbook` | `skill_manager` | `skill_manager_output` | None (pure) | 1 |
| **ApplyStep** | `skill_manager_output` | `skillbook` (real) | — | Applies update batch to skillbook | 1 |
| **DeduplicateStep** | `global_sample_index` | `manager` (DeduplicationManagerLike), `skillbook` (real) | — | Consolidates similar skills | 1 |
| **CheckpointStep** | `global_sample_index` | `skillbook` (real) | — | Saves skillbook to disk | 1 |
| **ObservabilityStep** | `skillbook` | — | — | Logs metrics to Opik | 1 |
| **PersistStep** | `skillbook` | `target_path` | — | Writes skillbook to CLAUDE.md or similar | 1 |

**Requires vs Injected:** `Requires` lists context fields read by the step — validated by the pipeline engine at construction time. The `skillbook` field on the context is a `SkillbookView` (read-only). Steps that **write** to the skillbook (TagStep, ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor injection — marked as "(real)" in the table. These injected dependencies are not tracked by `requires`/`provides`.

**`trace` as the universal learning input:** The learning tail's *entry point* (ReflectStep) requires only `trace` and `skillbook` from the context — subsequent steps in the tail chain off ReflectStep's output (`reflection`). In the standard ACE pipeline, `EvaluateStep` bundles the structured fields (`sample`, `agent_output`, and optionally environment feedback) into a `trace` dict. In TraceAnalyser, `_build_context` places the raw trace directly. In integrations, the execute step provides `trace` from its framework's native output. This means the learning tail is agnostic to trace format — `ReflectStep` passes `ctx.trace` to the Reflector, which is responsible for making sense of whatever it receives.

Steps with `provides = —` are pure side-effect steps (`provides = frozenset()`). They mutate shared state (skillbook) or write to external systems (disk, Opik) but add no new fields to the context.

### AgentStep

```python
class AgentStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output"})

    def __init__(self, agent: AgentLike) -> None:
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
    requires = frozenset({"sample", "agent_output"})
    provides = frozenset({"trace"})

    def __init__(self, environment: TaskEnvironment | None = None) -> None:
        self.environment = environment

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = {
            "question": ctx.sample.question,
            "context": ctx.sample.context,
            "ground_truth": ctx.sample.ground_truth,
            "reasoning": ctx.agent_output.reasoning,
            "answer": ctx.agent_output.final_answer,
            "skill_ids": ctx.agent_output.skill_ids,
        }
        if self.environment:
            result = self.environment.evaluate(
                sample=ctx.sample, agent_output=ctx.agent_output,
            )
            trace["feedback"] = result.feedback
        return ctx.replace(trace=trace)
```

Bridges the execute head (typed ACE objects) to the learning tail (raw traces). Always bundles the structured fields into a `trace` dict. Optionally evaluates the agent output against a `TaskEnvironment` — when provided via constructor, the environment's feedback is included in the trace. When no environment is provided, the trace still contains the agent's output, question, context, and ground truth — the Reflector can learn from these directly. The `TaskEnvironment` is injected at construction time (not on the context) to keep the context free of per-runner dependencies. This also means different `ACE` instances can use different environments without changing the pipeline shape.

### ReflectStep

```python
class ReflectStep:
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    async_boundary = True
    max_workers = 3

    def __init__(self, reflector: ReflectorLike) -> None:
        self.reflector = reflector

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = ctx.trace

        if isinstance(trace, dict):
            # Structured trace from EvaluateStep — extract known fields
            agent_output = AgentOutput(
                reasoning=trace.get("reasoning", ""),
                final_answer=trace.get("answer", ""),
                skill_ids=trace.get("skill_ids", []),
            )
            reflection = self.reflector.reflect(
                question=trace.get("question", ""),
                agent_output=agent_output,
                skillbook=ctx.skillbook,
                ground_truth=trace.get("ground_truth"),
                feedback=trace.get("feedback"),
            )
        else:
            # Raw trace from TraceAnalyser or integration — pass as-is
            reflection = self.reflector.reflect(
                question="",
                agent_output=AgentOutput(reasoning="", final_answer=""),
                skillbook=ctx.skillbook,
                trace=trace,
            )

        return ctx.replace(reflection=reflection)
```

Pure — produces a reflection object, no side effects. Handles two trace formats: (1) when `ctx.trace` is a dict (from EvaluateStep), it extracts known fields and calls the Reflector's existing API with typed arguments; (2) when `ctx.trace` is any other object (from TraceAnalyser or integrations), it passes the raw trace via `**kwargs` for the Reflector to handle directly. Declares `async_boundary = True` — everything from here onward runs in a background thread pool. This lets the execute head return fast while learning continues.

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
    requires = frozenset({"reflection", "skillbook"})
    provides = frozenset({"skill_manager_output"})

    max_workers = 1

    def __init__(self, skill_manager: SkillManagerLike) -> None:
        self.skill_manager = skill_manager

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        output = self.skill_manager.update(
            reflection=ctx.reflection,     # ReflectorOutput
            skillbook=ctx.skillbook,       # SkillbookView (read-only)
        )
        return ctx.replace(skill_manager_output=output)
```

Pure — generates update operations from the `ReflectorOutput` and the current skillbook state. The Reflector has already done the heavy lifting of analysing the trace; the SkillManager turns those insights into concrete skillbook operations (ADD, UPDATE, TAG, REMOVE). Does not mutate the skillbook. `max_workers = 1` because the skill manager reads the current skillbook state and concurrent calls would see stale data.

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

    def __init__(self, manager: DeduplicationManagerLike, skillbook: Skillbook, *, interval: int = 10) -> None:
        self.manager = manager
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx
        report = self.manager.get_similarity_report(self.skillbook)
        if report:
            logger.info("DeduplicateStep: similarity report at sample %d:\n%s",
                        ctx.global_sample_index, report)
        return ctx
```

Optional side-effect step — consolidates similar skills in `self.skillbook` (injected). Appended to the pipeline by factory methods when a `DeduplicationManagerLike` is provided. The step takes a protocol, not the concrete `DeduplicationManager` — this keeps `ace_next` decoupled from the deduplication implementation in `ace/`. Stateless — uses `ctx.global_sample_index` (computed by the runner) with a configurable `interval` (default 10) to skip most invocations. Deduplication involves O(n²) similarity comparisons across all skills, so running it on every sample would be expensive as the skillbook grows.

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
    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        metrics = {"skill_count": len(ctx.skillbook)}
        if ctx.reflection:
            metrics["key_insight"] = ctx.reflection.key_insight
            metrics["learnings_count"] = len(ctx.reflection.extracted_learnings)
        if ctx.skill_manager_output:
            metrics["operations_count"] = len(ctx.skill_manager_output.operations)
        if ctx.trace:
            metrics["trace_type"] = type(ctx.trace).__name__
        # Log to Opik
        ...
        return ctx
```

Optional side-effect step — logs metrics to Opik. Only requires `skillbook` (always present). Reads other context fields (`reflection`, `skill_manager_output`, `trace`) optionally — they may or may not be populated depending on the pipeline shape. This lets the same step work in both ACE and TraceAnalyser pipelines without breaking `requires` validation. Appended to the pipeline when Opik is installed.

### PersistStep

```python
class PersistStep:
    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __init__(self, target_path: str | Path, skillbook: Skillbook) -> None:
        self.target_path = Path(target_path)
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        self.skillbook.save_to_file(str(self.target_path))
        return ctx
```

Integration-specific side-effect step — writes the current skillbook to an external file (e.g., `CLAUDE.md` for Claude Code). Used by `ClaudeCodeACE` to persist learned strategies into the project's instruction file after each learning cycle. Unlike `CheckpointStep` (which saves the full skillbook JSON at intervals), `PersistStep` runs on every sample and writes in the target format expected by the integration. Receives the real `Skillbook` via constructor injection.

---

## Implementations

Concrete LLM-based implementations of the role protocols. Live in `ace_next/implementations/` — fully self-contained with no imports from `ace/`.

### Overview

| Class | Protocol | Method | Location |
|---|---|---|---|
| `Agent` | `AgentLike` | `generate()` | `implementations/agent.py` |
| `Reflector` | `ReflectorLike` | `reflect()` | `implementations/reflector.py` |
| `SkillManager` | `SkillManagerLike` | `update_skills()` | `implementations/skill_manager.py` |

All three share the same constructor pattern:

```python
def __init__(self, llm: LLMClientLike, prompt_template: str = DEFAULT_PROMPT, *, max_retries: int = 3) -> None:
```

The `llm` parameter must satisfy `LLMClientLike` — it must have both `complete()` and `complete_structured()`. No auto-wrapping with Instructor; callers pass pre-wrapped clients.

### Agent

Produces answers using the current skillbook of strategies. Formats the prompt with the skillbook, reflection, question, and context, then calls `llm.complete_structured(prompt, AgentOutput)`. After the LLM call, extracts cited skill IDs from the reasoning using `extract_cited_skill_ids()` (regex matching `[section-00001]` patterns).

```python
agent = Agent(llm)
output = agent.generate(
    question="What is the capital of France?",
    context="Answer concisely",
    skillbook=skillbook,
)
# output.final_answer == "Paris"
# output.skill_ids == ["geography-00001"]  (extracted from reasoning)
```

### Reflector

Analyzes agent outputs to extract lessons and improve strategies. Builds a skillbook excerpt from the agent's cited skill IDs (via `make_skillbook_excerpt()`), formats the prompt, and calls `llm.complete_structured(prompt, ReflectorOutput)`.

**SIMPLE mode only** — single-pass reflection. Recursive mode (where the Reflector iterates multiple times to deepen analysis) is an advanced feature deferred to a later version.

```python
reflector = Reflector(llm)
reflection = reflector.reflect(
    question="What is 2+2?",
    agent_output=agent_output,
    skillbook=skillbook,
    ground_truth="4",
    feedback="Correct!",
)
# reflection.key_insight, reflection.skill_tags, reflection.extracted_learnings
```

### SkillManager

Transforms reflections into actionable skillbook updates. Serializes the `ReflectorOutput` into a JSON dict, formats the prompt with progress and skillbook stats, and calls `llm.complete_structured(prompt, SkillManagerOutput)`.

**No dedup integration** — in `ace_next`, deduplication is handled by a separate `DeduplicateStep` in the pipeline. The SkillManager only produces `SkillManagerOutput`; it does not call a dedup manager itself.

```python
sm = SkillManager(llm)
output = sm.update_skills(
    reflection=reflection_output,
    skillbook=skillbook,
    question_context="Math problem solving",
    progress="5/10 correct",
)
skillbook.apply_update(output.update)
```

### Shared Helpers (`implementations/helpers.py`)

| Function | Purpose |
|---|---|
| `extract_cited_skill_ids(text)` | Regex `[section-00001]` → deduplicated list of IDs |
| `format_optional(value)` | Returns `"(none)"` for falsy values |
| `make_skillbook_excerpt(skillbook, skill_ids)` | Builds `[id] content` lines for cited skills |

### Prompt Templates (`implementations/prompts.py`)

Self-contained copy of the v2.1 prompts from `ace/prompts_v2_1.py`. The `{current_date}` placeholder is filled at import time via `datetime.now().strftime(...)`.

| Constant | Role |
|---|---|
| `AGENT_PROMPT` | Agent prompt with strategic problem-solving protocol |
| `REFLECTOR_PROMPT` | Reflector prompt with diagnostic analysis protocol |
| `SKILL_MANAGER_PROMPT` | SkillManager prompt with atomic strategy creation |
| `SKILLBOOK_USAGE_INSTRUCTIONS` | Shared text for skillbook usage guidance |

Also exports `wrap_skillbook_for_external_agent(skillbook)` — the canonical function for injecting skillbook context into external agentic systems.

---

## Deduplication

Skill deduplication subsystem. Lives in `ace_next/deduplication/` — fully self-contained with no imports from `ace/`.

### Overview

| Class | Role | Location |
|---|---|---|
| `SimilarityDetector` | Computes embeddings, detects similar pairs | `deduplication/detector.py` |
| `DeduplicationManager` | Coordinates detection and consolidation | `deduplication/manager.py` |
| `MergeOp`, `DeleteOp`, `KeepOp`, `UpdateOp` | Consolidation operation types | `deduplication/operations.py` |

### SimilarityDetector

Computes embeddings and detects similar skill pairs using cosine similarity. Supports two embedding providers:

- **LiteLLM** — uses `litellm.embedding()` for remote embedding models
- **sentence-transformers** — uses a local `SentenceTransformer` model (lazy-loaded)

Feature detection uses inline `importlib.import_module` checks instead of `ace.features`. Cosine similarity has a numpy implementation with a pure-Python fallback.

Key methods:
- `ensure_embeddings(skillbook)` — compute embeddings for all skills that lack one
- `detect_similar_pairs(skillbook, threshold)` — find all pairs above the similarity threshold
- Supports `within_section_only` mode (compare skills only within the same section)
- Respects existing `KEEP` decisions via `skillbook.has_keep_decision()`

### DeduplicationManager

Satisfies `DeduplicationManagerLike` protocol. Coordinates the full dedup workflow:

1. `get_similarity_report(skillbook)` — ensures embeddings, detects similar pairs, generates a formatted report for the SkillManager prompt. Returns `None` if dedup is disabled or too few pairs found.
2. `parse_consolidation_operations(response_data)` — parses `consolidation_operations` from SkillManager response JSON into typed operation objects.
3. `apply_operations(operations, skillbook)` — applies consolidation operations to the skillbook.

### Consolidation Operations

Four operation types, all dataclasses:

| Operation | Effect |
|---|---|
| `MergeOp` | Combine skills — accumulate counters into `keep_id`, soft-delete others, update content |
| `DeleteOp` | Soft-delete a redundant skill |
| `KeepOp` | Store a `SimilarityDecision` so the pair is not flagged again |
| `UpdateOp` | Refine a skill's content to differentiate it, clear its embedding |

`apply_consolidation_operations(operations, skillbook)` dispatches each operation to the appropriate apply function.

### Pipeline Integration

Deduplication runs as a separate `DeduplicateStep` in the pipeline, not inside the SkillManager role. The step is appended by factory methods when a `DeduplicationManagerLike` is provided:

```python
ace = ACE.from_roles(
    agent=agent, reflector=reflector, skill_manager=skill_manager,
    dedup_manager=DeduplicationManager(DeduplicationConfig(similarity_threshold=0.85)),
    dedup_interval=10,
)
# Pipeline: Agent → Evaluate → Reflect → Tag → Update → Apply → Deduplicate
```

---

## Integration Pattern

External frameworks (browser-use, LangChain, Claude Code) integrate via composable pipeline steps in `ace_next/integrations/`. Each integration provides three things:

1. **Result type** — an integration-specific dataclass (e.g. `BrowserResult`, `ClaudeCodeResult`)
2. **Execute step** — INJECT skillbook context + EXECUTE the framework, writes the result to `ctx.trace`
3. **ToTrace step** — converts the integration-specific result into the standardised trace dict that `ReflectStep` expects

Runners in `ace_next/runners/` compose these steps with `learning_tail()`.

### Core idea: execute → convert → learn

Each integration defines its own input/output format. A converter step acts as a compatibility layer between the integration-specific result and the learning tail's standardised trace dict.

```
Standard ACE:      [Agent → Evaluate]                          → [Reflect → Tag → Update → Apply]
                    ╰── execute (built-in) ──╯                    ╰──────── learn (shared) ──────╯
                         provides: trace (dict) ─────────────────► requires: trace

Browser-use:       [BrowserExecute] → [BrowserToTrace]         → [Reflect → Tag → Update → Apply]
                    ╰── execute ────╯   ╰── convert ──╯           ╰──────── learn (shared) ──────╯
                    provides: trace      rewrites trace             requires: trace
                    (BrowserResult)      (BrowserResult → dict)

TraceAnalyser:     [_build_context]                            → [Reflect → Tag → Update → Apply]
                    ╰── sets ctx.trace (raw object) ───────╯      ╰──────── learn (shared) ──────╯
```

The execute step writes an integration-specific result type to `ctx.trace`. The ToTrace step reads that result and rewrites `ctx.trace` with a standardised dict. The learning tail only ever sees the dict.

### Two-step contract

Every integration provides two steps that chain together:

**Step 1 — Execute step** (INJECT + EXECUTE):

```python
class SomeExecuteStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(self, framework_client) -> None:
        self.framework_client = framework_client

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task = ctx.sample                               # raw input (string, dict, etc.)
        enhanced = self._inject(task, ctx.skillbook)    # prepend skillbook context
        result = self.framework_client.run(enhanced)    # framework-specific execution
        return ctx.replace(trace=SomeResult(...))       # integration-specific result type
```

**Step 2 — ToTrace step** (convert to standardised dict):

```python
class SomeToTrace:
    requires = frozenset({"trace"})
    provides = frozenset({"trace"})         # overwrites trace with standardised dict

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        r: SomeResult = ctx.trace
        trace = {
            "question": r.task,
            "reasoning": r.execution_trace,     # integration-specific formatting
            "answer": r.output,
            "skill_ids": r.cited_skill_ids,
            "feedback": f"Task {'succeeded' if r.success else 'failed'}",
            "ground_truth": None,
        }
        return ctx.replace(trace=trace)
```

The standardised trace dict keys match what `ReflectStep` expects: `question`, `reasoning`, `answer`, `skill_ids`, `feedback`, `ground_truth`.

### Why two steps instead of one

Splitting execute from trace conversion gives three benefits:

1. **Independent testability** — test the execute step with a mock framework without worrying about trace format; test the ToTrace step with a fixture result without running a real framework.
2. **Reusability** — the execute step can be used standalone (without learning) to get framework-specific results. The ToTrace step can be swapped for a custom converter.
3. **Separation of concerns** — framework interaction logic stays in the execute step; trace formatting stays in the converter. Neither knows about the other's internals.

### Result types

Each integration defines its own result dataclass:

| Integration | Result type | Key fields |
|---|---|---|
| Browser-use | `BrowserResult` | `task`, `success`, `output`, `error`, `steps_count`, `duration_seconds`, `cited_skill_ids`, `chronological_steps`, `raw_history` |
| Claude Code | `ClaudeCodeResult` | `task`, `success`, `output`, `execution_trace`, `returncode`, `error` |
| LangChain | `LangChainResult` | `task`, `output`, `result_type` (simple/agent/langgraph/error), `success`, `error`, `intermediate_steps`, `messages`, `raw_result` |

### Example — browser-use execute step

```python
class BrowserExecuteStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(self, browser_llm, browser=None, **agent_kwargs) -> None:
        self.browser_llm = browser_llm
        self.browser = browser
        self.agent_kwargs = agent_kwargs

    async def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task: str = ctx.sample      # raw task string, not a Sample object

        # INJECT — prepend skillbook context
        enhanced_task = self._inject(task, ctx.skillbook)

        # EXECUTE — run browser-use agent
        agent = Agent(task=enhanced_task, llm=self.browser_llm, **self.agent_kwargs)
        history = await agent.run()

        # Build integration-specific result
        result = BrowserResult(
            task=task, success=True, output=history.final_result(),
            steps_count=history.number_of_steps(),
            chronological_steps=..., raw_history=history,
        )
        return ctx.replace(trace=result)
```

### Composing into a runner

Runners compose execute step + ToTrace step + learning tail:

```python
class BrowserUse(ACERunner):
    """Browser-use agent with ACE learning pipeline."""

    @classmethod
    def from_roles(cls, *, browser_llm, reflector, skill_manager,
                   skillbook=None, **kwargs):
        skillbook = skillbook or Skillbook()
        steps = [
            BrowserExecuteStep(browser_llm),
            BrowserToTrace(),
            *learning_tail(reflector, skill_manager, skillbook, **kwargs),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(self, tasks, epochs=1, *, wait=True):
        return self._run(tasks, epochs=epochs, wait=wait)

    def _build_context(self, task, *, epoch, total_epochs, index, total,
                       global_sample_index, **_):
        return ACEStepContext(
            sample=task,    # raw string — not wrapped in Sample
            skillbook=SkillbookView(self.skillbook),
            epoch=epoch, total_epochs=total_epochs,
            step_index=index, total_steps=total,
            global_sample_index=global_sample_index,
        )
```

The pattern is the same for every integration: subclass `ACERunner`, compose `[ExecuteStep, ToTrace, *learning_tail()]` in the factory, accept raw inputs (strings, dicts) in `run()`, and map them to `ACEStepContext` in `_build_context()`. No `Sample` wrapping — the raw input goes directly on `ctx.sample`.

### `learning_tail()` — reusable learning steps

Every integration assembles the same `[Reflect → Tag → Update → Apply]` suffix with optional dedup and checkpoint steps. Rather than duplicating this wiring in every factory method, `learning_tail()` returns the standard step list:

```python
# ace_next/steps/__init__.py

def learning_tail(
    reflector: ReflectorLike,
    skill_manager: SkillManagerLike,
    skillbook: Skillbook,
    *,
    dedup_manager: DeduplicationManagerLike | None = None,
    dedup_interval: int = 10,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 10,
) -> list:
    """Return the standard ACE learning steps.

    Use this when building custom integrations that provide their own
    execute step(s) but want the standard learning pipeline.
    """
    steps: list = [
        ReflectStep(reflector),
        TagStep(skillbook),
        UpdateStep(skill_manager),
        ApplyStep(skillbook),
    ]
    if dedup_manager:
        steps.append(DeduplicateStep(dedup_manager, skillbook, interval=dedup_interval))
    if checkpoint_dir:
        steps.append(CheckpointStep(checkpoint_dir, skillbook, interval=checkpoint_interval))
    return steps
```

Integration factories become shorter and less error-prone:

```python
class BrowserUse(ACERunner):
    @classmethod
    def from_roles(cls, *, browser_llm, reflector, skill_manager, skillbook=None, **kwargs):
        skillbook = skillbook or Skillbook()
        steps = [
            BrowserExecuteStep(browser_llm),
            BrowserToTrace(),
            *learning_tail(reflector, skill_manager, skillbook, **kwargs),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)
```

Power users building fully custom pipelines can also use it:

```python
from ace_next.steps import learning_tail

skillbook = Skillbook.load_from_file("expert.json")
steps = [
    MyCustomExecuteStep(my_agent),
    MyValidationStep(),  # custom step before learning
    *learning_tail(reflector, skill_manager, skillbook, dedup_manager=dedup),
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
    def __init__(self, llm, skillbook, *, environment=None, ...):
        self.agent = Agent(llm, ...)
        self.reflector = Reflector(llm, ...)
        self.skill_manager = SkillManager(llm, ...)
        self.skillbook = skillbook
        self.environment = environment
        self._ace: ACE | None = None
        self._analyser: TraceAnalyser | None = None

    def _get_ace(self) -> ACE:
        if self._ace is None:
            self._ace = ACE.from_roles(
                agent=self.agent,
                reflector=self.reflector,
                skill_manager=self.skill_manager,
                skillbook=self.skillbook,
                environment=self.environment,
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

    def learn(self, samples, epochs=1, wait=True):
        """Delegate to ACE. Environment is set at construction time."""
        return self._get_ace().run(samples, epochs=epochs, wait=wait)

    def learn_from_traces(self, traces, epochs=1, wait=True):
        """Delegate to TraceAnalyser."""
        return self._get_analyser().run(traces, epochs=epochs, wait=wait)


class ACEAgent:
    """Browser-use convenience wrapper. Delegates to BrowserUse."""

    def __init__(self, browser_llm, reflector, skill_manager, **kwargs):
        self.runner = BrowserUse.from_roles(
            browser_llm=browser_llm, reflector=reflector,
            skill_manager=skill_manager, **kwargs
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
ace_next/
  __init__.py               ← Public API re-exports
  context.py                ← ACEStepContext, SkillbookView, ACESample
  skill.py                  ← Skill, SimilarityDecision
  skillbook.py              ← Skillbook
  updates.py                ← UpdateOperation, UpdateBatch
  outputs.py                ← AgentOutput, ReflectorOutput, SkillManagerOutput, etc.
  environments.py           ← Sample, TaskEnvironment, SimpleEnvironment, EnvironmentResult
  protocols/                ← Role protocols (one file per protocol)
    __init__.py
    agent.py                ← AgentLike
    reflector.py            ← ReflectorLike
    skill_manager.py        ← SkillManagerLike
    deduplication.py        ← DeduplicationConfig, DeduplicationManagerLike
    llm.py                  ← LLMClientLike
  implementations/          ← Concrete LLM-based role implementations
    __init__.py             ← Exports Agent, Reflector, SkillManager
    agent.py                ← Agent (implements AgentLike)
    reflector.py            ← Reflector (implements ReflectorLike)
    skill_manager.py        ← SkillManager (implements SkillManagerLike)
    helpers.py              ← Shared utilities (extract_cited_skill_ids, etc.)
    prompts.py              ← Default v2.1 prompt templates
  deduplication/            ← Skill deduplication subsystem
    __init__.py             ← Exports DeduplicationManager, SimilarityDetector, etc.
    detector.py             ← SimilarityDetector (embeddings + cosine similarity)
    manager.py              ← DeduplicationManager (implements DeduplicationManagerLike)
    operations.py           ← ConsolidationOperation types + apply logic
    prompts.py              ← Similarity report generation
  steps/                    ← Pipeline steps (one file per class)
    __init__.py             ← learning_tail() helper
    agent.py                ← AgentStep
    evaluate.py             ← EvaluateStep
    reflect.py              ← ReflectStep
    tag.py                  ← TagStep
    update.py               ← UpdateStep
    apply.py                ← ApplyStep
    deduplicate.py          ← DeduplicateStep
    checkpoint.py           ← CheckpointStep
    observability.py        ← ObservabilityStep
    persist.py              ← PersistStep
  runners/                    ← Runner classes (compose Pipeline, manage epoch loop)
    __init__.py               ← Re-exports ACERunner, TraceAnalyser, ACE, BrowserUse, LangChain, ClaudeCode
    base.py                   ← ACERunner base class
    trace_analyser.py         ← TraceAnalyser (learning tail only)
    ace.py                    ← ACE (full adaptive pipeline)
    browser_use.py            ← BrowserUse runner (BrowserExecuteStep + BrowserToTrace + learning_tail)
    langchain.py              ← LangChain runner (LangChainExecuteStep + LangChainToTrace + learning_tail)
    claude_code.py            ← ClaudeCode runner (ClaudeCodeExecuteStep + ClaudeCodeToTrace + learning_tail)
  integrations/               ← Integration steps (execute + result type + trace converter)
    __init__.py               ← Exports steps, result types, ToTrace converters, wrap_skillbook_context
    browser_use.py            ← BrowserExecuteStep, BrowserResult, BrowserToTrace
    langchain.py              ← LangChainExecuteStep, LangChainResult, LangChainToTrace
    claude_code.py            ← ClaudeCodeExecuteStep, ClaudeCodeResult, ClaudeCodeToTrace
```

Each integration provides: (1) an execute step, (2) a result type, and (3) a ToTrace converter step. Runners in `ace_next/runners/` compose these with `learning_tail()`. For offline analysis, raw trace objects are passed directly to TraceAnalyser.

### What moves where

| Old location | New location | Notes |
|---|---|---|
| `ace/adaptation.py` | Deleted | Replaced by `ace_next/runners/` |
| `ace/async_learning.py` | Deleted | Replaced by pipeline engine `async_boundary` |
| `ace/environments.py` | `ace_next/environments.py` | `Sample`, `EnvironmentResult`, `TaskEnvironment`, `SimpleEnvironment` (copied) |
| `ace/roles.py` (protocols) | `ace_next/protocols/` | Protocols extracted from role classes |
| `ace/roles.py` (implementations) | `ace_next/implementations/` | Concrete `Agent`, `Reflector`, `SkillManager` classes |
| `ace/llm.py` (interface) | `ace_next/protocols/llm.py` | `LLMClientLike` protocol |
| `ace/prompts_v2_1.py` | `ace_next/implementations/prompts.py` | v2.1 prompt templates (self-contained copy) |
| `ace/deduplication/` | `ace_next/deduplication/` | Full dedup subsystem (detector, manager, operations, prompts) |
| `ace2/` | Deleted | Superseded by this design |
| New | `ace_next/steps/tag.py` | TagStep (split from ReflectStep) |
| New | `ace_next/steps/apply.py` | ApplyStep (split from UpdateStep) |
| New | `ace_next/steps/deduplicate.py` | DeduplicateStep (extracted from SkillManager) |
| New | `ace_next/steps/checkpoint.py` | CheckpointStep |
| New | `ace_next/steps/observability.py` | ObservabilityStep |
| New | `ace_next/steps/persist.py` | PersistStep |
| New | `ace_next/runners/` | ACERunner, TraceAnalyser, ACE, BrowserUse, LangChain, ClaudeCode |
| `ace/integrations/browser_use.py` | `ace_next/integrations/browser_use.py` + `ace_next/runners/browser_use.py` | Split into execute step + result type + ToTrace converter + runner |
| `ace/integrations/langchain.py` | `ace_next/integrations/langchain.py` + `ace_next/runners/langchain.py` | Split into execute step + result type + ToTrace converter + runner |
| `ace/integrations/claude_code.py` | `ace_next/integrations/claude_code.py` + `ace_next/runners/claude_code.py` | Split into execute step + result type + ToTrace converter + runner |

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
from ace_next import TraceAnalyser
from ace.llm_providers.litellm_client import LiteLLMClient
from ace.llm_providers.instructor_client import wrap_with_instructor

llm = wrap_with_instructor(LiteLLMClient(model="gpt-4o-mini"))

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
analyser = TraceAnalyser.from_client(llm)
results = analyser.run(traces, epochs=2)
analyser.save("travel_agent.json")
```

### ACE — live Q&A training

```python
from ace_next import ACE, Sample, SimpleEnvironment
from ace.llm_providers.litellm_client import LiteLLMClient
from ace.llm_providers.instructor_client import wrap_with_instructor

llm = wrap_with_instructor(LiteLLMClient(model="gpt-4o-mini"))

samples = [
    Sample(question="Capital of France?", ground_truth="Paris"),
    Sample(question="Largest ocean?", ground_truth="Pacific"),
]

# Environment provided at construction — EvaluateStep uses it to generate feedback
ace = ACE.from_client(llm, environment=SimpleEnvironment())
results = ace.run(samples, epochs=3)
ace.save("geography.json")
```

### ACE — without environment

```python
# No environment — trace still contains agent output + ground truth
# The Reflector learns from ground-truth comparison directly
ace = ACE.from_client(llm)
results = ace.run(samples, epochs=3)
```

### ACE — single-pass with iterable

```python
# Any Iterable works with epochs=1 (consumed once, not replayed)
samples = load_samples_from_csv("eval_set.csv")  # returns a list or generator

ace = ACE.from_client(llm)
results = ace.run(samples, epochs=1)
```

### ACE — with checkpoints and deduplication

```python
from ace_next import ACE, Agent, Reflector, SkillManager, SimpleEnvironment
from ace_next.deduplication import DeduplicationManager
from ace_next.protocols.deduplication import DeduplicationConfig

ace = ACE.from_roles(
    agent=Agent(llm),
    reflector=Reflector(llm),
    skill_manager=SkillManager(llm),
    environment=SimpleEnvironment(),
    dedup_manager=DeduplicationManager(DeduplicationConfig(similarity_threshold=0.85)),
    checkpoint_dir="./checkpoints",
    checkpoint_interval=10,
)
# Pipeline: Agent → Evaluate → Reflect → Tag → Update → Apply → Deduplicate → Checkpoint
results = ace.run(samples, epochs=3)
```

### Integration — browser-use runner

```python
from ace_next.integrations.browser_use import BrowserACE
from browser_use import Agent as BrowserAgent

browser_agent = BrowserAgent(llm=ChatOpenAI(model="gpt-4o"))
runner = BrowserACE.from_client(
    browser_agent,
    ace_client=llm,  # Must satisfy LLMClientLike (complete + complete_structured)
)

# Live execution + learning
results = runner.run(["Find top HN post", "Check weather in Tokyo"])
runner.save("browser_expert.json")
```

### Fire-and-forget — get results while learning continues

```python
ace = ACE.from_client(llm)

# wait=False: returns after foreground steps (Agent + Evaluate)
# Background learning (Reflect → Tag → Update → Apply) continues
results = ace.run(samples, epochs=1, wait=False)

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
from ace_next import TraceAnalyser, ACE, Skillbook

# Phase 1: build skillbook from historical traces
analyser = TraceAnalyser.from_client(llm, skillbook=Skillbook())
analyser.run(historical_traces, epochs=3)
skillbook = analyser.skillbook

# Phase 2: deploy with live learning
ace = ACE.from_client(llm, skillbook=skillbook)
ace.run(live_samples, epochs=1)
ace.save("production.json")
```

---

## Potential Improvements

Issues acknowledged but deferred from this version of the spec.

**Streaming / lazy iteration:**
`_run()` eagerly materializes the full iterable into a list of `ACEStepContext` objects before passing them to `Pipeline.run()`. For a large generator with `epochs=1`, the entire input gets buffered into memory. True streaming would require the pipeline to accept an iterator and process items one-at-a-time (e.g., `for ctx in contexts: pipeline.run_one(ctx)`), or an async iterator pattern with `asyncio.as_completed`. This is a deliberate simplification — batch materialization keeps the epoch loop and error handling straightforward. Revisit if memory pressure from large single-pass runs becomes a real problem.

**Builder API for custom pipelines (speculative):**
The current API offers two extremes: factory methods (`from_client`, `from_roles`) that hide the pipeline entirely, and manual `Pipeline([...])` construction that requires understanding `ACEStepContext`, `SkillbookView`, step contracts, and `ACERunner` subclassing. Users who want to insert a custom step between Reflect and Update, or swap the execute head while keeping the learning tail, fall into a gap where neither approach serves them well.

One possible direction is a builder API that would bridge this gap:

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

This would only be worth pursuing when there is evidence of users building custom pipelines with `learning_tail()` and hitting friction with the manual wiring. The `learning_tail()` helper (see Integration Pattern section) covers the most common customisation — custom execute step + standard learning — without a builder. A builder adds value when users need fine-grained insertion points (between existing steps) or want to compose from presets without understanding the step internals. The main risk is that a builder mirrors the step list, adding a second construction path to document, test, and keep in sync. It can also hide the `requires`/`provides` contracts — when a validation step is inserted at the wrong position, the error comes from the pipeline engine (field missing) rather than the builder (wrong position name), making debugging indirect. Mitigate by having the builder validate the final step chain at `build()` time and surfacing clear errors.

**Skillbook rollback and versioning:**
Currently the skillbook is mutated in place with no way to undo a bad update. If the LLM hallucinates a harmful skill or a batch degrades overall quality, the only recovery is restoring from a checkpoint file. A lightweight versioning mechanism — e.g., snapshotting skillbook state at epoch boundaries or before each `ApplyStep`, with a `rollback(to_version)` method — would enable automatic revert when a validation metric degrades, A/B comparison between skillbook versions, and safer experimentation with aggressive learning rates. This could live as a `VersionedSkillbook` wrapper or as an optional `SnapshotStep` inserted before `ApplyStep`. Deferred because the current checkpoint-to-disk approach covers the most common recovery scenario (resume after crash), and in-memory versioning adds memory overhead proportional to skillbook size times number of snapshots.

---

## What Was Rejected and Why

**Runner extends Pipeline:**
Making TraceAnalyser and ACE subclasses of `Pipeline` was considered. Rejected — the runner is not a pipeline. It owns the epoch loop. Composition (`self.pipeline`) keeps responsibilities separate.

**Cross-sample state (reflection window):**
A rolling window of recent reflections that persists across samples was considered, with variants: on the runner, on `StepContext`, on step instances, via a shared mediator object. All rejected — each sample should be independent. The only cross-sample coupling is the skillbook itself, which evolves as samples are processed. Adding a reflection window complicates the model (reset between epochs, eventual consistency with background steps, ordering issues with concurrent workers) for marginal benefit.

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

**Instructor auto-wrapping in implementations:**
The old `ace/roles.py` auto-wraps LLM clients with Instructor if `complete_structured` is missing (duck-typing check + fallback). Rejected for `ace_next` — auto-wrapping creates a hidden dependency on `ace.llm_providers.instructor_client` and masks what the implementation actually requires. In `ace_next`, `LLMClientLike` explicitly requires both `complete()` and `complete_structured()`. Callers wrap their LLM clients before passing them in. This makes the requirement visible at the call site and keeps implementations dependency-free.

**Recursive Reflector:**
The old `ace/reflector/` subsystem supports recursive mode where the Reflector iterates multiple times to deepen analysis. Rejected for the initial `ace_next` implementation — recursive mode pulls in significant additional complexity (iteration control, convergence detection, intermediate result accumulation) for marginal benefit on most workloads. The `Reflector` class implements SIMPLE mode only (single-pass reflection). Recursive mode can be added later as an opt-in capability without changing the `ReflectorLike` protocol.

**Observability decorator on implementations:**
The old `ace/roles.py` uses `@maybe_track()` decorators for Opik tracing on every role method. Rejected — `ObservabilityStep` already handles metrics at the pipeline level with full visibility into all context fields. Adding per-method decorators would double-count and create coupling between implementations and the observability system.

**Deduplication inside SkillManager:**
The old `ace/roles.py` SkillManager integrates with `DeduplicationManager` directly — calling `get_similarity_report()` before the LLM call and `apply_operations_from_response()` after. Rejected for `ace_next` — deduplication is now a separate `DeduplicateStep` in the pipeline. This is cleaner separation: the SkillManager role only produces `SkillManagerOutput`, and deduplication runs at a configurable interval as an independent pipeline step. The step takes a `DeduplicationManagerLike` protocol, keeping it decoupled from the concrete implementation.

**Shared `ace_next/features.py` module:**
Creating a centralized feature detection module (like `ace/features.py`) for optional dependency checks was considered. Rejected — the only code that needs feature detection is `deduplication/detector.py`, which uses a local `_has(module)` helper with `importlib.import_module`. A shared module would add a file for a single 4-line function. If more code needs feature detection in the future, the helper can be promoted to a shared location.
