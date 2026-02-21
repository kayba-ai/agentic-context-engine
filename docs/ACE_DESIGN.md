# ACE Architecture Design

Specification for rewriting the legacy `ace/` module to use the pipeline engine.

---

## Goals

1. Replace the monolithic `ACEBase` / `OfflineACE` / `OnlineACE` in `adaptation.py` with pipeline-based classes.
2. Rename to match what each class actually does:
   - **TraceAnalyser**: takes pre-recorded traces, outputs a skillbook. Replaces the concept of "offline" learning.
   - **ACE**: the live adaptive pipeline. Replaces the concept of "online" learning. Also supports multi-epoch batch runs.
3. Clean OOP: shared base class, composition over inheritance, pluggable steps.
4. Unify the integration pattern — external frameworks produce `Trace` objects, TraceAnalyser consumes them.

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

### Trace

A pre-recorded execution record. Framework-agnostic — works for browser-use histories, LangChain runs, Claude Code transcripts, or any external system.

```python
@dataclass
class Trace:
    """Pre-recorded execution data for offline analysis."""
    task: str                           # what was asked
    output: str                         # what was produced
    feedback: str                       # how it was evaluated
    ground_truth: str | None = None     # expected answer (optional)
    reasoning: str = ""                 # agent's reasoning / execution trace
    context: str = ""                   # task context
    metadata: dict = field(default_factory=dict)
    id: str | None = None
```

TraceAnalyser builds `agent_output` and `environment_result` internally from these fields, so callers never need to import ACE role types.

**Conversion helpers** on `Trace`:

```python
def to_agent_output(self) -> AgentOutput:
    return AgentOutput(
        reasoning=self.reasoning or f"Task: {self.task}",
        final_answer=self.output,
        skill_ids=[],
        raw=self.metadata,
    )

def to_environment_result(self) -> EnvironmentResult:
    return EnvironmentResult(
        feedback=self.feedback,
        ground_truth=self.ground_truth,
    )
```

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

---

## Class Hierarchy

```
ACERunner (shared infrastructure: epoch loop, delegates to Pipeline.run())
├── TraceAnalyser       — [ReflectStep → UpdateStep]; input = Trace
├── ACE                 — [AgentStep → EvaluateStep → ReflectStep → UpdateStep]; input = Sample + Environment
├── BrowserACE          — [BrowserExecuteStep → ReflectStep → UpdateStep]; input = tasks
├── LangChainACE        — [LangChainExecuteStep → ReflectStep → UpdateStep]; input = chain inputs
└── ClaudeCodeACE       — [ClaudeCodeExecuteStep → ReflectStep → UpdateStep → PersistStep]; input = tasks
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
```

### Responsibilities

| Concern | Owner |
|---|---|
| Epoch loop + Iterable validation | `ACERunner._run()` |
| Per-sample iteration + error isolation | `Pipeline.run()` |
| Foreground/background split | `Pipeline.run()` (via `async_boundary`) |
| Concurrent workers | `Pipeline.run(workers=N)` |
| Checkpoints | `CheckpointStep` (in the pipeline) |
| Background drain | `Pipeline.wait_for_background()` |
| Skillbook I/O | `save(path)` / `load(path)` on the runner |

Each sample is independent — no state persists across samples. The skillbook is a shared mutable object passed through the context; that is the only cross-sample coupling.

### Generic run loop

Every subclass delegates to `_run()`. The only thing that varies per subclass is (1) the public `run()` signature and (2) the `_build_context()` method that maps input items to `StepContext`.

```python
def _run(
    self,
    items: Sequence | Iterable,
    *,
    epochs: int,
) -> list[SampleResult]:
    if epochs > 1 and not isinstance(items, Sequence):
        raise ValueError("Multi-epoch requires a Sequence, not a consumed Iterable.")

    results: list[SampleResult] = []

    for epoch in range(1, epochs + 1):
        contexts = [
            self._build_context(item, epoch=epoch, total_epochs=epochs,
                                index=idx, total=len(items) if isinstance(items, Sequence) else idx)
            for idx, item in enumerate(items, start=1)
        ]
        epoch_results = self.pipeline.run(contexts)
        results.extend(epoch_results)

    self.pipeline.wait_for_background()
    return results
```

The runner builds `StepContext` objects and hands them to `Pipeline.run()`, which handles iteration, error isolation, foreground/background split, and concurrent workers. The runner only owns the epoch loop.

---

## TraceAnalyser

Analyses pre-recorded traces without executing an agent. Runs `ReflectStep → UpdateStep` only.

### When to use

- You have execution logs from an external system (browser-use, LangChain, custom agent, human sessions).
- You want to build or refine a skillbook from historical data.
- You want to re-analyse the same data multiple times (multi-epoch) to extract deeper patterns.

### Pipeline

```
[ReflectStep] → [UpdateStep]
```

No AgentStep, no EvaluateStep. The trace already contains the agent's output and the evaluation feedback.

### Context building

TraceAnalyser converts each `Trace` into a `StepContext` with `agent_output` and `environment_result` pre-filled:

```python
def _build_context(self, trace: Trace, *, epoch, total_epochs, index, total) -> StepContext:
    return StepContext(
        sample=trace,              # Trace doubles as the "sample" for step access
        skillbook=self.skillbook,
        environment=None,          # no live environment
        agent_output=trace.to_agent_output(),
        environment_result=trace.to_environment_result(),
        epoch=epoch,
        total_epochs=total_epochs,
        step_index=index,
        total_steps=total,
    )
```

Each sample is independent — no state carries over from previous samples.

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
        traces: Sequence[Trace],
        epochs: int = 1,
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | None = None,
    ) -> list[SampleResult]: ...
```

Note: no `environment` parameter. The evaluation is already baked into the trace.

### Multi-epoch semantics

Each epoch re-processes all traces with the current (evolving) skillbook. Early epochs extract obvious patterns; later epochs refine and consolidate.

```
Epoch 1:  trace₁ → trace₂ → ... → traceₙ   (skillbook grows)
Epoch 2:  trace₁ → trace₂ → ... → traceₙ   (skillbook refines)
Epoch 3:  trace₁ → trace₂ → ... → traceₙ   (diminishing returns)
```

Each sample is independent. The only thing that evolves across samples (and epochs) is the skillbook itself — a shared mutable object on the context.

### run() — delegates to _run()

```python
def run(self, traces, epochs=1, checkpoint_interval=None, checkpoint_dir=None):
    return self._run(
        traces,
        epochs=epochs,
        build_context=self._build_context,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
    )
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
[AgentStep] → [EvaluateStep] → [ReflectStep] → [UpdateStep]
```

### Context building

```python
def _build_context(self, sample, *, epoch, total_epochs, index, total) -> StepContext:
    return StepContext(
        sample=sample,
        skillbook=self.skillbook,
        environment=self.environment,
        epoch=epoch,
        total_epochs=total_epochs,
        step_index=index,
        total_steps=total,
    )
```

Each sample is independent. The skillbook is the only shared mutable object.

### Interface

```python
class ACE(ACERunner):
    """Live adaptive pipeline: Agent → Evaluate → Reflect → Update."""

    @classmethod
    def from_client(cls, client, *, skillbook=None, **kwargs) -> "ACE": ...

    @classmethod
    def from_roles(cls, *, agent, reflector, skill_manager, skillbook=None, **kwargs) -> "ACE": ...

    def run(
        self,
        samples: Sequence[Sample] | Iterable[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        checkpoint_interval: int | None = None,
        checkpoint_dir: str | None = None,
    ) -> list[SampleResult]: ...
```

### Single-pass vs multi-epoch

A single class handles both use cases. `epochs=1` gives streaming / single-pass behaviour. `epochs > 1` gives batch training.

```python
# Single pass (was OnlineACE)
results = ace.run(sample_stream, environment, epochs=1)

# Multi-epoch batch (was OfflineACE)
results = ace.run(training_set, environment, epochs=3)
```

When `samples` is an `Iterable` (not `Sequence`), `epochs` must be `1` — you cannot replay a consumed generator. `_run()` raises `ValueError` if `epochs > 1` and `samples` is not a `Sequence`.

### run() — delegates to _run()

```python
def run(self, samples, environment, epochs=1, checkpoint_interval=None, checkpoint_dir=None):
    self.environment = environment  # stored for _build_context
    return self._run(
        samples,
        epochs=epochs,
        build_context=self._build_context,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
    )
```

Same pattern as TraceAnalyser — the public `run()` stores any extra state (here, `environment`), then delegates entirely to `_run()`.

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
| `dedup_config` | `None` | Installs DeduplicationManager on SkillManager |

---

## Steps

Reusable step implementations live in `ace/steps/`. Each is a single class in a single file. All satisfy the `StepProtocol` from the pipeline engine.

### AgentStep

```python
class AgentStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output"})

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def __call__(self, ctx: StepContext) -> StepContext:
        agent_output = self.agent.generate(
            question=ctx.sample.question,
            context=ctx.sample.context,
            skillbook=ctx.skillbook,
            sample=ctx.sample,
        )
        return ctx.replace(agent_output=agent_output)
```

Stateless. Delegates to `self.agent.generate()`.

### EvaluateStep

```python
class EvaluateStep:
    requires = frozenset({"sample", "agent_output", "environment"})
    provides = frozenset({"environment_result"})

    def __call__(self, ctx: StepContext) -> StepContext: ...
```

Stateless. Delegates to `ctx.environment.evaluate()`.

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

    def __call__(self, ctx: StepContext) -> StepContext:
        reflection = self.reflector.reflect(...)

        # Tag skills on the shared (mutable) skillbook
        for tag in reflection.skill_tags:
            try:
                ctx.skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                continue

        return ctx.replace(reflection=reflection)
```

Declares `async_boundary = True` — everything from here onward runs in a background thread pool. This lets AgentStep + EvaluateStep return fast while learning continues.

Stateless across samples. The only side effect is tagging skills on the shared skillbook.

### UpdateStep

```python
class UpdateStep:
    requires = frozenset({"reflection", "skillbook", "sample", "environment_result", "agent_output"})
    provides = frozenset({"skill_manager_output"})

    max_workers = 1

    def __init__(self, skill_manager: SkillManager) -> None: ...
    def __call__(self, ctx: StepContext) -> StepContext: ...
```

`max_workers = 1` serialises skillbook writes — only one UpdateStep mutates the skillbook at a time.

Side effects:
- Applies update batch to the shared skillbook (`skillbook.apply_update(batch)`).
- Attaches insight-source provenance metadata to operations.

### CheckpointStep

Optional tail step that periodically saves the skillbook to disk. Stateless — derives the checkpoint decision from context fields.

```python
class CheckpointStep:
    requires = frozenset({"skillbook"})
    provides = frozenset()                  # pure side-effect, no new context fields

    def __init__(self, directory: str | Path, *, interval: int = 10) -> None:
        self.directory = Path(directory)
        self.interval = interval

    def __call__(self, ctx: StepContext) -> StepContext:
        # Global sample number across epochs
        global_index = (ctx.epoch - 1) * ctx.total_steps + ctx.step_index

        if global_index % self.interval != 0:
            return ctx                      # nothing to do

        self.directory.mkdir(parents=True, exist_ok=True)
        ctx.skillbook.save_to_file(str(self.directory / f"checkpoint_{global_index}.json"))
        ctx.skillbook.save_to_file(str(self.directory / "latest.json"))
        return ctx
```

Key points:
- **Stateless.** Uses `ctx.epoch`, `ctx.total_steps`, and `ctx.step_index` to compute the global sample number. No internal counter, no reset needed.
- **`provides` is empty** — it only writes to disk, does not modify the context.
- **Placement:** Append after UpdateStep. When `async_boundary` is set, checkpoints happen in the background tail alongside Reflect + Update.
- **`max_workers` not set** — inherits default of 1 from the pipeline engine, which is correct (disk writes should be serialised).

---

## Integration Pattern

External frameworks (browser-use, LangChain, Claude Code) integrate by providing **execute steps** that compose into an `ACERunner`. No separate pipeline classes per integration — just steps that plug into the standard runner infrastructure.

### Core idea: integrations are sub-pipelines

An integration provides the "execute" part. The "learn" part (`ReflectStep → UpdateStep`) is always the same. The runner composes them:

```
Standard ACE:      [AgentStep → EvaluateStep] → [ReflectStep → UpdateStep]
                    ╰── execute (built-in) ──╯   ╰──── learn (shared) ────╯

Browser-use:       [BrowserExecuteStep]        → [ReflectStep → UpdateStep]
                    ╰── execute (integration) ╯   ╰──── learn (shared) ────╯

LangChain:         [LangChainExecuteStep]      → [ReflectStep → UpdateStep]

Claude Code:       [ClaudeCodeExecuteStep]     → [ReflectStep → UpdateStep → PersistStep]
```

Each integration provides one or more execute steps. These steps `provide` at minimum `{"agent_output", "environment_result"}` — the same fields that `AgentStep + EvaluateStep` produce in the standard ACE pipeline. This is the contract that `ReflectStep` requires.

### Execute step contract

Every integration execute step must satisfy:

```python
class SomeExecuteStep:
    requires = frozenset({"sample", "skillbook"})                     # minimum
    provides = frozenset({"agent_output", "environment_result"})      # minimum

    def __call__(self, ctx: StepContext) -> StepContext:
        # Run framework, evaluate result
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

    async def __call__(self, ctx: StepContext) -> StepContext:
        # Inject skillbook context into the browser agent
        skillbook_context = ctx.skillbook.as_prompt()

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
        reflector = Reflector(ace_client)
        skill_manager = SkillManager(ace_client)

        steps = [
            BrowserExecuteStep(browser_agent),
            ReflectStep(reflector, **kwargs),
            UpdateStep(skill_manager),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    def run(self, tasks, *, epochs=1, **kwargs):
        samples = [Sample(question=t) if isinstance(t, str) else t for t in tasks]
        return self._run(
            samples,
            epochs=epochs,
            build_context=self._build_context,
            **kwargs,
        )

    def _build_context(self, sample, *, epoch, total_epochs, index, total):
        return StepContext(
            sample=sample,
            skillbook=self.skillbook,
            epoch=epoch,
            total_epochs=total_epochs,
            step_index=index,
            total_steps=total,
        )
```

The pattern is the same for every integration: subclass `ACERunner`, provide a factory that wires the execute step(s) + learning tail, override `run()` with the integration-specific signature, and implement `_build_context()`.

### TraceAnalyser — batch learning from recorded executions

Integrations also support offline learning. When an integration records execution history (browser-use AgentHistory, LangChain intermediate_steps, Claude Code transcripts), it can convert them to `Trace` objects and feed them to TraceAnalyser:

```python
# Record browser executions
histories = [await agent.run(task) for task in tasks]

# Convert to traces
traces = [browser_history_to_trace(h, task) for h, task in zip(histories, tasks)]

# Batch analysis
analyser = TraceAnalyser.from_client(llm_client)
analyser.run(traces, epochs=2)
analyser.save("browser_expert.json")
```

Each integration provides a `to_trace()` converter:

| Integration | Converter | Source |
|---|---|---|
| browser-use | `browser_history_to_trace(history, task)` | `AgentHistoryList` |
| LangChain | `langchain_result_to_trace(result, input)` | `Dict` with `intermediate_steps` or `messages` |
| Claude Code | `transcript_to_trace(transcript, task)` | Claude Code transcript |

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

The user-facing wrappers (ACELiteLLM, ACEAgent, ACELangChain) remain as convenience classes. Internally, they create the appropriate runner:

```python
class ACELiteLLM:
    def ask(self, question) -> str:
        """Use current skillbook to answer."""
        ...

    def learn(self, samples, environment, epochs=1):
        """Delegate to ACE."""
        ace = ACE.from_roles(
            agent=self.agent,
            reflector=self.reflector,
            skill_manager=self.skill_manager,
            skillbook=self.skillbook,
        )
        return ace.run(samples, environment, epochs=epochs)

    def learn_from_traces(self, traces, epochs=1):
        """Delegate to TraceAnalyser."""
        analyser = TraceAnalyser.from_roles(
            reflector=self.reflector,
            skill_manager=self.skill_manager,
            skillbook=self.skillbook,
        )
        return analyser.run(traces, epochs=epochs)


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
    update.py               ← UpdateStep
  trace_analyser.py         ← TraceAnalyser class
  ace.py                    ← ACE class
  runner.py                 ← ACERunner base class
  trace.py                  ← Trace dataclass
  integrations/
    browser_use/
      runner.py             ← BrowserACE (ACERunner subclass)
      steps/
        execute.py          ← BrowserExecuteStep
      converter.py          ← browser_history_to_trace()
    langchain/
      runner.py             ← LangChainACE (ACERunner subclass)
      steps/
        execute.py          ← LangChainExecuteStep
      converter.py          ← langchain_result_to_trace()
    claude_code/
      runner.py             ← ClaudeCodeACE (ACERunner subclass)
      steps/
        execute.py          ← ClaudeCodeExecuteStep
        persist.py          ← PersistStep
      converter.py          ← transcript_to_trace()
    litellm.py              ← ACELiteLLM (high-level wrapper)
    base.py                 ← wrap_skillbook_context (unchanged)
```

Each integration provides: (1) an execute step, (2) an `ACERunner` subclass that composes the step with the learning tail, and (3) a `to_trace()` converter for offline analysis. No separate pipeline classes.

### What moves where

| Old location | New location | Notes |
|---|---|---|
| `ace/adaptation.py` | Deleted | Replaced by `runner.py`, `ace.py`, `trace_analyser.py` |
| `ace/async_learning.py` | Deleted | Replaced by pipeline engine `async_boundary` |
| `ace/environments.py` | Unchanged | `Sample`, `EnvironmentResult`, `TaskEnvironment`, `SimpleEnvironment` stay |
| `ace2/steps/*.py` | `ace/steps/*.py` | Steps move into main package |
| `ace2/pipelines/offline.py` | `ace/ace.py` | ACE replaces both offline and online |
| `ace2/pipelines/online.py` | `ace/ace.py` | Merged into ACE |
| New | `ace/trace.py` | New `Trace` dataclass |
| New | `ace/trace_analyser.py` | New `TraceAnalyser` class |
| New | `ace/runner.py` | New shared `ACERunner` base |
| `ace/integrations/browser_use.py` | `ace/integrations/browser_use/` | Split into runner + steps + converter |
| `ace/integrations/langchain.py` | `ace/integrations/langchain/` | Split into runner + steps + converter |
| `ace/integrations/claude_code.py` | `ace/integrations/claude_code/` | Split into runner + steps + converter |

---

## Async Behaviour

Both TraceAnalyser and ACE inherit async capabilities from the pipeline engine. No custom async machinery is needed.

### ReflectStep as async boundary

`ReflectStep.async_boundary = True` means: when the pipeline processes a sample, everything before ReflectStep (Agent, Evaluate) runs in the foreground, and everything from ReflectStep onwards runs in a background thread pool.

```
sample 1:  [AgentStep] [EvaluateStep] ──fire──► [ReflectStep] [UpdateStep]  (background)
sample 2:  [AgentStep] [EvaluateStep] ──fire──► [ReflectStep] [UpdateStep]  (background)
                                       ↑
                                 async_boundary
```

For TraceAnalyser, there is no AgentStep or EvaluateStep in the foreground. The boundary still applies — context building is foreground, Reflect + Update are background:

```
trace 1:  [build_context] ──fire──► [ReflectStep] [UpdateStep]  (background)
trace 2:  [build_context] ──fire──► [ReflectStep] [UpdateStep]  (background)
```

### Controlling concurrency

| Knob | Where | Effect |
|---|---|---|
| `ReflectStep.max_workers = 3` | Step class attribute | Up to 3 reflections run in parallel |
| `UpdateStep.max_workers = 1` | Step class attribute | Serialises skillbook writes |
| `wait_for_background(timeout)` | Runner method | Blocks until background threads drain |

No custom `AsyncLearningPipeline` class, no manual thread management, no `asyncio.create_task` for background learning. The pipeline engine handles all of it.

---

## Error Handling

Follows the pipeline engine's error model without additions.

**Per-sample isolation:** A failing sample does not abort the run. The runner catches the exception, records it in `SampleResult.error` and `SampleResult.failed_at`, and continues to the next sample.

```python
def _process_one(self, ctx: StepContext) -> SampleResult:
    result = SampleResult(sample=ctx.sample, output=None, error=None, failed_at=None)
    try:
        result.output = self.pipeline(ctx)
    except Exception as exc:
        result.error = exc
        result.failed_at = type(exc).__name__
    return result
```

**Background failures:** Captured and attached to `SampleResult` by the pipeline engine. The runner calls `wait_for_background()` at the end to ensure all results are complete.

**No retry logic in the runner.** Retries are the responsibility of individual steps (e.g., LLM call retries via `tenacity` in the role classes).

---

## Observability

Observability (Opik integration, token tracking) moves out of the runner and into a dedicated step or hook.

### Option: ObservabilityStep

```python
class ObservabilityStep:
    requires = frozenset({"sample", "agent_output", "environment_result", "reflection", "skill_manager_output", "skillbook"})
    provides = frozenset()   # side-effect only

    def __call__(self, ctx: StepContext) -> StepContext:
        # Log metrics to Opik
        ...
        return ctx  # unchanged
```

Inserted at the end of the pipeline:

```python
ACE pipeline:            [Agent, Evaluate, Reflect, Update, Observability]
TraceAnalyser pipeline:  [Reflect, Update, Observability]
```

### Why move it out of the runner

The legacy `ACEBase._track_observability_data()` mixes observability concerns with orchestration logic. A step is:
- Independently testable
- Optional (omit it if Opik is not installed)
- Composable (add it to any pipeline)

---

## Pipeline Engine Change

One change to `Pipeline.run()` / `run_async()` is needed: accept pre-built `StepContext` objects, not only raw samples.

### Current behaviour

```python
# pipeline/pipeline.py — run_async, line 336
ctx = StepContext(sample=sample)   # always creates a bare context
```

This means the runner cannot set `skillbook`, `environment`, `epoch`, etc. before the pipeline starts processing.

### Required behaviour

If an item is already a `StepContext`, use it directly. Otherwise wrap it as before:

```python
async def process_one(item: Any) -> SampleResult:
    async with sem:
        ctx = item if isinstance(item, StepContext) else StepContext(sample=item)
        ...
```

This is the **only** change to the pipeline engine. Everything else — epoch loop, Iterable validation, context building — stays in the runner.

### Why not a `context_factory` callback?

A factory callback was considered but is unnecessary complexity. The runner already knows how to build contexts (`_build_context()`). It just needs the pipeline to pass them through unchanged. An `isinstance` check is simpler, has no new API surface, and is backwards-compatible — existing code that passes raw samples still works.

---

## Usage Examples

### TraceAnalyser — learn from browser-use history

```python
from ace import TraceAnalyser, Trace
from ace.llm_providers import LiteLLMClient

# Convert recorded browser sessions to traces
traces = [
    Trace(
        task="Find the cheapest flight to Tokyo",
        output="$450 on ANA, departing March 15",
        feedback="Correct price found in 8 steps",
        reasoning="Step 1: Navigate to Google Flights...",
    ),
    Trace(
        task="Book a hotel in Shibuya",
        output="Failed: could not find checkout button",
        feedback="Task failed after 15 steps — checkout button was behind a cookie modal",
        reasoning="Step 1: Navigate to Booking.com...",
    ),
]

# Analyse
analyser = TraceAnalyser.from_client(LiteLLMClient(model="gpt-4o-mini"))
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

### ACE — single-pass streaming

```python
def sample_stream():
    while True:
        yield get_next_sample()

ace = ACE.from_client(LiteLLMClient(model="gpt-4o-mini"))
results = ace.run(sample_stream(), environment, epochs=1)
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

## What Was Rejected and Why

**Runner extends Pipeline:**
Making TraceAnalyser and ACE subclasses of `Pipeline` was considered (and is what the current `ace2/OfflineACE` does). Rejected — the runner is not a pipeline. It owns the epoch loop. Composition (`self.pipeline`) keeps responsibilities separate.

**Cross-sample state (reflection window):**
A rolling window of recent reflections that persists across samples was considered, with variants: on the runner, on `StepContext`, on step instances, via a shared mediator object. All rejected — each sample should be independent. The only cross-sample coupling is the skillbook itself, which evolves as samples are processed. Adding a reflection window complicates the model (reset between epochs, eventual consistency with background steps, ordering issues with concurrent workers) for marginal benefit.

**Separate Online and Offline classes:**
Keeping two runner classes for single-pass and multi-epoch was considered. Rejected — the only difference is `epochs=1` vs `epochs > 1`, which is a parameter, not a class distinction. ACE handles both. TraceAnalyser is a separate class because its input type is fundamentally different (`Trace` vs `Sample + Environment`), not because of epoch count.

**Trace as a Sample subclass:**
Making `Trace` inherit from `Sample` was considered so that steps could accept either. Rejected — `Trace` has `output`, `feedback`, and `reasoning` which are outputs, not inputs. A Trace is a complete execution record; a Sample is an input to execution. Conflating them muddies the type contract.

**Steps that accept both Trace and Sample:**
Making ReflectStep and UpdateStep polymorphic over input type was considered. Rejected — steps always receive `StepContext` with the same named fields. The runner is responsible for building the context correctly. Steps do not need to know whether the data came from a Trace or from live execution.

**Observability in the runner:**
Keeping observability logic in `ACERunner._track_observability_data()` was considered. Rejected — it mixes concerns. A dedicated `ObservabilityStep` is independently testable, optional, and composable.

**Custom AsyncLearningPipeline:**
The legacy `ace/async_learning.py` implements a manual thread pool with reflector and skill manager queues. Rejected — the pipeline engine's `async_boundary` and `max_workers` provide the same functionality with less code and consistent semantics.

**Per-integration pipeline classes:**
Having each integration define its own pipeline class (`BrowserUsePipeline`, `LangChainPipeline`, `ClaudeCodePipeline`) was considered. Rejected — every integration pipeline has the same learning tail (`ReflectStep → UpdateStep`); only the execute step differs. Separate pipeline classes duplicate the learning tail wiring and the runner infrastructure. Instead, integrations provide execute steps that compose into an `ACERunner` subclass, reusing the shared `_run()` loop and epoch logic.

**Checkpoints in the runner:**
Having the runner own checkpoint logic was considered. Rejected — a `CheckpointStep` at the end of the pipeline tail keeps checkpointing within the pipeline formalism. The step counts samples internally and periodically saves the skillbook. No special runner support needed.
