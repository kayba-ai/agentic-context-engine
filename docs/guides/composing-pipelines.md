# Composing Custom Pipelines

ACE is built on a composable pipeline engine. Every runner (`ACE`, `BrowserUse`,
`LangChain`, `ClaudeCode`, `TraceAnalyser`) is a thin wrapper around a `Pipeline`
made of steps. You can compose your own pipelines by mixing and matching these
steps — or writing custom ones.

## Three Levels of ACE

| Level | Pattern | Control |
|-------|---------|---------|
| **Zero-config** | `ACELiteLLM.from_model("gpt-4o-mini")` | Roles + pipeline auto-created |
| **Role customisation** | `ACE.from_roles(agent=..., reflector=..., ...)` | Custom roles, pipeline auto-composed |
| **Pipeline composition** | `Pipeline([AgentStep(...), ...])` | Full control over step ordering |

This guide covers **Level 3** — composing pipelines directly.

## Anatomy of an ACE Pipeline

Every ACE pipeline is a sequence of steps, each with a `requires`/`provides`
contract that declares what context fields it reads and writes:

```
AgentStep ─────> EvaluateStep ─────> ReflectStep ─────> UpdateStep ─────> ApplyStep
  provides:        provides:           provides:          provides:         (mutates
  agent_output     trace                reflections       skill_manager     skillbook)
                                                                        _output
```

The pipeline validates these contracts at construction time — if a step requires
a field that no earlier step provides, you'll get an error immediately.

## Composing from Steps

All pipeline classes and ACE steps are importable from `ace`:

```python
from ace import (
    # Pipeline engine
    Pipeline, Branch, MergeStrategy, StepProtocol, SampleResult,
    # ACE context
    ACEStepContext, SkillbookView,
    # Roles
    Agent, Reflector, SkillManager,
    # Steps
    AgentStep, EvaluateStep, learning_tail,
    # Types
    Sample, Skillbook, SimpleEnvironment,
)

skillbook = Skillbook()

pipe = Pipeline([
    AgentStep(Agent("gpt-4o-mini")),
    EvaluateStep(SimpleEnvironment()),
    *learning_tail(Reflector("gpt-4o-mini"), SkillManager("gpt-4o-mini"), skillbook),
])
```

## Using `learning_tail()`

The `learning_tail()` helper returns the standard learning step sequence:

```python
from ace import learning_tail, Reflector, SkillManager, Skillbook

steps = learning_tail(
    Reflector(llm),
    SkillManager(llm),
    Skillbook(),
    dedup_manager=my_dedup_manager,      # optional
    checkpoint_dir="/tmp/checkpoints",    # optional
)
# Returns: [ReflectStep, UpdateStep, ApplyStep,
#           DeduplicateStep, CheckpointStep]
```

Use it when building custom integrations that provide their own execute step but
want the standard learning pipeline.

## Inspecting Runner Presets with `build_steps()`

Every runner has a `build_steps()` classmethod that returns the step list it
would use internally. This lets you inspect, modify, and recompose:

```python
from ace import ACE, Pipeline, ACERunner, Skillbook

# Get the default steps
steps = ACE.build_steps(
    agent=my_agent,
    reflector=my_reflector,
    skill_manager=my_skill_manager,
    environment=my_env,
)

# Insert a custom step after EvaluateStep
steps.insert(2, MyLoggingStep())

# Build your own pipeline and runner
skillbook = Skillbook()
pipe = Pipeline(steps)
runner = ACERunner(pipeline=pipe, skillbook=skillbook)
results = runner.run(samples)
```

All runners support `build_steps()`: `ACE`, `BrowserUse`, `ClaudeCode`,
`LangChain`, and `TraceAnalyser`.

## Writing Custom Steps

A step is any object satisfying `StepProtocol` — no base class needed:

```python
from ace import ACEStepContext

class MyLoggingStep:
    requires = frozenset({"agent_output"})
    provides = frozenset()

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        print(f"Agent answered: {ctx.agent_output.final_answer}")
        return ctx
```

Key rules:

- `requires`: frozenset of context field names this step reads
- `provides`: frozenset of context field names this step writes
- `__call__`: receives and returns `ACEStepContext` (use `ctx.replace(...)` for updates)
- Steps should be stateless — no internal counters

## Mixing Integrations

You can compose steps from different integrations into one pipeline. For example,
combining a browser-use execute step with custom learning:

```python
from ace import Pipeline, learning_tail, Reflector, SkillManager, Skillbook
from ace.integrations.browser_use import BrowserExecuteStep, BrowserToTrace

skillbook = Skillbook()
pipe = Pipeline([
    BrowserExecuteStep(browser_llm),
    BrowserToTrace(),
    MyCustomFilterStep(),  # your custom step
    *learning_tail(Reflector(llm), SkillManager(llm), skillbook),
])
```

Integration steps live in `ace.integrations` since they have
framework-specific dependencies.

## Running the Pipeline

### With a runner

The simplest way to run a custom pipeline is through `ACERunner`:

```python
from ace import ACERunner, Sample, Skillbook

runner = ACERunner(pipeline=pipe, skillbook=skillbook)
results = runner.run(
    [Sample(question="What is 2+2?", ground_truth="4")],
    epochs=1,
)
```

### Directly

You can also run the pipeline directly by constructing contexts yourself:

```python
from ace import Pipeline, ACEStepContext, SkillbookView, Sample, Skillbook

ctx = ACEStepContext(
    sample=Sample(question="What is 2+2?", ground_truth="4"),
    skillbook=SkillbookView(skillbook),
)

results = pipe.run([ctx])
pipe.wait_for_background()  # wait for async learning steps
```

## Branching (Parallel Steps)

The pipeline engine supports parallel branches for steps that can run
concurrently:

```python
from ace import Pipeline, Branch, MergeStrategy

pipe = Pipeline([
    AgentStep(agent),
    Branch(
        [EvaluateStep(env_a), EvaluateStep(env_b)],
        merge=MergeStrategy.LAST,
    ),
    *learning_tail(reflector, skill_manager, skillbook),
])
```

See the [Pipeline Engine docs](../pipeline/branching.md) for full branching
and merge strategy details.

## Using RRStep (Recursive Reflector)

`RRStep` satisfies both `StepProtocol` and `ReflectorLike`, so it can be used
in two ways:

### As a drop-in reflector replacement

Pass it anywhere a `Reflector` is expected:

```python
from ace import ACELiteLLM
from ace.rr import RRStep, RRConfig

ace = ACELiteLLM.from_model("gpt-4o-mini", reflector=RRStep("gpt-4o-mini", config=RRConfig(max_requests=10)))
```

### As a pipeline step

Place it directly in a pipeline (it provides `reflections`):

```python
from ace import Pipeline, learning_tail, SkillManager, Skillbook
from ace.rr import RRStep, RRConfig

skillbook = Skillbook()
rr = RRStep("gpt-4o-mini", config=RRConfig(max_requests=15))

pipe = Pipeline([
    MyExecuteStep(),
    MyToTrace(),
    rr,  # replaces ReflectStep — provides "reflections"
    *learning_tail(None, SkillManager("gpt-4o-mini"), skillbook)[1:],  # skip ReflectStep
])
```

### With recursion enabled

Allow the RR to decompose large batch inputs via recursive child sessions:

```python
from ace.rr import RRStep, RRConfig

rr = RRStep(
    "gpt-4o",
    config=RRConfig(max_requests=40, max_depth=1),  # depth=1 allows one level of recursion
)
```

## Available Steps

All steps are importable from `ace`:

| Step | Purpose |
|------|---------|
| `AgentStep` | Execute Agent role |
| `EvaluateStep` | Run TaskEnvironment evaluation |
| `ReflectStep` | Run Reflector role (async boundary) |
| `UpdateStep` | Run SkillManager to generate updates |
| `ApplyStep` | Apply updates to skillbook |
| `DeduplicateStep` | Merge near-duplicate skills |
| `CheckpointStep` | Save skillbook to disk |
| `LoadTracesStep` | Load JSONL trace files |
| `ExportSkillbookMarkdownStep` | Export skillbook as markdown |
| `ObservabilityStep` | Generic observability hook |
| `PersistStep` | Persist step output |
| `OpikStep` | Log traces to Opik |
| `RRStep` | Recursive Reflector |

Integration steps (in `ace.integrations`):

| Step | Integration |
|------|-------------|
| `BrowserExecuteStep` / `BrowserToTrace` | browser-use |
| `LangChainExecuteStep` / `LangChainToTrace` | LangChain |
| `ClaudeCodeExecuteStep` / `ClaudeCodeToTrace` | Claude Code |
| `ClaudeSDKExecuteStep` / `ClaudeSDKToTrace` | Anthropic Python SDK |
| `OpenClawToTraceStep` | OpenClaw |
