# ACE2 Pipeline Architecture

ACE2 reimplements the ACE core on top of the generic pipeline engine (`pipeline/`).
This document records the design choices, file layout, and migration path.

## Why ACE2?

The original ACE implementation (`ace/adaptation.py`) hard-codes the four-step loop
(Agent → Evaluate → Reflect → Update) inside monolithic `OfflineACE` / `OnlineACE`
classes. This makes it difficult to:

- Swap, reorder, or skip steps
- Run steps in parallel or in background threads
- Share the pipeline engine with non-ACE workloads
- Test individual steps in isolation

ACE2 decomposes the same logic into discrete **steps** that plug into the generic
pipeline engine, giving us all of the above for free.

## File Layout

```
ace2/
├── __init__.py
├── steps/
│   ├── __init__.py          # re-exports all steps
│   ├── agent.py             # AgentStep
│   ├── evaluate.py          # EvaluateStep
│   ├── reflect.py           # ReflectStep
│   └── update.py            # UpdateStep
└── pipelines/
    ├── __init__.py           # ace_pipeline() factory + re-exports
    ├── offline.py            # OfflineACE runner
    └── online.py             # OnlineACE runner
```

## The Four Steps

Each step implements the `StepProtocol` (requires/provides/`__call__`).
The pipeline engine validates that every step's `requires` are satisfied
by earlier steps or by the initial context seeded by the runner.

| Step | Requires | Provides | Notes |
|------|----------|----------|-------|
| **AgentStep** | `sample`, `skillbook` | `agent_output` | Wraps `Agent.generate()`. Reads `recent_reflections` from context but does NOT declare it as a requirement (see below). |
| **EvaluateStep** | `sample`, `agent_output`, `environment` | `environment_result` | Stateless — no constructor args. Calls `environment.evaluate()`. |
| **ReflectStep** | `sample`, `agent_output`, `environment_result`, `skillbook` | `reflection`, `recent_reflections` | `async_boundary=True`, `max_workers=3`. Tags skills and maintains rolling reflection window. |
| **UpdateStep** | `reflection`, `skillbook`, `sample`, `environment_result`, `agent_output` | `skill_manager_output` | `max_workers=1` to serialise skillbook writes. Attaches insight-source provenance. |

### Pipeline Wiring

```python
Pipeline()
    .then(AgentStep(agent))
    .then(EvaluateStep())
    .then(ReflectStep(reflector, reflection_window=3))
    .then(UpdateStep(skill_manager))
```

## Design Decisions

### Composition over Inheritance

The runners (`OfflineACE`, `OnlineACE`) hold a `Pipeline` by composition — they
are **not** subclasses of `Pipeline`. This matches the spec and keeps the runner
logic (epoch loops, checkpointing, error capture) separate from the pipeline
engine's step-chaining logic.

### `recent_reflections` is Cross-Sample State

`recent_reflections` is a rolling window of serialised reflections that carries
across samples. It is:

- **Seeded** by the runner (starts as empty tuple `()`)
- **Grown** by `ReflectStep` (appends new reflection, trims to window size)
- **Propagated** by the runner between samples (`recent_reflections = out_ctx.recent_reflections`)
- **Read** by `AgentStep` to build the reflection context string

Critically, `AgentStep` does **not** declare `recent_reflections` in its `requires`
set. If it did, the pipeline validator would see a circular dependency (AgentStep
requires something that ReflectStep provides, but ReflectStep comes after AgentStep).
Instead, the runner guarantees it is always present in the initial `StepContext`.

### Async Boundaries and Worker Limits

- `ReflectStep` sets `async_boundary = True` — the pipeline engine can run it
  (and everything after it) in a background thread pool, so the Agent + Evaluate
  foreground path returns quickly.
- `ReflectStep` uses `max_workers = 3` — multiple reflections can run concurrently.
- `UpdateStep` uses `max_workers = 1` — skillbook mutations are serialised to
  avoid race conditions, even when multiple ReflectSteps finish in parallel.

### Insight Source Provenance

`UpdateStep` calls `build_insight_source()` before applying updates to the
skillbook. This attaches provenance metadata (sample question, epoch, step,
error identification) to each operation so we can trace where a skill came from.

### Deduplication Wiring

Both runners accept an optional `dedup_config` parameter in `from_roles()` /
`from_client()`. If provided and the `SkillManager` doesn't already have a
`DeduplicationManager`, one is created and attached. This matches the old
`ACEBase` behaviour.

## API Surface

### Quick Start (from_client)

The simplest way — a single LLM client, all roles created internally:

```python
from ace.llm_providers import LiteLLMClient
from ace2.pipelines import OfflineACE, OnlineACE

client = LiteLLMClient(model="gpt-4o-mini")

# Offline: multi-epoch training
ace = OfflineACE.from_client(client)
results = ace.run(train_samples, environment, epochs=3)

# Online: single-pass streaming
ace = OnlineACE.from_client(client)
results = ace.run(test_samples, environment)
```

### Custom Roles (from_roles)

Full control over role configuration:

```python
from ace.roles import Agent, Reflector, SkillManager
from ace.prompts_v2_1 import PromptManager

pm = PromptManager()
ace = OfflineACE.from_roles(
    agent=Agent(client, prompt_template=pm.get_agent_prompt()),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    skillbook=skillbook,
    reflection_window=5,
    dedup_config=dedup_config,
)
```

### Manual Pipeline

Build a custom pipeline with different steps or ordering:

```python
from pipeline import Pipeline
from ace2.steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep

pipe = (
    Pipeline()
    .then(AgentStep(agent))
    .then(EvaluateStep())
    .then(ReflectStep(reflector, reflection_window=5))
    .then(UpdateStep(skill_manager))
)

# Use with a runner or call directly
ace = OfflineACE(pipe, skillbook)
```

## Runner Behaviour

### OfflineACE

- Loops `epochs × samples`, building a `StepContext` per sample
- Propagates `recent_reflections` across samples within an epoch
- Supports checkpoint saving (`checkpoint_interval` + `checkpoint_dir`)
- Captures exceptions per-sample as `SampleResult.error`

### OnlineACE

- Single pass over an arbitrary iterable (list or generator)
- Same `StepContext` + error capture pattern
- No epoch concept — `epoch=1, total_epochs=1` always

## Testing

```bash
python -m pytest tests/test_ace2_pipeline.py -v
```

The test suite covers:
- Each step in isolation (6 tests)
- Pipeline wiring validation (1 test)
- OfflineACE end-to-end: single epoch, multi epoch, checkpointing, error capture (4 tests)
- OnlineACE end-to-end: single sample, streaming generator (2 tests)
- `from_client` shorthand for both runners (3 tests)

## Migration Path

ACE2 is currently a parallel implementation. The cut-over plan (TODO Part 5):

1. Move `ace2/steps/` and `ace2/pipelines/` into `ace/pipeline/`
2. Update `ace/__init__.py` to import from the new location
3. Deprecate `ace/adaptation.py` (`OfflineACE`, `OnlineACE`, `ACEBase`)
4. Update all examples and benchmarks to use the new imports
