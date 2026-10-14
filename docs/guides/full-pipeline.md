# Full Pipeline Guide

This guide walks through building a complete ACE pipeline from scratch — choosing components, defining an environment, running training, and saving results.

## Components

A full pipeline needs four things:

1. **LLM Client** — the language model powering all three roles
2. **Three Roles** — Agent, Reflector, SkillManager
3. **Environment** — evaluates agent outputs
4. **Samples** — training data with questions and ground truth

## Step 1: Create the Roles

Each role takes a model string directly. Supports any [LiteLLM model](https://docs.litellm.ai/) or PydanticAI-native identifier:

```python
from ace import Agent, Reflector, SkillManager

agent = Agent("gpt-4o-mini")
reflector = Reflector("gpt-4o-mini")
skill_manager = SkillManager("gpt-4o-mini")
```

Optionally use a cheaper model for learning:

```python
agent = Agent("gpt-4o")
reflector = Reflector("gpt-4o-mini")
skill_manager = SkillManager("gpt-4o-mini")
```

## Step 3: Define an Environment

The environment evaluates agent outputs. Extend `TaskEnvironment` and implement `evaluate()`:

```python
from ace import TaskEnvironment, EnvironmentResult

class MathEnvironment(TaskEnvironment):
    def evaluate(self, sample, agent_output):
        correct = str(sample.ground_truth).lower() in str(agent_output.final_answer).lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else f"Incorrect. Expected: {sample.ground_truth}",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )
```

Or use the built-in `SimpleEnvironment` for basic ground-truth matching:

```python
from ace import SimpleEnvironment

environment = SimpleEnvironment()
```

## Step 4: Prepare Samples

```python
from ace import Sample

samples = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
    Sample(question="Capital of France?", context="", ground_truth="Paris"),
    Sample(question="Who wrote Hamlet?", context="", ground_truth="Shakespeare"),
]
```

## Step 5: Build and Run the Pipeline

```python
from ace import ACE

runner = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=environment,
)

results = runner.run(samples, epochs=3)
```

## Step 6: Save the Skillbook

```python
runner.save("trained.json")
print(f"Learned {len(runner.skillbook.skills())} strategies")
```

## Complete Example

```python
from ace import (
    ACE, Agent, Reflector, SkillManager,
    Sample, SimpleEnvironment,
)

# Roles (each takes a model string directly)
agent = Agent("gpt-4o-mini")
reflector = Reflector("gpt-4o-mini")
skill_manager = SkillManager("gpt-4o-mini")

# Pipeline
runner = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=SimpleEnvironment(),
)

# Training data
samples = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
    Sample(question="Capital of France?", context="", ground_truth="Paris"),
]

# Train and save
results = runner.run(samples, epochs=3)
runner.save("trained.json")
```

## Checkpoints

Save the skillbook automatically during long training runs:

```python
runner = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=environment,
    checkpoint_dir="./checkpoints",
    checkpoint_interval=10,  # Save every 10 samples
)
```

This creates:

- `ace_checkpoint_10.json`, `ace_checkpoint_20.json`, etc.
- `ace_latest.json` (always the most recent)

## Deduplication

Prevent duplicate skills from accumulating (requires `uv add ace-framework[deduplication]`):

```python
from ace import DeduplicationConfig, DeduplicationManager

dedup = DeduplicationManager(DeduplicationConfig(
    enabled=True,
    embedding_model="text-embedding-3-small",
    similarity_threshold=0.85,
    consolidation_model="gpt-4o-mini",  # decides merge / delete / keep / update
))

runner = ACE.from_roles(
    ...,
    dedup_manager=dedup,
    dedup_interval=10,
)
```

Without `consolidation_model`, deduplication only logs a similarity report and does not change the skillbook. `ACELiteLLM` uses its own model when you pass a `dedup_config` without one.

## Custom Prompts

The default prompts are v2.1 and work well out of the box. You can pass your own templates via the `prompt_template` parameter:

```python
agent = Agent(llm, prompt_template="Your custom agent prompt with {skillbook}, {question}, {context}")
reflector = Reflector(llm, prompt_template="Your custom reflector prompt ...")
skill_manager = SkillManager(llm, prompt_template="Your custom skill manager prompt ...")
```

See [Prompt Engineering](prompts.md) for template variables and more examples.

## Testing Without API Calls

Use `test` as the model to get PydanticAI's built-in test model, or use `unittest.mock` to patch the agent's `run_sync` method:

```python
agent = Agent("test")
reflector = Reflector("test")
skill_manager = SkillManager("test")
```

## Observability

Add Opik tracing to any pipeline via `extra_steps` (requires `uv add ace-framework[observability]`):

```python
from ace import ACE, OpikStep, register_opik_litellm_callback

runner = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=environment,
    extra_steps=[OpikStep(project_name="my-experiment")],
)

# Optionally add per-LLM-call cost tracking
register_opik_litellm_callback(project_name="my-experiment")
```

See [Opik Observability](../integrations/opik.md) for full details.

## Going Deeper: Manual Pipeline Composition

The `ACE.from_roles()` runner composes a `Pipeline` internally. You can build the
same pipeline yourself for full control over step ordering, branching, and
custom steps:

```python
from ace import Pipeline, AgentStep, EvaluateStep, learning_tail

pipe = Pipeline([
    AgentStep(agent, skillbook),
    EvaluateStep(environment),
    *learning_tail(reflector, skill_manager, skillbook),
])
```

See [Composing Pipelines](composing-pipelines.md) for the complete guide.

## What to Read Next

- [Composing Pipelines](composing-pipelines.md) — compose custom pipelines from steps
- [Async Learning](async-learning.md) — parallel Reflector execution
- [Prompt Engineering](prompts.md) — customize prompt templates
- [Integration Pattern](integration.md) — wrap existing agents instead
- [Opik Observability](../integrations/opik.md) — monitor costs and traces
