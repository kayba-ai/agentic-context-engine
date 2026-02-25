# LiteLLM Integration

`ACELiteLLM` is the simplest way to get a self-improving agent. It bundles Agent, Reflector, SkillManager, and Skillbook into a single class with `ask()` and `learn()` methods.

## Quick Start

```python
from ace_next import ACELiteLLM

agent = ACELiteLLM.from_model("gpt-4o-mini")

# Ask questions — learns patterns across them
answer = agent.ask("If all cats are animals, is Felix (a cat) an animal?")

# Save and reload
agent.save("learned.json")
```

## Parameters

### from_model()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | LiteLLM model identifier |
| `max_tokens` | `int` | `2048` | Max tokens for responses |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `api_key` | `str` | `None` | API key (or use env variable) |
| `base_url` | `str` | `None` | Custom API endpoint |
| `skillbook_path` | `str` | `None` | Path to load saved skillbook |
| `environment` | `TaskEnvironment` | `None` | Evaluation environment |
| `dedup_config` | `DeduplicationConfig` | `None` | Skill deduplication config |
| `is_learning` | `bool` | `True` | Enable/disable learning |

## Methods

### ask()

Direct agent call using the current skillbook:

```python
answer = agent.ask("Your question", context="Optional context")
```

### learn()

Run the full ACE learning pipeline over samples:

```python
from ace_next import Sample, SimpleEnvironment

samples = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
]
results = agent.learn(samples, environment=SimpleEnvironment(), epochs=3)
```

### learn_from_feedback()

Learn from the last `ask()` interaction:

```python
agent.ask("What is the capital of France?")
agent.learn_from_feedback(feedback="Correct!", ground_truth="Paris")
```

### learn_from_traces()

Learn from pre-recorded execution traces:

```python
results = agent.learn_from_traces(traces, epochs=1)
```

### Lifecycle

```python
agent.save("path.json")               # Save skillbook
agent.load("path.json")               # Load skillbook
agent.enable_learning()               # Turn on learning
agent.disable_learning()              # Turn off learning
agent.wait_for_background()           # Wait for async learning
agent.learning_stats                  # Background progress
agent.skillbook                       # Current Skillbook
agent.get_strategies()                # Formatted strategies
```

## Using a Cheaper Learning Model

Use a strong model for the Agent and a cheaper one for learning:

```python
from ace_next import ACELiteLLM, Agent, Reflector, SkillManager, LiteLLMClient

agent_llm = LiteLLMClient(model="gpt-4o")
learning_llm = LiteLLMClient(model="gpt-4o-mini")

ace = ACELiteLLM(
    llm=agent_llm,
    agent=Agent(agent_llm),
    reflector=Reflector(learning_llm),
    skill_manager=SkillManager(learning_llm),
)
```

## Deduplication

Prevent duplicate skills from accumulating:

```python
from ace_next import DeduplicationConfig

agent = ACELiteLLM.from_model(
    "gpt-4o-mini",
    dedup_config=DeduplicationConfig(
        enabled=True,
        embedding_model="text-embedding-3-small",
        similarity_threshold=0.85,
    ),
)
```

## Supported Providers

Any model supported by [LiteLLM](https://docs.litellm.ai/):

```python
# OpenAI
agent = ACELiteLLM.from_model("gpt-4o-mini")

# Anthropic
agent = ACELiteLLM.from_model("claude-sonnet-4-5-20250929")

# Google
agent = ACELiteLLM.from_model("gemini-pro")

# Local (Ollama)
agent = ACELiteLLM.from_model("ollama/llama2")

# Custom endpoint
agent = ACELiteLLM.from_model("gpt-4o-mini", base_url="https://your-endpoint.com")
```

## What to Read Next

- [Full Pipeline Guide](../guides/full-pipeline.md) — for more control over the pipeline
- [Async Learning](../guides/async-learning.md) — background learning with `wait=False`
- [Opik Observability](opik.md) — monitor costs and traces
