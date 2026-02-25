# LangChain Integration

The `LangChain` runner wraps any LangChain Runnable (chains, `AgentExecutor`, LangGraph graphs) with ACE learning. The runner extracts execution traces and learns strategies from them.

## Installation

```bash
pip install ace-framework[langchain]
```

## Quick Start

```python
from ace_next import LangChain

runner = LangChain.from_model(your_chain, ace_model="gpt-4o-mini")

results = runner.run([
    {"input": "Summarize this document"},
    {"input": "Extract key entities"},
])

runner.save("chain_expert.json")
```

## Parameters

### from_model()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runnable` | `Any` | — | LangChain Runnable (chain, AgentExecutor, graph) |
| `ace_model` | `str` | `"gpt-4o-mini"` | Model for Reflector + SkillManager |
| `ace_max_tokens` | `int` | `2048` | Max tokens for ACE LLM |
| `ace_llm` | `LLMClientLike` | `None` | Pre-built LLM for ACE roles |

### from_roles()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runnable` | `Any` | — | LangChain Runnable |
| `reflector` | `ReflectorLike` | — | Reflector instance |
| `skill_manager` | `SkillManagerLike` | — | SkillManager instance |
| `skillbook_path` | `str` | `None` | Load saved skillbook |
| `output_parser` | `Callable` | `None` | Custom output extraction |
| `dedup_config` | `DeduplicationConfig` | `None` | Deduplication config |
| `checkpoint_dir` | `str` | `None` | Checkpoint directory |

## Methods

```python
results = runner.run(inputs, epochs=1)      # Run with learning
results = runner.invoke(single_input)       # Single input convenience
runner.save("path.json")                    # Save skillbook
runner.wait_for_background()                # Wait for async learning
```

## How It Works

1. **INJECT** — Skillbook strategies are added to the chain input
2. **EXECUTE** — LangChain runs the chain normally
3. **Extract trace** — ACE extracts intermediate steps, tool calls, and reasoning
4. **LEARN** — Reflector analyzes the trace, SkillManager updates the skillbook

The runner handles simple chains, `AgentExecutor` (with `intermediate_steps`), and LangGraph graphs automatically.

## Input Types

The runner accepts any input your chain expects:

```python
# String input
runner.run(["What is ACE?"])

# Dict input
runner.run([{"input": "query", "context": "..."}])

# Message list
runner.run([[HumanMessage(content="Hello")]])
```

## Resuming from a Saved Skillbook

```python
runner = LangChain.from_model(
    your_chain,
    ace_model="gpt-4o-mini",
    skillbook_path="chain_expert.json",
)
```

## What to Read Next

- [Integration Pattern](../guides/integration.md) — how the INJECT/EXECUTE/LEARN pattern works
- [Insight Levels](../concepts/insight-levels.md) — meso-level learning from traces
- [Opik Observability](opik.md) — monitor chain execution costs
