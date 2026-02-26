# Browser-Use Integration

The `BrowserUse` runner wraps [browser-use](https://github.com/browser-use/browser-use) with ACE learning. The agent automates browser tasks and learns strategies from each run — improving navigation, element selection, and error recovery over time.

## Installation

```bash
pip install ace-framework[browser-use]
```

## Quick Start

```python
from ace_next import BrowserUse
from langchain_openai import ChatOpenAI

runner = BrowserUse.from_model(
    browser_llm=ChatOpenAI(model="gpt-4o"),
    ace_model="gpt-4o-mini",
)

results = runner.run("Find the top post on Hacker News")
runner.save("browser_expert.json")
```

## Parameters

### from_model()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `browser_llm` | `Any` | — | LLM for browser-use execution |
| `ace_model` | `str` | `"gpt-4o-mini"` | Model for Reflector + SkillManager |
| `ace_max_tokens` | `int` | `2048` | Max tokens for ACE LLM |
| `ace_llm` | `LLMClientLike` | `None` | Pre-built LLM for ACE roles |

### from_roles()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `browser_llm` | `Any` | — | LLM for browser-use execution |
| `reflector` | `ReflectorLike` | — | Reflector instance |
| `skill_manager` | `SkillManagerLike` | — | SkillManager instance |
| `skillbook_path` | `str` | `None` | Load saved skillbook |
| `browser` | `Browser` | `None` | browser-use Browser instance |
| `agent_kwargs` | `dict` | `None` | Extra kwargs for browser-use Agent |
| `dedup_config` | `DeduplicationConfig` | `None` | Deduplication config |
| `checkpoint_dir` | `str` | `None` | Checkpoint directory |

## Methods

```python
results = runner.run(tasks, epochs=1)       # Run with learning
runner.save("path.json")                    # Save skillbook
runner.wait_for_background()                # Wait for async learning
runner.get_strategies()                     # View learned strategies
```

## How It Works

1. **INJECT** — Skillbook strategies are added to the task prompt
2. **EXECUTE** — browser-use runs the task (navigation, clicks, form fills)
3. **Extract trace** — ACE extracts a chronological trace of agent thoughts, actions, and results
4. **LEARN** — Reflector analyzes the full trace, SkillManager updates the skillbook

The extracted trace includes:

- Agent reasoning at each step
- Browser actions taken (click, type, navigate)
- Page observations
- Success/failure of each action

## Running Multiple Tasks

```python
results = runner.run([
    "Find the top post on Hacker News",
    "Search for ACE framework on GitHub",
    "Check the weather in NYC",
])
```

## Example: Domain Checker

```python
from ace_next import BrowserUse
from langchain_openai import ChatOpenAI

runner = BrowserUse.from_model(
    browser_llm=ChatOpenAI(model="gpt-4o"),
    ace_model="gpt-4o-mini",
)

domains = ["example.com", "test.org", "sample.net"]
for domain in domains:
    runner.run(f"Check if {domain} is available for registration")

# After several runs, the agent learns:
# - Which registrar sites to use
# - How to navigate the domain search UI
# - How to interpret availability results
runner.save("domain_checker.json")
```

## Resuming from a Saved Skillbook

```python
runner = BrowserUse.from_model(
    browser_llm=ChatOpenAI(model="gpt-4o"),
    ace_model="gpt-4o-mini",
    skillbook_path="browser_expert.json",
)
```

## What to Read Next

- [Integration Pattern](../guides/integration.md) — how the INJECT/EXECUTE/LEARN pattern works
- [The Skillbook](../concepts/skillbook.md) — how learned strategies are stored
- [Opik Observability](opik.md) — monitor browser automation costs
