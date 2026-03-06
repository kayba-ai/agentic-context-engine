# Integration Pattern

Use the integration pattern when you have an **existing agent** (browser-use, LangChain, Claude Code, or a custom framework) and want to add ACE learning on top.

!!! note "Full Pipeline vs Integration"
    The [Full Pipeline](full-pipeline.md) uses all three ACE roles. The integration pattern skips the ACE Agent — your external agent handles execution, and ACE only learns from the results.

## Three Steps

Every integration follows the same pattern:

```
1. INJECT   — Add skillbook strategies to the agent's context
2. EXECUTE  — Run the external agent normally
3. LEARN    — Reflector + SkillManager update the skillbook
```

## Using Built-In Runners

ACE provides runners for popular frameworks. Each uses `from_model()` for quick setup or `from_roles()` for full control:

=== "Browser-Use"

    ```python
    from ace_next import BrowserUse
    from langchain_openai import ChatOpenAI

    runner = BrowserUse.from_model(
        browser_llm=ChatOpenAI(model="gpt-4o"),
        ace_model="gpt-4o-mini",
    )
    results = runner.run(["Find top HN post", "Check weather in NYC"])
    runner.save("browser_expert.json")
    ```

=== "LangChain"

    ```python
    from ace_next import LangChain

    runner = LangChain.from_model(your_chain, ace_model="gpt-4o-mini")
    results = runner.run([{"input": "Summarize this document"}])
    runner.save("chain_expert.json")
    ```

=== "Claude Code"

    ```python
    from ace_next import ClaudeCode

    runner = ClaudeCode.from_model(working_dir="./my_project")
    results = runner.run(["Add tests for utils.py", "Fix the login bug"])
    runner.save("code_expert.json")
    ```

## Construction Patterns

All integration runners offer two construction paths:

### from_model() — Quick Setup

Builds ACE roles automatically from a model string:

```python
runner = BrowserUse.from_model(
    browser_llm=ChatOpenAI(model="gpt-4o"),
    ace_model="gpt-4o-mini",       # Model for Reflector + SkillManager
    skillbook_path="saved.json",   # Optional: resume from saved skillbook
)
```

### from_roles() — Full Control

Provide pre-built role instances:

```python
from ace_next import Reflector, SkillManager, LiteLLMClient

learning_llm = LiteLLMClient(model="gpt-4o-mini")

runner = BrowserUse.from_roles(
    browser_llm=ChatOpenAI(model="gpt-4o"),
    reflector=Reflector(learning_llm),
    skill_manager=SkillManager(learning_llm),
    skillbook_path="saved.json",
    dedup_config=my_dedup_config,
    checkpoint_dir="./checkpoints",
)
```

## Common Options

All integration runners share these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `skillbook` | Existing `Skillbook` instance | `None` (creates empty) |
| `skillbook_path` | Path to load skillbook from | `None` |
| `dedup_config` | Deduplication configuration | `None` |
| `dedup_interval` | Samples between dedup runs | `10` |
| `checkpoint_dir` | Directory for checkpoint files | `None` |
| `checkpoint_interval` | Samples between checkpoints | `10` |

## Lifecycle Methods

All runners expose:

```python
runner.save("path.json")              # Save skillbook
runner.wait_for_background()          # Wait for async learning
runner.learning_stats                 # Background progress dict
runner.skillbook                      # Current Skillbook instance
runner.get_strategies()               # Formatted strategies string
```

## Building a Custom Integration

For frameworks not covered by the built-in runners, you can compose a custom pipeline using steps.

The pattern: **Execute Step** (runs your agent) + **ToTrace Step** (extracts learning signal) + **learning_tail()** (standard learning pipeline).

```python
from pipeline import Pipeline
from ace_next import Skillbook, Reflector, SkillManager, LiteLLMClient
from ace_next.steps import learning_tail
from ace_next.runners import ACERunner

# Your custom execute step would implement the Step protocol
# See the Pipeline Engine docs for details on building custom steps

learning_llm = LiteLLMClient(model="gpt-4o-mini")
skillbook = Skillbook()

steps = [
    MyCustomExecuteStep(...),
    MyCustomToTraceStep(),
    *learning_tail(
        Reflector(learning_llm),
        SkillManager(learning_llm),
        skillbook,
    ),
]

runner = ACERunner(pipeline=Pipeline(steps), skillbook=skillbook)
```

See [Pipeline Engine: Building Custom Steps](../pipeline/custom-steps.md) for the Step protocol.

## What to Read Next

- [LiteLLM Integration](../integrations/litellm.md) — simplest self-improving agent
- [Browser-Use Integration](../integrations/browser-use.md) — browser automation details
- [LangChain Integration](../integrations/langchain.md) — chain/agent wrapping
- [Claude Code Integration](../integrations/claude-code.md) — coding tasks
