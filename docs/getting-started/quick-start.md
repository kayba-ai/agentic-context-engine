# Quick Start

Get a self-learning agent running in under a minute.

## Simplest Example

```python
from ace_next import ACELiteLLM

agent = ACELiteLLM.from_model("gpt-4o-mini")

# Ask related questions — the agent learns patterns across them
answer1 = agent.ask("If all cats are animals, is Felix (a cat) an animal?")
answer2 = agent.ask("If all birds fly, can penguins (birds) fly?")

print(f"Learned {len(agent.skillbook.skills())} strategies")

# Save and reload later
agent.save("my_agent.json")
```

## Choose Your Integration

=== "LiteLLM"

    The simplest path. Supports 100+ LLM providers.

    ```python
    from ace_next import ACELiteLLM

    agent = ACELiteLLM.from_model("gpt-4o-mini")
    answer = agent.ask("Your question")
    agent.save("learned.json")
    ```

=== "LangChain"

    Wrap any LangChain Runnable (chains, agents, graphs) with learning.

    ```python
    from ace_next import LangChain

    runner = LangChain.from_model(your_chain, ace_model="gpt-4o-mini")
    results = runner.run([{"input": "Your task"}])
    runner.save("chain_expert.json")
    ```

=== "Browser-Use"

    Browser automation that learns navigation patterns.

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

=== "Claude Code"

    Self-improving coding agent using the Claude Code CLI.

    ```python
    from ace_next import ClaudeCode

    runner = ClaudeCode.from_model(working_dir="./my_project")
    results = runner.run("Add unit tests for utils.py")
    runner.save("coding_expert.json")
    ```

## Full Pipeline Example

For full control, use the three ACE roles directly:

```python
from ace_next import (
    ACE, Agent, Reflector, SkillManager,
    LiteLLMClient, Sample, SimpleEnvironment,
)

# Create LLM and roles
llm = LiteLLMClient(model="gpt-4o-mini")
agent = Agent(llm)
reflector = Reflector(llm)
skill_manager = SkillManager(llm)

# Build the adaptive pipeline
runner = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=SimpleEnvironment(),
)

# Train on samples
samples = [
    Sample(question="What is the capital of France?", context="", ground_truth="Paris"),
    Sample(question="What is 2 + 2?", context="", ground_truth="4"),
]

results = runner.run(samples, epochs=2)
print(f"Learned {len(runner.skillbook.skills())} strategies")
runner.save("trained.json")
```

## Loading Saved Agents

```python
from ace_next import ACELiteLLM

# Resume from a saved skillbook
agent = ACELiteLLM.from_model("gpt-4o-mini", skillbook_path="my_agent.json")
answer = agent.ask("New question")  # Uses previously learned strategies
```

## Trying Different Models

```python
from ace_next import ACELiteLLM

# OpenAI
agent = ACELiteLLM.from_model("gpt-4o-mini")

# Anthropic
agent = ACELiteLLM.from_model("claude-sonnet-4-5-20250929")

# Google
agent = ACELiteLLM.from_model("gemini-pro")

# Local (Ollama)
agent = ACELiteLLM.from_model("ollama/llama2")
```

## What to Read Next

- [How ACE Works](../concepts/overview.md) — understand the three-role architecture
- [The Skillbook](../concepts/skillbook.md) — how strategies are stored and evolve
- [Full Pipeline Guide](../guides/full-pipeline.md) — build custom ACE pipelines
- [Integrations](../integrations/index.md) — LangChain, Browser-Use, Claude Code
