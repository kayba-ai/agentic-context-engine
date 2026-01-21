<img src="https://framerusercontent.com/images/XBGa12hY8xKYI6KzagBxpbgY4.png" alt="Kayba Logo" width="1080"/>

# Agent Prompt Optimizer

![GitHub stars](https://img.shields.io/github/stars/kayba-ai/agentic-context-engine?style=social)
[![Discord](https://img.shields.io/discord/1429935408145236131?label=Discord&logo=discord&logoColor=white&color=5865F2)](https://discord.gg/mqCqH7sTyK)
[![Twitter Follow](https://img.shields.io/twitter/follow/kaybaai?style=social)](https://twitter.com/kaybaai)
[![PyPI version](https://badge.fury.io/py/ace-framework.svg)](https://badge.fury.io/py/ace-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## The Problem with Manual System Prompting

- Time-consuming iteration cycles of trial and error
- Prompt drift and regression as you patch edge cases
- No systematic learning from agent failures
- Knowledge stays in your head instead of in the prompt
- Hard to manage as prompts scale

## The Solution

ACE (Agentic Context Engine) automatically optimizes your agent's system prompt by learning from execution. It observes agent runs, analyzes what strategies worked and what failed, then generates actionable insights for the system prompt.

`Traces / Conversations` **→** `ACE` **→** `Prompt Suggestions`

You put in past traces or conversations. ACE handles agentic system prompting by learning from mistakes. You receive improved system prompt suggestions.

**How it works:**

0. **Prepare your data** - Place trace/conversation .md files in a folder. The more detailed your traces, the better the insights.

1. **ReplayAgent** - Simulates an agent for offline learning from your trace/conversation
2. **Reflector** - Analyzes each conversation to identify what worked, what failed, and why
3. **SkillManager** - Transforms reflections into atomic, actionable prompt strategies/insights
4. **Deduplicator** - Consolidates similar strategies/insights using embeddings to keep the output clean
5. **Skillbook** - Output file stores all prompt strategies/insights in a human-readable format you can review and implement

All insights are stored in a **human-readable skillbook** including prompt suggestion, reasoning and evidence. You can review, edit, or selectively apply any generated strategy. ACE can even suggest strategies that contradict your system prompt when it identifies flaws in the original design.

## Implementation

### Agentic System Prompting = ACE Offline Adapter

Process past traces and conversations in batch to generate insights **without the agent running**.

**Use case:** Periodic automated system prompt revision. Feed historical data, let ACE analyze patterns, then have a human review and choose what to implement.

```python
from ace import (
    Skillbook,
    Sample,
    OfflineACE,
    Reflector,
    SkillManager,
    ReplayAgent,
    SimpleEnvironment,
)
from ace.llm_providers.litellm_client import LiteLLMClient, LiteLLMConfig
from ace.prompts_v2_1 import PromptManager

# 1. Initialize LLM client
config = LiteLLMConfig(
    model="claude-sonnet-4-5-20250929",
    max_tokens=8192,
    temperature=0.1,
)
llm = LiteLLMClient(config=config)
prompt_mgr = PromptManager()

# 2. Create ACE components
skillbook = Skillbook()
agent = ReplayAgent()  # Dummy agent that replays conversations
reflector = Reflector(llm=llm, prompt_template=prompt_mgr.get_reflector_prompt())
skill_manager = SkillManager(llm=llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

# 3. Load past conversations as samples
samples = [
    Sample(
        question="Your task description here",
        context="The full conversation/trace content",
        ground_truth="",  # Empty for analysis tasks
        metadata={"source": "conversation_1"}
    ),
    # ... more historical data
]

# 4. Create adapter and run
environment = SimpleEnvironment()
adapter = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
)
results = adapter.run(samples, environment, epochs=1)

# 5. Review and save generated skills
print(adapter.skillbook.as_prompt())
adapter.skillbook.save_to_file("offline_adapter_skillbook.json")
```

**Tip:** Enable deduplication to automatically consolidate similar skills during learning. This keeps the skillbook clean.

```python
from ace import DeduplicationConfig

dedup_config = DeduplicationConfig(
    enabled=True,
    similarity_threshold=0.85,
)

adapter = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    dedup_config=dedup_config,
)
```

#### Async Mode

For large batches, enable async learning so the Reflector and SkillManager process in the background:

```python
adapter = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,
    max_reflector_workers=3,
)

results = adapter.run(samples, environment)
```

#### Checkpoints

Save skillbook periodically during long training runs:

```python
results = adapter.run(
    samples=samples,
    environment=environment,
    epochs=3,
    checkpoint_interval=10,  # Save every 10 samples
    checkpoint_dir="./checkpoints",
)
```

### Agentic Prompting at Runtime = Online Adapter

Fully autonomous self-improving agents at runtime. The agent learns from every interaction, generates insights, and injects them into future contexts automatically - no manual intervention required.

**Use case:** Continuous improvement in production where agents get better with every run.

See the [Quick Start Guide](../../docs/QUICKSTART.md) for setup instructions.

- **LiteLLM:** [`examples/litellm/`](../litellm/) - Make a new agent that self-learns
- **LangChain:** [`examples/langchain/`](../langchain/) - Wrap your existing agent with self-improving

## FAQ

**Can I combine manual prompts with ACE skills?**
Yes. ACE skills complement your base prompts. Start with manual prompts and let ACE build domain-specific expertise on top.

**What if ACE suggests something that contradicts my system prompt?**
Review it. ACE may have identified a flaw in your original design. The skillbook is human-readable JSON - you decide what to keep.

**Can I share skills between agents?**
Yes. Skillbooks are portable JSON files with human readable text.

## Next Steps

- Explore [examples](../) in this repository
- Read the [main documentation](https://github.com/kayba-ai/agentic-context-engine)
- Join our [Discord](https://discord.gg/mqCqH7sTyK) for tips and support
