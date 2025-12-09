# ACE Framework Documentation

> **Agentic Context Engineering (ACE)** - Build self-improving AI agents that learn from experience

**Version:** 0.7.0 | **Python:** 3.11+ | **License:** MIT

---

## Quick Navigation

| Getting Started | Core Concepts | Integrations | Advanced |
|-----------------|---------------|--------------|----------|
| [Quick Start](QUICK_START.md) | [Complete Guide](COMPLETE_GUIDE_TO_ACE.md) | [Integration Guide](INTEGRATION_GUIDE.md) | [Prompt Engineering](PROMPT_ENGINEERING.md) |
| [Setup Guide](SETUP_GUIDE.md) | [Architecture](ARCHITECTURE.md) | [API Reference](API_REFERENCE.md) | [Testing Guide](TESTING_GUIDE.md) |

---

## What is ACE?

ACE enables AI agents to **learn from their successes and failures** through in-context learning. Instead of expensive fine-tuning, ACE maintains a living "skillbook" of strategies that evolves based on execution feedback.

```python
from ace import ACELiteLLM

# Create a self-improving agent
agent = ACELiteLLM(model="gpt-4o-mini")

# Agent learns from each interaction
answer = agent.ask("What is the capital of France?")

# Save learned strategies for reuse
agent.save_skillbook("my_agent.json")
```

**Key Benefits:**
- **Self-Improving:** Agents get smarter with each task
- **No Fine-Tuning:** All learning happens in-context
- **Transparent:** View exactly what your agent learned
- **Cost-Efficient:** 16-62% token savings with TOON format

---

## Documentation Overview

### 1. Getting Started

| Document | Description | Time |
|----------|-------------|------|
| **[Quick Start](QUICK_START.md)** | Get running with ACE in minutes | 5 min |
| **[Setup Guide](SETUP_GUIDE.md)** | Development environment setup | 10 min |

### 2. Understanding ACE

| Document | Description | Audience |
|----------|-------------|----------|
| **[Complete Guide to ACE](COMPLETE_GUIDE_TO_ACE.md)** | Core concepts, architecture, how ACE works | Everyone |
| **[Architecture Deep Dive](ARCHITECTURE.md)** | Detailed system architecture, data flow, internals | Developers |

### 3. Using ACE

| Document | Description | Use Case |
|----------|-------------|----------|
| **[API Reference](API_REFERENCE.md)** | Complete API documentation | All development |
| **[Integration Guide](INTEGRATION_GUIDE.md)** | Add ACE to existing agents | Browser-use, LangChain, Custom |
| **[Prompt Guide](PROMPTS.md)** | Prompt versions and customization | Prompt engineering |

### 4. Advanced Topics

| Document | Description |
|----------|-------------|
| **[Prompt Engineering](PROMPT_ENGINEERING.md)** | Advanced prompt techniques for ACE |
| **[Testing Guide](TESTING_GUIDE.md)** | Testing patterns and best practices |
| **[Developer Guide](DEVELOPER_GUIDE.md)** | Contributing to ACE framework |

---

## Core Components

### The Three ACE Roles

ACE uses three specialized LLM roles (same base model, different prompts):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ACE LEARNING LOOP                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐     ┌─────────────┐     ┌────────────┐     ┌─────────────┐ │
│  │  Query  │────►│    Agent    │────►│  Reflector │────►│SkillManager │ │
│  └─────────┘     └─────────────┘     └────────────┘     └─────────────┘ │
│                        │                   │                   │         │
│                        ▼                   ▼                   ▼         │
│                   Executes            Analyzes            Updates        │
│                   using               outcome &           skillbook      │
│                   skillbook           identifies          with new       │
│                                       learnings           strategies     │
│                                                                          │
│                              ┌────────────┐                              │
│                              │  Skillbook │◄─────────────────────────────┤
│                              │ (Context)  │                              │
│                              └────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

| Role | Purpose | Key Method |
|------|---------|------------|
| **Agent** | Produces answers using skillbook strategies | `agent.generate()` |
| **Reflector** | Analyzes what worked/failed | `reflector.reflect()` |
| **SkillManager** | Updates skillbook with learnings | `skill_manager.update_skills()` |

### The Skillbook

The **Skillbook** is ACE's knowledge store - a collection of learned strategies:

```python
from ace import Skillbook

skillbook = Skillbook()

# Add a strategy
skillbook.add_skill(
    section="reasoning",
    content="Break complex problems into smaller steps"
)

# Save and load
skillbook.save_to_file("learned.json")
loaded = Skillbook.load_from_file("learned.json")
```

Each **Skill** tracks:
- `content`: The strategy text
- `helpful/harmful/neutral`: Success metrics
- `section`: Category for organization

---

## Ready-to-Use Integrations

### ACELiteLLM - Simple Q&A Agent

```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="gpt-4o-mini")
answer = agent.ask("Your question")
agent.save_skillbook("learned.json")
```

### ACEAgent - Browser Automation

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

agent = ACEAgent(llm=ChatBrowserUse(), ace_model="gpt-4o-mini")
await agent.run(task="Find top HN post")
agent.save_skillbook("browser_expert.json")
```

### ACELangChain - LangChain Wrapper

```python
from ace import ACELangChain

ace_chain = ACELangChain(runnable=your_chain)
result = ace_chain.invoke({"question": "Your task"})
ace_chain.save_skillbook("chain_learned.json")
```

**[Full Integration Guide →](INTEGRATION_GUIDE.md)**

---

## Key Features

### Async Learning Mode

Process samples in parallel while learning happens in background:

```python
adapter = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,
    max_reflector_workers=3
)

results = adapter.run(samples, environment)
adapter.wait_for_learning()  # Optional: wait for completion
```

### Checkpoint Saving

Save skillbook during long training runs:

```python
results = adapter.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=10,  # Save every 10 samples
    checkpoint_dir="./checkpoints"
)
```

### Skill Deduplication

Automatically consolidate similar strategies:

```python
from ace import DeduplicationConfig

config = DeduplicationConfig(
    enabled=True,
    similarity_threshold=0.85
)

agent = ACELiteLLM(model="gpt-4o-mini", dedup_config=config)
```

### Production Monitoring

Track costs and performance with Opik:

```bash
pip install ace-framework[observability]
export OPIK_API_KEY="your-key"
```

---

## Version 0.7.0 Terminology

ACE 0.7.0 introduced clearer terminology:

| Old Term (pre-0.7) | New Term (0.7+) |
|--------------------|-----------------|
| Playbook | **Skillbook** |
| Bullet | **Skill** |
| Generator | **Agent** |
| Curator | **SkillManager** |
| OfflineAdapter | **OfflineACE** |
| OnlineAdapter | **OnlineACE** |
| DeltaOperation | **UpdateOperation** |
| DeltaBatch | **UpdateBatch** |

---

## Prompt Versions

| Version | Status | Use Case |
|---------|--------|----------|
| **v1.0** (`prompts.py`) | Stable | Tutorials, simple tasks |
| v2.0 (`prompts_v2.py`) | **Deprecated** | Don't use |
| **v2.1** (`prompts_v2_1.py`) | **Recommended** | Production (+17% success rate) |

```python
from ace.prompts_v2_1 import PromptManager

mgr = PromptManager()
agent = Agent(llm, prompt_template=mgr.get_agent_prompt())
```

**[Full Prompt Guide →](PROMPTS.md)**

---

## Installation

```bash
# Basic
pip install ace-framework

# With extras
pip install ace-framework[browser-use]      # Browser automation
pip install ace-framework[langchain]        # LangChain
pip install ace-framework[observability]    # Opik monitoring
pip install ace-framework[all]              # Everything
```

**Development:**
```bash
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync  # Recommended: Uses UV package manager
```

---

## Examples

Browse the `examples/` directory for ready-to-run code:

| Category | Description |
|----------|-------------|
| `examples/litellm/` | Basic ACE usage, async learning, deduplication |
| `examples/langchain/` | LangChain chain/agent integration |
| `examples/browser-use/` | Browser automation with ACE |
| `examples/local-models/` | Ollama, LM Studio integration |
| `examples/prompts/` | Prompt comparison and engineering |

**Quick Example:**
```bash
uv run python examples/litellm/simple_ace_example.py
```

---

## Research Background

ACE is based on the paper:

> **"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"**
> Stanford University & SambaNova Systems
> [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)

Key findings:
- **+17.1 percentage points** improvement on AppWorld benchmark
- **86.9% lower adaptation latency** vs fine-tuning
- No training data or weight updates required

---

## Community & Support

- **GitHub:** [kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine)
- **Discord:** [Join our community](https://discord.gg/mqCqH7sTyK)
- **Issues:** [Report bugs or request features](https://github.com/kayba-ai/agentic-context-engine/issues)

---

## Document Index

### Getting Started
- [Quick Start Guide](QUICK_START.md) - Get running in 5 minutes
- [Setup Guide](SETUP_GUIDE.md) - Development environment setup

### Core Documentation
- [Complete Guide to ACE](COMPLETE_GUIDE_TO_ACE.md) - Core concepts and architecture
- [Architecture Deep Dive](ARCHITECTURE.md) - Detailed system internals
- [API Reference](API_REFERENCE.md) - Complete API documentation

### Integration & Usage
- [Integration Guide](INTEGRATION_GUIDE.md) - Add ACE to existing agents
- [Prompt Guide](PROMPTS.md) - Prompt versions and customization
- [Prompt Engineering](PROMPT_ENGINEERING.md) - Advanced prompt techniques

### Development
- [Developer Guide](DEVELOPER_GUIDE.md) - Contributing to ACE
- [Testing Guide](TESTING_GUIDE.md) - Testing patterns and practices

### Reference
- [Changelog](../CHANGELOG.md) - Version history
- [Contributing](../CONTRIBUTING.md) - How to contribute

---

**Last Updated:** December 2025 | **Version:** 0.7.0
