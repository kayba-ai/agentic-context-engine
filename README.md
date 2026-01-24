<img src="https://framerusercontent.com/images/XBGa12hY8xKYI6KzagBxpbgY4.png" alt="Kayba Logo" width="1080"/>

# Agentic Context Engine (ACE)

![GitHub stars](https://img.shields.io/github/stars/kayba-ai/agentic-context-engine?style=social)
[![Discord](https://img.shields.io/discord/1429935408145236131?label=Discord&logo=discord&logoColor=white&color=5865F2)](https://discord.gg/mqCqH7sTyK)
[![Twitter Follow](https://img.shields.io/twitter/follow/kaybaai?style=social)](https://twitter.com/kaybaai)
[![PyPI version](https://badge.fury.io/py/ace-framework.svg)](https://badge.fury.io/py/ace-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI agents that get smarter with every task**

⭐ Star this repo if you find it useful!

---

## What is ACE?

ACE enables AI agents to **learn from their execution feedback**—what works, what doesn't—and continuously improve. No fine-tuning, no training data, just automatic in-context learning.

The framework maintains a **Skillbook**: a living document of strategies that evolves with each task. When your agent succeeds, ACE extracts patterns. When it fails, ACE learns what to avoid. All learning happens transparently in context.

- **Self-Improving**: Agents autonomously get smarter with each task
- **20-35% Better Performance**: Proven improvements on complex tasks
- **49% Token Reduction**: Demonstrated in browser automation benchmarks
- **No Context Collapse**: Preserves valuable knowledge over time

---

## LLM Quickstart
1. Direct your favorite coding agent (Cursor, Claude Code, Codex, etc) to [Quick Start Guide](docs/QUICK_START.md)
2. Prompt away!

---

## Quick Start

### 1. Install

```bash
pip install ace-framework
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Run

```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="gpt-4o-mini")

answer = agent.ask("What does Kayba's ACE framework do?")
print(answer)  # "ACE allows AI agents to remember and learn from experience!"
```

**Done! Your agent learns automatically from each interaction.**

[→ Quick Start Guide](docs/QUICK_START.md) | [→ Setup Guide](docs/SETUP_GUIDE.md)

---

## Use Cases

### Enhance Existing Agents
Wrap your existing agent (browser-use, LangChain, custom) with ACE learning. Your agent executes tasks normally while ACE analyzes results and builds a skillbook of effective strategies.

### Automated System Prompting
The Skillbook acts as an evolving system prompt that automatically improves based on execution feedback—no manual prompt engineering required.

### Claude Code with Learning
Run coding tasks with Claude Code while ACE learns patterns from each execution, building expertise over time for your specific codebase and workflows.

### Build Self-Improving Agents
Create new agents with built-in learning for customer support, data extraction, code generation, research, content creation, and task automation.

---

## Demos

### The Seahorse Emoji Challenge

A challenge where LLMs often hallucinate that a seahorse emoji exists (it doesn't).

<img src="examples/seahorse-emoji-ace.gif" alt="Seahorse Emoji ACE Demo" width="70%"/>

In this example:
1. The agent incorrectly outputs a horse emoji
2. ACE reflects on the mistake without external feedback
3. On the second attempt, the agent correctly realizes there is no seahorse emoji

[→ Try it yourself](examples/litellm/seahorse_emoji_ace.py)

### Browser Automation

**Online Shopping Demo**: ACE vs baseline agent shopping for 5 grocery items.

<img src="examples/browser-use/online-shopping/results-online-shopping-brwoser-use.png" alt="Online Shopping Demo Results" width="70%"/>

In this example:
- ACE learns to navigate the website over 10 attempts
- Performance stabilizes and step count decreases by 29.8%
- Token costs reduce 49.0% for base agent and 42.6% including ACE overhead

[→ Try it yourself & see all demos](examples/browser-use/README.md)

### Claude Code Loop

In this example, Claude Code is enhanced with ACE and self-reflects after each execution while translating the ACE library from Python to TypeScript.

**Python → TypeScript Translation:**

| Metric | Result |
|--------|--------|
| Duration | ~4 hours |
| Commits | 119 |
| Lines written | ~14k |
| Outcome | Zero build errors, all tests passing |
| API cost | ~$1.5 (Sonnet for learning) |

[→ Claude Code Loop](examples/claude-code-loop/)

---

## Integrations

ACE integrates with popular agent frameworks:

| Integration | ACE Class | Use Case |
|-------------|-----------|----------|
| LiteLLM | `ACELiteLLM` | Simple self-improving agent |
| LangChain | `ACELangChain` | Wrap LangChain chains/agents |
| browser-use | `ACEAgent` | Browser automation |
| Claude Code | `ACEClaudeCode` | Claude Code CLI |
| Opik | `OpikIntegration` | Production monitoring and cost tracking |

[→ Integration Guide](docs/INTEGRATION_GUIDE.md) | [→ Examples](examples/)

---

## How Does ACE Work?

*Based on the [ACE research framework](https://arxiv.org/abs/2510.04618) from Stanford & SambaNova.*

ACE uses three specialized roles that work together:
1. **Agent** - Creates a plan using learned skills and executes the task
2. **Reflector** - Analyzes what worked and what didn't after execution
3. **SkillManager** - Updates the skillbook with new strategies based on reflection

**Important:** The three ACE roles are different specialized prompts using the same language model, not separate models.

ACE teaches your agent and internalizes:
- **Successes** → Extract patterns that work
- **Failures** → Learn what to avoid
- **Tool usage** → Discover which tools work best for which tasks
- **Edge cases** → Remember rare scenarios and how to handle them

The magic happens in the **Skillbook**—a living document of skills that evolves with experience.
**Key innovation:** All learning happens **in context** through incremental updates—no fine-tuning, no training data, and complete transparency into what your agent learned.

```mermaid
---
config:
  look: neo
  theme: neutral
---
flowchart LR
    Skillbook[("`**Skillbook**<br>(Evolving Context)<br><br>•Strategy Skills<br> Helpful skills <br> Harmful patterns <br> Neutral observations`")]
    Start(["**Query** <br>User prompt or question"]) --> Agent["**Agent** <br>Executes task using skillbook"]
    Agent --> Reflector
    Skillbook -. Provides Context .-> Agent
    Environment["**Task Environment**<br>Evaluates answer<br>Provides feedback"] -- Feedback+ <br>Optional Ground Truth --> Reflector
    Reflector["**Reflector**<br>Analyzes and provides feedback what was helpful/harmful"]
    Reflector --> SkillManager["**SkillManager**<br>Produces improvement updates"]
    SkillManager --> UpdateOps["**Merger** <br>Updates the skillbook with updates"]
    UpdateOps -- Incremental<br>Updates --> Skillbook
    Agent <--> Environment
```

---

## Documentation

- [Quick Start Guide](docs/QUICK_START.md) - Get running in 5 minutes
- [Setup Guide](docs/SETUP_GUIDE.md) - Installation, configuration, providers
- [Integration Guide](docs/INTEGRATION_GUIDE.md) - Add ACE to existing agents
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Complete Guide to ACE](docs/COMPLETE_GUIDE_TO_ACE.md) - Deep dive into concepts
- [Prompt Engineering](docs/PROMPT_ENGINEERING.md) - Advanced prompt techniques
- [Agentic System Prompting](examples/agentic-system-prompting/README.md) - Automatically generate prompt improvements from past traces
- [Examples](examples/) - Ready-to-run code examples
- [Benchmarks](benchmarks/README.md) - Evaluate ACE performance
- [Changelog](CHANGELOG.md) - Recent changes

---

## Contributing

We love contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

---

## Acknowledgment

Based on the [ACE paper](https://arxiv.org/abs/2510.04618) and inspired by [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952).

If you use ACE in your research, please cite:
```bibtex
@article{zhang2024ace,title={Agentic Context Engineering},author={Zhang et al.},journal={arXiv:2510.04618},year={2024}}
```

---

<div align="center">

**⭐ Star this repo if you find it useful!**

**Built with ❤️ by [Kayba](https://kayba.ai) and the open-source community.**

</div>
