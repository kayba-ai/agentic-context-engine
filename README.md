<a href="https://kayba.ai"><img src="assets/kayba-banner.png" alt="Kayba - Make your agents self-improve from experience" width="1080"/></a>

# Agentic Context Engine (ACE)

![GitHub stars](https://img.shields.io/github/stars/kayba-ai/agentic-context-engine?style=social)
[![Kayba Website](https://img.shields.io/badge/kayba.ai-6B8BA8?style=flat&logo=data:image/png;base64,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&logoColor=white)](https://kayba.ai)
[![Discord](https://img.shields.io/discord/1429935408145236131?label=Discord&logo=discord&logoColor=white&color=5865F2)](https://discord.gg/mqCqH7sTyK)
[![Twitter Follow](https://img.shields.io/twitter/follow/kaybaai?style=social)](https://twitter.com/kaybaai)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://kayba-ai.github.io/agentic-context-engine/latest/)

> [!TIP]
> ### Try our hosted solution for free at [kayba.ai](https://kayba.ai): automated agent self-improvement from your terminal. CLI + dashboard that analyzes traces, surfaces failures, and ships improvements directly from Claude Code, Codex, and more.
> [![Kayba Pro](https://img.shields.io/badge/Kayba_Pro-Start_Free_Trial-4A6B80?style=for-the-badge)](https://kayba.ai)

---

**AI agents don't learn from experience.** They repeat the same mistakes every session, forget what worked, and ignore what failed. ACE adds a persistent learning loop that makes them better over time.

<img src="examples/seahorse-emoji-ace.gif" alt="ACE learns from mistakes in real time" width="70%"/>

> The agent claims a seahorse emoji exists. ACE reflects on the error, and on the next attempt, the agent responds correctly — without human intervention.

---

## Proven Results

| Metric | Result | Context |
|:-------|:-------|:--------|
| **2x consistency** | Doubles pass^4 on Tau2 airline benchmark | 15 learned strategies, no reward signals |
| **49% token reduction** | Browser automation costs cut nearly in half | 10-run learning curve |
| **$1.50 learning cost** | Claude Code translated 14k lines to TypeScript | Zero build errors, all tests passing |

---

## Quick Start

```bash
uv add ace-framework
```

**Option A** — Interactive setup (recommended):

```bash
ace setup            # Walks you through model selection, API keys, and connection validation
```

**Option B** — Manual configuration:

```bash
export OPENAI_API_KEY="your-key"    # or ANTHROPIC_API_KEY, or any of 100+ supported providers
```

Then use it:

```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="gpt-4o-mini")

# First attempt — the agent may hallucinate
answer = agent.ask("Is there a seahorse emoji?")

# Feed a correction — ACE extracts a strategy and updates the Skillbook
agent.learn_from_feedback("There is no seahorse emoji in Unicode.")

# Subsequent calls benefit from the learned strategy
answer = agent.ask("Is there a seahorse emoji?")

# Inspect what the agent has learned
print(agent.get_strategies())
```

No fine-tuning, no training data, no vector database.

[-> Quick Start Guide](https://kayba-ai.github.io/agentic-context-engine/latest/getting-started/quick-start/) | [-> Setup Guide](https://kayba-ai.github.io/agentic-context-engine/latest/getting-started/setup/) | [-> Hosted API: Where Do Traces Come From?](https://kayba-ai.github.io/agentic-context-engine/latest/integrations/hosted-api/#where-do-traces-come-from)

---

## How It Works

ACE maintains a **Skillbook** — a persistent collection of strategies that evolves with every task. Three specialized roles manage the learning loop:

| Role | Responsibility |
|:-----|:---------------|
| **Agent** | Executes tasks, enhanced with Skillbook strategies |
| **Reflector** | Analyzes execution traces to extract what worked and what failed |
| **SkillManager** | Curates the Skillbook — adds, refines, and removes strategies |

The **Recursive Reflector** is the key innovation: instead of summarizing traces in a single pass, it writes and executes Python code in a sandboxed environment to programmatically search for patterns, isolate errors, and iterate until it finds actionable insights.

```mermaid
flowchart LR
    Skillbook[(Skillbook)]
    Start([Task]) --> Agent[Agent]
    Agent <--> Environment[Environment]
    Environment -- Trace --> Reflector[Reflector]
    Reflector --> SkillManager[SkillManager]
    SkillManager -- Updates --> Skillbook
    Skillbook -. Strategies .-> Agent
```

All roles are backed by [PydanticAI](https://ai.pydantic.dev/) agents with structured output validation. PydanticAI routes to 100+ LLM providers through its LiteLLM integration, with native support for OpenAI, Anthropic, Google, Bedrock, Groq, and more.

*Based on the [ACE paper](https://arxiv.org/abs/2510.04618) (Stanford & SambaNova) and [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952).*

---

## Runners

| Runner | Class | Description |
|:-------|:------|:------------|
| **LiteLLM** | `ACELiteLLM` | Batteries-included agent with `.ask()`, `.learn()`, `.save()` — accepts any [LiteLLM model string](https://docs.litellm.ai/docs/providers) |
| **Core** | `ACE` | Full learning loop with batch epochs and evaluation |
| **Trace Analyser** | `TraceAnalyser` | Learn from pre-recorded traces without re-running tasks |
| **browser-use** | `BrowserUse` | Browser automation that improves with each run |
| **LangChain** | `LangChain` | Wrap any LangChain chain or agent with learning |
| **Claude Code** | `ClaudeCode` | Claude Code CLI tasks with learning |

```bash
uv add 'ace-framework[browser-use]'    # Browser automation
uv add 'ace-framework[langchain]'      # LangChain
uv add 'ace-framework[logfire]'        # Observability (auto-instruments PydanticAI)
uv add 'ace-framework[mcp]'            # MCP server for IDE integration
uv add 'ace-framework[deduplication]'  # Embedding-based skill deduplication
```

Have existing agent logs? Extract strategies from them directly:

```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="gpt-4o-mini")
agent.learn_from_traces(your_existing_traces)
print(agent.get_strategies())
```

[-> Examples](examples/)

---

## Benchmarks

### Tau2 — Multi-Step Agentic Tasks

[tau2-bench](https://github.com/sierra-research/tau2-bench) by Sierra Research: airline domain tasks requiring tool use and policy adherence. Claude Haiku 4.5 agent, strategies learned on the train split with no reward signals, evaluated on the held-out test split.

<img src="benchmarks/tasks/tau_bench/Tau2Benchmark Result Haiku4.5.png" alt="Tau2 Benchmark — ACE doubles consistency at pass^4" width="35%"/>

*pass^k = probability all k independent attempts succeed. ACE doubles consistency at pass^4 with 15 learned strategies.*

### Claude Code — Autonomous Translation

ACE + Claude Code translated this library from Python to TypeScript with zero supervision:

| Metric | Result |
|:-------|:-------|
| Duration | ~4 hours |
| Commits | 119 |
| Lines written | ~14,000 |
| Build errors | 0 |
| Tests | All passing |
| Learning cost | ~$1.50 |

---

## Pipeline Architecture

ACE is built on a composable pipeline engine. Each step declares what it requires and what it produces:

```
AgentStep -> EvaluateStep -> ReflectStep -> UpdateStep -> ApplyStep -> DeduplicateStep
```

Use `learning_tail()` for the standard learning sequence, or compose custom pipelines:

```python
from ace import Pipeline, AgentStep, EvaluateStep, learning_tail

steps = [AgentStep(agent), EvaluateStep(env)] + learning_tail(reflector, skill_manager, skillbook)
pipeline = Pipeline(steps)
```

The pipeline engine ([`pipeline/`](pipeline/)) is framework-agnostic with `requires`/`provides` contracts, immutable context, and error isolation. See [Pipeline Design](docs/design/PIPELINE_DESIGN.md) and [Architecture](docs/design/ACE_ARCHITECTURE.md).

---

## CLI

| Command | Description |
|:--------|:------------|
| `ace setup` | Interactive setup — model selection, API keys, connection validation |
| `ace models <query>` | Search available models with pricing |
| `ace validate <model>` | Test a model connection |
| `ace config` | Show current configuration |
| `kayba` | Cloud CLI — upload traces, fetch insights, manage prompts |
| `ace-mcp` | MCP server for IDE integration |

---

## Documentation

- [Full Documentation](https://kayba-ai.github.io/agentic-context-engine/latest/) — Guides, API reference, examples
- [Quick Start](https://kayba-ai.github.io/agentic-context-engine/latest/getting-started/quick-start/) — 5-minute setup
- [Setup Guide](https://kayba-ai.github.io/agentic-context-engine/latest/getting-started/setup/) — Configuration and providers
- [Hosted API Guide](https://kayba-ai.github.io/agentic-context-engine/latest/integrations/hosted-api/) — Hosted CLI, trace upload, prompt install
- [Architecture](docs/design/ACE_ARCHITECTURE.md) — Core concepts and system design
- [Code Reference](docs/design/ACE_REFERENCE.md) — Implementations, API, usage examples
- [Design Decisions](docs/design/ACE_DECISIONS.md) — Rejected alternatives and rationale
- [Pipeline Engine](docs/design/PIPELINE_DESIGN.md) — Step composition and context flow
- [Examples](examples/) — Runnable demos
- [Changelog](CHANGELOG.md) — Version history

---

## Contributing

Contributions are welcome. See [Contributing Guidelines](CONTRIBUTING.md).

---

<div align="center">

**Built by [Kayba](https://kayba.ai) and the open-source community.**

</div>
