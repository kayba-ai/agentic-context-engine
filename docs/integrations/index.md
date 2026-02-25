# Integrations Overview

ACE provides runners for popular agentic frameworks. Each runner adds self-improving learning to an existing agent with minimal code changes.

## Available Integrations

| Runner | Framework | Input | Insight Level |
|--------|-----------|-------|--------------|
| [`ACELiteLLM`](litellm.md) | LiteLLM (100+ providers) | Questions | Micro |
| [`LangChain`](langchain.md) | LangChain Runnables | Chain inputs | Meso |
| [`BrowserUse`](browser-use.md) | browser-use | Task strings | Meso |
| [`ClaudeCode`](claude-code.md) | Claude Code CLI | Task strings | Meso |
| [Opik](opik.md) | Opik observability | — | Monitoring |

## The Pattern

All integration runners follow the same three-step pattern:

```
1. INJECT   — Add skillbook strategies to the agent's context
2. EXECUTE  — Run the external agent normally
3. LEARN    — Reflector + SkillManager update the skillbook
```

## Quick Construction

Every runner offers a `from_model()` factory that builds ACE roles automatically:

```python
from ace_next import BrowserUse, LangChain, ClaudeCode

# Browser automation
browser = BrowserUse.from_model(browser_llm=my_llm, ace_model="gpt-4o-mini")

# LangChain chain/agent
chain = LangChain.from_model(my_runnable, ace_model="gpt-4o-mini")

# Claude Code CLI
coder = ClaudeCode.from_model(working_dir="./project", ace_model="gpt-4o-mini")
```

## Shared Features

All runners share these capabilities:

- **Skillbook persistence** — `save()` / load via `skillbook_path`
- **Checkpointing** — automatic saves during long runs
- **Deduplication** — prevent duplicate skills
- **Background learning** — `wait=False` for async learning
- **Progress tracking** — `learning_stats` property

## Which Integration Should I Use?

- **Building a Q&A or reasoning agent?** Use [ACELiteLLM](litellm.md)
- **Have an existing LangChain chain or agent?** Use [LangChain](langchain.md)
- **Automating browser tasks?** Use [BrowserUse](browser-use.md)
- **Running coding tasks with Claude Code?** Use [ClaudeCode](claude-code.md)
- **Want to monitor costs and traces?** Add [Opik](opik.md)
- **Using a different framework?** See the [Integration Guide](../guides/integration.md) to build a custom runner
