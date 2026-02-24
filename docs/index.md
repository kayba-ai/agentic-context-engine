# ACE Framework

**Agentic Context Engineering** — a framework for self-improving language model agents.

ACE enables AI agents to learn from their own execution feedback through three collaborative roles: **Agent**, **Reflector**, and **SkillManager**.

## How It Works

```
Sample → Agent → Environment → Reflector → SkillManager → Skillbook
                    (feedback)   (analyzes)   (updates)     (context)
```

The **Skillbook** accumulates strategies across runs, making every subsequent agent call smarter.

## Quick Links

- [Quick Start](getting-started/quick-start.md) — up and running in minutes
- [Complete ACE Guide](guides/complete-guide.md) — deep dive into the framework
- [API Reference](api/index.md) — full class and method documentation

## Install

```bash
pip install ace-framework
```

## Paper

This framework implements the method from:

> *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*
> arXiv:2510.04618
