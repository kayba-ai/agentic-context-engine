![Kayba Agent Prompt Optimizer](agent_prompt_optimizer.png)

# Agent Prompt Optimizer - Turn agents' failures into better prompts 🔄

![GitHub stars](https://img.shields.io/github/stars/kayba-ai/agentic-context-engine?style=social)
[![Discord](https://img.shields.io/discord/1429935408145236131?label=Discord&logo=discord&logoColor=white&color=5865F2)](https://discord.gg/mqCqH7sTyK)
[![Twitter Follow](https://img.shields.io/twitter/follow/kaybaai?style=social)](https://twitter.com/kaybaai)
[![PyPI version](https://badge.fury.io/py/ace-framework.svg)](https://badge.fury.io/py/ace-framework)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## The Problem with Traditional Prompt Engineering
- Time-consuming manual iteration cycles
- Prompt drift and regression issues
- Lack of systematic learning from failures
- One-size-fits-all prompts that don't adapt to specific use cases

## How ACE Replaces Manual Prompt Engineering
- **Automatic Skill Generation**: ACE observes agent execution and generates context-specific skills
- **Continuous Improvement**: Every interaction makes your agent smarter without manual intervention
- **Learn from Production**: Real-world usage patterns directly improve agent behavior
- **No More Prompt Regression**: Skills are versioned and scored based on actual performance
- **No labels, no fine-tuning: just prompt updates learned from real-time experience.

### Real-World Impact
- **💰 49% Token Reduction**: Demonstrated in browser automation tasks
- **📈 20-35% Performance Gains**: Across complex reasoning tasks
- **🚀 Zero Manual Effort**: Set it and forget it - agents improve autonomously**

![Kayba Prompt Optimizer Improvements](prompt-optimizer-improvements.png)


### How It Works
1. **Execute**: Your agent performs tasks as usual
2. **Reflect**: ACE analyzes what worked and what didn't
3. **Learn**: Automatically generates and refines skills
4. **Persist**: Save optimized skillbooks for consistent performance
5. **Update prompt**: Skills are automatically injected in future runs

## ⚡ Quick Start

#### Enhance your existing agents
```python
from ace import ACELangChain

# Wrap your existing LangChain agent
ace_chain = ACELangChain(runnable=your_existing_chain)
result = ace_chain.invoke({"input": "Your task"})  # Learns from every execution
```

#### Create a self-improving agent
```python
from ace import ACELiteLLM

# Before: Static prompt engineering
agent = LiteLLMClient(model="gpt-4", system_prompt="You are a helpful assistant...")

# After: Dynamic skill learning
agent = ACELiteLLM(model="gpt-4")  # Starts simple, learns automatically
answer = agent.ask("Complex reasoning task")
print(f"Learned {len(agent.skillbook.skills())} skills automatically")
```

#### Let browser automation agents self-optimize
```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

# Enhanced browser automation that learns optimal strategies
agent = ACEAgent(llm=ChatBrowserUse(), ace_model="gpt-4")
await agent.run(task="Navigate complex website")  # Improves with each interaction
```

### Use Cases for Prompt Optimization
- Customer support agents learning from successful resolutions
- Code generation agents adapting to your codebase patterns
- Data extraction agents learning document structures
- Browser automation agents optimizing navigation strategies

### Metrics & Monitoring
- Track skill evolution over time
- Measure performance improvements
- Monitor token usage reduction
- Analyze which skills contribute most to success

## ❓ FAQ

**Can I combine manual prompts with ACE skills?**
Yes! ACE skills complement your base prompts. Start with manual prompts and let ACE build domain-specific expertise autonomously.

**How quickly do agents improve?**
Performance gains appear as soon as within 10 interactions.

**What if a skill performs poorly?**
ACE automatically scores skills based on outcomes. Poor-performing skills get lower scores and are used less frequently.

**Can I share learned skills between agents?**
Absolutely! Skillbooks are portable JSON files that can be shared across agents handling similar tasks.

## Next Steps
- Try the [examples](../) in this folder
- Read more about [ACE on GitHub](https://github.com/kayba-ai/agentic-context-engine)
- Join our [Discord](https://discord.gg/mqCqH7sTyK) for optimization tips
