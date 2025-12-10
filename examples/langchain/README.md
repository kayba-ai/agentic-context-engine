# LangChain + ACE Integration Examples

This directory contains examples demonstrating how to use ACELangChain to add learning capabilities to LangChain chains and agents.

## Overview

ACELangChain wraps any LangChain Runnable (chains, agents, custom components) and adds ACE learning capabilities. The runnable executes normally, but ACE learns from results to improve future executions.

## Examples

### 1. simple_chain_example.py

Demonstrates basic LangChain chain integration with ACE learning.

**Features:**
- Basic LLM chain with learning
- Reusing learned strategies across sessions
- Learning control (enable/disable)
- Async chain execution
- Custom output parsers

**Requirements:**
```bash
pip install ace-framework[langchain]
# or: pip install langchain-core langchain-openai
```

**Usage:**
```bash
export OPENAI_API_KEY="your-api-key"
python simple_chain_example.py
```

**What you'll see:**
- The chain answers questions and learns from each execution
- Strategies are saved to `simple_chain_learned.json`
- Next run loads previous knowledge and improves further

### 2. agent_with_tools_example.py

Shows how to use ACE with LangChain AgentExecutor for **meso-level learning**.

**Features:**
- Real AgentExecutor with tool-calling agent
- **Meso-level learning** - captures full reasoning traces
- Comparison of micro vs meso learning
- Async AgentExecutor execution

**Requirements:**
```bash
pip install ace-framework[langchain]
# or: pip install langchain langchain-openai
```

**Usage:**
```bash
export OPENAI_API_KEY="your-api-key"
python agent_with_tools_example.py
```

**What you'll see:**
- AgentExecutor reasoning with tool calls (verbose output)
- ACE learning from full execution trace (thoughts, actions, observations)
- Comparison showing richer learning with meso vs micro
- Strategies saved to `agent_executor_learned.json`

## Key Concepts

### Insight Levels (Micro vs Meso)

ACE learns at different levels depending on the setup:

| Level | Context | What ACE Sees | Use Case |
|-------|---------|---------------|----------|
| **Micro** | With TaskEnvironment | Request → response → ground truth/feedback | OfflineACE, OnlineACE |
| **Meso** | AgentExecutor | Thoughts → tool calls → observations → answer | ACELangChain with agents |

**Micro-level** (full ACE loop): Uses TaskEnvironment with ground truth to evaluate correctness. The Reflector learns from whether the answer was right or wrong.

**Meso-level** (AgentExecutor): ACE captures the full reasoning trace (thoughts, tool calls, observations) without external ground truth. Learns from execution patterns.

```python
# Micro: Full ACE loop with environment
adapter = OfflineACE(skillbook, agent, reflector, skill_manager)
adapter.run(samples, environment)  # Micro learning with ground truth

# Meso: AgentExecutor reasoning trace (automatic!)
agent_executor = AgentExecutor(agent=agent, tools=tools)
ace_agent = ACELangChain(runnable=agent_executor)  # Meso learning
result = ace_agent.invoke({"input": "question"})
```

ACE **automatically detects** AgentExecutor and enables meso-level trace extraction.

### The ACE Learning Pattern

All examples follow the same three-step pattern:

1. **INJECT**: ACE adds learned strategies to the input
2. **EXECUTE**: Your LangChain chain/agent runs normally
3. **LEARN**: ACE analyzes results and updates strategies

### Skillbook Persistence

Learned strategies are saved as JSON files:

```python
# Save learned knowledge
ace_chain.save_skillbook("my_expert.json")

# Load in next session
ace_chain = ACELangChain(
    runnable=chain,
    skillbook_path="my_expert.json"
)
```

### Learning Control

You can control when learning occurs:

```python
# Disable learning for warmup
ace_chain.disable_learning()
result = ace_chain.invoke("test")

# Enable learning for production
ace_chain.enable_learning()
result = ace_chain.invoke("real task")
```

## Common Patterns

### Basic Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ace.integrations import ACELangChain

# Create chain
prompt = ChatPromptTemplate.from_template("Answer: {input}")
chain = prompt | ChatOpenAI(temperature=0)

# Wrap with ACE
ace_chain = ACELangChain(runnable=chain)

# Use it
result = ace_chain.invoke({"input": "What is ACE?"})
```

### Custom Output Parser

```python
def parse_json_output(result):
    """Extract specific field from complex output."""
    return result["data"]["answer"]

ace_chain = ACELangChain(
    runnable=chain,
    output_parser=parse_json_output
)
```

### Async Execution

```python
async def main():
    result = await ace_chain.ainvoke({"input": "Question"})
    print(result.content)

import asyncio
asyncio.run(main())
```

## Troubleshooting

### "LangChain is not installed"

Install LangChain dependencies:

```bash
pip install ace-framework[langchain]
# or: pip install langchain-core langchain-openai
```

### "Please set OPENAI_API_KEY"

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Or use a `.env` file:

```
OPENAI_API_KEY=sk-...
```

### Examples don't learn anything

Check that:
1. Learning is enabled: `ace_chain.is_learning == True`
2. Skillbook is being saved: `ace_chain.save_skillbook("file.json")`
3. Check file contents: `cat file.json`

### High API costs

- Use cheaper model: `ace_model="gpt-4o-mini"`
- Disable learning for testing: `is_learning=False`
- Learn only every N tasks (batch learning)

## Next Steps

1. Run the examples to see ACE learning in action
2. Adapt the patterns to your own LangChain chains/agents
3. See [INTEGRATION_GUIDE.md](../../docs/INTEGRATION_GUIDE.md) for advanced patterns
4. Check [API_REFERENCE.md](../../docs/API_REFERENCE.md) for complete API docs

## Resources

- **Integration Guide**: `docs/INTEGRATION_GUIDE.md` (includes all integration patterns)
- **API Reference**: `docs/API_REFERENCE.md`
- **Discord Community**: Join our [Discord](https://discord.gg/mqCqH7sTyK)
