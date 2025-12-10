# Quick Start Guide

Get your first self-learning AI agent running in 5 minutes.

---

## Simple Quickstart

### Step 1: Install

```bash
pip install ace-framework
```

### Step 2: Set API Key

```bash
export OPENAI_API_KEY="your-key-here"
# Or: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

### Step 3: Create Your Agent

```python
from ace import ACELiteLLM

# Create agent that learns automatically
agent = ACELiteLLM(model="gpt-4o-mini")

# Ask questions - it learns from each interaction
answer1 = agent.ask("What is 2+2?")
print(f"Answer: {answer1}")

answer2 = agent.ask("What is the capital of France?")
print(f"Answer: {answer2}")

# Agent now has learned strategies!
print(f"Learned {len(agent.skillbook.skills())} strategies")

# Save for later
agent.save_skillbook("my_agent.json")
```

### Step 4: Run It

```bash
python my_first_ace.py
```

**That's it!** Your agent:
- **Learned automatically** from each interaction
- **Built a skillbook** of successful strategies
- **Saved knowledge** for reuse

---

## Load a Saved Agent

```python
from ace import ACELiteLLM

# Load previously trained agent
agent = ACELiteLLM.from_skillbook("my_agent.json", model="gpt-4o-mini")

# Use it immediately with prior knowledge
answer = agent.ask("New question")
```

---

## Use Different Models

ACE works with 100+ LLM providers via LiteLLM:

```python
# OpenAI
agent = ACELiteLLM(model="gpt-4o-mini")

# Anthropic Claude
agent = ACELiteLLM(model="claude-3-5-sonnet-20241022")

# Google Gemini
agent = ACELiteLLM(model="gemini-pro")

# Local Ollama
agent = ACELiteLLM(model="ollama/llama2")
```

---

## Understanding the Full ACE Pipeline

For more control, use the full ACE components directly:

```python
from ace import OfflineACE, Agent, Reflector, SkillManager
from ace import LiteLLMClient, Skillbook, Sample, TaskEnvironment, EnvironmentResult


# Custom environment that checks answers
class SimpleEnvironment(TaskEnvironment):
    def evaluate(self, sample, agent_output):
        correct = str(sample.ground_truth).lower() in str(agent_output.final_answer).lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth
        )


# Initialize LLM client
client = LiteLLMClient(model="gpt-4o-mini")

# Create ACE components (three roles)
agent = Agent(client)               # Produces answers using skillbook
reflector = Reflector(client)       # Analyzes what worked/failed
skill_manager = SkillManager(client)  # Updates skillbook with learnings

# Create the ACE orchestrator
adapter = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)

# Training samples
samples = [
    Sample(question="What is the capital of France?", ground_truth="Paris"),
    Sample(question="What is 2 + 2?", ground_truth="4"),
    Sample(question="Who wrote Romeo and Juliet?", ground_truth="Shakespeare")
]

# Train the agent
print("Training agent...")
results = adapter.run(samples, SimpleEnvironment(), epochs=2)

# Save learned strategies
adapter.skillbook.save_to_file("my_agent.json")
print(f"Learned {len(adapter.skillbook.skills())} strategies")

# Test with new question
test_output = agent.generate(
    question="What is 5 + 3?",
    context="",
    skillbook=adapter.skillbook
)
print(f"Test: What is 5 + 3? -> {test_output.final_answer}")
```

### The Three ACE Roles

| Role | Purpose |
|------|---------|
| **Agent** | Executes tasks using skillbook strategies |
| **Reflector** | Analyzes outcomes to identify what worked/failed |
| **SkillManager** | Updates skillbook with new strategies |

### Two Adaptation Modes

| Mode | Use Case |
|------|----------|
| **OfflineACE** | Train on batch of samples (multiple epochs) |
| **OnlineACE** | Learn from each task in real-time |

---

## Integrate with Existing Agents

### Browser Automation (browser-use)

```bash
pip install ace-framework[browser-use]
```

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

agent = ACEAgent(llm=ChatBrowserUse())
await agent.run(task="Find the top post on Hacker News")
agent.save_skillbook("browser_expert.json")
```

### LangChain

```bash
pip install ace-framework[langchain]
```

```python
from ace import ACELangChain

ace_chain = ACELangChain(runnable=your_langchain_chain)
result = ace_chain.invoke({"question": "Your task"})
ace_chain.save_skillbook("chain_learned.json")
```

See [Integration Guide](INTEGRATION_GUIDE.md) for complete patterns.

---

## Common Patterns

### Online Learning (Learn While Running)

```python
from ace import OnlineACE

adapter = OnlineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)

# Process tasks one by one, learning from each
for sample in samples:
    result = adapter.run([sample], environment)
    # Skillbook updates after each task
```

### Async Learning (Background Processing)

```python
adapter = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,
    max_reflector_workers=3
)

# Returns fast - learning happens in background
results = adapter.run(samples, environment)

# Check progress
print(adapter.learning_stats)

# Wait when needed (e.g., before saving)
adapter.wait_for_learning()
adapter.skillbook.save_to_file("learned.json")
```

### Custom Evaluation Environment

```python
class MathEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        try:
            result = eval(output.final_answer)
            correct = result == sample.ground_truth
            return EnvironmentResult(
                feedback=f"Result: {result}. {'Correct!' if correct else 'Wrong'}",
                ground_truth=sample.ground_truth
            )
        except:
            return EnvironmentResult(
                feedback="Invalid math expression",
                ground_truth=sample.ground_truth
            )
```

### Checkpoint Saving

```python
results = adapter.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=10,  # Save every 10 samples
    checkpoint_dir="./checkpoints"
)
# Creates: checkpoint_10.json, checkpoint_20.json, latest.json
```

---

## Recommended Prompts (v2.1)

For best performance, use v2.1 prompts (+17% success rate):

```python
from ace.prompts_v2_1 import PromptManager

mgr = PromptManager()
agent = Agent(client, prompt_template=mgr.get_agent_prompt())
reflector = Reflector(client, prompt_template=mgr.get_reflector_prompt())
skill_manager = SkillManager(client, prompt_template=mgr.get_skill_manager_prompt())
```

---

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade ace-framework
```

**API key not working?**
- Verify key is correct: `echo $OPENAI_API_KEY`
- Try different model: `ACELiteLLM(model="gpt-3.5-turbo")`

**JSON parsing errors?**
- Increase max_tokens: `LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)`
- Use v2.1 prompts (more robust)

**Need help?**
- [GitHub Issues](https://github.com/kayba-ai/agentic-context-engine/issues)
- [Discord Community](https://discord.gg/mqCqH7sTyK)

---

## Next Steps

| Resource | Description |
|----------|-------------|
| **[Integration Guide](INTEGRATION_GUIDE.md)** | Add ACE to existing agents |
| **[Complete Guide](COMPLETE_GUIDE_TO_ACE.md)** | Deep dive into ACE concepts |
| **[API Reference](API_REFERENCE.md)** | Full API documentation |
| **[Examples](../examples/)** | Ready-to-run code examples |

---

**Ready to build production agents?** Check out the [Integration Guide](INTEGRATION_GUIDE.md) for browser automation, LangChain, and custom agent patterns.
