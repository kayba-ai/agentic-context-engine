# 🚀 ACE Framework Quick Start

Get your first self-learning AI agent running!

---

## 🚀 Simple Quickstart (5 minutes)

The fastest way to get started with ACE.

### Step 1: Install

```bash
pip install ace-framework
```

### Step 2: Set API Key

```bash
export OPENAI_API_KEY="your-key-here"
# Or: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

### Step 3: Create `my_first_ace.py`

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
print(f"✅ Learned {len(agent.skillbook.skills())} strategies")

# Save for later
agent.save_skillbook("my_agent.json")
```

### Step 4: Run It

```bash
python my_first_ace.py
```

### What Just Happened?

Your agent:
- **Learned automatically** from each interaction
- **Built a skillbook** of successful strategies
- **Saved knowledge** for reuse

That's it! You now have a self-improving AI agent.

---

## 🎓 Advanced Tutorial: Understanding ACE Internals (15 minutes)

Want to understand how ACE works under the hood? This section shows the full architecture with Agent, Reflector, and SkillManager roles.

### Full Pipeline Example

```python
from ace import OfflineACE, Agent, Reflector, SkillManager
from ace import LiteLLMClient, Sample, TaskEnvironment, EnvironmentResult


# Simple environment that checks if answer contains the ground truth
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
agent = Agent(client)              # Produces answers
reflector = Reflector(client)      # Analyzes performance
skill_manager = SkillManager(client)  # Updates skillbook

# Create adapter to orchestrate everything
adapter = OfflineACE(agent=agent, reflector=reflector, skill_manager=skill_manager)

# Create training samples
samples = [
    Sample(question="What is the capital of France?", context="", ground_truth="Paris"),
    Sample(question="What is 2 + 2?", context="", ground_truth="4"),
    Sample(question="Who wrote Romeo and Juliet?", context="", ground_truth="Shakespeare")
]

# Train the agent
print("Training agent...")
results = adapter.run(samples, SimpleEnvironment(), epochs=2)

# Save learned strategies
adapter.skillbook.save_to_file("my_agent.json")
print(f"✅ Agent trained! Learned {len(adapter.skillbook.skills())} strategies")

# Test with new question
test_output = agent.generate(
    question="What is 5 + 3?",
    context="",
    skillbook=adapter.skillbook
)
print(f"\nTest question: What is 5 + 3?")
print(f"Answer: {test_output.final_answer}")
```

Expected output:
```
Training agent...
✅ Agent trained! Learned 3 strategies

Test question: What is 5 + 3?
Answer: 8
```

### Understanding the Architecture

**Three ACE Roles:**
1. **Agent** - Executes tasks using skillbook strategies
2. **Reflector** - Analyzes what worked/didn't work
3. **SkillManager** - Updates skillbook with new strategies

**Two Adaptation Modes:**
- **OfflineACE** - Train on batch of samples (shown above)
- **OnlineACE** - Learn from each task in real-time

---

## Next Steps

### Load Saved Agent

```python
from ace import ACELiteLLM

# Load previously trained agent
agent = ACELiteLLM.from_skillbook("my_agent.json", model="gpt-4o-mini")

# Use it immediately
answer = agent.ask("New question")
```

Or with full pipeline:

```python
from ace import Skillbook, Agent, LiteLLMClient

# Load skillbook
skillbook = Skillbook.load_from_file("my_agent.json")

# Use with agent
client = LiteLLMClient(model="gpt-4o-mini")
agent = Agent(client)
output = agent.generate(
    question="New question",
    context="",
    skillbook=skillbook
)
```

### Try Different Models

```python
# Anthropic Claude
agent = ACELiteLLM(model="claude-3-5-sonnet-20241022")

# Google Gemini
agent = ACELiteLLM(model="gemini-pro")

# Local Ollama
agent = ACELiteLLM(model="ollama/llama2")
```

### Add ACE to Existing Agents

Already have an agent? Wrap it with ACE learning:

**Browser Automation:**
```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

agent = ACEAgent(llm=ChatBrowserUse())
await agent.run(task="Your task")  # Learns automatically
```

**LangChain:**
```python
from ace import ACELangChain

ace_chain = ACELangChain(runnable=your_langchain_chain)
result = ace_chain.invoke({"question": "Your task"})
```

See [Integration Guide](INTEGRATION_GUIDE.md) for details.

---

## Common Patterns

### Online Learning (Learn While Running)

```python
from ace import OnlineACE

adapter = OnlineACE(skillbook, agent, reflector, skill_manager)

# Process tasks one by one, learning from each
for task in tasks:
    result = adapter.process(task, environment)
```

### Custom Evaluation

```python
class MathEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        try:
            result = eval(output.final_answer)
            correct = result == sample.ground_truth
            return EnvironmentResult(
                feedback=f"Result: {result}. {'✓' if correct else '✗'}",
                ground_truth=sample.ground_truth
            )
        except:
            return EnvironmentResult(
                feedback="Invalid math expression",
                ground_truth=sample.ground_truth
            )
```

---

## Learn More

- **[Integration Guide](INTEGRATION_GUIDE.md)** - Add ACE to existing agents
- **[Complete Guide](COMPLETE_GUIDE_TO_ACE.md)** - Deep dive into ACE concepts
- **[Examples](../examples/)** - Real-world examples
  - [Browser Automation](../examples/browser-use/) - Self-improving browser agents
  - [LangChain Integration](../examples/langchain/) - Wrap chains with learning
  - [Custom Integration](../examples/custom_integration_example.py) - Any agent pattern

---

## Troubleshooting

**Import errors?**
```bash
pip install --upgrade ace-framework
```

**API key not working?**
- Verify key is correct: `echo $OPENAI_API_KEY`
- Try different model: `ACELiteLLM(model="gpt-3.5-turbo")`

**Need help?**
- [GitHub Issues](https://github.com/kayba-ai/agentic-context-engine/issues)
- [Discord Community](https://discord.com/invite/mqCqH7sTyK)

---

**Ready to build production agents?** Check out the [Integration Guide](INTEGRATION_GUIDE.md) for browser automation, LangChain, and custom agent patterns.
