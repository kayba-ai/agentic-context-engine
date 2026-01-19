# ACE Framework Quick Start

Get your first self-learning AI agent running!

---

## Installation

```bash
pip install ace-framework
```

Set your API key:

```bash
export OPENAI_API_KEY="your-key-here"
# Or: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

---

## Integration Examples

### ACELiteLLM - Simple Self-Improving Agent

```python
from ace import ACELiteLLM

# Create self-improving agent
agent = ACELiteLLM(model="gpt-4o-mini")

# Ask related questions - agent learns patterns
answer1 = agent.ask("If all cats are animals, is Felix (a cat) an animal?")
answer2 = agent.ask("If all birds fly, can penguins (birds) fly?")  # Learns to check assumptions!
answer3 = agent.ask("If all metals conduct electricity, does copper conduct electricity?")

# View learned strategies
print(f"Learned {len(agent.skillbook.skills())} reasoning skills")

# Save for reuse
agent.save_skillbook("my_agent.json")

# Load and continue
agent2 = ACELiteLLM(model="gpt-4o-mini", skillbook_path="my_agent.json")
```

### ACELangChain - Wrap LangChain Chains/Agents

Best for multi-step workflows and tool-using agents.

```python
from ace import ACELangChain

ace_chain = ACELangChain(runnable=your_langchain_chain)
result = ace_chain.invoke({"question": "Your task"})  # Learns automatically
```

### ACEAgent - Browser Automation (browser-use)

Drop-in replacement for `browser_use.Agent` with automatic learning.

```bash
pip install ace-framework[browser-use]
```

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

# Two LLMs: ChatBrowserUse for browser, gpt-4o-mini for ACE learning
agent = ACEAgent(
    llm=ChatBrowserUse(),      # Browser execution
    ace_model="gpt-4o-mini"    # ACE learning
)

await agent.run(task="Find top Hacker News post")
agent.save_skillbook("hn_expert.json")

# Reuse learned knowledge
agent = ACEAgent(llm=ChatBrowserUse(), skillbook_path="hn_expert.json")
await agent.run(task="New task")  # Starts smart!
```

[→ Browser Use Guide](../examples/browser-use/README.md)

### ACEClaudeCode - Claude Code CLI

Self-improving coding agent using Claude Code.

```python
from ace import ACEClaudeCode

agent = ACEClaudeCode(
    working_dir="./my_project",
    ace_model="gpt-4o-mini"
)

# Execute coding tasks - agent learns from each
result = agent.run(task="Add unit tests for utils.py")
agent.save_skillbook("coding_expert.json")

# Reuse learned knowledge
agent = ACEClaudeCode(working_dir="./project", skillbook_path="coding_expert.json")
```

[→ Claude Code Loop Example](../examples/claude-code-loop/)

---

## Advanced Tutorial: Understanding ACE Internals

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
agent = ACELiteLLM(model="gpt-4o-mini", skillbook_path="my_agent.json")

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


---

## Common Patterns

### Online Learning (Learn While Running)

```python
from ace import OnlineACE

adapter = OnlineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)

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
