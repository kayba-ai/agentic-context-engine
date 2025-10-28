# 🚀 ACE Framework Quick Start Guide

Get your first self-improving AI agent running in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- An API key for your preferred LLM provider (OpenAI, Anthropic, etc.)

## Installation

```bash
pip install ace-framework
```

## Your First ACE Agent

### Step 1: Set up your environment

Create a `.env` file with your API key:

```bash
# For OpenAI
OPENAI_API_KEY=your-key-here

# For Anthropic
ANTHROPIC_API_KEY=your-key-here

# For Google
GOOGLE_API_KEY=your-key-here
```

### Step 2: Create a simple agent

Create a file `my_first_ace.py`:

```python
from ace import OfflineAdapter, Generator, Reflector, Curator, Playbook
from ace import LiteLLMClient, Sample, TaskEnvironment, EnvironmentResult


# Create a simple environment for evaluating answers
class SimpleEnvironment(TaskEnvironment):
    """Basic environment for testing - checks if ground truth appears in answer."""

    def evaluate(self, sample, generator_output):
        # Simple substring matching (case-insensitive)
        correct = str(sample.ground_truth).lower() in str(generator_output.final_answer).lower()

        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )


# Initialize the LLM client
client = LiteLLMClient(model="gpt-4o-mini")  # or claude-3-haiku, gemini-pro, etc.

# Create the three ACE components
generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)

# Create an adapter to orchestrate everything
adapter = OfflineAdapter(
    generator=generator,
    reflector=reflector,
    curator=curator
)

# Set up the environment (evaluates answers)
environment = SimpleEnvironment()

# Create training samples
samples = [
    Sample(
        question="What is the capital of France?",
        context="Provide a direct answer",
        ground_truth="Paris"
    ),
    Sample(
        question="What is 2 + 2?",
        context="Show the calculation",
        ground_truth="4"
    ),
    Sample(
        question="Who wrote Romeo and Juliet?",
        context="Name the author",
        ground_truth="William Shakespeare"
    )
]

# Train the agent (it learns strategies from these examples)
print("Training agent...")
results = adapter.run(samples, environment, epochs=2)

# Save the learned strategies
adapter.playbook.save_to_file("my_trained_agent.json")
print(f"Agent trained! Learned {len(adapter.playbook.bullets())} strategies")

# Test with a new question
test_sample = Sample(
    question="What is 5 + 3?",
    context="Provide the answer"
)

print("\nTesting with new question:", test_sample.question)
output = generator.generate(
    question=test_sample.question,
    context=test_sample.context,
    playbook=adapter.playbook
)
print("Answer:", output.final_answer)
print("Reasoning:", output.reasoning)
```

### Step 3: Run your agent

```bash
python my_first_ace.py
```

Expected output:
```
Training agent...
Agent trained! Learned 3 strategies

Testing with new question: What is 5 + 3?
Answer: 8
Reasoning: Using direct calculation strategy learned from training...
```

## What Just Happened?

Your agent:
1. **Learned** from the training examples
2. **Reflected** on what strategies work
3. **Built a playbook** of successful approaches
4. **Applied** those strategies to solve a new problem

## Next Steps

### 1. Load and Continue Training

```python
from ace import Playbook

# Load a previously trained agent
playbook = Playbook.load_from_file("my_trained_agent.json")

# Continue training with new examples
adapter = OfflineAdapter(generator, reflector, curator, playbook=playbook)
```

### 2. Try Different Models

```python
# OpenAI GPT-4
client = LiteLLMClient(model="gpt-4")

# Anthropic Claude
client = LiteLLMClient(model="claude-3-5-sonnet-20241022")

# Google Gemini
client = LiteLLMClient(model="gemini-pro")

# Local Ollama
client = LiteLLMClient(model="ollama/llama2")
```

### 3. Online Learning (Learn While Running)

```python
from ace import OnlineAdapter

adapter = OnlineAdapter(
    playbook=playbook,
    generator=generator,
    reflector=reflector,
    curator=curator
)

# Process tasks one by one, learning from each
for task in tasks:
    result = adapter.process(task, environment)
    print(f"Processed: {task.question}")
    print(f"Playbook now has {len(adapter.playbook.bullets())} strategies")
```

### 4. Custom Environments

```python
from ace import TaskEnvironment, EnvironmentResult

class MathEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        try:
            # Evaluate mathematical correctness
            result = eval(output.final_answer)
            correct = (result == eval(sample.ground_truth))

            return EnvironmentResult(
                feedback="Correct!" if correct else "Incorrect",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 1.0 if correct else 0.0}
            )
        except:
            return EnvironmentResult(
                feedback="Invalid mathematical expression",
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0}
            )
```

## Common Patterns

### Math Problem Solver

```python
math_samples = [
    Sample("What is 10 * 5?", "Calculate", "50"),
    Sample("What is 100 / 4?", "Calculate", "25"),
    Sample("What is 7 + 8?", "Calculate", "15"),
]
```

### Code Generator

```python
code_samples = [
    Sample(
        "Write a Python function to add two numbers",
        "Include type hints",
        "def add(a: int, b: int) -> int:\n    return a + b"
    ),
]
```

### Q&A System

```python
qa_samples = [
    Sample(
        "What is machine learning?",
        "Explain in simple terms",
        "Machine learning is a type of AI that allows computers to learn from data without being explicitly programmed."
    ),
]
```

## Troubleshooting

### API Key Issues

If you get authentication errors:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

### Rate Limiting

Add delays between calls:
```python
import time

for sample in samples:
    result = adapter.process(sample, environment)
    time.sleep(1)  # Wait 1 second between calls
```

### Memory Issues

Use online learning for large datasets:
```python
# Instead of loading all samples at once
adapter = OnlineAdapter(...)
for sample in large_dataset:
    adapter.process(sample, environment)
```

## Learn More

- [API Reference](API_REFERENCE.md) - Complete documentation
- [Examples](../examples/) - More complex examples
- [Setup Guide](SETUP_GUIDE.md) - Detailed configuration
- [Testing Guide](TESTING_GUIDE.md) - Best practices

## Get Help

- [GitHub Issues](https://github.com/Kayba-ai/agentic-context-engine/issues)
- [Discord Community](#) - Coming soon!

---

**Ready for more?** Check out the [full documentation](API_REFERENCE.md) or explore [advanced examples](../examples/).
