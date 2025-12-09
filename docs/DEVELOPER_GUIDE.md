# Developer Guide: Implementation Patterns and Best Practices

**Version**: 0.5.1
**Last Updated**: January 2025
**Audience**: Developers implementing ACE in their projects
**Reading Time**: 50-60 minutes

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Implementation Patterns](#implementation-patterns)
4. [Common Use Cases](#common-use-cases)
5. [Best Practices](#best-practices)
6. [Advanced Techniques](#advanced-techniques)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Production Deployment](#production-deployment)
10. [Migration Guides](#migration-guides)

---

## Introduction

This guide provides practical implementation patterns for developers building with the ACE framework. Whether you're adding learning to an existing agent or building a new one from scratch, you'll find concrete examples and best practices here.

### Quick Decision Matrix

Choose your implementation pattern based on your use case:

| Scenario | Recommended Pattern | Why |
|----------|-------------------|-----|
| Building new Q&A agent | ACELiteLLM | All-in-one, simple API |
| Adding learning to browser-use | ACEAgent | Pre-built integration |
| Adding learning to LangChain | ACELangChain | Pre-built integration |
| Custom agent framework | Full ACE Pipeline | Maximum control |
| Production deployment | Hybrid Architecture | Balance learning & performance |
| Research/experimentation | OfflineAdapter | Reproducible experiments |

---

## Getting Started

### Installation

```bash
# Basic installation
pip install ace-framework

# With observability (recommended)
pip install ace-framework[observability]

# All features
pip install ace-framework[all]

# Development (contributors)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync
```

### Minimal Working Example (52 lines)

```python
from ace import (
    Playbook, Generator, Reflector, Curator,
    OfflineAdapter, Sample, SimpleEnvironment
)
from ace.llm_providers import LiteLLMClient

# 1. Setup LLM
llm = LiteLLMClient(model="gpt-4", api_key="your-key")

# 2. Create components
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# 3. Prepare training data
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="What is 5+3?", ground_truth="8"),
    Sample(question="What is 10-7?", ground_truth="3"),
]

# 4. Train
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(samples, SimpleEnvironment(), epochs=2)

# 5. Save learned strategies
playbook.save_to_file("math_playbook.json")

# 6. Use in production
loaded_playbook = Playbook.load_from_file("math_playbook.json")
output = generator.generate("What is 7+6?", "", loaded_playbook)
print(output.final_answer)  # "13"
```

**Run it**:
```bash
export OPENAI_API_KEY="your-key"
python minimal_example.py
```

---

## Implementation Patterns

### Pattern 1: Quick Start with ACELiteLLM

**When to use**: Prototyping, simple Q&A, classification

**Implementation**:

```python
from ace.integrations import ACELiteLLM

# Initialize
agent = ACELiteLLM(
    model="gpt-4",
    api_key="your-key",
    playbook_path="./strategies.json"  # Auto-save
)

# Ask questions - it learns automatically
answer1 = agent.ask(
    question="What is the capital of France?",
    ground_truth="Paris"
)
print(answer1)  # "Paris"

# Second question uses learned strategies
answer2 = agent.ask(
    question="What is the capital of Germany?",
    ground_truth="Berlin"
)
print(answer2)  # "Berlin"

# Playbook automatically saved to strategies.json
print(f"Learned {len(agent.playbook.bullets)} strategies")
```

**Batch training**:

```python
from ace import Sample, SimpleEnvironment

# Prepare training data
samples = [
    Sample(q, ground_truth=gt)
    for q, gt in [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 5+5?", "10"),
    ]
]

# Train
results = agent.learn(samples, SimpleEnvironment(), epochs=3)

# Analyze results
accuracy = sum(1 for r in results if "Correct" in r.environment_result.feedback) / len(results)
print(f"Accuracy: {accuracy:.1%}")
```

**Loading existing playbook**:

```python
# Load agent with pre-trained strategies
agent = ACELiteLLM.from_playbook(
    playbook_path="./pretrained_strategies.json",
    model="gpt-4"
)

# Use immediately with learned strategies
answer = agent.ask("What is the capital of Spain?")
```

---

### Pattern 2: Full ACE Pipeline with Custom Environment

**When to use**: Need custom evaluation logic, maximum control

**Implementation**:

```python
from ace import (
    Playbook, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult
)
from ace.llm_providers import LiteLLMClient
from ace.prompts_v2_1 import PromptManager

# 1. Custom evaluation environment
class SentimentEnvironment(TaskEnvironment):
    """Evaluate sentiment classification."""

    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        # Extract sentiment from answer
        answer_lower = answer.lower()
        if "positive" in answer_lower:
            predicted = "positive"
        elif "negative" in answer_lower:
            predicted = "negative"
        else:
            predicted = "neutral"

        # Compare with ground truth
        correct = predicted == ground_truth
        feedback = f"{'Correct' if correct else 'Incorrect'} - predicted {predicted}"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={
                "accuracy": 1.0 if correct else 0.0,
                "predicted": predicted
            }
        )

# 2. Setup with production prompts
llm = LiteLLMClient(model="gpt-4")
playbook = Playbook()

prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())

# 3. Training data
samples = [
    Sample("This movie was amazing!", ground_truth="positive"),
    Sample("Terrible experience, very disappointed.", ground_truth="negative"),
    Sample("It was okay, nothing special.", ground_truth="neutral"),
    # ... more samples
]

# 4. Train with checkpoints
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(
    samples=samples,
    environment=SentimentEnvironment(),
    epochs=3,
    checkpoint_interval=10,
    checkpoint_dir="./checkpoints"
)

# 5. Analyze results
from collections import Counter

predictions = [r.environment_result.metrics["predicted"] for r in results]
ground_truths = [r.sample.ground_truth for r in results]

accuracy = sum(p == g for p, g in zip(predictions, ground_truths)) / len(results)
print(f"Accuracy: {accuracy:.1%}")

# Confusion matrix
print("\nPrediction distribution:")
print(Counter(predictions))

# 6. Save for production
playbook.save_to_file("sentiment_playbook.json")
```

---

### Pattern 3: Browser Automation Integration

**When to use**: Adding learning to browser-use agents

**Implementation**:

```python
from ace.integrations import ACEAgent
from browser_use import Agent
from langchain_openai import ChatOpenAI
from ace.llm_providers import LiteLLMClient
import asyncio

# 1. Create browser agent (your existing setup)
browser_llm = ChatOpenAI(model="gpt-4")
browser_agent = Agent(
    task="Go to example.com and check if the domain is available for purchase",
    llm=browser_llm
)

# 2. Wrap with ACE learning
learning_llm = LiteLLMClient(model="gpt-4")
ace_browser = ACEAgent(
    agent=browser_agent,
    llm=learning_llm,
    playbook_path="./browser_strategies.json"
)

# 3. Execute with learning
async def main():
    result = await ace_browser.run()
    print(f"Result: {result}")
    print(f"Learned strategies: {len(ace_browser.playbook.bullets)}")

asyncio.run(main())

# 4. Reuse learned strategies
# Create new agent with same playbook
browser_agent2 = Agent(
    task="Fill out the contact form on example.com",
    llm=browser_llm
)

ace_browser2 = ACEAgent(
    agent=browser_agent2,
    llm=learning_llm,
    playbook_path="./browser_strategies.json"  # Reuse strategies
)

result2 = asyncio.run(ace_browser2.run())
```

**Multiple tasks with shared playbook**:

```python
tasks = [
    "Check domain availability on namecheap.com",
    "Fill out registration form on example.com",
    "Search for 'Python tutorials' on google.com",
]

async def run_all_tasks():
    results = []
    for task in tasks:
        agent = Agent(task=task, llm=browser_llm)
        ace_agent = ACEAgent(agent, learning_llm, "./shared_browser.json")
        result = await ace_agent.run()
        results.append(result)
    return results

results = asyncio.run(run_all_tasks())
print(f"Final playbook size: {len(ace_agent.playbook.bullets)}")
```

---

### Pattern 4: LangChain Workflow Integration

**When to use**: Adding learning to LangChain chains/agents

**Implementation**:

```python
from ace.integrations import ACELangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ace.llm_providers import LiteLLMClient

# 1. Your existing LangChain setup
langchain_llm = ChatOpenAI(model="gpt-4")
prompt = PromptTemplate.from_template(
    "Translate the following to French: {text}"
)
chain = LLMChain(llm=langchain_llm, prompt=prompt)

# 2. Wrap with ACE
learning_llm = LiteLLMClient(model="gpt-4")
ace_chain = ACELangChain(
    chain=chain,
    llm=learning_llm,
    playbook_path="./translation_strategies.json"
)

# 3. Execute with learning
result = ace_chain.run(
    {"text": "Hello, how are you?"},
    ground_truth="Bonjour, comment allez-vous?"
)
print(result)

# 4. Batch processing
test_cases = [
    ("Hello", "Bonjour"),
    ("Thank you", "Merci"),
    ("Goodbye", "Au revoir"),
]

for english, french in test_cases:
    result = ace_chain.run({"text": english}, ground_truth=french)
    print(f"{english} → {result}")

print(f"Learned {len(ace_chain.playbook.bullets)} translation strategies")
```

**With LangChain Agents**:

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Define tools
def calculator(query: str) -> str:
    return str(eval(query))

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for arithmetic calculations"
    )
]

# Create agent
langchain_agent = initialize_agent(
    tools,
    langchain_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Wrap with ACE
ace_agent = ACELangChain(
    chain=langchain_agent,
    llm=learning_llm,
    playbook_path="./agent_strategies.json"
)

# Execute
result = ace_agent.run(
    "What is 25 multiplied by 17?",
    ground_truth="425"
)
```

---

### Pattern 5: Online Learning in Production

**When to use**: Real-time learning from live traffic

**Implementation**:

```python
from ace import OnlineAdapter, Generator, Reflector, Curator, Playbook
from ace.llm_providers import LiteLLMClient
import threading
import queue

class ProductionAgent:
    """Production agent with continuous learning."""

    def __init__(self, model: str, playbook_path: str):
        # Setup
        self.llm = LiteLLMClient(model=model)
        self.playbook = Playbook.load_from_file(playbook_path)
        self.generator = Generator(self.llm)
        self.reflector = Reflector(self.llm)
        self.curator = Curator(self.llm)

        # Online adapter for learning
        self.adapter = OnlineAdapter(
            self.playbook,
            self.generator,
            self.reflector,
            self.curator
        )

        # Feedback queue
        self.feedback_queue = queue.Queue()

        # Start background learner
        self.learner_thread = threading.Thread(
            target=self._background_learner,
            daemon=True
        )
        self.learner_thread.start()

    def query(self, question: str) -> str:
        """Handle user query (fast path - no learning)."""
        output = self.generator.generate(question, "", self.playbook)
        return output.final_answer

    def submit_feedback(self, question: str, answer: str, feedback: str):
        """Submit feedback for learning (async)."""
        self.feedback_queue.put((question, answer, feedback))

    def _background_learner(self):
        """Background thread for continuous learning."""
        from ace import Sample, SimpleEnvironment

        environment = SimpleEnvironment()

        while True:
            # Wait for feedback
            question, answer, feedback = self.feedback_queue.get()

            # Create sample
            sample = Sample(question=question)

            # Learn (updates playbook in-place)
            self.adapter.process(sample, environment)

            # Periodic save
            if self.adapter.step_number % 10 == 0:
                self.playbook.save_to_file("production_playbook.json")

# Usage
agent = ProductionAgent(model="gpt-4", playbook_path="./init_playbook.json")

# Handle user query (fast)
answer = agent.query("What is the capital of France?")
print(answer)

# Submit feedback later (async learning)
agent.submit_feedback(
    question="What is the capital of France?",
    answer=answer,
    feedback="Correct"
)

# Agent continues learning in background
```

---

## Common Use Cases

### Use Case 1: Question Answering

```python
from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment

# Initialize
agent = ACELiteLLM(model="gpt-4", playbook_path="./qa_playbook.json")

# Training data (FAQ)
faq_data = [
    ("What are your business hours?", "9 AM to 5 PM weekdays"),
    ("Do you offer refunds?", "Yes, within 30 days"),
    ("How do I contact support?", "Email support@example.com"),
    # ... more FAQs
]

samples = [Sample(q, ground_truth=a) for q, a in faq_data]

# Train
results = agent.learn(samples, SimpleEnvironment(), epochs=3)

# Production usage
user_question = "When are you open?"
answer = agent.ask(user_question)
print(answer)  # Should relate to business hours
```

### Use Case 2: Code Generation

```python
from ace import (
    Playbook, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult
)
from ace.llm_providers import LiteLLMClient

class CodeEnvironment(TaskEnvironment):
    """Evaluate generated code."""

    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        # Extract code from answer
        import re
        code_match = re.search(r'```python\n(.*?)\n```', answer, re.DOTALL)
        if not code_match:
            return EnvironmentResult(
                feedback="No code found in answer",
                metrics={"score": 0.0}
            )

        code = code_match.group(1)

        # Try to execute
        try:
            exec(code)
            feedback = "Code executed successfully"
            score = 1.0
        except Exception as e:
            feedback = f"Code failed: {str(e)}"
            score = 0.0

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"score": score}
        )

# Setup
llm = LiteLLMClient(model="gpt-4")
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# Training data
samples = [
    Sample("Write a function to check if a number is prime"),
    Sample("Write a function to reverse a string"),
    Sample("Write a function to find the factorial"),
]

# Train
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(samples, CodeEnvironment(), epochs=2)

# Check success rate
success_rate = sum(r.environment_result.metrics["score"] for r in results) / len(results)
print(f"Success rate: {success_rate:.1%}")
```

### Use Case 3: Data Extraction

```python
from ace.integrations import ACELiteLLM
import json

# Initialize
agent = ACELiteLLM(model="gpt-4", playbook_path="./extraction_playbook.json")

# Example: Extract structured data from text
text = """
John Doe is a software engineer at Acme Corp.
He can be reached at john.doe@example.com or (555) 123-4567.
His office is located in San Francisco, CA.
"""

question = f"""
Extract the following information from the text:
- Name
- Job title
- Company
- Email
- Phone
- Location

Text: {text}

Return as JSON.
"""

answer = agent.ask(question)
print(answer)

# Parse extracted data
try:
    data = json.loads(answer)
    print(f"Extracted: {data}")
except:
    print("Failed to parse JSON")
```

### Use Case 4: Sentiment Analysis

```python
from ace.integrations import ACELiteLLM
from ace import Sample, TaskEnvironment, EnvironmentResult

class SentimentEnvironment(TaskEnvironment):
    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        # Extract sentiment
        answer_lower = answer.lower()
        if "positive" in answer_lower:
            predicted = "positive"
        elif "negative" in answer_lower:
            predicted = "negative"
        else:
            predicted = "neutral"

        correct = predicted == ground_truth
        return EnvironmentResult(
            feedback=f"{'Correct' if correct else 'Incorrect'}",
            ground_truth=ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0}
        )

# Initialize
agent = ACELiteLLM(model="gpt-4", playbook_path="./sentiment_playbook.json")

# Training data
reviews = [
    ("This product is amazing! Highly recommend.", "positive"),
    ("Terrible quality, waste of money.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    # ... more reviews
]

samples = [
    Sample(f"Classify sentiment: {review}", ground_truth=sentiment)
    for review, sentiment in reviews
]

# Train
results = agent.learn(samples, SentimentEnvironment(), epochs=3)

# Use in production
new_review = "Absolutely love this! Best purchase ever."
answer = agent.ask(f"Classify sentiment: {new_review}")
print(answer)
```

---

## Best Practices

### 1. Prompt Engineering

**Use production prompts (v2.1)**:

```python
from ace.prompts_v2_1 import PromptManager
from ace import Generator, Reflector, Curator

prompt_mgr = PromptManager()

generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

**Customize retry prompts**:

```python
# For multilingual models
generator = Generator(
    llm,
    retry_prompt="\n\n[日本語] 有効なJSONのみを返してください。"
)

# For specific model quirks
generator = Generator(
    llm,
    retry_prompt="\n\nIMPORTANT: Return ONLY a valid JSON object, no markdown."
)
```

### 2. Playbook Management

**Initialize with seed strategies**:

```python
from ace import Playbook, Bullet

playbook = Playbook()

# Add domain knowledge upfront
seed_bullets = [
    Bullet(
        id="reasoning-00001",
        section="reasoning",
        content="Break complex questions into sub-questions"
    ),
    Bullet(
        id="validation-00001",
        section="validation",
        content="Always validate input types before processing"
    ),
]

for bullet in seed_bullets:
    playbook.add_bullet(bullet)

# Save template
playbook.save_to_file("seed_playbook.json")
```

**Version your playbooks**:

```python
import json
from datetime import datetime

# Add metadata
playbook_data = playbook.to_dict()
playbook_data['metadata'] = {
    'version': '1.2.0',
    'created': datetime.now().isoformat(),
    'domain': 'customer-support',
    'samples_trained': 500,
    'accuracy': 0.92
}

# Save with version
with open(f'playbook_v1.2.0.json', 'w') as f:
    json.dump(playbook_data, f, indent=2)
```

**Prune underperforming strategies**:

```python
# Remove harmful bullets
def prune_playbook(playbook: Playbook, threshold: int = 3):
    """Remove bullets with harmful > helpful * threshold."""
    to_remove = []
    for bullet_id, bullet in playbook.bullets.items():
        if bullet.harmful > bullet.helpful * threshold:
            to_remove.append(bullet_id)

    for bullet_id in to_remove:
        playbook.remove_bullet(bullet_id)
        print(f"Removed harmful bullet: {bullet_id}")

    return len(to_remove)

# Prune after training
removed = prune_playbook(playbook)
print(f"Removed {removed} harmful bullets")
```

### 3. Error Handling

**Wrap LLM calls**:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def safe_generate(generator, question, playbook):
    """Generate with retry on failure."""
    return generator.generate(question, "", playbook)

# Usage
try:
    output = safe_generate(generator, question, playbook)
except Exception as e:
    print(f"Failed after retries: {e}")
    # Fallback logic
```

**Validate playbook before use**:

```python
def validate_playbook(playbook: Playbook) -> bool:
    """Check playbook integrity."""
    if len(playbook.bullets) == 0:
        print("Warning: Empty playbook")
        return False

    # Check for duplicate content
    contents = [b.content for b in playbook.bullets.values()]
    if len(contents) != len(set(contents)):
        print("Warning: Duplicate bullet content")

    # Check ID format
    for bullet_id in playbook.bullets.keys():
        if not re.match(r'^[a-z_]+-\d{5}$', bullet_id):
            print(f"Invalid bullet ID: {bullet_id}")
            return False

    return True

# Before loading
playbook = Playbook.load_from_file("playbook.json")
if not validate_playbook(playbook):
    print("Playbook validation failed!")
```

### 4. Cost Optimization

**Use cheaper models for learning**:

```python
from ace.llm_providers import LiteLLMClient

# Expensive model for production
production_llm = LiteLLMClient(model="gpt-4")

# Cheap model for learning
learning_llm = LiteLLMClient(model="gpt-3.5-turbo")

# Use cheap model for Reflector/Curator
generator = Generator(production_llm)  # User-facing
reflector = Reflector(learning_llm)   # Background
curator = Curator(learning_llm)       # Background
```

**Batch operations**:

```python
# Don't learn from every sample
learn_every_n = 10

for i, sample in enumerate(samples):
    # Always generate
    output = generator.generate(sample.question, "", playbook)

    # Only learn periodically
    if i % learn_every_n == 0:
        reflection = reflector.reflect(...)
        curation = curator.curate(...)
        playbook.apply_delta(curation.delta_batch)
```

**Track costs**:

```python
# Install: pip install ace-framework[observability]

from ace.llm_providers import LiteLLMClient

# Automatic cost tracking
llm = LiteLLMClient(model="gpt-4")

# View in Opik dashboard
# https://www.comet.com/opik
```

### 5. Testing

**Unit test with DummyLLMClient**:

```python
from ace.llm import DummyLLMClient
import json

def test_generator():
    # Setup mock
    llm = DummyLLMClient()
    llm.queue(json.dumps({
        "reasoning": "Test reasoning",
        "final_answer": "Test answer"
    }))

    generator = Generator(llm)
    playbook = Playbook()

    # Test
    output = generator.generate("Test question", "", playbook)

    assert output.reasoning == "Test reasoning"
    assert output.final_answer == "Test answer"

test_generator()
```

**Integration test with real LLM**:

```python
def test_full_pipeline():
    """Test complete adaptation flow."""
    llm = LiteLLMClient(model="gpt-3.5-turbo")  # Cheap for testing
    playbook = Playbook()
    generator = Generator(llm)
    reflector = Reflector(llm)
    curator = Curator(llm)

    adapter = OfflineAdapter(playbook, generator, reflector, curator)

    samples = [Sample("Test question", ground_truth="Test answer")]
    results = adapter.run(samples, SimpleEnvironment(), epochs=1)

    assert len(results) == 1
    assert len(playbook.bullets) > 0

test_full_pipeline()
```

---

## Advanced Techniques

### 1. Multi-Domain Playbooks

**Separate playbooks per domain**:

```python
class MultiDomainAgent:
    """Agent with multiple playbooks."""

    def __init__(self, model: str):
        self.llm = LiteLLMClient(model=model)
        self.generator = Generator(self.llm)

        # Load domain-specific playbooks
        self.playbooks = {
            "math": Playbook.load_from_file("math_playbook.json"),
            "geography": Playbook.load_from_file("geography_playbook.json"),
            "science": Playbook.load_from_file("science_playbook.json"),
        }

    def query(self, question: str, domain: str):
        """Query with domain-specific playbook."""
        playbook = self.playbooks.get(domain, Playbook())
        output = self.generator.generate(question, "", playbook)
        return output.final_answer

# Usage
agent = MultiDomainAgent(model="gpt-4")

answer = agent.query("What is 2+2?", domain="math")
print(answer)

answer = agent.query("What is the capital of France?", domain="geography")
print(answer)
```

### 2. Playbook Merging

**Combine multiple playbooks**:

```python
def merge_playbooks(*playbooks: Playbook) -> Playbook:
    """Merge multiple playbooks into one."""
    merged = Playbook()

    for playbook in playbooks:
        for bullet_id, bullet in playbook.bullets.items():
            # Avoid duplicates
            if bullet_id not in merged.bullets:
                merged.add_bullet(bullet)
            else:
                # Merge counters
                merged.bullets[bullet_id].helpful += bullet.helpful
                merged.bullets[bullet_id].harmful += bullet.harmful
                merged.bullets[bullet_id].neutral += bullet.neutral

    return merged

# Usage
math_playbook = Playbook.load_from_file("math.json")
science_playbook = Playbook.load_from_file("science.json")

combined = merge_playbooks(math_playbook, science_playbook)
combined.save_to_file("combined.json")
```

### 3. A/B Testing

**Compare different strategies**:

```python
from ace import Sample, SimpleEnvironment

def ab_test(samples, environment, playbook_a, playbook_b):
    """Compare two playbooks."""
    llm = LiteLLMClient(model="gpt-4")
    generator = Generator(llm)

    results_a = []
    results_b = []

    for sample in samples:
        # Test A
        output_a = generator.generate(sample.question, "", playbook_a)
        eval_a = environment.evaluate(
            sample.question,
            output_a.final_answer,
            sample.ground_truth
        )
        results_a.append(eval_a.metrics["accuracy"])

        # Test B
        output_b = generator.generate(sample.question, "", playbook_b)
        eval_b = environment.evaluate(
            sample.question,
            output_b.final_answer,
            sample.ground_truth
        )
        results_b.append(eval_b.metrics["accuracy"])

    accuracy_a = sum(results_a) / len(results_a)
    accuracy_b = sum(results_b) / len(results_b)

    print(f"Playbook A: {accuracy_a:.1%}")
    print(f"Playbook B: {accuracy_b:.1%}")
    print(f"Winner: {'A' if accuracy_a > accuracy_b else 'B'}")

# Usage
playbook_v1 = Playbook.load_from_file("v1.json")
playbook_v2 = Playbook.load_from_file("v2.json")

ab_test(test_samples, SimpleEnvironment(), playbook_v1, playbook_v2)
```

### 4. Curriculum Learning

**Progressive difficulty**:

```python
from ace import Sample, OfflineAdapter

# Organize samples by difficulty
easy_samples = [Sample(...) for ... in easy_data]
medium_samples = [Sample(...) for ... in medium_data]
hard_samples = [Sample(...) for ... in hard_data]

# Train progressively
adapter = OfflineAdapter(playbook, generator, reflector, curator)

# Stage 1: Easy
print("Stage 1: Easy samples")
adapter.run(easy_samples, environment, epochs=3)

# Stage 2: Medium
print("Stage 2: Medium samples")
adapter.run(medium_samples, environment, epochs=2)

# Stage 3: Hard
print("Stage 3: Hard samples")
adapter.run(hard_samples, environment, epochs=2)

# Final playbook optimized for hard tasks
playbook.save_to_file("curriculum_playbook.json")
```

---

## Troubleshooting

### Issue 1: JSON Parse Failures

**Symptoms**: `JSONDecodeError` during generation/reflection/curation

**Solutions**:

1. **Use retry prompts**:
```python
generator = Generator(
    llm,
    retry_prompt="\n\nReturn ONLY valid JSON, no other text."
)
```

2. **Check model compatibility**:
```python
# Some models need explicit instructions
prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
```

3. **Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show full LLM responses
output = generator.generate(question, "", playbook)
```

### Issue 2: Playbook Not Learning

**Symptoms**: Playbook stays empty or doesn't grow

**Solutions**:

1. **Check Curator output**:
```python
curation = curator.curate(reflection, playbook)
print(curation.reasoning)
print(len(curation.delta_batch.operations))

# Should see operations being generated
```

2. **Verify delta application**:
```python
before_count = len(playbook.bullets)
playbook.apply_delta(curation.delta_batch)
after_count = len(playbook.bullets)

print(f"Bullets: {before_count} → {after_count}")
```

3. **Check for duplicate prevention**:
```python
# Curator may skip adding duplicate strategies
# This is expected behavior, not a bug
```

### Issue 3: Poor Performance

**Symptoms**: Low accuracy, unhelpful strategies

**Solutions**:

1. **Use better prompts**:
```python
# Switch from v1.0 to v2.1
from ace.prompts_v2_1 import PromptManager
prompt_mgr = PromptManager()
# Use prompt_mgr.get_*_prompt()
```

2. **Train longer**:
```python
# More epochs
results = adapter.run(samples, environment, epochs=5)  # Instead of 1-2
```

3. **Better evaluation**:
```python
# Implement stricter environment
class StrictEnvironment(TaskEnvironment):
    def evaluate(self, question, answer, ground_truth):
        # Exact match instead of substring
        correct = answer.strip() == ground_truth.strip()
        return EnvironmentResult(...)
```

### Issue 4: High Costs

**Symptoms**: Unexpectedly high API bills

**Solutions**:

1. **Use cheaper models for learning**:
```python
production_llm = LiteLLMClient(model="gpt-4")
learning_llm = LiteLLMClient(model="gpt-3.5-turbo")

generator = Generator(production_llm)  # User-facing
reflector = Reflector(learning_llm)   # Background
curator = Curator(learning_llm)       # Background
```

2. **Track costs**:
```python
pip install ace-framework[observability]
# View costs in Opik dashboard
```

3. **Reduce learning frequency**:
```python
# Only learn from failures
if "incorrect" in feedback.lower():
    reflection = reflector.reflect(...)
    curation = curator.curate(...)
    playbook.apply_delta(...)
```

---

## Performance Optimization

### 1. Token Optimization

**Use TOON format** (automatic):
```python
# Already optimized by default
playbook_str = playbook.as_prompt()  # Uses TOON
# 16-62% token savings vs JSON
```

**Limit playbook size**:
```python
def limit_playbook_size(playbook: Playbook, max_bullets: int = 100):
    """Keep only top-performing bullets."""
    bullets = sorted(
        playbook.bullets.values(),
        key=lambda b: b.helpful - b.harmful,
        reverse=True
    )

    if len(bullets) > max_bullets:
        # Remove worst performers
        to_remove = bullets[max_bullets:]
        for bullet in to_remove:
            playbook.remove_bullet(bullet.id)

limit_playbook_size(playbook, max_bullets=50)
```

### 2. Parallel Processing

**Process samples in parallel** (advanced):
```python
from concurrent.futures import ThreadPoolExecutor

def process_sample_parallel(sample):
    # Each thread gets its own Generator
    llm = LiteLLMClient(model="gpt-4")
    generator = Generator(llm)
    output = generator.generate(sample.question, "", playbook)
    return output

with ThreadPoolExecutor(max_workers=5) as executor:
    outputs = list(executor.map(process_sample_parallel, samples))
```

**Note**: Reflector/Curator/Playbook updates should still be sequential

### 3. Caching

**Cache common questions**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(question: str) -> str:
    output = generator.generate(question, "", playbook)
    return output.final_answer

# Usage
answer = cached_generate("What is 2+2?")  # LLM call
answer = cached_generate("What is 2+2?")  # Cached
```

---

## Production Deployment

### Architecture 1: Offline Training + Static Serving

**Best for**: Stable domains, infrequent retraining

```python
# Training script (offline)
def train_playbook():
    llm = LiteLLMClient(model="gpt-4")
    playbook = Playbook()
    generator = Generator(llm)
    reflector = Reflector(llm)
    curator = Curator(llm)

    adapter = OfflineAdapter(playbook, generator, reflector, curator)
    results = adapter.run(training_samples, environment, epochs=5)

    playbook.save_to_file("production_playbook.json")
    return playbook

# Run once per week
playbook = train_playbook()

# Serving script (production)
class ProductionAPI:
    def __init__(self):
        self.llm = LiteLLMClient(model="gpt-4")
        self.generator = Generator(self.llm)
        self.playbook = Playbook.load_from_file("production_playbook.json")

    def query(self, question: str) -> str:
        output = self.generator.generate(question, "", self.playbook)
        return output.final_answer

# Deploy
api = ProductionAPI()

@app.route("/query", methods=["POST"])
def query():
    question = request.json["question"]
    answer = api.query(question)
    return {"answer": answer}
```

### Architecture 2: Continuous Learning

**Best for**: Dynamic domains, frequent updates

```python
# See "Pattern 5: Online Learning in Production" above
```

### Architecture 3: Hybrid (Recommended)

**Best for**: Balance between stability and adaptability

```python
class HybridAgent:
    def __init__(self, model: str, base_playbook_path: str):
        self.llm = LiteLLMClient(model=model)
        self.generator = Generator(self.llm)
        self.reflector = Reflector(self.llm)
        self.curator = Curator(self.llm)

        # Load trained base playbook
        self.base_playbook = Playbook.load_from_file(base_playbook_path)

        # Runtime playbook (starts as copy)
        self.runtime_playbook = Playbook.from_dict(self.base_playbook.to_dict())

        # Feedback collection
        self.feedback_buffer = []

    def query(self, question: str) -> str:
        """Fast path: Use runtime playbook."""
        output = self.generator.generate(question, "", self.runtime_playbook)
        return output.final_answer

    def submit_feedback(self, question: str, answer: str, feedback: str):
        """Collect feedback for batch retraining."""
        self.feedback_buffer.append((question, answer, feedback))

    def retrain(self, environment):
        """Periodic retraining from feedback."""
        if len(self.feedback_buffer) < 100:
            return

        samples = [
            Sample(q) for q, a, f in self.feedback_buffer
        ]

        # Online adaptation
        adapter = OnlineAdapter(
            self.runtime_playbook,
            self.generator,
            self.reflector,
            self.curator
        )

        for sample in samples:
            adapter.process(sample, environment)

        # Save updated playbook
        self.runtime_playbook.save_to_file("runtime_playbook.json")

        # Clear buffer
        self.feedback_buffer.clear()

# Usage
agent = HybridAgent(model="gpt-4", base_playbook_path="base_playbook.json")

# Handle queries (fast)
answer = agent.query("User question")

# Collect feedback
agent.submit_feedback("User question", answer, "Correct")

# Retrain periodically (e.g., daily cron job)
agent.retrain(environment)
```

---

## Migration Guides

### From No ACE to ACE

**Before** (static agent):
```python
def answer_question(question: str) -> str:
    llm = LiteLLMClient(model="gpt-4")
    response = llm.complete(f"Answer this question: {question}")
    return response.text
```

**After** (with ACE):
```python
from ace.integrations import ACELiteLLM

agent = ACELiteLLM(model="gpt-4", playbook_path="./strategies.json")

def answer_question(question: str, ground_truth: str = None) -> str:
    return agent.ask(question, ground_truth=ground_truth)
```

### From v1.0 Prompts to v2.1

**Before**:
```python
generator = Generator(llm)  # Uses v1.0 prompts by default
```

**After**:
```python
from ace.prompts_v2_1 import PromptManager

prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

---

## Summary

This developer guide covered:

✅ **Getting Started**: Installation and minimal example
✅ **Implementation Patterns**: 5 common patterns with code
✅ **Use Cases**: Q&A, code generation, data extraction, sentiment
✅ **Best Practices**: Prompts, playbooks, errors, costs, testing
✅ **Advanced Techniques**: Multi-domain, merging, A/B testing, curriculum
✅ **Troubleshooting**: Common issues and solutions
✅ **Optimization**: Token, parallel, caching techniques
✅ **Production**: 3 deployment architectures
✅ **Migration**: Upgrading from non-ACE or v1.0

### Quick Reference

**Most Common Pattern**:
```python
from ace.integrations import ACELiteLLM
agent = ACELiteLLM(model="gpt-4", playbook_path="./playbook.json")
answer = agent.ask(question, ground_truth=truth)
```

**Production Pattern**:
```python
from ace.prompts_v2_1 import PromptManager
prompt_mgr = PromptManager()
# Use prompt_mgr for Generator/Reflector/Curator
# Deploy with cost tracking (observability)
```

---

**Related Documentation**:
- [COMPREHENSIVE_GUIDE.md](./COMPREHENSIVE_GUIDE.md) - High-level overview
- [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md) - Technical architecture
- [COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md) - Complete API reference
- [DATA_FLOW_GUIDE.md](./DATA_FLOW_GUIDE.md) - Data flow details

---

*Last updated: January 2025*
*Framework version: 0.5.1*
