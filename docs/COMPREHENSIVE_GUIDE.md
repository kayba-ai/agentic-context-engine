# Comprehensive Guide to the Agentic Context Engineering Framework

**Version**: 0.5.1
**Last Updated**: January 2025
**Reading Time**: 30-45 minutes

## Table of Contents

1. [Introduction](#introduction)
2. [What is Agentic Context Engineering?](#what-is-agentic-context-engineering)
3. [Framework Architecture Overview](#framework-architecture-overview)
4. [Core Concepts](#core-concepts)
5. [How the Framework Works](#how-the-framework-works)
6. [Usage Patterns](#usage-patterns)
7. [Key Components at a Glance](#key-components-at-a-glance)
8. [Quick Start Examples](#quick-start-examples)
9. [Advanced Features](#advanced-features)
10. [Production Deployment](#production-deployment)
11. [Related Guides](#related-guides)

---

## Introduction

The **Agentic Context Engineering (ACE)** framework is a production-ready Python library that enables AI agents to **learn from their execution feedback** through a sophisticated three-role architecture. Based on the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618), this framework provides a structured approach to self-improving AI systems.

### What Problems Does ACE Solve?

Traditional AI agents:
- ❌ Repeat the same mistakes
- ❌ Can't learn from experience within a session
- ❌ Require manual prompt engineering for each domain
- ❌ Have no memory of what works and what doesn't

ACE framework agents:
- ✅ Learn from execution feedback automatically
- ✅ Build a knowledge base ("playbook") of strategies
- ✅ Adapt to new domains through reflection
- ✅ Improve performance over time without retraining

### Who Should Use This Framework?

- **AI Engineers**: Building production AI agents that need to improve over time
- **Researchers**: Studying self-improving AI systems and meta-learning
- **Product Developers**: Creating AI applications that learn from user interactions
- **Enterprise Teams**: Deploying reliable AI systems with observability and cost tracking

---

## What is Agentic Context Engineering?

**Agentic Context Engineering** is a method where AI systems improve themselves by:

1. **Executing** tasks (generating answers, performing actions)
2. **Reflecting** on their performance (analyzing what worked/failed)
3. **Curating** a knowledge base (updating strategies based on learnings)

### The Three Roles

The framework implements three specialized AI roles that work together:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  User Question → [GENERATOR] → Answer                  │
│                       ↓                                 │
│                  [ENVIRONMENT]                          │
│                   (evaluates)                           │
│                       ↓                                 │
│                  [REFLECTOR]                            │
│                   (analyzes)                            │
│                       ↓                                 │
│                   [CURATOR]                             │
│                   (updates)                             │
│                       ↓                                 │
│                   Playbook ←──────────┐                │
│            (knowledge base) ───────────┘                │
│                       │                                 │
│                       └──→ Used by Generator            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 1. Generator (Execution Role)
- **Purpose**: Produces answers/executes tasks
- **Input**: Question + Current playbook strategies
- **Output**: Reasoned answer with cited strategies
- **Analogy**: The student taking the exam

#### 2. Reflector (Analysis Role)
- **Purpose**: Analyzes performance and identifies root causes
- **Input**: Question + Answer + Feedback + Playbook
- **Output**: Analysis of what worked/failed, strategy classifications
- **Analogy**: The teacher reviewing the exam

#### 3. Curator (Knowledge Management Role)
- **Purpose**: Updates the knowledge base incrementally
- **Input**: Reflection + Current playbook
- **Output**: Delta operations (ADD/UPDATE/TAG/REMOVE strategies)
- **Analogy**: The curriculum designer updating course materials

### The Playbook: Your Agent's Knowledge Base

The **Playbook** is a structured context store containing learned strategies:

```json
{
  "bullets": {
    "reasoning-00001": {
      "content": "Break complex questions into smaller sub-questions",
      "section": "reasoning",
      "helpful": 5,
      "harmful": 0,
      "neutral": 1
    },
    "edge_cases-00001": {
      "content": "Check for empty input before processing",
      "section": "edge_cases",
      "helpful": 3,
      "harmful": 0,
      "neutral": 0
    }
  }
}
```

**Key Features**:
- **Incremental Updates**: Uses delta operations (not full rewrites)
- **Performance Tracking**: Each strategy has helpful/harmful/neutral counters
- **Token Efficient**: TOON encoding reduces tokens by 16-62%
- **Persistent**: Save/load from JSON files
- **Debuggable**: Human-readable markdown format for inspection

---

## Framework Architecture Overview

### Directory Structure

```
agentic-context-engine/
├── ace/                          # Core framework library
│   ├── playbook.py              # Knowledge base management
│   ├── roles.py                 # Generator, Reflector, Curator
│   ├── delta.py                 # Incremental update operations
│   ├── adaptation.py            # Orchestration (Offline/Online)
│   ├── llm.py                   # LLM client interface
│   ├── prompts_v2_1.py          # Production prompts (RECOMMENDED)
│   ├── llm_providers/           # LiteLLM, LangChain clients
│   ├── integrations/            # ACELiteLLM, ACEAgent, ACELangChain
│   └── observability/           # Opik cost tracking & tracing
├── benchmarks/                   # Scientific evaluation framework
├── examples/                     # 30+ production examples
├── tests/                        # 11,337 lines of tests
├── docs/                         # Comprehensive documentation
└── scripts/                      # Research & benchmarking scripts
```

### Core Modules (4,500+ lines)

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `playbook.py` | Knowledge storage | `Bullet`, `Playbook` |
| `roles.py` | Three AI roles | `Generator`, `Reflector`, `Curator` |
| `delta.py` | Incremental updates | `DeltaOperation`, `DeltaBatch` |
| `adaptation.py` | Orchestration | `OfflineAdapter`, `OnlineAdapter` |
| `llm.py` | LLM abstraction | `LLMClient`, `TransformersLLMClient` |
| `integrations/` | Framework wrappers | `ACELiteLLM`, `ACEAgent`, `ACELangChain` |

**For detailed technical architecture**: See [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md)

---

## Core Concepts

### 1. Bullets: Individual Strategy Entries

A **Bullet** is a single learned strategy:

```python
from ace import Bullet

bullet = Bullet(
    id="reasoning-00001",
    section="reasoning",
    content="Always verify input types before processing",
    helpful=3,
    harmful=0,
    neutral=1
)
```

**Bullet Lifecycle**:
1. Created by Curator (ADD operation)
2. Used by Generator (cited in reasoning)
3. Evaluated by Reflector (tagged helpful/harmful/neutral)
4. Updated by Curator (UPDATE/TAG operations)
5. Removed if consistently harmful (REMOVE operation)

### 2. Delta Operations: Incremental Updates

Instead of regenerating the entire playbook, ACE uses **delta operations**:

```python
from ace import DeltaOperation, DeltaBatch

operations = [
    DeltaOperation(type="ADD", section="reasoning",
                   content="New strategy to learn"),
    DeltaOperation(type="TAG", bullet_id="reasoning-00001",
                   metadata={"helpful": 1}),
    DeltaOperation(type="REMOVE", bullet_id="edge_cases-00005")
]

batch = DeltaBatch(
    reasoning="Refining strategies based on recent failures",
    operations=operations
)

# Apply to playbook
playbook.apply_delta(batch)
```

**Benefits**:
- More efficient (fewer tokens, faster)
- Preserves playbook history
- Enables incremental learning
- Easier to debug

### 3. Task Environments: Evaluation Logic

A **TaskEnvironment** provides feedback on agent performance:

```python
from ace import TaskEnvironment, EnvironmentResult

class MathEnvironment(TaskEnvironment):
    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        # Your evaluation logic
        correct = self.check_math(answer, ground_truth)
        feedback = "Correct!" if correct else "Incorrect calculation"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0}
        )
```

### 4. Two Usage Patterns

#### Pattern A: Full ACE Pipeline (New Agents)

**When to use**: Building a new agent from scratch

```python
from ace import OfflineAdapter, Generator, Reflector, Curator, Playbook
from ace.llm_providers import LiteLLMClient

# Setup
llm = LiteLLMClient(model="gpt-4")
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# Train
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(training_samples, environment, epochs=3)
```

**Flow**: ACE Generator executes → Environment evaluates → Reflector analyzes → Curator updates

#### Pattern B: Integration Pattern (Existing Agents)

**When to use**: Adding learning to existing systems (browser-use, LangChain, custom agents)

```python
from ace.integrations import ACEAgent
from browser_use import Agent

# Your existing agent
browser_agent = Agent(task="Buy tickets", llm=llm)

# Wrap with ACE learning
ace_agent = ACEAgent(
    agent=browser_agent,
    llm=llm,
    playbook_path="./browser_playbook.json"
)

# Execute with learning
result = await ace_agent.run()
# Playbook automatically updated with learnings
```

**Flow**: External agent executes → Reflector analyzes → Curator updates

**For detailed usage patterns**: See [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)

---

## How the Framework Works

### Offline Adaptation (Batch Training)

Use **OfflineAdapter** when you have training samples upfront:

```python
from ace import OfflineAdapter, Sample

# Define training samples
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="What is the capital of France?", ground_truth="Paris"),
    # ... more samples
]

# Create adapter
adapter = OfflineAdapter(playbook, generator, reflector, curator)

# Run training with checkpoints
results = adapter.run(
    samples=samples,
    environment=environment,
    epochs=3,                    # Multiple passes
    checkpoint_interval=10,      # Save every 10 samples
    checkpoint_dir="./checkpoints"
)

# Analyze results
for result in results:
    print(f"Q: {result.sample.question}")
    print(f"Answer: {result.generator_output.final_answer}")
    print(f"Feedback: {result.environment_result.feedback}")
    print(f"Playbook size: {len(result.playbook_snapshot.bullets)}")
```

**Use Cases**:
- Training on question-answering datasets
- Learning from historical interaction logs
- Multi-epoch refinement of strategies

### Online Adaptation (Streaming)

Use **OnlineAdapter** for real-time learning:

```python
from ace import OnlineAdapter

# Create adapter
adapter = OnlineAdapter(playbook, generator, reflector, curator)

# Process samples as they arrive
for sample in incoming_stream:
    result = adapter.process(sample, environment)

    # Playbook is updated after each sample
    print(f"Current playbook size: {len(adapter.playbook.bullets)}")
```

**Use Cases**:
- Live user interactions
- Production deployments
- Continuous learning systems

### Integration Mode (External Agents)

Use **integration classes** to wrap existing agents:

```python
# 1. Simple Q&A with ACELiteLLM
from ace.integrations import ACELiteLLM

agent = ACELiteLLM(model="gpt-4")
answer = agent.ask("What is the capital of France?")
# Automatically learns from interaction

# 2. Browser automation with ACEAgent
from ace.integrations import ACEAgent
from browser_use import Agent

browser_agent = Agent(task="Check domain availability", llm=llm)
ace_browser = ACEAgent(agent=browser_agent, llm=llm)
result = await ace_browser.run()
# Learns web automation strategies

# 3. LangChain workflows with ACELangChain
from ace.integrations import ACELangChain
from langchain.chains import LLMChain

chain = LLMChain(llm=langchain_llm, prompt=prompt)
ace_chain = ACELangChain(chain=chain, llm=llm)
result = ace_chain.run(input_data)
# Learns from chain execution
```

**For detailed data flows**: See [DATA_FLOW_GUIDE.md](./DATA_FLOW_GUIDE.md)

---

## Usage Patterns

### Pattern 1: Quick Start with ACELiteLLM

**Best for**: Prototyping, simple Q&A, classification tasks

```python
from ace.integrations import ACELiteLLM

# Initialize (bundles Generator+Reflector+Curator+Playbook)
agent = ACELiteLLM(model="gpt-4")

# Ask questions - it learns automatically
answer1 = agent.ask("What is 2+2?")
answer2 = agent.ask("What is 3+3?")
# After each question, playbook is updated

# Save learned strategies
agent.save_playbook("math_strategies.json")

# Load for later use
agent2 = ACELiteLLM.from_playbook("math_strategies.json", model="gpt-4")
```

### Pattern 2: Custom Agent with Full Control

**Best for**: Production systems, custom evaluation logic

```python
from ace import (
    Playbook, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult
)
from ace.llm_providers import LiteLLMClient

# 1. Create components
llm = LiteLLMClient(model="gpt-4")
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# 2. Define custom evaluation
class CustomEnvironment(TaskEnvironment):
    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        # Your custom logic
        success = self.custom_check(answer, ground_truth)
        return EnvironmentResult(
            feedback="Success" if success else "Failed",
            ground_truth=ground_truth,
            metrics={"score": 1.0 if success else 0.0}
        )

# 3. Train
samples = [Sample(question=q, ground_truth=gt) for q, gt in data]
environment = CustomEnvironment()

adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(samples, environment, epochs=3)

# 4. Deploy with learned playbook
playbook.save_to_file("production_playbook.json")
```

### Pattern 3: Integrate with Existing Systems

**Best for**: Adding learning to browser-use, LangChain, CrewAI, custom agents

```python
from ace.integrations import ACELangChain
from langchain.agents import initialize_agent, Tool

# Your existing LangChain setup
tools = [Tool(name="Calculator", func=calculator)]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# Wrap with ACE learning
ace_agent = ACELangChain(
    chain=agent,
    llm=llm,
    playbook_path="./agent_strategies.json"
)

# Execute - ACE learns from results
result = ace_agent.run("Calculate 25 * 17")
# Playbook updated with tool usage strategies
```

### Pattern 4: Production Deployment with Observability

**Best for**: Enterprise deployments, cost monitoring

```bash
# Install with observability
pip install ace-framework[observability]
```

```python
from ace.llm_providers import LiteLLMClient
from ace.integrations import ACELiteLLM

# Automatic token tracking when Opik installed
agent = ACELiteLLM(model="gpt-4")

# All LLM calls automatically tracked
answer = agent.ask("Complex question...")

# View costs at: https://www.comet.com/opik
# - Token usage per role (Generator/Reflector/Curator)
# - Cost per interaction
# - Playbook evolution metrics
```

---

## Key Components at a Glance

### 1. Playbook (`ace/playbook.py`)

```python
from ace import Playbook, Bullet

# Create playbook
playbook = Playbook()

# Add strategies manually
bullet = Bullet(
    id="custom-001",
    section="validation",
    content="Always validate email format before processing"
)
playbook.add_bullet(bullet)

# Apply delta updates
playbook.apply_delta(delta_batch)

# Persistence
playbook.save_to_file("strategies.json")
playbook2 = Playbook.load_from_file("strategies.json")

# Get token-efficient format for LLM
prompt_context = playbook.as_prompt()  # TOON format

# Get human-readable format for debugging
print(str(playbook))  # Markdown format

# Statistics
stats = playbook.stats()
# {'total_bullets': 15, 'sections': {'reasoning': 5, ...}, ...}
```

### 2. Generator (`ace/roles.py`)

```python
from ace import Generator
from ace.llm_providers import LiteLLMClient

llm = LiteLLMClient(model="gpt-4")
generator = Generator(llm)

output = generator.generate(
    question="What is the capital of France?",
    context="",
    playbook=playbook,
    reflection=""  # Optional previous reflection
)

print(output.reasoning)      # Thought process
print(output.final_answer)   # "Paris"
print(output.bullet_ids)     # ["reasoning-00001", "geography-00003"]
```

### 3. Reflector (`ace/roles.py`)

```python
from ace import Reflector

reflector = Reflector(llm)

reflection = reflector.reflect(
    question="What is the capital of France?",
    generator_output=generator_output,
    feedback="Correct! Well done.",
    ground_truth="Paris",
    playbook=playbook
)

print(reflection.reasoning)
print(reflection.error_identification)
print(reflection.root_cause_analysis)
print(reflection.bullet_tags)
# [
#   {"bullet_id": "geography-00003", "tag": "helpful", "justification": "..."},
#   {"bullet_id": "reasoning-00001", "tag": "neutral", "justification": "..."}
# ]
```

### 4. Curator (`ace/roles.py`)

```python
from ace import Curator

curator = Curator(llm)

curation = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="Geography question about European capitals"
)

print(curation.reasoning)
print(curation.delta_batch)  # DeltaBatch with operations

# Apply updates
playbook.apply_delta(curation.delta_batch)
```

### 5. LLM Clients (`ace/llm_providers/`)

```python
# LiteLLM (100+ providers)
from ace.llm_providers import LiteLLMClient

# OpenAI
llm = LiteLLMClient(model="gpt-4")

# Anthropic
llm = LiteLLMClient(model="claude-3-opus-20240229")

# Google
llm = LiteLLMClient(model="gemini/gemini-pro")

# Local via Ollama
llm = LiteLLMClient(model="ollama/llama2")

# LangChain
from ace.llm_providers import LangChainClient
from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(model="gpt-4")
llm = LangChainClient(llm=langchain_llm)

# Transformers (local)
from ace.llm import TransformersLLMClient

llm = TransformersLLMClient(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto"
)
```

**For detailed component documentation**: See [COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md)

---

## Quick Start Examples

### Example 1: Minimal Working Example (52 lines)

```python
from ace import (
    Playbook, Generator, Reflector, Curator,
    OfflineAdapter, Sample, SimpleEnvironment
)
from ace.llm_providers import LiteLLMClient

# Setup
llm = LiteLLMClient(model="gpt-4")
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# Training data
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="What is 5+3?", ground_truth="8"),
]

# Environment
environment = SimpleEnvironment()

# Train
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(samples, environment, epochs=2)

# Save
playbook.save_to_file("math_playbook.json")

# Test
test_output = generator.generate(
    question="What is 7+6?",
    context="",
    playbook=playbook
)
print(test_output.final_answer)
```

### Example 2: Browser Automation

```python
from ace.integrations import ACEAgent
from browser_use import Agent
from langchain_openai import ChatOpenAI

# Your browser agent
llm = ChatOpenAI(model="gpt-4")
browser_agent = Agent(
    task="Go to example.com and check if domain is available",
    llm=llm
)

# Wrap with ACE
ace_browser = ACEAgent(
    agent=browser_agent,
    llm=llm,
    playbook_path="./browser_strategies.json"
)

# Execute (async)
import asyncio
result = asyncio.run(ace_browser.run())

print(f"Result: {result}")
# Playbook now contains learned web automation strategies
```

### Example 3: Cost Tracking

```python
# Install: pip install ace-framework[observability]

from ace.llm_providers import LiteLLMClient
from ace.integrations import ACELiteLLM

# Automatic tracking when Opik installed
agent = ACELiteLLM(model="gpt-4")

# Execute
for question in questions:
    answer = agent.ask(question)
    print(f"Q: {question}\nA: {answer}\n")

# View in Opik dashboard:
# - Token usage per role
# - Cost per interaction
# - Playbook evolution
# - Performance metrics

# Dashboard: https://www.comet.com/opik
```

---

## Advanced Features

### 1. Checkpoint Saving During Training

Save playbook snapshots during long training runs:

```python
adapter = OfflineAdapter(playbook, generator, reflector, curator)

results = adapter.run(
    samples=large_dataset,
    environment=environment,
    epochs=5,
    checkpoint_interval=50,      # Save every 50 samples
    checkpoint_dir="./checkpoints"
)

# Creates:
# - checkpoint_50.json
# - checkpoint_100.json
# - checkpoint_150.json
# - latest.json (always most recent)
```

**Use Cases**:
- Resume training after interruption
- Compare playbook evolution
- Early stopping based on validation

### 2. Prompt Version Management

Choose the right prompts for your use case:

```python
from ace.prompts_v2_1 import PromptManager

# Production prompts (v2.1) - RECOMMENDED
prompt_mgr = PromptManager()

generator = Generator(
    llm,
    prompt_template=prompt_mgr.get_generator_prompt()
)
reflector = Reflector(
    llm,
    prompt_template=prompt_mgr.get_reflector_prompt()
)
curator = Curator(
    llm,
    prompt_template=prompt_mgr.get_curator_prompt()
)

# v2.1 improvements:
# - +17% success rate over v1.0
# - MCP integration
# - Better error handling
# - Token efficiency
```

**Versions**:
- `prompts.py` (v1.0): Simple, tutorial-friendly
- `prompts_v2.py` (v2.0): Enhanced
- `prompts_v2_1.py` (v2.1): **RECOMMENDED** for production

### 3. Custom Retry Prompts

Customize JSON parsing retry behavior:

```python
# Default (English)
generator = Generator(llm)

# Custom for specific models
generator = Generator(
    llm,
    retry_prompt="\n\nPlease return ONLY valid JSON."
)

# Multilingual
generator = Generator(
    llm,
    retry_prompt="\n\n[日本語] 有効なJSONのみを返してください。"
)
```

**Benefits**:
- Reduces parse failures by 7-12%
- Supports multilingual models
- Consistent across all roles

### 4. Feature Detection

Check available optional features:

```python
from ace.features import (
    has_opik, has_litellm, has_langchain,
    has_transformers, get_available_features,
    print_feature_status
)

# Check individual features
if has_opik():
    print("Opik observability available")

# Get all features
features = get_available_features()
# {
#   'opik': True,
#   'litellm': True,
#   'langchain': False,
#   'transformers': True,
#   ...
# }

# Pretty print
print_feature_status()
# ACE Framework Features:
#   ✓ opik (Observability)
#   ✓ litellm (LLM Provider)
#   ✗ langchain (Not installed)
#   ✓ transformers (Local Models)
```

### 5. Benchmarking Framework

Scientific evaluation with train/test splits:

```bash
# Run benchmark
python scripts/run_benchmark.py simple_qa --limit 50

# Compare baseline vs ACE
python scripts/run_benchmark.py simple_qa --limit 50 --compare

# Multi-epoch training
python scripts/run_benchmark.py finer_ord --limit 100 --epochs 3

# Analyze results
python scripts/analyze_ace_results.py
python scripts/explain_ace_performance.py
```

**Features**:
- Train/test split
- Multiple epochs
- Baseline comparison
- Metrics tracking
- Results visualization

---

## Production Deployment

### Installation for Production

```bash
# Core functionality
pip install ace-framework

# With cost tracking (RECOMMENDED)
pip install ace-framework[observability]

# Enterprise features
pip install ace-framework[all]
```

### Best Practices

#### 1. Use Production Prompts

```python
from ace.prompts_v2_1 import PromptManager

prompt_mgr = PromptManager()
# Use prompt_mgr.get_*_prompt() for all roles
```

#### 2. Enable Cost Tracking

```python
# Automatic with observability installed
pip install ace-framework[observability]

# Set Opik API key
export OPIK_API_KEY="your-key"
```

#### 3. Persist Playbooks

```python
# Save after training
playbook.save_to_file("production_playbook.json")

# Load in production
playbook = Playbook.load_from_file("production_playbook.json")
```

#### 4. Monitor Performance

```python
# Track results
results = adapter.run(samples, environment, epochs=3)

# Analyze
successful = [r for r in results if "correct" in r.environment_result.feedback.lower()]
accuracy = len(successful) / len(results)
print(f"Accuracy: {accuracy:.2%}")

# View in Opik dashboard
# - Token usage trends
# - Cost per interaction
# - Success rate over time
```

#### 5. Version Playbooks

```python
import datetime

# Add metadata
playbook_data = playbook.to_dict()
playbook_data['metadata'] = {
    'version': '1.2.0',
    'created': datetime.datetime.now().isoformat(),
    'domain': 'customer-support',
    'training_samples': 500
}

# Save with version
import json
with open('playbook_v1.2.0.json', 'w') as f:
    json.dump(playbook_data, f, indent=2)
```

### Deployment Architectures

#### Architecture 1: Offline Training + Online Serving

```python
# Training phase (offline)
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(training_samples, environment, epochs=5)
playbook.save_to_file("trained_playbook.json")

# Serving phase (production)
playbook = Playbook.load_from_file("trained_playbook.json")
generator = Generator(llm)

# No Reflector/Curator in production - just use learned playbook
for user_query in production_stream:
    output = generator.generate(user_query, "", playbook)
    return output.final_answer
```

#### Architecture 2: Continuous Learning

```python
# Production with ongoing learning
adapter = OnlineAdapter(playbook, generator, reflector, curator)

# Process samples continuously
for sample in production_stream:
    result = adapter.process(sample, environment)

    # Periodically save playbook
    if result.step_number % 100 == 0:
        playbook.save_to_file(f"playbook_step_{result.step_number}.json")
```

#### Architecture 3: Hybrid (Recommended)

```python
# Use trained playbook + periodic retraining
playbook = Playbook.load_from_file("trained_playbook.json")

# Production serving (fast)
generator = Generator(llm)
output = generator.generate(query, "", playbook)

# Collect feedback asynchronously
feedback_queue.append((query, output, user_feedback))

# Periodic retraining (batch)
if len(feedback_queue) >= 100:
    samples = [Sample(q, feedback=f) for q, o, f in feedback_queue]
    adapter = OfflineAdapter(playbook, generator, reflector, curator)
    adapter.run(samples, environment, epochs=1)
    playbook.save_to_file("playbook_updated.json")
    feedback_queue.clear()
```

---

## Related Guides

This comprehensive guide provides a high-level overview. For deeper dives into specific topics, see:

### Technical Documentation
- **[ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md)** - Detailed technical architecture, design decisions, and module relationships
- **[COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md)** - Complete API reference for all classes and methods
- **[DATA_FLOW_GUIDE.md](./DATA_FLOW_GUIDE.md)** - How data flows through the system with detailed examples
- **[DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)** - Implementation patterns, best practices, and common use cases

### Existing Documentation
- **[QUICK_START.md](./QUICK_START.md)** - 5-minute getting started guide
- **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)** - Integrate ACE with existing agents (39KB guide)
- **[API_REFERENCE.md](./API_REFERENCE.md)** - Complete API documentation
- **[PROMPT_ENGINEERING.md](./PROMPT_ENGINEERING.md)** - Advanced prompt techniques
- **[TESTING_GUIDE.md](./TESTING_GUIDE.md)** - Testing strategies and test suite overview
- **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - Environment setup and installation

### Repository Files
- **[README.md](../README.md)** - Project overview and quick links
- **[CLAUDE.md](../CLAUDE.md)** - Instructions for AI assistants
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - How to contribute to the project
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history and release notes

---

## Summary

The **Agentic Context Engineering framework** provides:

✅ **Self-improving AI agents** through three-role architecture
✅ **Production-ready** with cost tracking and observability
✅ **Flexible integration** with existing systems (browser-use, LangChain, etc.)
✅ **Token-efficient** knowledge storage (16-62% savings)
✅ **Scientifically validated** with benchmarking framework
✅ **Comprehensive testing** (11,337 lines of tests)
✅ **Extensive documentation** (100+ KB of guides)
✅ **30+ examples** for various use cases

### Quick Navigation

**Getting Started**:
1. Read [QUICK_START.md](./QUICK_START.md) (5 minutes)
2. Try `examples/simple_ace_example.py`
3. Explore [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for patterns

**Advanced Usage**:
1. Study [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md)
2. Review [COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md)
3. Understand [DATA_FLOW_GUIDE.md](./DATA_FLOW_GUIDE.md)

**Production Deployment**:
1. Install with `[observability]` extras
2. Use v2.1 prompts
3. Enable cost tracking
4. Follow deployment architectures above

---

**Need Help?**
- Documentation: `docs/` directory
- Examples: `examples/` directory
- Issues: https://github.com/kayba-ai/agentic-context-engine/issues
- Paper: arXiv:2510.04618

---

*Last updated: January 2025*
*Framework version: 0.5.1*
