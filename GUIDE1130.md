# ACE Framework Comprehensive Documentation Guide (November 2024)

## Table of Contents

1. [Overview](#overview)
2. [Architecture Summary](#architecture-summary)
3. [Module Reference](#module-reference)
4. [Data Flow](#data-flow)
5. [Usage Patterns](#usage-patterns)
6. [File Reference Index](#file-reference-index)
7. [Configuration Reference](#configuration-reference)
8. [Advanced Topics](#advanced-topics)

---

## Overview

The **Agentic Context Engine (ACE)** is a framework for building self-improving AI agents. Based on the research paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618), ACE enables agents to learn from execution feedback through three collaborative roles: Generator, Reflector, and Curator.

### Key Concepts

| Concept | Description | Location |
|---------|-------------|----------|
| **Playbook** | Structured context store containing learned strategies (bullets) | `ace/playbook.py` |
| **Delta Operations** | Incremental updates to the playbook (ADD, UPDATE, TAG, REMOVE) | `ace/delta.py` |
| **Three Roles** | Generator, Reflector, Curator - all using the same base LLM | `ace/roles.py` |
| **Adaptation Loops** | Offline (training) and Online (continuous) learning | `ace/adaptation.py` |
| **TOON Format** | Token-Oriented Object Notation for efficient LLM context | `playbook.as_prompt()` |

### Insight Levels

ACE operates at three insight levels based on what the Reflector analyzes:

| Level | Scope | Implementation | Use Case |
|-------|-------|----------------|----------|
| **Micro** | Single interaction + environment | `OfflineAdapter`, `OnlineAdapter` with `TaskEnvironment` | Q&A, classification with ground truth |
| **Meso** | Full agent run (reasoning trace) | `_learn_with_trace()` | Browser automation, multi-step agents |
| **Macro** | Cross-run analysis | Future enhancement | Pattern recognition across sessions |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACE FRAMEWORK ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │    Generator    │───►│    Environment  │───►│    Reflector    │         │
│  │  (produces      │    │  (evaluates     │    │  (analyzes      │         │
│  │   answers)      │    │   answers)      │    │   outcomes)     │         │
│  └────────┬────────┘    └─────────────────┘    └────────┬────────┘         │
│           │                                              │                  │
│           │                                              ▼                  │
│           │                                    ┌─────────────────┐         │
│           │                                    │     Curator     │         │
│           │                                    │  (updates       │         │
│           │                                    │   playbook)     │         │
│           │                                    └────────┬────────┘         │
│           │                                              │                  │
│           ▼                                              ▼                  │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                        PLAYBOOK                                 │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │        │
│  │  │ Bullet 1 │  │ Bullet 2 │  │ Bullet 3 │  │ Bullet N │       │        │
│  │  │ H:5 U:0  │  │ H:3 U:2  │  │ H:0 U:4  │  │ H:? U:?  │       │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Patterns

**1. Full ACE Pipeline** (for new agents)
- Sample → Generator → Environment → Reflector → Curator → Playbook
- Use for: Q&A tasks, classification, building agents from scratch

**2. Integration Pattern** (for existing agents)
- External agent executes → Reflector analyzes → Curator updates
- Use for: browser-use, LangChain, custom agents
- Reference: `ace/integrations/base.py:1-97`

---

## Module Reference

### Core Modules (`ace/`)

#### `playbook.py` - Knowledge Storage
**Purpose:** Structured context store for learned strategies

**Key Classes:**
- `Bullet` - Single playbook entry with helpful/harmful/neutral counters
- `Playbook` - Collection of bullets organized by sections
- `SimilarityDecision` - Record of deduplication decisions

**Key Methods:**
```python
playbook.add_bullet(section, content)      # Add new strategy
playbook.tag_bullet(bullet_id, "helpful")  # Update effectiveness counters
playbook.apply_delta(delta_batch)          # Apply Curator updates
playbook.as_prompt()                       # TOON format for LLM context
playbook.save_to_file("path.json")         # Persist to disk
Playbook.load_from_file("path.json")       # Load from disk
```

**File Reference:** `ace/playbook.py:1-428`

---

#### `delta.py` - Playbook Mutations
**Purpose:** Define incremental operations for playbook updates

**Operation Types:**
- `ADD` - Create new bullet
- `UPDATE` - Modify existing bullet content/metadata
- `TAG` - Update helpful/harmful/neutral counters
- `REMOVE` - Delete a bullet

**Key Classes:**
- `DeltaOperation` - Single mutation
- `DeltaBatch` - Bundle of operations with reasoning

**File Reference:** `ace/delta.py:1-86`

---

#### `roles.py` - The Three ACE Roles
**Purpose:** Generator, Reflector, and Curator implementations

**Generator** (`roles.py:138-253`)
- Produces answers using playbook strategies
- Auto-wraps with Instructor for structured outputs
- Extracts bullet citations from reasoning

**Reflector** (`roles.py:491-596`)
- Analyzes generator outputs and environment feedback
- Classifies bullet effectiveness (helpful/harmful/neutral)
- Extracts learnings for Curator

**Curator** (`roles.py:612-791`)
- Transforms reflections into playbook updates
- Generates delta operations
- Supports deduplication manager

**ReplayGenerator** (`roles.py:256-438`)
- Replays pre-recorded responses for offline training
- Supports dict-based and sample-based modes

**Output Classes:**
- `GeneratorOutput` - reasoning, final_answer, bullet_ids
- `ReflectorOutput` - reasoning, bullet_tags, extracted_learnings
- `CuratorOutput` - delta batch

**File Reference:** `ace/roles.py:1-805`

---

#### `adaptation.py` - Training Loops
**Purpose:** Orchestrate offline and online adaptation

**Key Classes:**

**`OfflineAdapter`** (`adaptation.py:536-723`)
- Multiple epochs over training samples
- Checkpoint saving support
- Async learning support
```python
adapter = OfflineAdapter(playbook, generator, reflector, curator)
results = adapter.run(samples, environment, epochs=3,
                      checkpoint_interval=10, checkpoint_dir="./checkpoints")
```

**`OnlineAdapter`** (`adaptation.py:726-849`)
- Sequential processing of streaming samples
- Continuous deployment learning

**Supporting Classes:**
- `Sample` - Task instance (question, context, ground_truth)
- `TaskEnvironment` - Abstract evaluation interface
- `SimpleEnvironment` - Built-in ground truth matching
- `EnvironmentResult` - Feedback from evaluation
- `AdapterStepResult` - Complete processing result

**File Reference:** `ace/adaptation.py:1-850`

---

#### `async_learning.py` - Parallel Learning
**Purpose:** Background learning with parallel Reflector execution

```
Sample 1 ──► Generator ──► Env ──► Reflector ─┐
Sample 2 ──► Generator ──► Env ──► Reflector ─┼──► [Queue] ──► Curator ──► Playbook
Sample 3 ──► Generator ──► Env ──► Reflector ─┘              (serialized)
           (parallel)           (parallel)
```

**Key Classes:**
- `LearningTask` - Input to Reflector (from main thread)
- `ReflectionResult` - Output from Reflector (goes to Curator queue)
- `ThreadSafePlaybook` - Thread-safe playbook wrapper
- `AsyncLearningPipeline` - Orchestrates parallel learning

**Benefits:**
- Generator returns immediately (fast response)
- Multiple Reflectors run concurrently (parallel LLM calls)
- Single Curator processes queue sequentially (safe playbook updates)

**File Reference:** `ace/async_learning.py:1-550`

---

#### `llm.py` - LLM Abstractions
**Purpose:** Base interfaces for LLM integration

**Key Classes:**
- `LLMClient` - Abstract base class
- `LLMResponse` - Container for LLM outputs
- `DummyLLMClient` - Testing/dry runs
- `TransformersLLMClient` - Local models via transformers

**File Reference:** `ace/llm.py:1-201`

---

#### `prompts_v2_1.py` - State-of-the-Art Prompts
**Purpose:** Enhanced prompt templates with MCP techniques

**Prompt Versions:**
| Version | Description | Recommendation |
|---------|-------------|----------------|
| v1.0 | Simple, minimal | Tutorials only |
| v2.0 | Enhanced structure | Development |
| v2.1 | **RECOMMENDED** | Production (+17% success) |

**Features in v2.1:**
- Quick reference summaries
- Imperative language (CRITICAL/MANDATORY/REQUIRED)
- Atomicity scoring for strategies
- Visual indicators (✓/✗/⚠️)
- Quality metrics

**Key Functions:**
- `PromptManager` - Version selection and domain-specific prompts
- `wrap_playbook_for_external_agent()` - Format playbook for integrations
- `validate_prompt_output_v2_1()` - Output validation with metrics

**File Reference:** `ace/prompts_v2_1.py:1-1622`

---

### LLM Providers (`ace/llm_providers/`)

#### `litellm_client.py` - Production LLM Client
**Purpose:** Unified access to 100+ LLM providers

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google (Gemini)
- Cohere, Meta, Mistral, and more

**Key Features:**
- Automatic API key detection from environment
- Fallback model support via Router
- Claude parameter conflict resolution (temperature vs top_p)
- Opik integration for cost tracking

**Configuration:**
```python
from ace.llm_providers import LiteLLMClient, LiteLLMConfig

# Simple
client = LiteLLMClient(model="gpt-4o-mini")

# With config
config = LiteLLMConfig(
    model="claude-3-sonnet-20240229",
    temperature=0.0,
    max_tokens=2048,
    fallbacks=["gpt-4o", "gpt-3.5-turbo"]
)
client = LiteLLMClient(config=config)
```

**File Reference:** `ace/llm_providers/litellm_client.py:1-728`

---

#### `instructor_client.py` - Structured Outputs
**Purpose:** Automatic Pydantic validation with intelligent retry

**Features:**
- Wraps LiteLLM with Instructor
- Automatic validation
- Feeds validation errors back to LLM

**Usage:**
```python
from ace.llm_providers.instructor_client import wrap_with_instructor

base_llm = LiteLLMClient(model="gpt-4")
llm = wrap_with_instructor(base_llm, max_retries=3)
```

**Note:** Generator, Reflector, and Curator auto-wrap with Instructor if not already wrapped.

**File Reference:** `ace/llm_providers/instructor_client.py:1-227`

---

### Integrations (`ace/integrations/`)

#### `litellm.py` - ACELiteLLM
**Purpose:** Quick-start learning agent

**Use When:**
- Q&A, classification, reasoning
- Prototyping ACE
- Simple tasks without external frameworks

```python
from ace import ACELiteLLM, Sample, SimpleEnvironment

agent = ACELiteLLM(model="gpt-4o-mini")

# Ask questions
answer = agent.ask("What is the capital of France?")

# Learn from samples
samples = [Sample(question="...", ground_truth="...")]
agent.learn(samples, SimpleEnvironment(), epochs=1)

# Provide feedback
agent.learn_from_feedback(feedback="correct")

# Save/load playbook
agent.save_playbook("agent.json")
agent.load_playbook("agent.json")
```

**File Reference:** `ace/integrations/litellm.py:1-503`

---

#### `browser_use.py` - ACEAgent
**Purpose:** Browser automation with learning

**Key Differences from Full ACE:**
- **No ACE Generator** - browser-use executes directly
- **Meso-level insight** - analyzes full execution trace
- **Playbook injection** - adds strategies to task context

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

agent = ACEAgent(
    llm=ChatBrowserUse(),      # Browser execution LLM
    ace_model="gpt-4o-mini"    # ACE learning LLM
)

history = await agent.run(task="Find top HN post")
agent.save_playbook("hn_expert.json")
```

**Async Learning:**
```python
agent = ACEAgent(llm=llm, async_learning=True)
await agent.run(task="Task 1")
await agent.run(task="Task 2")  # Learning from Task 1 happens in background
await agent.wait_for_learning()  # Wait before saving
agent.save_playbook("learned.json")
```

**File Reference:** `ace/integrations/browser_use.py:1-671`

---

#### `langchain.py` - ACELangChain
**Purpose:** Wrap LangChain chains/agents with learning

**File Reference:** `ace/integrations/langchain.py`

---

#### `base.py` - Integration Utilities
**Purpose:** Base classes and utilities for custom integrations

**Key Function:**
```python
from ace.integrations.base import wrap_playbook_context

# Inject learned strategies into external agent task
task_with_context = f"{task}\n\n{wrap_playbook_context(playbook)}"
```

**Integration Pattern (Three Steps):**
1. **INJECT**: Add playbook context to agent input
2. **EXECUTE**: External agent runs normally
3. **LEARN**: ACE analyzes results and updates playbook

**File Reference:** `ace/integrations/base.py:1-186`

---

### Deduplication (`ace/deduplication/`)

**Purpose:** Prevent duplicate strategies in playbook

**Key Components:**
- `DeduplicationConfig` - Configuration settings
- `DeduplicationManager` - Coordinate detection and operations
- `SimilarityDetector` - Compute embeddings, find similar pairs

**Consolidation Operations:**
- `MERGE` - Combine similar bullets
- `DELETE` - Soft-delete redundant bullets
- `KEEP` - Mark similar bullets as intentionally separate
- `UPDATE` - Refine content to differentiate

**Usage:**
```python
from ace.deduplication import DeduplicationConfig, DeduplicationManager

config = DeduplicationConfig(
    similarity_threshold=0.85,
    embedding_model="text-embedding-3-small"
)
curator = Curator(llm, dedup_manager=DeduplicationManager(config))
```

**File Reference:** `ace/deduplication/manager.py:1-186`

---

### Observability (`ace/observability/`)

**Purpose:** Production monitoring with Opik

**Features:**
- Automatic token and cost tracking
- Role performance logging
- Playbook evolution tracking
- Adaptation metrics

**Setup:**
```bash
pip install ace-framework[observability]
```

**Configuration:**
```python
from ace.observability import configure_opik

configure_opik(project_name="my-ace-project")
```

**File Reference:** `ace/observability/opik_integration.py:1-385`

---

### Benchmarks (`benchmarks/`)

**Purpose:** Scientific evaluation framework

**Components:**
- `base.py` - Base classes (BenchmarkConfig, DataLoader, BenchmarkEnvironment)
- `environments.py` - Task environment implementations
- `manager.py` - Benchmark execution
- `loaders/` - Dataset loaders (HuggingFace, etc.)

**Running Benchmarks:**
```bash
uv run python scripts/run_benchmark.py simple_qa --limit 50
uv run python scripts/run_benchmark.py simple_qa --limit 50 --compare  # Baseline vs ACE
```

**File Reference:** `benchmarks/base.py:1-135`

---

## Data Flow

### Full ACE Pipeline

```
┌──────────────┐
│    Sample    │
│  (question,  │
│  ground_truth)│
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   Generator  │────►│  Environment │
│  (playbook)  │     │  (evaluate)  │
└──────────────┘     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Reflector  │
                     │  (analyze)   │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Curator    │
                     │  (update)    │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Playbook   │
                     │ (apply_delta)│
                     └──────────────┘
```

### Integration Pattern

```
┌──────────────┐
│  Your Agent  │
│ (browser-use,│
│  LangChain)  │
└──────┬───────┘
       │
       │ inject playbook context
       ▼
┌──────────────┐
│   Execute    │
│    Task      │
└──────┬───────┘
       │
       │ execution results
       ▼
┌──────────────┐     ┌──────────────┐
│   Reflector  │────►│   Curator    │
│  (analyze    │     │  (update     │
│   results)   │     │   playbook)  │
└──────────────┘     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Playbook   │
                     └──────────────┘
```

---

## Usage Patterns

### Pattern 1: Simple Q&A Agent

```python
from ace import ACELiteLLM, Sample, SimpleEnvironment

# Create agent
agent = ACELiteLLM(model="gpt-4o-mini")

# Learn from examples
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="Capital of France?", ground_truth="Paris"),
]
agent.learn(samples, SimpleEnvironment(), epochs=1)

# Use learned knowledge
answer = agent.ask("What is 5+3?")
agent.save_playbook("math_agent.json")
```

### Pattern 2: Custom Environment

```python
from ace import TaskEnvironment, EnvironmentResult, Sample

class CustomEnvironment(TaskEnvironment):
    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        # Your evaluation logic
        is_correct = self.check_answer(
            generator_output.final_answer,
            sample.ground_truth
        )

        return EnvironmentResult(
            feedback="Correct!" if is_correct else f"Expected: {sample.ground_truth}",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if is_correct else 0.0}
        )
```

### Pattern 3: Browser Automation

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

# Create learning browser agent
agent = ACEAgent(
    llm=ChatBrowserUse(),
    ace_model="gpt-4o-mini",
    playbook_path="existing_knowledge.json"  # Optional
)

# Run tasks - agent learns from each
await agent.run(task="Navigate to example.com and click login")
await agent.run(task="Fill out the contact form")

# Save learned strategies
agent.save_playbook("browser_expert.json")
```

### Pattern 4: Async Learning

```python
from ace import ACELiteLLM, Sample, SimpleEnvironment

agent = ACELiteLLM(model="gpt-4o-mini")

# Enable async learning
results = agent.learn(
    samples, SimpleEnvironment(),
    async_learning=True,
    max_reflector_workers=3
)

# Results return immediately
for r in results:
    print(r.generator_output.final_answer)

# Wait when needed
agent.wait_for_learning(timeout=60.0)
agent.save_playbook("learned.json")
```

### Pattern 5: Custom Integration

```python
from ace import Playbook, Reflector, Curator
from ace.llm_providers import LiteLLMClient
from ace.roles import GeneratorOutput
from ace.integrations.base import wrap_playbook_context

# Setup
playbook = Playbook()
llm = LiteLLMClient(model="gpt-4o-mini")
reflector = Reflector(llm)
curator = Curator(llm)

# 1. INJECT: Add learned strategies
task = "Your task"
if playbook.bullets():
    task = f"{task}\n\n{wrap_playbook_context(playbook)}"

# 2. EXECUTE: Your agent runs
result = your_agent.execute(task)

# 3. LEARN: ACE learns from results
generator_output = GeneratorOutput(
    reasoning=f"Task: {task}",
    final_answer=result.output,
    bullet_ids=[],
    raw={"success": result.success}
)

reflection = reflector.reflect(
    question=task,
    generator_output=generator_output,
    playbook=playbook,
    feedback=f"Task {'succeeded' if result.success else 'failed'}"
)

curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context=f"task: {task}",
    progress="Executing task"
)

playbook.apply_delta(curator_output.delta)
playbook.save_to_file("learned.json")
```

---

## File Reference Index

### Core Framework

| File | Purpose | Key Exports |
|------|---------|-------------|
| `ace/__init__.py` | Package entry point | All public APIs |
| `ace/playbook.py` | Knowledge storage | `Bullet`, `Playbook` |
| `ace/delta.py` | Playbook mutations | `DeltaOperation`, `DeltaBatch` |
| `ace/roles.py` | Three ACE roles | `Generator`, `Reflector`, `Curator` |
| `ace/adaptation.py` | Training loops | `OfflineAdapter`, `OnlineAdapter` |
| `ace/async_learning.py` | Parallel learning | `AsyncLearningPipeline` |
| `ace/llm.py` | LLM abstractions | `LLMClient`, `DummyLLMClient` |
| `ace/prompts_v2_1.py` | Production prompts | `PromptManager` |
| `ace/features.py` | Dependency detection | `has_opik`, `has_litellm` |

### LLM Providers

| File | Purpose | Key Exports |
|------|---------|-------------|
| `ace/llm_providers/litellm_client.py` | 100+ providers | `LiteLLMClient` |
| `ace/llm_providers/instructor_client.py` | Structured outputs | `InstructorClient`, `wrap_with_instructor` |
| `ace/llm_providers/langchain_client.py` | LangChain support | `LangChainClient` |

### Integrations

| File | Purpose | Key Exports |
|------|---------|-------------|
| `ace/integrations/litellm.py` | Quick-start agent | `ACELiteLLM` |
| `ace/integrations/browser_use.py` | Browser automation | `ACEAgent` |
| `ace/integrations/langchain.py` | LangChain wrapper | `ACELangChain` |
| `ace/integrations/base.py` | Integration utilities | `wrap_playbook_context` |

### Deduplication

| File | Purpose | Key Exports |
|------|---------|-------------|
| `ace/deduplication/config.py` | Configuration | `DeduplicationConfig` |
| `ace/deduplication/manager.py` | Coordination | `DeduplicationManager` |
| `ace/deduplication/detector.py` | Similarity detection | `SimilarityDetector` |
| `ace/deduplication/operations.py` | Consolidation ops | `MergeOp`, `DeleteOp`, etc. |

### Observability

| File | Purpose | Key Exports |
|------|---------|-------------|
| `ace/observability/opik_integration.py` | Opik integration | `OpikIntegration` |
| `ace/observability/tracers.py` | Automatic tracing | `maybe_track` |

### Benchmarks

| File | Purpose | Key Exports |
|------|---------|-------------|
| `benchmarks/base.py` | Base classes | `BenchmarkConfig`, `BenchmarkEnvironment` |
| `benchmarks/environments.py` | Environments | Task-specific environments |
| `benchmarks/manager.py` | Execution | Benchmark runner |
| `benchmarks/loaders/huggingface.py` | Data loading | HuggingFace dataset loader |

### Examples

| Directory | Purpose |
|-----------|---------|
| `examples/litellm/` | ACELiteLLM examples |
| `examples/browser-use/` | Browser automation demos |
| `examples/langchain/` | LangChain integration |
| `examples/local-models/` | Ollama, LM Studio |
| `examples/prompts/` | Prompt engineering |
| `examples/helicone/` | Replay training |

### Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Development instructions |
| `README.md` | Project overview |
| `docs/QUICK_START.md` | Getting started |
| `docs/API_REFERENCE.md` | API documentation |
| `docs/INTEGRATION_GUIDE.md` | Adding ACE to existing agents |
| `docs/COMPLETE_GUIDE_TO_ACE.md` | Deep dive |
| `docs/PROMPTS.md` | Prompt versions |

---

## Configuration Reference

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `COHERE_API_KEY` | Cohere API key | - |
| `OPIK_DISABLED` | Disable Opik | `false` |
| `OPIK_API_KEY` | Opik API key | - |
| `BENCHMARK_CACHE_DIR` | Benchmark cache | HF default |

### LiteLLMConfig Options

```python
LiteLLMConfig(
    model="gpt-4o-mini",          # Model identifier
    api_key=None,                  # API key (auto-detect from env)
    api_base=None,                 # Custom API endpoint
    temperature=0.0,               # Sampling temperature
    max_tokens=2048,               # Max generation tokens
    top_p=None,                    # Nucleus sampling
    timeout=60,                    # Request timeout
    max_retries=3,                 # Retry attempts
    fallbacks=None,                # Fallback models
    track_cost=True,               # Enable cost tracking
    max_budget=None,               # Cost limit
    sampling_priority="temperature" # Claude param priority
)
```

### DeduplicationConfig Options

```python
DeduplicationConfig(
    enabled=True,                  # Enable deduplication
    similarity_threshold=0.85,     # Similarity threshold (0-1)
    embedding_model="text-embedding-3-small",
    min_pairs_to_report=1,         # Min similar pairs to report
)
```

### OfflineAdapter Options

```python
OfflineAdapter(
    playbook=playbook,
    generator=generator,
    reflector=reflector,
    curator=curator,
    max_refinement_rounds=1,       # Reflector refinement
    reflection_window=3,           # Recent reflections kept
    enable_observability=True,     # Opik tracking
    async_learning=False,          # Background learning
    max_reflector_workers=3,       # Parallel reflectors
    dedup_config=None,             # Deduplication config
)
```

---

## Advanced Topics

### Checkpoint Saving

```python
results = adapter.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=10,        # Save every 10 samples
    checkpoint_dir="./checkpoints"
)
# Creates: checkpoint_10.json, checkpoint_20.json, latest.json
```

### Fire-and-Forget Learning

```python
results = adapter.run(samples, environment, wait_for_learning=False)

# Use results immediately
for r in results:
    print(r.generator_output.final_answer)

# Check progress anytime
print(adapter.learning_stats)

# Wait when needed
adapter.wait_for_learning(timeout=60.0)
playbook.save_to_file("learned.json")
```

### Custom Prompts

```python
from ace.prompts_v2_1 import PromptManager

prompt_mgr = PromptManager(default_version="2.1")

# Domain-specific prompts
math_generator = Generator(llm,
    prompt_template=prompt_mgr.get_generator_prompt(domain="math"))
code_generator = Generator(llm,
    prompt_template=prompt_mgr.get_generator_prompt(domain="code"))

# Or fully custom
custom_prompt = """
Your custom prompt with {playbook}, {question}, {context}, {reflection}
Return JSON with: reasoning, bullet_ids, final_answer
"""
generator = Generator(llm, prompt_template=custom_prompt)
```

### Instructor Configuration

```python
import instructor
from ace.llm_providers.instructor_client import InstructorClient

# Use different Instructor modes
llm = InstructorClient(
    llm=base_llm,
    mode=instructor.Mode.JSON,     # OpenAI structured outputs
    # mode=instructor.Mode.MD_JSON, # Markdown JSON (default, works everywhere)
    max_retries=3
)
```

### Thread Safety

```python
from ace.async_learning import ThreadSafePlaybook

# Wrap playbook for concurrent access
ts_playbook = ThreadSafePlaybook(playbook)

# Lock-free reads (eventual consistency)
prompt = ts_playbook.as_prompt()
bullets = ts_playbook.bullets()

# Locked writes (thread-safe)
ts_playbook.apply_delta(delta_batch)
ts_playbook.tag_bullet(bullet_id, "helpful")
```

---

## Related Documentation

- **[CLAUDE.md](CLAUDE.md)** - Development commands and project structure
- **[docs/QUICK_START.md](docs/QUICK_START.md)** - Getting started guide
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation
- **[docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)** - Adding ACE to existing agents
- **[docs/COMPLETE_GUIDE_TO_ACE.md](docs/COMPLETE_GUIDE_TO_ACE.md)** - Deep dive into ACE
- **[docs/PROMPTS.md](docs/PROMPTS.md)** - Prompt version details
- **[examples/README.md](examples/README.md)** - Example index
- **[benchmarks/README.md](benchmarks/README.md)** - Benchmark documentation

---

*Generated: November 30, 2024*
*Version: ACE Framework v0.6.0*
