# Agentic Context Engineering: Complete Guide

**How ACE enables AI agents to improve through in-context learning instead of fine-tuning.**

---

## Table of Contents

- [What is ACE?](#what-is-agentic-context-engineering)
- [The Core Problem](#the-core-problem)
- [How ACE Works](#how-ace-works)
- [Key Technical Innovations](#key-technical-innovations)
- [Performance Results](#performance-results)
- [When to Use ACE](#when-to-use-ace)
- [ACE vs Other Approaches](#ace-vs-other-approaches)
- [Getting Started](#getting-started)

---

## What is Agentic Context Engineering?

Agentic Context Engineering (ACE) is a framework introduced by researchers at Stanford University and SambaNova Systems that enables AI agents to improve performance by dynamically curating their own context through execution feedback.

**Key Innovation:** Instead of updating model weights through expensive fine-tuning cycles, ACE treats context as a living "skillbook" that evolves based on what strategies actually work in practice.

**Research Paper:** [Agentic Context Engineering (arXiv:2510.04618)](https://arxiv.org/abs/2510.04618)

---

## The Core Problem

Modern AI agents face a fundamental limitation: they don't learn from execution history. When an agent makes a mistake, developers must manually intervene—editing prompts, adjusting parameters, or fine-tuning the model.

**Traditional approaches have major drawbacks:**

| Problem | Impact |
|---------|--------|
| **Repetitive failures** | Agents lack institutional memory |
| **Manual intervention** | Doesn't scale as complexity increases |
| **Expensive adaptation** | Fine-tuning costs $10,000+ per cycle and takes weeks |
| **Black box improvement** | Unclear what changed or why |

---

## How ACE Works

ACE introduces a three-role architecture where specialized LLM instances collaborate to build and maintain a dynamic knowledge base called the "skillbook."

### The Three Roles

All three roles use the **same base LLM** with different specialized prompts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ACE LEARNING LOOP                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Query ──► Agent ──► Environment ──► Reflector ──► SkillManager        │
│               │           │              │              │                │
│               ▼           ▼              ▼              ▼                │
│           Executes    Evaluates      Analyzes       Updates             │
│           using       answer &       outcome &      skillbook           │
│           skillbook   provides       identifies     with new            │
│                       feedback       learnings      strategies          │
│                                                                          │
│                         ┌────────────┐                                   │
│                         │  Skillbook │◄──────────────────────────────────┤
│                         │ (Evolving) │                                   │
│                         └────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1. Agent - Task Execution

The Agent performs the actual work using strategies from the skillbook.

```python
from ace import Agent, LiteLLMClient, Skillbook

client = LiteLLMClient(model="gpt-4o-mini")
agent = Agent(client)

output = agent.generate(
    question="What is 2+2?",
    context="Show your work",
    skillbook=skillbook
)

print(output.reasoning)      # Step-by-step thought process
print(output.final_answer)   # "4"
print(output.skill_ids)      # Skills cited: ["math-00001"]
```

**Key Capabilities:**
- Operates like a traditional agent but with access to learned knowledge
- Cites skillbook strategies in reasoning using `[section-00001]` notation
- Automatic citation extraction for tracking

#### 2. Reflector - Performance Analysis

The Reflector analyzes execution outcomes without human supervision.

```python
from ace import Reflector

reflector = Reflector(client)

reflection = reflector.reflect(
    question="What is 2+2?",
    agent_output=output,
    skillbook=skillbook,
    ground_truth="4",
    feedback="Correct!"
)

print(reflection.key_insight)         # Main lesson learned
print(reflection.error_identification) # What went wrong (if any)
print(reflection.skill_tags)          # Skills marked helpful/harmful
```

**Key Capabilities:**
- Identifies which strategies worked, which failed, and why
- Generates structured insights for skillbook updates
- Tags cited skills as helpful, harmful, or neutral

#### 3. SkillManager - Knowledge Management

The SkillManager transforms reflections into actionable skillbook updates.

```python
from ace import SkillManager

skill_manager = SkillManager(client)

output = skill_manager.update_skills(
    reflection=reflection,
    skillbook=skillbook,
    question_context="Math problems"
)

# Apply updates
skillbook.apply_update(output.update)
```

**Update Operations:**
- **ADD**: Insert new skill to skillbook
- **UPDATE**: Modify existing skill content
- **TAG**: Update helpful/harmful/neutral counts
- **REMOVE**: Delete ineffective skills

### The Skillbook

The skillbook stores learned strategies as structured "skills" with metadata:

```json
{
  "id": "reasoning-00042",
  "section": "reasoning",
  "content": "When querying financial data, filter by date range first to reduce result set size",
  "helpful": 12,
  "harmful": 1,
  "neutral": 3,
  "status": "active"
}
```

**Key Properties:**
- **TOON Format**: Token-optimized representation saves 16-62% tokens
- **Section Organization**: Skills grouped by category (reasoning, extraction, etc.)
- **Success Tracking**: helpful/harmful/neutral counters for each skill
- **Persistence**: Save/load to JSON for reuse across sessions

### The Learning Cycle

1. **Execution:** Agent receives a task and retrieves relevant skillbook skills
2. **Action:** Agent executes using retrieved strategies (citing them in reasoning)
3. **Evaluation:** Environment provides feedback (correct/incorrect, metrics)
4. **Reflection:** Reflector analyzes the execution outcome
5. **Curation:** SkillManager updates the skillbook with update operations
6. **Iteration:** Process repeats, skillbook grows more refined over time

---

## Key Technical Innovations

### Update Operations (Preventing Context Collapse)

A critical insight from the ACE paper: LLMs exhibit **brevity bias** when asked to rewrite context. They compress information, losing crucial details.

ACE solves this through **update operations**—incremental modifications that never ask the LLM to regenerate entire contexts:

```python
from ace.updates import UpdateOperation, UpdateBatch

# Example batch of operations
batch = UpdateBatch(
    reasoning="Added new math strategy, tagged existing as helpful",
    operations=[
        UpdateOperation(type="ADD", section="math", content="Always check units"),
        UpdateOperation(type="TAG", skill_id="math-00001", metadata={"helpful": 1})
    ]
)

skillbook.apply_update(batch)
```

This preserves the exact wording and structure of learned knowledge.

### TOON Format (Token Efficiency)

ACE uses Token-Oriented Object Notation (TOON) for skillbook serialization:

```python
# TOON format (for LLM consumption) - 16-62% token savings
skillbook.as_prompt()

# Markdown format (for human debugging)
str(skillbook)
```

### Skill Deduplication

As agents learn, they may generate similar but differently-worded strategies. ACE prevents skillbook bloat through embedding-based deduplication:

```python
from ace import DeduplicationConfig

config = DeduplicationConfig(
    enabled=True,
    similarity_threshold=0.85,
    embedding_model="text-embedding-3-small"
)

# SkillManager can now output consolidation operations
# MERGE: Combine similar skills
# DELETE: Remove redundant skill
# KEEP: Mark as intentionally separate
# UPDATE: Refine to differentiate
```

### Insight Levels

The Reflector can analyze execution at three different levels of scope:

| Level | Scope | What's Analyzed | Learning Quality |
|-------|-------|-----------------|------------------|
| **Micro** | Single interaction + environment | Request → response → ground truth/feedback | Learns from correctness |
| **Meso** | Full agent run | Reasoning traces (thoughts, tool calls, observations) | Learns from execution patterns |
| **Macro** | Cross-run analysis | Patterns across multiple executions | Comprehensive (future) |

**Micro-level insights** come from the full ACE adaptation loop with environment feedback and ground truth. The Reflector knows whether the answer was correct and learns from that evaluation. Used by OfflineACE and OnlineACE.

**Meso-level insights** come from full agent runs with intermediate steps—the agent's thoughts, tool calls, and observations—but without external ground truth. The Reflector learns from the execution patterns themselves. Used by integration wrappers like ACEAgent (browser-use) and ACELangChain.

### Async Learning Mode

For latency-sensitive applications, ACE supports async learning where the Agent returns immediately while Reflector and SkillManager process in the background:

```
┌───────────────────────────────────────────────────────────────────────┐
│                       ASYNC LEARNING PIPELINE                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Sample 1 ──► Agent ──► Env ──► Reflector ─┐                         │
│  Sample 2 ──► Agent ──► Env ──► Reflector ─┼──► Queue ──► SkillManager│
│  Sample 3 ──► Agent ──► Env ──► Reflector ─┘           (serialized)   │
│             (parallel)        (parallel)                              │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Why this architecture:**
- **Parallel Reflectors**: Safe to parallelize (read-only analysis, no skillbook writes)
- **Serialized SkillManager**: Must be sequential (writes to skillbook, handles deduplication)
- **3x faster learning**: Reflector LLM calls run concurrently

**Usage:**
```python
from ace import OfflineACE

adapter = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,        # Enable async mode
    max_reflector_workers=3     # Parallel Reflector threads
)

results = adapter.run(samples, environment)  # Fast - learning in background

# Control methods
adapter.learning_stats        # Check progress
adapter.wait_for_learning()   # Block until complete
adapter.stop_async_learning() # Shutdown pipeline
```

### Checkpoint Saving

Save skillbook during long training runs:

```python
results = adapter.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=10,      # Save every 10 samples
    checkpoint_dir="./checkpoints"
)
# Creates: checkpoint_10.json, checkpoint_20.json, ..., latest.json
```

**Use Cases:**
- Resume training after interruption
- Compare skillbook evolution over time
- Early stopping based on validation metrics

---

## Performance Results

The Stanford team evaluated ACE across multiple benchmarks:

### AppWorld Agent Benchmark
- **+17.1 percentage points** improvement vs base LLM (≈40% relative improvement)
- Tested on complex multi-step tasks requiring tool use and reasoning

### Finance Domain (FiNER)
- **+8.6 percentage points** improvement on financial reasoning tasks

### Adaptation Efficiency
- **86.9% lower adaptation latency** compared to existing context-adaptation methods

### Browser Automation (ACE Framework Results)
- **29.8% fewer steps** (57.2 vs 81.5)
- **49.0% token reduction** (595k vs 1,166k)
- **42.6% cost reduction** (including ACE overhead)

**Key Insight:** Performance improvements compound over time. As the skillbook grows, agents make fewer mistakes on similar tasks, creating a positive feedback loop.

---

## When to Use ACE

### Best Fit Use Cases

| Use Case | Why ACE Works |
|----------|---------------|
| **Software Development Agents** | Learn project-specific patterns (naming, error handling), build bug solutions |
| **Customer Support Automation** | Learn escalation criteria, communication patterns, edge cases |
| **Data Analysis Agents** | Learn efficient query patterns, visualization choices, baseline expectations |
| **Research Assistants** | Learn search strategies, citation patterns, reliable sources |
| **Browser Automation** | Learn navigation patterns, form filling strategies, error recovery |

### When NOT to Use ACE

ACE may not be the right fit when:

- **Single-use tasks:** No benefit from learning if task never repeats
- **Perfect first-time execution required:** ACE learns through iteration
- **Purely factual retrieval:** Traditional RAG may be more appropriate
- **Real-time constraints:** Learning adds latency (though async mode helps)

---

## ACE vs Other Approaches

### vs Fine-Tuning

| Aspect | ACE | Fine-Tuning |
|--------|-----|-------------|
| **Speed** | Immediate (after single execution) | Days to weeks |
| **Cost** | Inference only | $10K+ per iteration |
| **Interpretability** | Readable skillbook | Black box weights |
| **Reversibility** | Edit/remove strategies easily | Requires retraining |
| **Data Requirements** | None (learns from feedback) | Large training datasets |

### vs RAG

| Aspect | ACE | RAG |
|--------|-----|-----|
| **Knowledge Source** | Learned from execution | Static documents |
| **Update Mechanism** | Autonomous curation | Manual updates |
| **Content Type** | Strategies, patterns | Facts, references |
| **Optimization** | Self-improving | Requires query tuning |

### vs Prompt Engineering

| Aspect | ACE | Manual Prompting |
|--------|-----|------------------|
| **Scalability** | Automatic improvement | Manual effort |
| **Adaptation** | Task-specific learning | Generic prompts |
| **Maintenance** | Self-updating | Ongoing manual work |
| **Transparency** | Visible in skillbook | Hidden in prompts |

---

## Architecture Patterns

### Full ACE Pipeline

For building new agents from scratch:

```
Sample → Agent → Environment → Reflector → SkillManager → Skillbook
```

**Use when:** Building new agent, Q&A tasks, classification, reasoning tasks

**Components:** Skillbook + Agent + Reflector + SkillManager

```python
from ace import OfflineACE, Agent, Reflector, SkillManager

adapter = OfflineACE(
    agent=Agent(llm),
    reflector=Reflector(llm),
    skill_manager=SkillManager(llm)
)

results = adapter.run(samples, environment, epochs=3)
```

### Integration Pattern

For wrapping existing agents:

```
INJECT → EXECUTE (external agent) → LEARN
```

**Use when:** Wrapping browser-use, LangChain, CrewAI, or custom agents

**Components:** Skillbook + Reflector + SkillManager (NO ACE Agent)

```python
from ace import ACEAgent, ACELangChain

# Browser automation
agent = ACEAgent(llm=ChatBrowserUse())
await agent.run(task="Find top HN post")

# LangChain
ace_chain = ACELangChain(runnable=your_chain)
result = ace_chain.invoke({"question": "Your task"})
```

See [Integration Guide](INTEGRATION_GUIDE.md) for complete patterns.

---

## Prompt Versions

ACE includes multiple prompt versions with different trade-offs:

| Version | Status | Use Case | Performance |
|---------|--------|----------|-------------|
| v1.0 | Stable | Tutorials, learning | Baseline |
| v2.0 | **Deprecated** | Don't use | +12% |
| v2.1 | **Recommended** | Production | **+17%** |

```python
from ace.prompts_v2_1 import PromptManager

mgr = PromptManager()
agent = Agent(llm, prompt_template=mgr.get_agent_prompt())
reflector = Reflector(llm, prompt_template=mgr.get_reflector_prompt())
skill_manager = SkillManager(llm, prompt_template=mgr.get_skill_manager_prompt())
```

---

## Getting Started

### Quick Start (5 minutes)

```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="gpt-4o-mini")
answer = agent.ask("What is the capital of France?")
agent.save_skillbook("learned.json")
```

### Resources

| Resource | Description |
|----------|-------------|
| **[Quick Start Guide](QUICK_START.md)** | Get running in 5 minutes |
| **[Integration Guide](INTEGRATION_GUIDE.md)** | Add ACE to existing agents |
| **[API Reference](API_REFERENCE.md)** | Complete API documentation |
| **[Examples](../examples/)** | Ready-to-run code examples |
| **[Architecture](ARCHITECTURE.md)** | Detailed system internals |

### Community

- **Discord:** [Join our community](https://discord.gg/mqCqH7sTyK)
- **GitHub:** [kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine)

---

## Additional Resources

### Research
- [Original ACE Paper (arXiv)](https://arxiv.org/abs/2510.04618)
- [Dynamic Cheatsheet Paper](https://arxiv.org/abs/2504.07952) - Related work

### Citation

If you use ACE in your research, please cite:

```bibtex
@article{zhang2024ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang et al.},
  journal={arXiv:2510.04618},
  year={2024}
}
```

---

**Last Updated:** December 2025 | **Version:** 0.7.0
