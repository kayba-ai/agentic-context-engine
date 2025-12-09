# ACE Framework Architecture

Detailed technical architecture of the Agentic Context Engineering (ACE) framework.

**Version:** 0.7.0

---

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Module Structure](#module-structure)
- [Key Design Patterns](#key-design-patterns)
- [Async Learning Architecture](#async-learning-architecture)
- [Deduplication System](#deduplication-system)
- [Observability Integration](#observability-integration)

---

## System Overview

ACE is a framework for building self-improving AI agents that learn from execution feedback through in-context learning.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ACE FRAMEWORK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ADAPTATION LAYER                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ OfflineACE  │  │  OnlineACE  │  │ Integrations (ACELiteLLM,   │  │   │
│  │  │ (batch)     │  │ (streaming) │  │ ACEAgent, ACELangChain)     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           ROLE LAYER                                  │   │
│  │  ┌─────────┐      ┌───────────┐      ┌──────────────┐                │   │
│  │  │  Agent  │ ───► │ Reflector │ ───► │ SkillManager │                │   │
│  │  └─────────┘      └───────────┘      └──────────────┘                │   │
│  │       │                 │                    │                        │   │
│  │       └─────────────────┼────────────────────┘                        │   │
│  │                         ▼                                             │   │
│  │                   ┌───────────┐                                       │   │
│  │                   │ Skillbook │                                       │   │
│  │                   └───────────┘                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        PROVIDER LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │   │
│  │  │ LiteLLMClient│  │TransformersClient│  │ LangChainClient       │   │   │
│  │  └─────────────┘  └─────────────────┘  └────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       SUPPORT SYSTEMS                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │ Deduplication│  │ Observability│  │ Async Learning Pipeline  │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### The Three Roles

ACE uses three specialized LLM roles that share the same base model but use different prompts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ROLE INTERACTION                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sample ──► ┌─────────┐     ┌─────────────┐     ┌────────────────┐      │
│             │  Agent  │────►│ Environment │────►│   Reflector    │      │
│             └─────────┘     └─────────────┘     └────────────────┘      │
│                  │                                      │               │
│                  │ Uses                                 │ Analyzes      │
│                  ▼                                      ▼               │
│             ┌─────────────────────────────────────────────┐             │
│             │               Skillbook                      │             │
│             │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│             │  │ Skill 1 │ │ Skill 2 │ │ Skill N │       │             │
│             │  └─────────┘ └─────────┘ └─────────┘       │             │
│             └─────────────────────────────────────────────┘             │
│                                      ▲                                   │
│                                      │ Updates                           │
│                              ┌───────────────┐                          │
│                              │ SkillManager  │◄──── Reflection          │
│                              └───────────────┘                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Agent

**Purpose:** Execute tasks using learned strategies from skillbook

**Location:** `ace/roles.py`

**Key Interface:**
```python
class Agent:
    def generate(
        self,
        question: str,
        context: str,
        skillbook: Skillbook,
        reflection: str | None = None
    ) -> AgentOutput
```

**Output Structure:**
```python
@dataclass
class AgentOutput:
    reasoning: str        # Step-by-step thought process
    final_answer: str     # The answer
    skill_ids: List[str]  # Cited skill IDs (auto-extracted)
    raw: Dict[str, Any]   # Raw LLM response
```

**Key Features:**
- Automatic Instructor wrapping for structured output
- Citation extraction using regex `[section-00001]`
- Support for reflection-based retry

#### Reflector

**Purpose:** Analyze execution outcomes and extract learnings

**Location:** `ace/roles.py`

**Key Interface:**
```python
class Reflector:
    def reflect(
        self,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: str | None = None,
        feedback: str | None = None
    ) -> ReflectorOutput
```

**Output Structure:**
```python
@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    extracted_learnings: List[ExtractedLearning]
    skill_tags: List[SkillTag]
    raw: Dict[str, Any]
```

**Key Features:**
- Creates skillbook excerpt with only cited skills
- Distinguishes outcome-based vs strategy-based learning
- Structured atomic learning extraction

#### SkillManager

**Purpose:** Transform reflections into skillbook updates

**Location:** `ace/roles.py`

**Key Interface:**
```python
class SkillManager:
    def update_skills(
        self,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str | None = None,
        progress: str | None = None
    ) -> SkillManagerOutput
```

**Output Structure:**
```python
@dataclass
class SkillManagerOutput:
    update: UpdateBatch
    consolidation_operations: List[ConsolidationOp] | None
    raw: Dict[str, Any]
```

**Key Features:**
- Optional deduplication integration
- Consolidation operations for skill merging
- Progress awareness for context

### Skillbook System

**Location:** `ace/skillbook.py`

```
┌─────────────────────────────────────────────────────────────────────┐
│                           SKILLBOOK                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Skills Dict                              │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │  "reasoning-00001" → Skill(                          │   │    │
│  │  │      id="reasoning-00001",                           │   │    │
│  │  │      section="reasoning",                            │   │    │
│  │  │      content="Break problems into smaller steps",    │   │    │
│  │  │      helpful=12, harmful=1, neutral=3,               │   │    │
│  │  │      status="active"                                 │   │    │
│  │  │  )                                                   │   │    │
│  │  ├──────────────────────────────────────────────────────┤   │    │
│  │  │  "extraction-00001" → Skill(...)                     │   │    │
│  │  ├──────────────────────────────────────────────────────┤   │    │
│  │  │  "navigation-00001" → Skill(...)                     │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Methods:                                                            │
│  - add_skill(section, content) → Skill                              │
│  - update_skill(skill_id, content) → Skill                          │
│  - tag_skill(skill_id, tag, increment) → Skill                      │
│  - remove_skill(skill_id, soft) → bool                              │
│  - as_prompt() → str (TOON format)                                  │
│  - apply_update(batch) → None                                       │
│  - save_to_file(path) / load_from_file(path)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Skill ID Format:** `{section_prefix}-{sequential_id}`
- Examples: `reasoning-00001`, `extraction-00042`, `navigation-00003`

**Serialization Formats:**
- `as_prompt()`: TOON format for LLM (16-62% token savings)
- `str(skillbook)`: Markdown format for debugging
- `save_to_file()`: JSON with full metadata

### Update Operations

**Location:** `ace/updates.py`

```python
@dataclass
class UpdateOperation:
    type: Literal["ADD", "UPDATE", "TAG", "REMOVE"]
    section: str | None = None
    content: str | None = None
    skill_id: str | None = None
    metadata: Dict[str, int] = None

@dataclass
class UpdateBatch:
    reasoning: str
    operations: List[UpdateOperation]
```

**Operation Types:**

| Type | Purpose | Required Fields |
|------|---------|-----------------|
| ADD | Create new skill | section, content |
| UPDATE | Modify skill content | skill_id, content |
| TAG | Update counters | skill_id, metadata |
| REMOVE | Delete skill | skill_id |

---

## Data Flow

### Full ACE Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            FULL ACE PIPELINE                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. INPUT                                                                   │
│     Sample(question, context, ground_truth)                                │
│                         │                                                   │
│                         ▼                                                   │
│  2. AGENT EXECUTION                                                        │
│     ┌─────────────────────────────────────────┐                            │
│     │ Agent.generate(question, skillbook)     │                            │
│     │   → Formats prompt with skillbook       │                            │
│     │   → Calls LLM                           │                            │
│     │   → Parses AgentOutput                  │                            │
│     │   → Extracts cited skill_ids            │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  3. ENVIRONMENT EVALUATION                                                 │
│     ┌─────────────────────────────────────────┐                            │
│     │ Environment.evaluate(sample, output)    │                            │
│     │   → Compares to ground_truth            │                            │
│     │   → Returns EnvironmentResult           │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  4. REFLECTION                                                             │
│     ┌─────────────────────────────────────────┐                            │
│     │ Reflector.reflect(question, output,     │                            │
│     │                   skillbook, feedback)  │                            │
│     │   → Creates cited skills excerpt        │                            │
│     │   → Analyzes outcome                    │                            │
│     │   → Extracts learnings                  │                            │
│     │   → Tags skills as helpful/harmful      │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  5. SKILL MANAGEMENT                                                       │
│     ┌─────────────────────────────────────────┐                            │
│     │ SkillManager.update_skills(reflection)  │                            │
│     │   → Generates UpdateBatch               │                            │
│     │   → Optional: deduplication check       │                            │
│     │   → Returns SkillManagerOutput          │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  6. SKILLBOOK UPDATE                                                       │
│     ┌─────────────────────────────────────────┐                            │
│     │ skillbook.apply_update(batch)           │                            │
│     │   → ADD: Creates new skills             │                            │
│     │   → UPDATE: Modifies existing           │                            │
│     │   → TAG: Updates counters               │                            │
│     │   → REMOVE: Deletes skills              │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  7. OUTPUT                                                                 │
│     ACEStepResult(sample, agent_output, environment_result,                │
│                   reflection, skill_manager_output)                        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Integration Pattern

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          INTEGRATION PATTERN                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. INJECT (optional)                                                      │
│     ┌─────────────────────────────────────────┐                            │
│     │ wrap_skillbook_context(skillbook)       │                            │
│     │   → Formats skillbook for external agent │                            │
│     │   → Prepends to task description         │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  2. EXECUTE (external agent)                                               │
│     ┌─────────────────────────────────────────┐                            │
│     │ your_agent.execute(enhanced_task)       │                            │
│     │   → Browser-use, LangChain, custom      │                            │
│     │   → Returns execution result + trace    │                            │
│     └─────────────────────────────────────────┘                            │
│                         │                                                   │
│                         ▼                                                   │
│  3. LEARN                                                                  │
│     ┌─────────────────────────────────────────┐                            │
│     │ Create AgentOutput adapter              │                            │
│     │ Reflector.reflect(...)                  │                            │
│     │ SkillManager.update_skills(...)         │                            │
│     │ skillbook.apply_update(...)             │                            │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
ace/
├── __init__.py              # Package exports
├── adaptation.py            # OfflineACE, OnlineACE orchestration
├── async_learning.py        # Async learning pipeline
├── features.py              # Optional dependency detection
├── llm.py                   # LLMClient interface, DummyLLMClient
├── prompts.py               # v1.0 prompts (simple)
├── prompts_v2.py            # v2.0 prompts (deprecated)
├── prompts_v2_1.py          # v2.1 prompts (recommended)
├── roles.py                 # Agent, Reflector, SkillManager
├── skillbook.py             # Skill, Skillbook classes
├── updates.py               # UpdateOperation, UpdateBatch
│
├── llm_providers/           # LLM client implementations
│   ├── instructor_client.py # Instructor wrapper
│   ├── litellm_client.py    # LiteLLM (100+ providers)
│   └── langchain_client.py  # LangChain integration
│
├── integrations/            # External framework wrappers
│   ├── base.py              # Integration utilities
│   ├── browser_use.py       # ACEAgent
│   ├── langchain.py         # ACELangChain
│   └── litellm.py           # ACELiteLLM
│
├── deduplication/           # Skill deduplication
│   ├── config.py            # DeduplicationConfig
│   ├── detector.py          # SimilarityDetector
│   ├── manager.py           # DeduplicationManager
│   ├── operations.py        # Consolidation operations
│   └── prompts.py           # Dedup prompt helpers
│
└── observability/           # Production monitoring
    ├── opik_integration.py  # Opik integration
    └── tracers.py           # @maybe_track decorator

benchmarks/                  # Benchmark framework
├── base.py                  # Base classes
├── environments.py          # Task environments
├── manager.py               # Benchmark orchestration
├── processors.py            # Data processing
└── loaders/                 # Dataset loaders

examples/                    # 35+ example scripts
├── litellm/                 # Basic ACE examples
├── langchain/               # LangChain integration
├── browser-use/             # Browser automation
├── local-models/            # Ollama, LM Studio
└── prompts/                 # Prompt engineering

tests/                       # Test suite (19 modules)
├── test_adaptation.py
├── test_async_learning.py
├── test_roles.py
├── test_skillbook.py
└── integrations/
```

---

## Key Design Patterns

### 1. Role Abstraction

All roles inherit common LLM interaction patterns:

```python
class Role:
    def __init__(self, llm: LLMClient, prompt_template: str, max_retries: int = 3):
        self.llm = wrap_with_instructor(llm)  # Auto-wrap for structured output
        self.prompt_template = prompt_template
        self.max_retries = max_retries
```

### 2. Instructor Integration

Automatic Pydantic validation for LLM outputs:

```python
# Roles automatically wrap LLM with Instructor
llm = wrap_with_instructor(llm) if not is_instructor_wrapped(llm) else llm

# Structured output with validation
response = llm.complete_structured(prompt, response_model=AgentOutput)
```

### 3. Defensive Parsing

UpdateOperations use defensive parsing:

```python
@classmethod
def from_json(cls, data: Dict) -> "UpdateOperation":
    # Filter invalid tag names
    if data.get("type") == "TAG":
        metadata = {k: v for k, v in data.get("metadata", {}).items()
                    if k in ("helpful", "harmful", "neutral")}
    # Type coercion, default values, etc.
```

### 4. Graceful Degradation

Optional features checked at runtime:

```python
from ace.features import has_opik, has_litellm

if has_opik():
    # Use Opik tracing
else:
    # Fall back to no-op
```

### 5. Thread-Safe Skillbook

For async learning:

```python
class ThreadSafeSkillbook:
    def __init__(self, skillbook: Skillbook):
        self._skillbook = skillbook
        self._lock = RWLock()

    def read(self) -> Skillbook:
        # Lock-free read for eventual consistency
        return self._skillbook

    def write(self, update: UpdateBatch):
        with self._lock.write():
            self._skillbook.apply_update(update)
```

---

## Async Learning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASYNC LEARNING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PRODUCER (Main Thread)                        │    │
│  │  ┌─────────┐    ┌─────────────┐    ┌────────────────────────────┐   │    │
│  │  │ Sample  │───►│    Agent    │───►│ Submit to Reflector Queue  │   │    │
│  │  └─────────┘    └─────────────┘    └────────────────────────────┘   │    │
│  │                                              │                       │    │
│  │  Returns immediately ◄──────────────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   REFLECTOR WORKERS (N Threads)                      │    │
│  │                                                                       │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │ Reflector #1  │  │ Reflector #2  │  │ Reflector #3  │            │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │    │
│  │          │                  │                  │                     │    │
│  │          └──────────────────┼──────────────────┘                     │    │
│  │                             ▼                                        │    │
│  │                    ┌────────────────┐                                │    │
│  │                    │ Curation Queue │                                │    │
│  │                    └────────────────┘                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   SKILLMANAGER WORKER (1 Thread)                     │    │
│  │                                                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                     SkillManager                              │   │    │
│  │  │  - Processes queue sequentially                               │   │    │
│  │  │  - Updates skillbook atomically                               │   │    │
│  │  │  - Handles deduplication                                      │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                              │                                       │    │
│  │                              ▼                                       │    │
│  │                    ┌────────────────┐                                │    │
│  │                    │   Skillbook    │                                │    │
│  │                    │ (Thread-Safe)  │                                │    │
│  │                    └────────────────┘                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Data Structures:**

```python
@dataclass
class LearningTask:
    sample: Sample
    agent_output: AgentOutput
    environment_result: EnvironmentResult
    epoch: int
    step_index: int
    timestamp: float

@dataclass
class ReflectionResult:
    task: LearningTask
    reflection: ReflectorOutput
    timestamp: float
```

**Why This Architecture:**
- **Reflector parallelization**: Safe (read-only, no writes)
- **SkillManager serialization**: Required (writes to skillbook)
- **3x speedup**: Multiple Reflector LLM calls concurrent
- **Eventual consistency**: Agent uses latest available skillbook

---

## Deduplication System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEDUPLICATION SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      DeduplicationConfig                               │  │
│  │  enabled: bool = True                                                  │  │
│  │  embedding_model: str = "text-embedding-3-small"                      │  │
│  │  similarity_threshold: float = 0.85                                   │  │
│  │  within_section_only: bool = True                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      SimilarityDetector                                │  │
│  │  1. Compute embeddings for skills (lazy, cached)                      │  │
│  │  2. Calculate cosine similarity between pairs                         │  │
│  │  3. Return pairs above threshold                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     DeduplicationManager                               │  │
│  │  1. get_similarity_report() → Formatted report for SkillManager       │  │
│  │  2. parse_consolidation_operations() → Extract from LLM response      │  │
│  │  3. apply_operations() → Execute MERGE/DELETE/KEEP/UPDATE             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Consolidation Operations:                                                   │
│  ┌────────┬──────────────────────────────────────────────────────────────┐  │
│  │ MERGE  │ Combine content, sum counters, soft-delete redundant         │  │
│  │ DELETE │ Soft-delete redundant skill                                  │  │
│  │ KEEP   │ Store decision in skillbook (avoid re-asking)                │  │
│  │ UPDATE │ Modify content to differentiate                              │  │
│  └────────┴──────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Observability Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Opik Integration                                │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │ @maybe_track Decorator                                          │  │  │
│  │  │   - Wraps Agent.generate()                                      │  │  │
│  │  │   - Wraps Reflector.reflect()                                   │  │  │
│  │  │   - Wraps SkillManager.update_skills()                          │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  Features:                                                             │  │
│  │  - Automatic span creation for role interactions                      │  │
│  │  - Token usage tracking per LLM call                                  │  │
│  │  - Cost estimation with provider pricing                              │  │
│  │  - Graceful degradation if Opik not installed                        │  │
│  │                                                                        │  │
│  │  Dashboard: https://www.comet.com/opik                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       Feature Detection                                │  │
│  │                                                                        │  │
│  │  from ace.features import (                                           │  │
│  │      has_opik,                                                        │  │
│  │      has_litellm,                                                     │  │
│  │      has_langchain,                                                   │  │
│  │      get_available_features                                           │  │
│  │  )                                                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Extension Points

### Custom LLM Client

```python
from ace.llm import LLMClient, LLMResponse

class CustomLLMClient(LLMClient):
    def complete(self, prompt: str) -> LLMResponse:
        # Your implementation
        response = your_api.generate(prompt)
        return LLMResponse(text=response.text, raw=response.raw)
```

### Custom Task Environment

```python
from ace import TaskEnvironment, EnvironmentResult

class CustomEnvironment(TaskEnvironment):
    def evaluate(self, sample: Sample, output: AgentOutput) -> EnvironmentResult:
        # Your evaluation logic
        correct = your_evaluation(sample, output)
        return EnvironmentResult(
            feedback="Pass" if correct else "Fail",
            ground_truth=sample.ground_truth,
            metrics={"score": 1.0 if correct else 0.0}
        )
```

### Custom Prompts

```python
CUSTOM_AGENT_PROMPT = """
# Your Domain-Specific Agent

Available Strategies:
{skillbook}

Task: {question}
Context: {context}
Previous Reflection: {reflection}

Output JSON with: reasoning, final_answer, skill_ids
"""

agent = Agent(llm, prompt_template=CUSTOM_AGENT_PROMPT)
```

---

## See Also

- [Complete Guide](COMPLETE_GUIDE_TO_ACE.md) - ACE concepts
- [API Reference](API_REFERENCE.md) - Complete API
- [Integration Guide](INTEGRATION_GUIDE.md) - Custom integrations
- [Prompt Guide](PROMPTS.md) - Prompt customization

---

**Last Updated:** December 2025 | **Version:** 0.7.0
