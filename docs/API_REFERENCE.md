# API Reference

Complete API documentation for ACE Framework v0.7.0.

---

## Table of Contents

- [Core Components](#core-components)
  - [Agent](#agent)
  - [Reflector](#reflector)
  - [SkillManager](#skillmanager)
- [Skillbook System](#skillbook-system)
  - [Skillbook](#skillbook)
  - [Skill](#skill)
- [Adaptation](#adaptation)
  - [OfflineACE](#offlineace)
  - [OnlineACE](#onlineace)
  - [Sample](#sample)
  - [TaskEnvironment](#taskenvironment)
- [Integrations](#integrations)
  - [ACELiteLLM](#acelitellm)
  - [ACEAgent](#aceagent)
  - [ACELangChain](#acelangchain)
- [LLM Clients](#llm-clients)
  - [LiteLLMClient](#litellmclient)
- [Updates](#updates)
  - [UpdateOperation](#updateoperation)
  - [UpdateBatch](#updatebatch)
- [Deduplication](#deduplication)
  - [DeduplicationConfig](#deduplicationconfig)
- [Prompts](#prompts)
  - [PromptManager](#promptmanager)
- [Types](#types)

---

## Core Components

### Agent

The Agent produces answers using the current skillbook of strategies.

```python
from ace import Agent, LiteLLMClient

client = LiteLLMClient(model="gpt-4o-mini")
agent = Agent(client)

output = agent.generate(
    question="What is 2+2?",
    context="Show your work",
    skillbook=skillbook,
    reflection=None  # Optional: reflection from previous attempt
)
```

#### Constructor

```python
Agent(
    llm: LLMClient,
    prompt_template: str = None,  # Custom prompt (default: v1.0)
    max_retries: int = 3          # JSON parse retry attempts
)
```

#### Methods

**`generate(question, context, skillbook, reflection=None) -> AgentOutput`**

Generate an answer using the skillbook.

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | str | The question to answer |
| `context` | str | Additional context |
| `skillbook` | Skillbook | Strategies to use |
| `reflection` | str \| None | Prior reflection for retry |

**Returns:** `AgentOutput`

#### AgentOutput

```python
@dataclass
class AgentOutput:
    reasoning: str           # Step-by-step thought process
    final_answer: str        # The generated answer
    skill_ids: List[str]     # Skills cited in reasoning
    raw: Dict[str, Any]      # Raw LLM response
```

---

### Reflector

The Reflector analyzes agent outputs to extract lessons and classify skill effectiveness.

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
```

#### Constructor

```python
Reflector(
    llm: LLMClient,
    prompt_template: str = None,  # Custom prompt
    max_retries: int = 3
)
```

#### Methods

**`reflect(question, agent_output, skillbook, ground_truth=None, feedback=None) -> ReflectorOutput`**

Analyze agent output and extract learnings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | str | Original question |
| `agent_output` | AgentOutput | Agent's response |
| `skillbook` | Skillbook | Current skillbook |
| `ground_truth` | str \| None | Expected answer |
| `feedback` | str \| None | Environment feedback |

**Returns:** `ReflectorOutput`

#### ReflectorOutput

```python
@dataclass
class ReflectorOutput:
    reasoning: str                    # Overall analysis
    error_identification: str         # What went wrong (if applicable)
    root_cause_analysis: str          # Why errors occurred
    correct_approach: str             # What should have been done
    key_insight: str                  # Main lesson learned
    extracted_learnings: List[ExtractedLearning]  # Atomic learnings
    skill_tags: List[SkillTag]        # Skill effectiveness tags
    raw: Dict[str, Any]
```

#### SkillTag

```python
@dataclass
class SkillTag:
    skill_id: str                               # e.g., "reasoning-00001"
    tag: Literal["helpful", "harmful", "neutral"]
```

---

### SkillManager

The SkillManager transforms reflections into skillbook updates.

```python
from ace import SkillManager

skill_manager = SkillManager(client)

output = skill_manager.update_skills(
    reflection=reflection,
    skillbook=skillbook,
    question_context="Math problems",
    progress="3/5 correct"
)

# Apply the updates
skillbook.apply_update(output.update)
```

#### Constructor

```python
SkillManager(
    llm: LLMClient,
    prompt_template: str = None,
    max_retries: int = 3,
    dedup_manager: DeduplicationManager = None  # Optional deduplication
)
```

#### Methods

**`update_skills(reflection, skillbook, question_context=None, progress=None) -> SkillManagerOutput`**

Generate skillbook updates based on reflection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reflection` | ReflectorOutput | Reflection to process |
| `skillbook` | Skillbook | Current skillbook |
| `question_context` | str \| None | Task context |
| `progress` | str \| None | Progress indicator |

**Returns:** `SkillManagerOutput`

#### SkillManagerOutput

```python
@dataclass
class SkillManagerOutput:
    update: UpdateBatch                                    # Updates to apply
    consolidation_operations: List[ConsolidationOp] | None # Dedup operations
    raw: Dict[str, Any]
```

---

## Skillbook System

### Skillbook

The Skillbook is ACE's knowledge store - a collection of learned strategies.

```python
from ace import Skillbook

skillbook = Skillbook()

# Add a skill
skill = skillbook.add_skill(
    section="reasoning",
    content="Break complex problems into smaller steps",
    metadata={"helpful": 5, "harmful": 0, "neutral": 1}
)

# Get all skills
all_skills = skillbook.skills()

# Get a specific skill
skill = skillbook.get_skill("reasoning-00001")

# Format for LLM consumption
prompt = skillbook.as_prompt()  # TOON format (compressed)

# Format for debugging
debug = str(skillbook)  # Markdown format (human-readable)

# Save and load
skillbook.save_to_file("learned.json")
loaded = Skillbook.load_from_file("learned.json")
```

#### Constructor

```python
Skillbook(skills: Dict[str, Skill] = None)
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `add_skill(section, content, metadata=None)` | Add new skill | `Skill` |
| `update_skill(skill_id, content=None, metadata=None)` | Update existing skill | `Skill` |
| `tag_skill(skill_id, tag, increment=1)` | Increment helpful/harmful/neutral | `Skill` |
| `remove_skill(skill_id, soft=False)` | Remove skill | `bool` |
| `get_skill(skill_id)` | Get skill by ID | `Skill \| None` |
| `skills()` | Get all active skills | `List[Skill]` |
| `stats()` | Get skillbook statistics | `Dict` |
| `as_prompt()` | TOON format for LLM | `str` |
| `apply_update(batch)` | Apply UpdateBatch | `None` |
| `save_to_file(path)` | Save to JSON | `None` |
| `load_from_file(path)` | Load from JSON | `Skillbook` |

#### Statistics

```python
stats = skillbook.stats()
# Returns:
# {
#   "sections": 3,
#   "skills": 15,
#   "tags": {"helpful": 45, "harmful": 5, "neutral": 10}
# }
```

---

### Skill

A single strategy entry in the skillbook.

```python
@dataclass
class Skill:
    id: str              # e.g., "reasoning-00001"
    section: str         # Category: "reasoning", "extraction", etc.
    content: str         # The strategy text
    helpful: int = 0     # Times marked helpful
    harmful: int = 0     # Times marked harmful
    neutral: int = 0     # Times marked neutral
    created_at: str      # ISO timestamp
    updated_at: str      # ISO timestamp
    embedding: List[float] | None  # For deduplication
    status: str = "active"  # "active" or "invalid" (soft delete)
```

---

## Adaptation

### OfflineACE

Train on a batch of samples with multiple epochs.

```python
from ace import OfflineACE

adapter = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    skillbook=None,              # Optional: existing skillbook
    async_learning=False,        # Enable async mode
    max_reflector_workers=3      # Workers for async mode
)

results = adapter.run(
    samples=samples,
    environment=environment,
    epochs=3,
    checkpoint_interval=10,      # Save every N samples
    checkpoint_dir="./checkpoints"
)
```

#### Constructor

```python
OfflineACE(
    agent: Agent,
    reflector: Reflector,
    skill_manager: SkillManager,
    skillbook: Skillbook = None,
    async_learning: bool = False,
    max_reflector_workers: int = 3
)
```

#### Methods

**`run(samples, environment, epochs=1, checkpoint_interval=None, checkpoint_dir=None, wait_for_learning=True) -> List[ACEStepResult]`**

Run offline adaptation loop.

| Parameter | Type | Description |
|-----------|------|-------------|
| `samples` | List[Sample] | Training samples |
| `environment` | TaskEnvironment | Evaluation environment |
| `epochs` | int | Number of passes over samples |
| `checkpoint_interval` | int \| None | Save every N samples |
| `checkpoint_dir` | str \| None | Checkpoint directory |
| `wait_for_learning` | bool | Wait for async learning |

**Async Learning Methods:**

| Method | Description |
|--------|-------------|
| `learning_stats` | Get queue sizes, completion counts |
| `wait_for_learning(timeout=None)` | Block until complete |
| `stop_async_learning(wait=True)` | Shutdown pipeline |

---

### OnlineACE

Learn from tasks sequentially in real-time.

```python
from ace import OnlineACE

adapter = OnlineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)

# Process one at a time
for sample in samples:
    results = adapter.run([sample], environment)
```

#### Constructor

```python
OnlineACE(
    agent: Agent,
    reflector: Reflector,
    skill_manager: SkillManager,
    skillbook: Skillbook = None
)
```

---

### Sample

A task instance for training/evaluation.

```python
from ace import Sample

sample = Sample(
    question="What is 2+2?",
    context="Calculate the sum",    # Optional context
    ground_truth="4",               # Optional expected answer
    metadata={"difficulty": "easy"} # Optional metadata
)
```

```python
@dataclass
class Sample:
    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

### TaskEnvironment

Abstract base class for task evaluation.

```python
from ace import TaskEnvironment, EnvironmentResult

class CustomEnvironment(TaskEnvironment):
    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        correct = sample.ground_truth.lower() in agent_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0}
        )
```

#### EnvironmentResult

```python
@dataclass
class EnvironmentResult:
    feedback: str                           # Informative feedback
    ground_truth: str | None = None         # Expected answer
    metrics: Dict[str, float] = None        # Optional metrics
```

#### SimpleEnvironment (Built-in)

Basic environment using substring matching:

```python
from ace import SimpleEnvironment

env = SimpleEnvironment()  # Checks if ground_truth appears in answer
```

---

## Integrations

### ACELiteLLM

Quick-start integration for simple conversational agents.

```python
from ace import ACELiteLLM

# Create agent
agent = ACELiteLLM(model="gpt-4o-mini")

# Ask questions (learns automatically)
answer = agent.ask("What is the capital of France?")

# Save/load skillbook
agent.save_skillbook("learned.json")

# Load existing skillbook
agent2 = ACELiteLLM.from_skillbook("learned.json", model="gpt-4o-mini")
```

#### Constructor

```python
ACELiteLLM(
    model: str,                           # LiteLLM model identifier
    skillbook: Skillbook = None,          # Existing skillbook
    ace_model: str = None,                # Model for learning (default: same)
    dedup_config: DeduplicationConfig = None,
    **llm_kwargs                          # Additional LiteLLM params
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `ask(question, context="")` | Ask question, auto-learn |
| `learn(samples, environment, epochs=1)` | Explicit batch learning |
| `save_skillbook(path)` | Save to file |
| `from_skillbook(path, model, **kwargs)` | Class method: load and create |

---

### ACEAgent

Self-improving browser automation agent using [browser-use](https://github.com/browser-use/browser-use).

```bash
pip install ace-framework[browser-use]
```

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

agent = ACEAgent(
    llm=ChatBrowserUse(model="gpt-4o"),
    ace_model="gpt-4o-mini",
    skillbook_path="browser_expert.json"  # Optional: load existing
)

# Run task (learns automatically)
await agent.run(task="Find top post on Hacker News")

# Save learned strategies
agent.save_skillbook("browser_expert.json")
```

#### Constructor

```python
ACEAgent(
    llm: ChatBrowserUse,
    skillbook: Skillbook = None,
    skillbook_path: str = None,          # Load from file
    ace_model: str = "gpt-4o-mini",
    is_learning: bool = True,
    dedup_config: DeduplicationConfig = None
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `run(task)` | Execute task with learning |
| `save_skillbook(path)` | Save to file |
| `enable_learning()` | Enable learning |
| `disable_learning()` | Disable learning |

---

### ACELangChain

Wrap LangChain Runnables with ACE learning.

```bash
pip install ace-framework[langchain]
```

```python
from ace import ACELangChain

# Wrap any LangChain Runnable
ace_chain = ACELangChain(
    runnable=your_chain,
    ace_model="gpt-4o-mini"
)

# Use like normal LangChain
result = ace_chain.invoke({"question": "Your task"})

# Save learned strategies
ace_chain.save_skillbook("chain_learned.json")
```

#### Constructor

```python
ACELangChain(
    runnable: Runnable,                   # LangChain Runnable
    skillbook: Skillbook = None,
    ace_model: str = "gpt-4o-mini",
    environment: TaskEnvironment = None,  # Custom evaluation
    dedup_config: DeduplicationConfig = None
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `invoke(input_dict)` | Run chain with learning |
| `save_skillbook(path)` | Save to file |
| `enable_learning()` | Enable learning |
| `disable_learning()` | Disable learning |

---

## LLM Clients

### LiteLLMClient

Production LLM client supporting 100+ providers.

```python
from ace import LiteLLMClient

# Basic usage
client = LiteLLMClient(model="gpt-4o-mini")

# With configuration
client = LiteLLMClient(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    timeout=60,
    fallbacks=["claude-3-haiku", "gpt-3.5-turbo"]
)

# Generate completion
response = client.complete("What is 2+2?")
print(response.text)
```

#### Constructor

```python
LiteLLMClient(
    model: str,                           # Model identifier
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 60,
    max_retries: int = 3,
    fallbacks: List[str] = None,          # Fallback models
    track_cost: bool = True,              # Enable cost tracking
    **kwargs                              # Additional LiteLLM params
)
```

#### Supported Providers

- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-haiku`
- Google: `gemini-pro`, `gemini-1.5-pro`
- Azure: `azure/your-deployment`
- AWS Bedrock: `bedrock/anthropic.claude-v2`
- Local: `ollama/llama2`, `ollama/mistral`

---

## Updates

### UpdateOperation

A single mutation to the skillbook.

```python
from ace.updates import UpdateOperation

# ADD new skill
add_op = UpdateOperation(
    type="ADD",
    section="reasoning",
    content="Always verify calculations"
)

# UPDATE existing skill
update_op = UpdateOperation(
    type="UPDATE",
    skill_id="reasoning-00001",
    content="Updated strategy text"
)

# TAG skill effectiveness
tag_op = UpdateOperation(
    type="TAG",
    skill_id="reasoning-00001",
    metadata={"helpful": 1}
)

# REMOVE skill
remove_op = UpdateOperation(
    type="REMOVE",
    skill_id="reasoning-00001"
)
```

#### Fields

```python
@dataclass
class UpdateOperation:
    type: Literal["ADD", "UPDATE", "TAG", "REMOVE"]
    section: str | None = None
    content: str | None = None
    skill_id: str | None = None
    metadata: Dict[str, int] = None
```

---

### UpdateBatch

Bundle of operations to apply atomically.

```python
@dataclass
class UpdateBatch:
    reasoning: str                  # SkillManager's explanation
    operations: List[UpdateOperation]

# Apply to skillbook
skillbook.apply_update(batch)
```

---

## Deduplication

### DeduplicationConfig

Configure skill deduplication using embeddings.

```python
from ace import DeduplicationConfig

config = DeduplicationConfig(
    enabled=True,
    embedding_model="text-embedding-3-small",
    embedding_provider="litellm",       # or "sentence_transformers"
    similarity_threshold=0.85,          # Pairs above this are similar
    within_section_only=True,           # Compare within same section
    local_model_name="all-MiniLM-L6-v2" # For sentence_transformers
)

# Use with any integration
agent = ACELiteLLM(model="gpt-4o-mini", dedup_config=config)
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | True | Enable deduplication |
| `embedding_model` | str | "text-embedding-3-small" | Embedding model |
| `embedding_provider` | str | "litellm" | Provider: litellm or sentence_transformers |
| `similarity_threshold` | float | 0.85 | Threshold for similarity |
| `within_section_only` | bool | True | Only compare within sections |
| `min_pairs_to_report` | int | 1 | Min pairs to include in report |

#### Consolidation Operations

When deduplication is enabled, SkillManager can output:

- **MERGE**: Combine similar skills
- **DELETE**: Remove redundant skill
- **KEEP**: Mark as intentionally separate
- **UPDATE**: Refine to differentiate

---

## Prompts

### PromptManager

Manage prompt versions for roles.

```python
from ace.prompts_v2_1 import PromptManager

# Create manager (v2.1 recommended)
mgr = PromptManager()

# Get prompts for each role
agent_prompt = mgr.get_agent_prompt()
reflector_prompt = mgr.get_reflector_prompt()
skill_manager_prompt = mgr.get_skill_manager_prompt()

# Use with roles
agent = Agent(client, prompt_template=agent_prompt)
reflector = Reflector(client, prompt_template=reflector_prompt)
skill_manager = SkillManager(client, prompt_template=skill_manager_prompt)
```

#### Prompt Versions

| Version | Module | Status | Performance |
|---------|--------|--------|-------------|
| v1.0 | `ace.prompts` | Stable | Baseline |
| v2.0 | `ace.prompts_v2` | **Deprecated** | +12% |
| v2.1 | `ace.prompts_v2_1` | **Recommended** | +17% |

#### Template Variables

**Agent:** `{skillbook}`, `{question}`, `{context}`, `{reflection}`
**Reflector:** `{skillbook}`, `{question}`, `{agent_output}`, `{feedback}`, `{ground_truth}`
**SkillManager:** `{skillbook}`, `{reflection}`, `{question_context}`, `{progress}`

---

## Types

### ACEStepResult

Result from a single ACE step.

```python
@dataclass
class ACEStepResult:
    sample: Sample
    agent_output: AgentOutput
    environment_result: EnvironmentResult
    reflection: ReflectorOutput | None
    skill_manager_output: SkillManagerOutput | None
    skillbook_snapshot: str           # JSON snapshot
    epoch: int
    step: int
    performance_score: float
```

### LLMResponse

Response from LLM client.

```python
@dataclass
class LLMResponse:
    text: str                    # Main response text
    raw: Dict[str, Any] | None   # Full API response
```

---

## Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google
export GOOGLE_API_KEY="your-key"

# Opik (observability)
export OPIK_API_KEY="your-key"

# Custom endpoint
export LITELLM_API_BASE="https://your-endpoint.com"
```

### Logging

```python
import logging

# Enable debug logging for ACE
logging.getLogger("ace").setLevel(logging.DEBUG)
```

---

## Best Practices

1. **Start with SimpleEnvironment** - Get basic training working first
2. **Use v2.1 prompts** - Better performance (+17% success rate)
3. **Set adequate max_tokens** - At least 2048 for reliable JSON
4. **Save skillbooks regularly** - Preserve learned strategies
5. **Use async learning** - For latency-sensitive applications
6. **Enable deduplication** - Prevent skillbook bloat
7. **Use fallback models** - Ensure reliability in production

---

## Examples

See the [examples/](../examples/) directory:

| Category | Examples |
|----------|----------|
| `litellm/` | Basic usage, async learning, deduplication |
| `langchain/` | Chain/agent integration |
| `browser-use/` | Browser automation |
| `local-models/` | Ollama, LM Studio |
| `prompts/` | Prompt engineering |

---

## See Also

- [Quick Start](QUICK_START.md) - Get started in 5 minutes
- [Complete Guide](COMPLETE_GUIDE_TO_ACE.md) - ACE concepts in depth
- [Integration Guide](INTEGRATION_GUIDE.md) - Add ACE to existing agents
- [Prompt Guide](PROMPTS.md) - Prompt customization
