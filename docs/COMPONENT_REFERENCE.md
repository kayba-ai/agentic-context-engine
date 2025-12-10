# Component Reference: Complete API Documentation

**Version**: 0.5.1
**Last Updated**: January 2025
**Audience**: Developers, API users
**Reading Time**: 45-60 minutes

## Table of Contents

1. [Core Data Structures](#core-data-structures)
2. [Playbook Module](#playbook-module)
3. [Delta Module](#delta-module)
4. [Roles Module](#roles-module)
5. [Adaptation Module](#adaptation-module)
6. [LLM Module](#llm-module)
7. [LLM Providers](#llm-providers)
8. [Integrations](#integrations)
9. [Observability](#observability)
10. [Features Module](#features-module)
11. [Prompt Management](#prompt-management)

---

## Core Data Structures

### Bullet

A single learned strategy entry in the playbook.

**Location**: `ace/playbook.py`

```python
@dataclass
class Bullet:
    """A single learned strategy entry."""

    id: str                  # Unique identifier (e.g., "reasoning-00001")
    section: str             # Section/category (e.g., "reasoning", "edge_cases")
    content: str             # The strategy text
    helpful: int = 0         # Number of times marked helpful
    harmful: int = 0         # Number of times marked harmful
    neutral: int = 0         # Number of times marked neutral
    created_at: str = field(default_factory=...)  # ISO timestamp
    updated_at: str = field(default_factory=...)  # ISO timestamp
```

**Example**:
```python
from ace import Bullet

bullet = Bullet(
    id="reasoning-00001",
    section="reasoning",
    content="Break complex questions into sub-questions",
    helpful=5,
    harmful=0,
    neutral=1
)

print(bullet.id)        # "reasoning-00001"
print(bullet.section)   # "reasoning"
print(bullet.content)   # "Break complex questions..."
```

**ID Format**: `{section}-{5-digit-number}`
- Section: lowercase, underscores allowed (e.g., "edge_cases")
- Number: zero-padded 5 digits (e.g., "00001", "00042")

---

## Playbook Module

### Playbook

Structured context store containing learned strategies.

**Location**: `ace/playbook.py`

#### Constructor

```python
def __init__(self)
```

Creates an empty playbook.

**Example**:
```python
from ace import Playbook

playbook = Playbook()
```

#### add_bullet

```python
def add_bullet(self, bullet: Bullet) -> None
```

Add a new bullet to the playbook.

**Parameters**:
- `bullet` (Bullet): The bullet to add

**Raises**:
- `ValueError`: If bullet with same ID already exists

**Example**:
```python
bullet = Bullet(id="test-00001", section="test", content="Test strategy")
playbook.add_bullet(bullet)
```

#### update_bullet

```python
def update_bullet(self, bullet_id: str, content: str) -> None
```

Update the content of an existing bullet.

**Parameters**:
- `bullet_id` (str): ID of bullet to update
- `content` (str): New content

**Raises**:
- `ValueError`: If bullet not found

**Example**:
```python
playbook.update_bullet("test-00001", "Updated strategy text")
```

#### tag_bullet

```python
def tag_bullet(
    self,
    bullet_id: str,
    tag: str,
    increment: int = 1
) -> None
```

Increment helpful/harmful/neutral counter for a bullet.

**Parameters**:
- `bullet_id` (str): ID of bullet to tag
- `tag` (str): "helpful", "harmful", or "neutral"
- `increment` (int): Amount to increment (default: 1)

**Raises**:
- `ValueError`: If bullet not found or invalid tag

**Example**:
```python
playbook.tag_bullet("test-00001", "helpful", increment=1)
```

#### remove_bullet

```python
def remove_bullet(self, bullet_id: str) -> None
```

Remove a bullet from the playbook.

**Parameters**:
- `bullet_id` (str): ID of bullet to remove

**Raises**:
- `ValueError`: If bullet not found

**Example**:
```python
playbook.remove_bullet("test-00001")
```

#### apply_delta

```python
def apply_delta(self, delta_batch: DeltaBatch) -> None
```

Apply a batch of delta operations to the playbook.

**Parameters**:
- `delta_batch` (DeltaBatch): Batch of operations to apply

**Example**:
```python
from ace import DeltaBatch, DeltaOperation

batch = DeltaBatch(
    reasoning="Adding new strategy",
    operations=[
        DeltaOperation(type="ADD", section="test", content="New strategy")
    ]
)
playbook.apply_delta(batch)
```

#### as_prompt

```python
def as_prompt(self) -> str
```

Convert playbook to TOON format (token-efficient) for LLM consumption.

**Returns**: String in TOON format

**Example**:
```python
prompt_context = playbook.as_prompt()
# Output:
# # Playbook
# ## reasoning
# - [reasoning-00001] Break questions into sub-questions (✓5 ✗0 ~1)
```

#### \_\_str\_\_

```python
def __str__(self) -> str
```

Convert playbook to human-readable markdown format.

**Returns**: String in markdown format

**Example**:
```python
print(str(playbook))
# Output:
# # Playbook (2 bullets)
# ## reasoning (1 bullet)
# - [reasoning-00001] Break questions... (✓5 ✗0 ~1)
```

#### to_dict / from_dict

```python
def to_dict(self) -> dict

@classmethod
def from_dict(cls, data: dict) -> "Playbook"
```

Convert to/from dictionary representation.

**Example**:
```python
# Serialize
data = playbook.to_dict()

# Deserialize
playbook2 = Playbook.from_dict(data)
```

#### save_to_file / load_from_file

```python
def save_to_file(self, filepath: str) -> None

@classmethod
def load_from_file(cls, filepath: str) -> "Playbook"
```

Persist to/load from JSON file.

**Parameters**:
- `filepath` (str): Path to JSON file

**Example**:
```python
# Save
playbook.save_to_file("my_playbook.json")

# Load
playbook2 = Playbook.load_from_file("my_playbook.json")
```

#### stats

```python
def stats(self) -> dict
```

Get playbook statistics.

**Returns**: Dict with counts and metrics

**Example**:
```python
stats = playbook.stats()
# {
#   'total_bullets': 15,
#   'sections': {'reasoning': 5, 'edge_cases': 3, ...},
#   'total_helpful': 45,
#   'total_harmful': 2,
#   'total_neutral': 8
# }
```

#### get_bullets_by_section

```python
def get_bullets_by_section(self, section: str) -> List[Bullet]
```

Get all bullets in a specific section.

**Parameters**:
- `section` (str): Section name

**Returns**: List of Bullet objects

**Example**:
```python
reasoning_bullets = playbook.get_bullets_by_section("reasoning")
```

---

## Delta Module

### DeltaOperation

A single incremental change to the playbook.

**Location**: `ace/delta.py`

```python
@dataclass
class DeltaOperation:
    """A single playbook mutation."""

    type: str                # "ADD", "UPDATE", "TAG", "REMOVE"
    section: str             # Target section
    content: str = ""        # For ADD/UPDATE
    bullet_id: str = ""      # For UPDATE/TAG/REMOVE
    metadata: dict = field(default_factory=dict)  # For TAG
```

**Operation Types**:

1. **ADD**: Create new bullet
```python
DeltaOperation(type="ADD", section="reasoning", content="New strategy")
```

2. **UPDATE**: Modify existing bullet
```python
DeltaOperation(type="UPDATE", bullet_id="reasoning-00001",
               section="reasoning", content="Updated strategy")
```

3. **TAG**: Update performance counters
```python
DeltaOperation(type="TAG", bullet_id="reasoning-00001",
               section="reasoning",
               metadata={"helpful": 1, "harmful": 0, "neutral": 0})
```

4. **REMOVE**: Delete bullet
```python
DeltaOperation(type="REMOVE", bullet_id="reasoning-00001", section="reasoning")
```

### DeltaBatch

A bundle of delta operations with reasoning.

```python
@dataclass
class DeltaBatch:
    """Bundle of operations with justification."""

    reasoning: str                    # Why these operations?
    operations: List[DeltaOperation]  # The changes
```

**Example**:
```python
from ace import DeltaBatch, DeltaOperation

batch = DeltaBatch(
    reasoning="Refining edge case handling based on failure",
    operations=[
        DeltaOperation(type="ADD", section="edge_cases", content="Check null"),
        DeltaOperation(type="TAG", bullet_id="reasoning-00001",
                      section="reasoning", metadata={"helpful": 1})
    ]
)
```

#### from_json / to_json

```python
@classmethod
def from_json(cls, json_str: str) -> "DeltaBatch"

def to_json(self) -> str
```

Serialize/deserialize to JSON.

**Example**:
```python
# Serialize
json_str = batch.to_json()

# Deserialize
batch2 = DeltaBatch.from_json(json_str)
```

---

## Roles Module

### Generator

Produces answers using playbook strategies.

**Location**: `ace/roles.py`

#### Constructor

```python
def __init__(
    self,
    llm: LLMClient,
    prompt_template: str = None,
    retry_prompt: str = None
)
```

**Parameters**:
- `llm` (LLMClient): LLM client for generation
- `prompt_template` (str, optional): Custom prompt template
- `retry_prompt` (str, optional): Custom retry message for JSON parsing failures

**Example**:
```python
from ace import Generator
from ace.llm_providers import LiteLLMClient

llm = LiteLLMClient(model="gpt-4")
generator = Generator(llm)

# With custom retry prompt
generator = Generator(llm, retry_prompt="\n\nReturn ONLY valid JSON.")
```

#### generate

```python
def generate(
    self,
    question: str,
    context: str,
    playbook: Playbook,
    reflection: str = ""
) -> GeneratorOutput
```

Generate answer with reasoning.

**Parameters**:
- `question` (str): The task/question to answer
- `context` (str): Additional context (can be empty string)
- `playbook` (Playbook): Current strategies to use
- `reflection` (str, optional): Previous reflection (for retry scenarios)

**Returns**: GeneratorOutput with reasoning, final_answer, bullet_ids

**Example**:
```python
output = generator.generate(
    question="What is 2+2?",
    context="Simple arithmetic",
    playbook=playbook
)

print(output.reasoning)      # "Breaking down: 2+2 means..."
print(output.final_answer)   # "4"
print(output.bullet_ids)     # ["reasoning-00001", "math-00003"]
```

### GeneratorOutput

```python
@dataclass
class GeneratorOutput:
    """Output from Generator role."""

    reasoning: str            # Thought process
    final_answer: str         # The answer
    bullet_ids: List[str]     # Cited strategy IDs
```

### ReplayGenerator

Deterministic generator for testing/replay.

```python
class ReplayGenerator(Generator):
    """Generator that replays recorded responses."""

    def __init__(self, responses: List[GeneratorOutput]):
        """
        Args:
            responses: List of pre-recorded responses to replay
        """
        self.responses = responses
        self.index = 0

    def generate(self, *args, **kwargs) -> GeneratorOutput:
        """Return next response in sequence."""
        output = self.responses[self.index]
        self.index += 1
        return output
```

**Use Cases**: Testing, deterministic benchmarking

---

### Reflector

Analyzes performance and classifies strategy contributions.

**Location**: `ace/roles.py`

#### Constructor

```python
def __init__(
    self,
    llm: LLMClient,
    prompt_template: str = None,
    retry_prompt: str = None
)
```

**Parameters**:
- `llm` (LLMClient): LLM client for reflection
- `prompt_template` (str, optional): Custom prompt template
- `retry_prompt` (str, optional): Custom retry message

**Example**:
```python
from ace import Reflector
from ace.llm_providers import LiteLLMClient

llm = LiteLLMClient(model="gpt-4")
reflector = Reflector(llm)
```

#### reflect

```python
def reflect(
    self,
    question: str,
    generator_output: GeneratorOutput,
    feedback: str,
    ground_truth: str = None,
    playbook: Playbook = None
) -> ReflectorOutput
```

Analyze performance and classify bullets.

**Parameters**:
- `question` (str): The original question
- `generator_output` (GeneratorOutput): Generator's output
- `feedback` (str): Environment feedback (e.g., "Correct" or "Incorrect")
- `ground_truth` (str, optional): Expected answer
- `playbook` (Playbook, optional): Current playbook (for context)

**Returns**: ReflectorOutput with analysis

**Example**:
```python
reflection = reflector.reflect(
    question="What is 2+2?",
    generator_output=generator_output,
    feedback="Correct! Well done.",
    ground_truth="4",
    playbook=playbook
)

print(reflection.reasoning)
print(reflection.error_identification)
print(reflection.bullet_tags)
```

### ReflectorOutput

```python
@dataclass
class ReflectorOutput:
    """Output from Reflector role."""

    reasoning: str                  # Analysis process
    error_identification: str       # What went wrong/right?
    root_cause_analysis: str        # Why?
    bullet_tags: List[dict]         # Strategy classifications

    # Example bullet_tags:
    # [
    #   {
    #     "bullet_id": "reasoning-00001",
    #     "tag": "helpful",
    #     "justification": "Led to correct answer"
    #   }
    # ]
```

---

### Curator

Updates playbook based on reflection.

**Location**: `ace/roles.py`

#### Constructor

```python
def __init__(
    self,
    llm: LLMClient,
    prompt_template: str = None,
    retry_prompt: str = None
)
```

**Parameters**:
- `llm` (LLMClient): LLM client for curation
- `prompt_template` (str, optional): Custom prompt template
- `retry_prompt` (str, optional): Custom retry message

**Example**:
```python
from ace import Curator
from ace.llm_providers import LiteLLMClient

llm = LiteLLMClient(model="gpt-4")
curator = Curator(llm)
```

#### curate

```python
def curate(
    self,
    reflection: ReflectorOutput,
    playbook: Playbook,
    question_context: str = ""
) -> CuratorOutput
```

Generate delta operations to update playbook.

**Parameters**:
- `reflection` (ReflectorOutput): Reflector's analysis
- `playbook` (Playbook): Current playbook state
- `question_context` (str, optional): Context about the question type

**Returns**: CuratorOutput with reasoning and delta_batch

**Example**:
```python
curation = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="Simple arithmetic"
)

print(curation.reasoning)
print(curation.delta_batch.operations)

# Apply updates
playbook.apply_delta(curation.delta_batch)
```

### CuratorOutput

```python
@dataclass
class CuratorOutput:
    """Output from Curator role."""

    reasoning: str            # Decision process
    delta_batch: DeltaBatch   # Operations to apply
```

---

## Adaptation Module

### Sample

A single task instance for training/testing.

**Location**: `ace/adaptation.py`

```python
@dataclass
class Sample:
    """A single task instance."""

    question: str               # The task/question
    context: str = ""           # Additional context
    ground_truth: str = None    # Expected answer (optional)
    metadata: dict = field(default_factory=dict)  # Custom data
```

**Example**:
```python
from ace import Sample

sample = Sample(
    question="What is 2+2?",
    context="Simple arithmetic",
    ground_truth="4",
    metadata={"difficulty": "easy"}
)
```

---

### EnvironmentResult

Feedback from task evaluation.

```python
@dataclass
class EnvironmentResult:
    """Evaluation feedback."""

    feedback: str               # Human-readable feedback
    ground_truth: str = None    # Expected answer
    metrics: dict = field(default_factory=dict)  # Performance metrics
```

**Example**:
```python
from ace import EnvironmentResult

result = EnvironmentResult(
    feedback="Correct! Well done.",
    ground_truth="4",
    metrics={"accuracy": 1.0, "time": 0.5}
)
```

---

### TaskEnvironment

Abstract base class for evaluation logic.

**Location**: `ace/adaptation.py`

```python
class TaskEnvironment(ABC):
    """Abstract task evaluation environment."""

    @abstractmethod
    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str = None
    ) -> EnvironmentResult:
        """
        Evaluate agent's answer.

        Args:
            question: The original question
            answer: Agent's answer
            ground_truth: Expected answer (optional)

        Returns:
            EnvironmentResult with feedback
        """
        pass
```

**Example Implementation**:
```python
from ace import TaskEnvironment, EnvironmentResult

class MathEnvironment(TaskEnvironment):
    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        try:
            result = eval(question.replace("What is ", "").replace("?", ""))
            correct = str(result) in answer
            feedback = "Correct!" if correct else "Incorrect"
            return EnvironmentResult(
                feedback=feedback,
                ground_truth=str(result),
                metrics={"accuracy": 1.0 if correct else 0.0}
            )
        except:
            return EnvironmentResult(
                feedback="Error evaluating",
                metrics={"accuracy": 0.0}
            )
```

### SimpleEnvironment

Built-in basic environment.

```python
class SimpleEnvironment(TaskEnvironment):
    """
    Simple environment that checks if ground truth appears in answer.

    Case-insensitive substring matching.
    """

    def evaluate(self, question: str, answer: str, ground_truth: str = None):
        if not ground_truth:
            return EnvironmentResult(feedback="No ground truth provided")

        correct = ground_truth.lower() in answer.lower()
        feedback = "Correct" if correct else "Incorrect"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0}
        )
```

**Example**:
```python
from ace import SimpleEnvironment

env = SimpleEnvironment()
result = env.evaluate("What is 2+2?", "The answer is 4", ground_truth="4")
print(result.feedback)  # "Correct"
```

---

### OfflineAdapter

Batch training with multiple epochs.

**Location**: `ace/adaptation.py`

#### Constructor

```python
def __init__(
    self,
    playbook: Playbook,
    generator: Generator,
    reflector: Reflector,
    curator: Curator
)
```

**Parameters**:
- `playbook` (Playbook): Playbook to update
- `generator` (Generator): Generator role
- `reflector` (Reflector): Reflector role
- `curator` (Curator): Curator role

**Example**:
```python
from ace import OfflineAdapter, Playbook, Generator, Reflector, Curator
from ace.llm_providers import LiteLLMClient

llm = LiteLLMClient(model="gpt-4")
playbook = Playbook()
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

adapter = OfflineAdapter(playbook, generator, reflector, curator)
```

#### run

```python
def run(
    self,
    samples: List[Sample],
    environment: TaskEnvironment,
    epochs: int = 1,
    checkpoint_interval: int = None,
    checkpoint_dir: str = "./checkpoints"
) -> List[AdapterStepResult]
```

Train on samples over multiple epochs.

**Parameters**:
- `samples` (List[Sample]): Training data
- `environment` (TaskEnvironment): Evaluation logic
- `epochs` (int, default=1): Number of passes through data
- `checkpoint_interval` (int, optional): Save playbook every N samples
- `checkpoint_dir` (str, default="./checkpoints"): Where to save checkpoints

**Returns**: List of AdapterStepResult for all samples across all epochs

**Example**:
```python
from ace import Sample, SimpleEnvironment

samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="What is 3+3?", ground_truth="6"),
]

environment = SimpleEnvironment()

results = adapter.run(
    samples=samples,
    environment=environment,
    epochs=3,
    checkpoint_interval=10,
    checkpoint_dir="./checkpoints"
)

# Analyze results
for result in results:
    print(f"Q: {result.sample.question}")
    print(f"A: {result.generator_output.final_answer}")
    print(f"Feedback: {result.environment_result.feedback}")
```

### OnlineAdapter

Real-time learning from streaming samples.

**Location**: `ace/adaptation.py`

#### Constructor

```python
def __init__(
    self,
    playbook: Playbook,
    generator: Generator,
    reflector: Reflector,
    curator: Curator
)
```

Same as OfflineAdapter.

#### process

```python
def process(
    self,
    sample: Sample,
    environment: TaskEnvironment
) -> AdapterStepResult
```

Process single sample with learning.

**Parameters**:
- `sample` (Sample): Single task instance
- `environment` (TaskEnvironment): Evaluation logic

**Returns**: AdapterStepResult with updated playbook

**Example**:
```python
from ace import OnlineAdapter, Sample, SimpleEnvironment

adapter = OnlineAdapter(playbook, generator, reflector, curator)
environment = SimpleEnvironment()

# Process samples as they arrive
for sample in incoming_stream:
    result = adapter.process(sample, environment)
    print(f"Processed sample {adapter.step_number}")
    print(f"Playbook size: {len(adapter.playbook.bullets)}")
```

### AdapterStepResult

Result from processing a single sample.

```python
@dataclass
class AdapterStepResult:
    """Result from one adaptation step."""

    sample: Sample                        # The input sample
    generator_output: GeneratorOutput     # Generator's response
    environment_result: EnvironmentResult # Evaluation feedback
    reflector_output: ReflectorOutput     # Analysis
    curator_output: CuratorOutput         # Updates
    playbook_snapshot: dict               # Playbook state after this step
    step_number: int = 0                  # Step counter (OnlineAdapter)
```

**Example**:
```python
result = adapter.process(sample, environment)

print(result.sample.question)
print(result.generator_output.final_answer)
print(result.environment_result.feedback)
print(len(result.playbook_snapshot['bullets']))
```

---

## LLM Module

### LLMResponse

Container for LLM output.

**Location**: `ace/llm.py`

```python
@dataclass
class LLMResponse:
    """LLM response container."""

    text: str       # Response text
    raw: Any = None # Optional raw response object
```

### LLMClient

Abstract interface for LLM providers.

**Location**: `ace/llm.py`

```python
class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate completion.

        Args:
            prompt: Input text
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse
        """
        pass
```

### DummyLLMClient

Testing stub for unit tests.

```python
class DummyLLMClient(LLMClient):
    """Mock LLM for testing."""

    def __init__(self):
        self.responses = []
        self.index = 0

    def queue(self, text: str):
        """Queue a response."""
        self.responses.append(text)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Return next queued response."""
        if self.index >= len(self.responses):
            raise ValueError("No more queued responses")
        text = self.responses[self.index]
        self.index += 1
        return LLMResponse(text=text)
```

**Example**:
```python
from ace.llm import DummyLLMClient

llm = DummyLLMClient()
llm.queue('{"reasoning": "...", "final_answer": "42"}')
llm.queue('{"reasoning": "...", "error_identification": "..."}')

response1 = llm.complete("Question 1")
response2 = llm.complete("Question 2")
```

### TransformersLLMClient

Local model support via HuggingFace.

```python
class TransformersLLMClient(LLMClient):
    """Local model via HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Args:
            model_name: HuggingFace model ID
            device_map: Device mapping ("auto", "cuda:0", etc.)
            torch_dtype: Tensor dtype ("float16", "float32", etc.)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
        """
```

**Example**:
```python
from ace.llm import TransformersLLMClient

llm = TransformersLLMClient(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype="float16",
    max_new_tokens=512
)

response = llm.complete("What is 2+2?")
print(response.text)
```

---

## LLM Providers

### LiteLLMClient

Universal LLM provider supporting 100+ models.

**Location**: `ace/llm_providers/litellm_client.py`

#### Constructor

```python
def __init__(
    self,
    model: str,
    api_key: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    fallback_models: List[str] = None,
    timeout: int = 60,
    **kwargs
)
```

**Parameters**:
- `model` (str): Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
- `api_key` (str, optional): API key (defaults to env variable)
- `temperature` (float, default=0.7): Sampling temperature
- `max_tokens` (int, default=2000): Max completion tokens
- `fallback_models` (List[str], optional): Fallback models on failure
- `timeout` (int, default=60): Request timeout in seconds
- `**kwargs`: Additional provider-specific parameters

**Supported Providers**:
- OpenAI: `gpt-4`, `gpt-3.5-turbo`, etc.
- Anthropic: `claude-3-opus-20240229`, `claude-2`, etc.
- Google: `gemini/gemini-pro`, `palm-2`, etc.
- Cohere: `cohere/command-nightly`
- Azure: `azure/gpt-4`
- AWS Bedrock: `bedrock/anthropic.claude-v2`
- Local: `ollama/llama2`, `ollama/mistral`, etc.
- 100+ more via LiteLLM

**Example**:
```python
from ace.llm_providers import LiteLLMClient

# OpenAI
llm = LiteLLMClient(model="gpt-4", api_key="sk-...")

# Anthropic
llm = LiteLLMClient(model="claude-3-opus-20240229", api_key="sk-ant-...")

# Google
llm = LiteLLMClient(model="gemini/gemini-pro", api_key="...")

# Local via Ollama
llm = LiteLLMClient(model="ollama/llama2")

# With fallback
llm = LiteLLMClient(
    model="gpt-4",
    fallback_models=["gpt-3.5-turbo", "claude-2"]
)

response = llm.complete("What is 2+2?")
```

#### complete

```python
def complete(self, prompt: str, **kwargs) -> LLMResponse
```

Generate completion with fallback support and cost tracking.

**Parameters**:
- `prompt` (str): Input text
- `**kwargs`: Override default parameters

**Returns**: LLMResponse

**Example**:
```python
response = llm.complete(
    "What is the capital of France?",
    temperature=0.5,
    max_tokens=100
)
print(response.text)
```

---

### LangChainClient

LangChain integration.

**Location**: `ace/llm_providers/langchain_client.py`

#### Constructor

```python
def __init__(self, llm):
```

**Parameters**:
- `llm`: Any LangChain Runnable (ChatOpenAI, ChatAnthropic, etc.)

**Example**:
```python
from ace.llm_providers import LangChainClient
from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(model="gpt-4")
llm = LangChainClient(llm=langchain_llm)

response = llm.complete("What is 2+2?")
```

---

## Integrations

### ACELiteLLM

Simple Q&A agent with learning (all-in-one).

**Location**: `ace/integrations/litellm.py`

#### Constructor

```python
def __init__(
    self,
    model: str,
    api_key: str = None,
    playbook_path: str = None,
    **kwargs
)
```

**Parameters**:
- `model` (str): LiteLLM model identifier
- `api_key` (str, optional): API key
- `playbook_path` (str, optional): Path to playbook JSON file
- `**kwargs`: Additional LiteLLM parameters

**Example**:
```python
from ace.integrations import ACELiteLLM

# Create agent
agent = ACELiteLLM(model="gpt-4", api_key="sk-...")

# Or with existing playbook
agent = ACELiteLLM(
    model="gpt-4",
    playbook_path="./my_playbook.json"
)
```

#### ask

```python
def ask(
    self,
    question: str,
    context: str = "",
    ground_truth: str = None
) -> str
```

Ask question and learn from result.

**Parameters**:
- `question` (str): The question
- `context` (str, optional): Additional context
- `ground_truth` (str, optional): Expected answer (for learning)

**Returns**: The answer (str)

**Example**:
```python
answer = agent.ask("What is 2+2?", ground_truth="4")
print(answer)  # "4"

# Playbook automatically updated
print(len(agent.playbook.bullets))  # May have new strategies
```

#### learn

```python
def learn(
    self,
    samples: List[Sample],
    environment: TaskEnvironment,
    epochs: int = 1
) -> List[AdapterStepResult]
```

Batch training on samples.

**Parameters**:
- `samples` (List[Sample]): Training data
- `environment` (TaskEnvironment): Evaluation logic
- `epochs` (int, default=1): Number of epochs

**Returns**: List of results

**Example**:
```python
from ace import Sample, SimpleEnvironment

samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="What is 3+3?", ground_truth="6"),
]

results = agent.learn(samples, SimpleEnvironment(), epochs=3)
```

#### save_playbook / from_playbook

```python
def save_playbook(self, path: str)

@classmethod
def from_playbook(cls, playbook_path: str, model: str, **kwargs)
```

Save/load learned strategies.

**Example**:
```python
# Save
agent.save_playbook("./math_agent.json")

# Load
agent2 = ACELiteLLM.from_playbook("./math_agent.json", model="gpt-4")
```

---

### ACEAgent

Browser automation integration (wraps browser-use).

**Location**: `ace/integrations/browser_use.py`

#### Constructor

```python
def __init__(
    self,
    agent,              # browser-use Agent
    llm: LLMClient,     # For Reflector/Curator
    playbook_path: str = None
)
```

**Parameters**:
- `agent`: browser-use Agent instance
- `llm` (LLMClient): LLM for learning (Reflector/Curator)
- `playbook_path` (str, optional): Path to playbook file

**Example**:
```python
from ace.integrations import ACEAgent
from browser_use import Agent
from langchain_openai import ChatOpenAI

# Your browser agent
browser_llm = ChatOpenAI(model="gpt-4")
browser_agent = Agent(
    task="Go to example.com and check if domain is available",
    llm=browser_llm
)

# Wrap with ACE
learning_llm = LiteLLMClient(model="gpt-4")
ace_browser = ACEAgent(
    agent=browser_agent,
    llm=learning_llm,
    playbook_path="./browser_playbook.json"
)
```

#### run

```python
async def run(self, **kwargs)
```

Execute browser task with learning.

**Parameters**:
- `**kwargs`: Additional parameters for browser agent

**Returns**: Browser agent result

**Example**:
```python
import asyncio

result = asyncio.run(ace_browser.run())
print(result)

# Playbook updated with web automation strategies
print(len(ace_browser.playbook.bullets))
```

---

### ACELangChain

LangChain workflow integration.

**Location**: `ace/integrations/langchain.py`

#### Constructor

```python
def __init__(
    self,
    chain,              # LangChain Runnable
    llm: LLMClient,     # For Reflector/Curator
    playbook_path: str = None
)
```

**Parameters**:
- `chain`: Any LangChain Runnable (LLMChain, Agent, etc.)
- `llm` (LLMClient): LLM for learning
- `playbook_path` (str, optional): Path to playbook file

**Example**:
```python
from ace.integrations import ACELangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Your LangChain setup
langchain_llm = ChatOpenAI(model="gpt-4")
prompt = PromptTemplate.from_template("Answer: {question}")
chain = LLMChain(llm=langchain_llm, prompt=prompt)

# Wrap with ACE
learning_llm = LiteLLMClient(model="gpt-4")
ace_chain = ACELangChain(
    chain=chain,
    llm=learning_llm,
    playbook_path="./chain_playbook.json"
)
```

#### run

```python
def run(self, input_data, ground_truth=None, **kwargs)
```

Execute chain with learning.

**Parameters**:
- `input_data`: Input for chain (dict or str)
- `ground_truth` (str, optional): Expected output
- `**kwargs`: Additional chain parameters

**Returns**: Chain output

**Example**:
```python
result = ace_chain.run(
    {"question": "What is 2+2?"},
    ground_truth="4"
)
print(result)
```

---

## Observability

### OpikIntegration

Enterprise-grade monitoring with Opik.

**Location**: `ace/observability/opik_integration.py`

**Features**:
- Automatic token usage tracking
- Cost calculation per LLM call
- Role-level attribution (Generator/Reflector/Curator)
- Playbook evolution tracking

**Usage**:
```bash
# Install
pip install ace-framework[observability]

# Set API key
export OPIK_API_KEY="your-key"
```

**Automatic Tracking**:
```python
from ace.llm_providers import LiteLLMClient

# Automatically tracks when Opik installed
llm = LiteLLMClient(model="gpt-4")
response = llm.complete("What is 2+2?")

# View at: https://www.comet.com/opik
# - Token usage
# - Cost per call
# - Role attribution
```

---

## Features Module

Detect available optional features.

**Location**: `ace/features.py`

### Functions

```python
def has_opik() -> bool
def has_litellm() -> bool
def has_langchain() -> bool
def has_transformers() -> bool
def has_torch() -> bool
def has_browser_use() -> bool
def has_playwright() -> bool
def get_available_features() -> dict
def print_feature_status() -> None
```

**Example**:
```python
from ace.features import (
    has_opik, has_litellm,
    get_available_features,
    print_feature_status
)

# Check individual features
if has_opik():
    print("Opik available")

# Get all features
features = get_available_features()
# {
#   'opik': True,
#   'litellm': True,
#   'langchain': False,
#   ...
# }

# Pretty print
print_feature_status()
# ACE Framework Features:
#   ✓ opik (Observability)
#   ✓ litellm (LLM Provider)
#   ✗ langchain (Not installed)
```

---

## Prompt Management

### PromptManager (v2.1)

Manage production-ready prompts.

**Location**: `ace/prompts_v2_1.py`

```python
class PromptManager:
    """Manage v2.1 production prompts."""

    def get_generator_prompt(self) -> str:
        """Get Generator prompt template."""
        return GENERATOR_PROMPT_v2_1

    def get_reflector_prompt(self) -> str:
        """Get Reflector prompt template."""
        return REFLECTOR_PROMPT_v2_1

    def get_curator_prompt(self) -> str:
        """Get Curator prompt template."""
        return CURATOR_PROMPT_v2_1
```

**Example**:
```python
from ace.prompts_v2_1 import PromptManager
from ace import Generator, Reflector, Curator
from ace.llm_providers import LiteLLMClient

prompt_mgr = PromptManager()
llm = LiteLLMClient(model="gpt-4")

# Use v2.1 prompts (RECOMMENDED)
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

### wrap_playbook_for_external_agent

Format playbook for non-ACE agents.

```python
def wrap_playbook_for_external_agent(playbook: Playbook) -> str:
    """
    Format playbook context for injection into external agents.

    Args:
        playbook: Playbook to format

    Returns:
        Formatted string for injection
    """
```

**Example**:
```python
from ace.prompts_v2_1 import wrap_playbook_for_external_agent

context = wrap_playbook_for_external_agent(playbook)

# Inject into external agent
agent_prompt = base_prompt + "\n\n" + context
```

---

## Summary

This component reference covers all public APIs in the ACE framework:

✅ **Core Data Structures**: Bullet, Playbook, DeltaOperation, DeltaBatch
✅ **Roles**: Generator, Reflector, Curator
✅ **Adaptation**: OfflineAdapter, OnlineAdapter, TaskEnvironment
✅ **LLM Clients**: LiteLLMClient, LangChainClient, TransformersLLMClient
✅ **Integrations**: ACELiteLLM, ACEAgent, ACELangChain
✅ **Observability**: OpikIntegration, automatic tracking
✅ **Utilities**: Features detection, prompt management

---

**Related Documentation**:
- [COMPREHENSIVE_GUIDE.md](./COMPREHENSIVE_GUIDE.md) - High-level overview
- [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md) - Technical architecture
- [DATA_FLOW_GUIDE.md](./DATA_FLOW_GUIDE.md) - Data flow examples
- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - Implementation patterns

---

*Last updated: January 2025*
*Framework version: 0.5.1*
