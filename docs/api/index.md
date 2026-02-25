# API Reference

Quick reference for the most-used classes and functions in `ace_next`.

## Runners

### ACELiteLLM

Simple self-improving conversational agent.

```python
from ace_next import ACELiteLLM

agent = ACELiteLLM.from_model("gpt-4o-mini")
```

| Method | Description |
|--------|-------------|
| `ask(question, context="")` | Generate an answer using the current skillbook |
| `learn(samples, environment, epochs=1, *, wait=True)` | Run the full ACE learning pipeline |
| `learn_from_feedback(feedback, ground_truth=None)` | Learn from the last `ask()` interaction |
| `learn_from_traces(traces, epochs=1, *, wait=True)` | Learn from pre-recorded execution traces |
| `save(path)` | Save skillbook to JSON |
| `load(path)` | Load skillbook from JSON |
| `enable_learning()` / `disable_learning()` | Toggle learning on/off |
| `wait_for_background(timeout=None)` | Wait for async learning to finish |
| `learning_stats` | Dict with background learning progress |
| `get_strategies()` | Formatted string of current strategies |

See [LiteLLM Integration](../integrations/litellm.md) for full details.

### ACE

Full adaptive pipeline (Agent + Reflector + SkillManager + Environment).

```python
from ace_next import ACE, Agent, Reflector, SkillManager, Skillbook, LiteLLMClient, SimpleEnvironment

client = LiteLLMClient(model="gpt-4o-mini")
runner = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    environment=SimpleEnvironment(),
    skillbook=Skillbook(),
)

results = runner.run(samples, epochs=3)
```

| Method | Description |
|--------|-------------|
| `run(samples, epochs=1, wait=True)` | Run adaptation loop, return `list[SampleResult]` |
| `save(path)` | Save skillbook |
| `wait_for_background(timeout=None)` | Wait for async learning |
| `learning_stats` | Background learning progress |

See [Full Pipeline Guide](../guides/full-pipeline.md).

### BrowserUse

Browser automation with learning.

```python
from ace_next import BrowserUse

runner = BrowserUse.from_model(browser_llm=my_llm, ace_model="gpt-4o-mini")
results = runner.run("Find the top post on Hacker News")
```

See [Browser-Use Integration](../integrations/browser-use.md).

### LangChain

Wrap LangChain Runnables with learning.

```python
from ace_next import LangChain

runner = LangChain.from_model(my_chain, ace_model="gpt-4o-mini")
results = runner.run([{"input": "Summarize this document"}])
```

See [LangChain Integration](../integrations/langchain.md).

### ClaudeCode

Claude Code CLI with learning.

```python
from ace_next import ClaudeCode

runner = ClaudeCode.from_model(working_dir="./project", ace_model="gpt-4o-mini")
results = runner.run("Add unit tests for utils.py")
```

See [Claude Code Integration](../integrations/claude-code.md).

---

## Roles

### Agent

Produces answers using the current skillbook.

```python
from ace_next import Agent

agent = Agent(llm)
output = agent.generate(
    question="What is 2+2?",
    context="",
    skillbook=skillbook,
    reflection=None,  # optional
)
```

**AgentOutput fields:**

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | The generated answer |
| `reasoning` | `str` | Step-by-step reasoning |
| `skill_ids` | `list[str]` | Skillbook strategies cited |
| `raw` | `dict` | Raw LLM response |

### Reflector

Analyzes what worked and what failed.

```python
from ace_next import Reflector

reflector = Reflector(llm)
reflection = reflector.reflect(
    question="What is 2+2?",
    agent_output=output,
    skillbook=skillbook,
    ground_truth="4",
    feedback="Correct!",
)
```

**ReflectorOutput fields:**

| Field | Type | Description |
|-------|------|-------------|
| `reasoning` | `str` | Analysis of the outcome |
| `error_identification` | `str` | What went wrong |
| `root_cause_analysis` | `str` | Why it went wrong |
| `correct_approach` | `str` | What should have been done |
| `key_insight` | `str` | Main lesson learned |
| `extracted_learnings` | `list[ExtractedLearning]` | Learnings with evidence and justification |
| `skill_tags` | `list[SkillTag]` | `(skill_id, tag)` pairs |
| `raw` | `dict` | Raw LLM response |

### SkillManager

Transforms reflections into skillbook updates.

```python
from ace_next import SkillManager

skill_manager = SkillManager(llm)
sm_output = skill_manager.update_skills(
    reflection=reflection,
    skillbook=skillbook,
    question_context="Math problems",
    progress="3/5 correct",
)
# Apply the updates
skillbook.apply_update(sm_output.update)
```

Returns a `SkillManagerOutput` with an `.update` field (`UpdateBatch`) and `.raw` field.

See [Roles](../concepts/roles.md) for full details.

---

## Skillbook

```python
from ace_next import Skillbook

skillbook = Skillbook()
```

| Method / Property | Description |
|-------------------|-------------|
| `add_skill(section, content, metadata=None)` | Add a strategy |
| `apply_update(update_batch)` | Apply update operations |
| `as_prompt()` | TOON format for LLM consumption |
| `save_to_file(path)` | Save to JSON |
| `Skillbook.load_from_file(path)` | Load from JSON |
| `stats()` | Section count, skill count, tag totals |
| `skills()` | List of all skills |

See [The Skillbook](../concepts/skillbook.md).

---

## Data Types

### Sample

```python
from ace_next import Sample

sample = Sample(
    question="What is 2+2?",
    context="Show your work",
    ground_truth="4",
)
```

### EnvironmentResult

```python
from ace_next import EnvironmentResult

result = EnvironmentResult(
    feedback="Correct!",
    ground_truth="4",
    metrics={"accuracy": 1.0},
)
```

### UpdateOperation

```python
from ace_next import UpdateOperation

op = UpdateOperation(
    type="ADD",
    section="Math",
    content="Break problems into smaller steps",
    skill_id="math-00001",
)
```

Operations: `ADD`, `UPDATE`, `TAG`, `REMOVE`. See [Update Operations](../concepts/updates.md).

### DeduplicationConfig

```python
from ace_next import DeduplicationConfig

config = DeduplicationConfig(
    enabled=True,
    embedding_model="text-embedding-3-small",
    similarity_threshold=0.85,
)
```

---

## Environments

Extend `TaskEnvironment` to provide evaluation feedback:

```python
from ace_next import TaskEnvironment, EnvironmentResult

class MyEnvironment(TaskEnvironment):
    def evaluate(self, sample, agent_output):
        correct = sample.ground_truth.lower() in agent_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )
```

A built-in `SimpleEnvironment` uses substring matching and is included for quick testing.

---

## LLM Clients

### LiteLLMClient

```python
from ace_next import LiteLLMClient

client = LiteLLMClient(model="gpt-4o-mini", temperature=0.0, max_tokens=2048)
response = client.complete("Hello")
```

Supports all [LiteLLM providers](https://docs.litellm.ai/) (OpenAI, Anthropic, Google, Ollama, etc.).

### InstructorClient

Wraps any LLM client with Pydantic validation for more reliable structured outputs:

```python
from ace_next import InstructorClient, LiteLLMClient

client = InstructorClient(LiteLLMClient(model="ollama/gemma3:1b"))
```

---

## Observability

### OpikStep

Append to any pipeline for automatic tracing and cost tracking:

```python
from ace_next import OpikStep

OpikStep(project_name="my-experiment", tags=["training"])
```

### register_opik_litellm_callback

Standalone LLM cost tracking without pipeline traces:

```python
from ace_next import register_opik_litellm_callback

register_opik_litellm_callback(project_name="my-experiment")
```

See [Opik Observability](../integrations/opik.md).

---

## Prompts

The default prompts are v2.1 (built into `ace_next`). Pass a custom template via `prompt_template`:

```python
agent = Agent(llm, prompt_template="Custom prompt with {skillbook}, {question}, {context}")
reflector = Reflector(llm, prompt_template="Custom reflector prompt ...")
skill_manager = SkillManager(llm, prompt_template="Custom skill manager prompt ...")
```

See [Prompt Engineering](../guides/prompts.md).
