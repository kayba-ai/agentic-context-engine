# API Reference

Quick reference for the most-used classes and functions in `ace`.

## Runners

### ACELiteLLM

Simple self-improving conversational agent.

```python
from ace import ACELiteLLM

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
from ace import ACE, Agent, Reflector, SkillManager, Skillbook, SimpleEnvironment

runner = ACE.from_roles(
    agent=Agent("gpt-4o-mini"),
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
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
from ace import BrowserUse

runner = BrowserUse.from_model(browser_llm=my_llm, ace_model="gpt-4o-mini")
results = runner.run("Find the top post on Hacker News")
```

See [Browser-Use Integration](../integrations/browser-use.md).

### LangChain

Wrap LangChain Runnables with learning.

```python
from ace import LangChain

runner = LangChain.from_model(my_chain, ace_model="gpt-4o-mini")
results = runner.run([{"input": "Summarize this document"}])
```

See [LangChain Integration](../integrations/langchain.md).

### ClaudeCode

Claude Code CLI with learning.

```python
from ace import ClaudeCode

runner = ClaudeCode.from_model(working_dir="./project", ace_model="gpt-4o-mini")
results = runner.run("Add unit tests for utils.py")
```

See [Claude Code Integration](../integrations/claude-code.md).

### ClaudeSDKExecuteStep / ClaudeSDKToTrace

Direct Anthropic Messages API steps for custom pipelines.

```python
from ace import Pipeline, Reflector, SkillManager, Skillbook, learning_tail
from ace.integrations import ClaudeSDKExecuteStep, ClaudeSDKToTrace

skillbook = Skillbook()
pipe = Pipeline([
    ClaudeSDKExecuteStep(model="claude-sonnet-4-20250514"),
    ClaudeSDKToTrace(),
    *learning_tail(Reflector("gpt-4o-mini"), SkillManager("gpt-4o-mini"), skillbook),
])
```

`ClaudeSDKResult` and `ToolCall` are Pydantic models, so token counts, latency,
tool calls, and serialization are validated before the learning tail consumes
the trace.

See [Claude SDK Integration](../integrations/claude-sdk.md).

---

## Roles

### Agent

Produces answers using the current skillbook.

```python
from ace import Agent

agent = Agent("gpt-4o-mini")
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
from ace import Reflector

reflector = Reflector("gpt-4o-mini")
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
from ace import SkillManager

skill_manager = SkillManager("gpt-4o-mini")
sm_output = skill_manager.update_skills(
    reflections=(reflection,),
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
from ace import Skillbook

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
from ace import Sample

sample = Sample(
    question="What is 2+2?",
    context="Show your work",
    ground_truth="4",
)
```

### EnvironmentResult

```python
from ace import EnvironmentResult

result = EnvironmentResult(
    feedback="Correct!",
    ground_truth="4",
    metrics={"accuracy": 1.0},
)
```

### UpdateOperation

```python
from ace import UpdateOperation

op = UpdateOperation(
    type="ADD",
    section="Math",
    content="Break problems into smaller steps",
    reflection_index=0,
    reflection_indices=[0, 1],
    skill_id="math-00001",
)
```

Operations: `ADD`, `UPDATE`, `TAG`, `REMOVE`. See [Update Operations](../concepts/updates.md).

### DeduplicationConfig

**Requires:** `uv add ace-framework[deduplication]`

```python
from ace import DeduplicationConfig

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
from ace import TaskEnvironment, EnvironmentResult

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

## Providers

### resolve_model

Resolve a model string to a PydanticAI model instance:

```python
from ace.providers import resolve_model

model = resolve_model("gpt-4o-mini")
```

Supports any [LiteLLM model](https://docs.litellm.ai/) or PydanticAI-native identifier.

### ACEModelConfig

Configuration for model selection per role:

```python
from ace.providers import ACEModelConfig

config = ACEModelConfig.from_toml("ace.toml")
agent_model = config.for_role("agent")
```

---

## Observability

### OpikStep

Append to any pipeline for automatic tracing and cost tracking:

```python
from ace import OpikStep

OpikStep(project_name="my-experiment", tags=["training"])
```

### register_opik_litellm_callback

Standalone LLM cost tracking without pipeline traces:

```python
from ace import register_opik_litellm_callback

register_opik_litellm_callback(project_name="my-experiment")
```

See [Opik Observability](../integrations/opik.md).

---

## Recursive Reflector (RR)

PydanticAI agent-based trace analyser with tools for code execution and sub-agent analysis.

### RRStep

Drop-in replacement for `Reflector` — satisfies both `StepProtocol` and `ReflectorLike`.

```python
from ace.rr import RRStep, RRConfig

rr = RRStep(
    "gpt-4o-mini",                          # Model string
    config=RRConfig(max_llm_calls=10),      # Configuration
)

# As drop-in reflector
ace = ACELiteLLM.from_model("gpt-4o-mini", reflector=rr)

# As pipeline step
pipe = Pipeline([..., rr, ...])
```

### RRConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `20` | Max REPL loop iterations |
| `timeout` | `30.0` | Per-execution timeout in seconds (Unix only) |
| `max_llm_calls` | `30` | Shared budget across main + sub-agent calls |
| `max_context_chars` | `50_000` | Message history trim threshold |
| `max_output_chars` | `20_000` | Per-execution output truncation limit |
| `enable_subagent` | `True` | Enable `ask_llm()` in sandbox |
| `subagent_model` | `None` | Sub-agent model (None = same as main) |
| `subagent_max_tokens` | `8192` | Max tokens for sub-agent responses |
| `subagent_temperature` | `0.3` | Sub-agent temperature |
| `subagent_system_prompt` | `None` | Custom sub-agent system prompt |
| `enable_fallback_synthesis` | `True` | Attempt LLM synthesis on timeout |

### Sandbox Functions

Available inside sandbox code generated by the LLM:

| Function | Description |
|----------|-------------|
| `FINAL(value)` | Submit final result dict (terminates the loop) |
| `FINAL_VAR(name)` | Submit a named variable as the result |
| `SHOW_VARS()` | Print available variables (debugging) |
| `ask_llm(question, context="", mode="analysis")` | Query sub-agent LLM |
| `llm_query(prompt)` | Alias for `ask_llm(prompt, "")` |

### TraceContext

Structured trace wrapper with factory methods:

| Factory | Input |
|---------|-------|
| `TraceContext.from_agent_output(output)` | `AgentOutput` |
| `TraceContext.from_conversation_history(msgs)` | `list[dict]` |
| `TraceContext.from_tau_simulation(msgs, system_prompt)` | TAU-bench messages |
| `TraceContext.from_browser_use(history)` | browser-use `AgentHistory` |
| `TraceContext.from_langchain(steps)` | LangChain intermediate steps |
| `TraceContext.from_reasoning_string(text)` | Raw reasoning string |
| `TraceContext.combine(traces)` | Merge multiple traces |

See [RR_DESIGN.md](../RR_DESIGN.md) for the full architecture reference.

---

## Prompts

The default prompts are v2.1 (built into `ace`). Pass a custom template via `prompt_template`:

```python
agent = Agent("gpt-4o-mini", prompt_template="Custom prompt with {skillbook}, {question}, {context}")
reflector = Reflector("gpt-4o-mini", prompt_template="Custom reflector prompt ...")
skill_manager = SkillManager("gpt-4o-mini", prompt_template="Custom skill manager prompt ...")
```

See [Prompt Engineering](../guides/prompts.md).
