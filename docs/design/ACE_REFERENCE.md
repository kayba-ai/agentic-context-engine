# ACE Code Reference

> Full code examples, API signatures, step implementations, and usage patterns.

For architecture and concepts, see [ACE_ARCHITECTURE.md](ACE_ARCHITECTURE.md).
For design decisions and rejected alternatives, see [ACE_DECISIONS.md](ACE_DECISIONS.md).

---

## Public API

All pipeline primitives, ACE steps, and context types are importable from `ace`:

```python
# Pipeline engine
from ace import Pipeline, Branch, MergeStrategy, StepProtocol, SampleResult

# ACE context
from ace import ACEStepContext, SkillbookView

# Runner base class (for custom runners)
from ace import ACERunner

# Core steps
from ace import (
    AgentStep, EvaluateStep, ReflectStep, UpdateStep,
    ApplyStep,
    DeduplicateStep, CheckpointStep, LoadTracesStep, ExportSkillbookMarkdownStep,
    ObservabilityStep, PersistStep, learning_tail,
)
```

Integration steps live in `ace.integrations` (they have framework-specific dependencies):

```python
from ace.integrations.browser_use import BrowserExecuteStep, BrowserToTrace
from ace.integrations.langchain import LangChainExecuteStep, LangChainToTrace
from ace.integrations.claude_code import ClaudeCodeExecuteStep, ClaudeCodeToTrace
from ace.integrations.claude_sdk import ClaudeSDKExecuteStep, ClaudeSDKToTrace
from ace.integrations.openclaw import OpenClawToTraceStep
```

Every runner also exposes a `build_steps()` classmethod that returns the step list it would compose internally.

---

## Core Type Definitions

### Sample

```python
@dataclass
class Sample:
    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: dict = field(default_factory=dict)
    id: str | None = None
```

### ACESample protocol

```python
class ACESample(Protocol):
    """Minimal interface that Sample satisfies."""

    @property
    def question(self) -> str: ...

    @property
    def context(self) -> str: ...

    @property
    def ground_truth(self) -> str | None: ...

    @property
    def metadata(self) -> dict: ...
```

### SkillbookView

```python
class SkillbookView:
    """Read-only projection of a Skillbook. Safe on a frozen context."""

    __slots__ = ("_sb",)

    def __init__(self, skillbook: Skillbook) -> None:
        self._sb = skillbook

    def as_prompt(self) -> str:
        return self._sb.as_prompt()

    def get_skill(self, skill_id: str) -> Skill | None:
        return self._sb.get_skill(skill_id)

    def skills(self, include_invalid: bool = False) -> list[Skill]:
        return self._sb.skills(include_invalid=include_invalid)

    def stats(self) -> dict[str, object]:
        return self._sb.stats()

    def __len__(self) -> int:
        return len(self._sb.skills())

    def __iter__(self):
        return iter(self._sb.skills())

    def __repr__(self) -> str:
        return f"SkillbookView({len(self)} skills)"
```

### ACEStepContext

```python
@dataclass(frozen=True)
class ACEStepContext(StepContext):
    """Immutable context for the ACE pipeline.

    The skillbook field is a SkillbookView (read-only). Steps that need to
    write to the skillbook receive the real Skillbook via constructor injection.
    """

    sample: ACESample | None = None
    skillbook: SkillbookView | None = None
    trace: object | None = None
    agent_output: AgentOutput | None = None
    reflections: tuple[ReflectorOutput, ...] = ()
    skill_manager_output: UpdateBatch | None = None
    epoch: int = 1
    total_epochs: int = 1
    step_index: int = 0
    total_steps: int | None = None
    global_sample_index: int = 0
```

---

## Protocol Definitions

All protocols live in `ace/protocols/` (one file per protocol, re-exported from `__init__.py`).

```python
class AgentLike(Protocol):
    def generate(self, question: str, context: str, skillbook: SkillbookView,
                 reflection: str | None = None, **kwargs) -> AgentOutput: ...

class ReflectorLike(Protocol):
    def reflect(self, question: str, agent_output: AgentOutput, skillbook: SkillbookView,
                ground_truth: str | None = None, feedback: str | None = None,
                **kwargs) -> ReflectorOutput: ...

class SkillManagerLike(Protocol):
    def update_skills(self, reflections: tuple[ReflectorOutput, ...],
                      skillbook: SkillbookView, question_context: str,
                      progress: str, **kwargs) -> SkillManagerOutput: ...

class DeduplicationManagerLike(Protocol):
    def get_similarity_report(self, skillbook: Skillbook) -> str | None: ...
```

---

## Step Implementations

### AgentStep

```python
class AgentStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output"})

    def __init__(self, agent: AgentLike) -> None:
        self.agent = agent

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        agent_output = self.agent.generate(
            question=ctx.sample.question,
            context=ctx.sample.context,
            skillbook=ctx.skillbook,       # SkillbookView (read-only)
            sample=ctx.sample,
        )
        return ctx.replace(agent_output=agent_output)
```

### EvaluateStep

Bridges the execute head (typed ACE objects) to the learning tail (raw traces). Optionally evaluates against a `TaskEnvironment`.

```python
class EvaluateStep:
    requires = frozenset({"sample", "agent_output"})
    provides = frozenset({"trace"})

    def __init__(self, environment: TaskEnvironment | None = None) -> None:
        self.environment = environment

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = {
            "question": ctx.sample.question,
            "context": ctx.sample.context,
            "ground_truth": ctx.sample.ground_truth,
            "reasoning": ctx.agent_output.reasoning,
            "answer": ctx.agent_output.final_answer,
            "skill_ids": ctx.agent_output.skill_ids,
        }
        if self.environment:
            result = self.environment.evaluate(
                sample=ctx.sample, agent_output=ctx.agent_output,
            )
            trace["feedback"] = result.feedback
        return ctx.replace(trace=trace)
```

### ReflectStep

Handles two trace formats: (1) dict from EvaluateStep — extracts known fields; (2) any other object from TraceAnalyser or integrations — passes raw trace via `**kwargs`.

```python
class ReflectStep:
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    async_boundary = True
    max_workers = 3

    def __init__(self, reflector: ReflectorLike) -> None:
        self.reflector = reflector

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = ctx.trace

        if isinstance(trace, dict):
            agent_output = AgentOutput(
                reasoning=trace.get("reasoning", ""),
                final_answer=trace.get("answer", ""),
                skill_ids=trace.get("skill_ids", []),
            )
            reflection = self.reflector.reflect(
                question=trace.get("question", ""),
                agent_output=agent_output,
                skillbook=ctx.skillbook,
                ground_truth=trace.get("ground_truth"),
                feedback=trace.get("feedback"),
            )
        else:
            reflection = self.reflector.reflect(
                question="",
                agent_output=AgentOutput(reasoning="", final_answer=""),
                skillbook=ctx.skillbook,
                trace=trace,
            )

        return ctx.replace(reflections=(reflection,))
```

### UpdateStep

Pure — generates update operations from reflections and current skillbook state.

```python
class UpdateStep:
    requires = frozenset({"reflections", "skillbook"})
    provides = frozenset({"skill_manager_output"})

    max_workers = 1

    def __init__(self, skill_manager: SkillManagerLike) -> None:
        self.skill_manager = skill_manager

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        output = self.skill_manager.update_skills(
            reflections=ctx.reflections,
            skillbook=ctx.skillbook,
            question_context=...,
            progress=...,
        )
        return ctx.replace(skill_manager_output=output.update)
```

### ApplyStep

Side-effect step — applies the enriched update batch to the real `Skillbook`.

```python
class ApplyStep:
    requires = frozenset({"skill_manager_output"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, skillbook: Skillbook) -> None:
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        self.skillbook.apply_update(ctx.skill_manager_output)
        return ctx
```

### DeduplicateStep

Optional — consolidates similar skills at a configurable interval.

```python
class DeduplicateStep:
    requires = frozenset({"global_sample_index"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, manager: DeduplicationManagerLike, skillbook: Skillbook, *, interval: int = 10) -> None:
        self.manager = manager
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx
        report = self.manager.get_similarity_report(self.skillbook)
        if report:
            logger.info("DeduplicateStep: similarity report at sample %d:\n%s",
                        ctx.global_sample_index, report)
        return ctx
```

### CheckpointStep

Optional — periodically saves the skillbook to disk.

```python
class CheckpointStep:
    requires = frozenset({"global_sample_index"})
    provides = frozenset()

    def __init__(self, directory: str | Path, skillbook: Skillbook, *, interval: int = 10) -> None:
        self.directory = Path(directory)
        self.skillbook = skillbook
        self.interval = interval

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        if ctx.global_sample_index % self.interval != 0:
            return ctx
        self.directory.mkdir(parents=True, exist_ok=True)
        self.skillbook.save_to_file(str(self.directory / f"checkpoint_{ctx.global_sample_index}.json"))
        self.skillbook.save_to_file(str(self.directory / "latest.json"))
        return ctx
```

### LoadTracesStep

Generic JSONL file loader — reads a file path from `ctx.sample`, parses each line as JSON.

```python
class LoadTracesStep:
    requires = frozenset({"sample"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        path = Path(ctx.sample)
        events: list[dict] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return ctx.replace(trace=events)
```

### PersistStep

Writes the current skillbook to an external file (e.g. `CLAUDE.md` for Claude Code).

```python
class PersistStep:
    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __init__(self, target_path: str | Path, skillbook: Skillbook) -> None:
        self.target_path = Path(target_path)
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        self.skillbook.save_to_file(str(self.target_path))
        return ctx
```

### ExportSkillbookMarkdownStep

Exports the skillbook as a human-readable markdown file, grouped by section.

```python
class ExportSkillbookMarkdownStep:
    requires = frozenset({"skillbook"})
    provides = frozenset()

    def __init__(self, path: str | Path, skillbook: Skillbook) -> None:
        self.path = Path(path)
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        # Rewrites the markdown file from the current skillbook state
        ...
        return ctx
```

---

## Factory Methods

### `learning_tail()` — reusable learning steps

```python
# ace/steps/__init__.py

def learning_tail(
    reflector: ReflectorLike,
    skill_manager: SkillManagerLike,
    skillbook: Skillbook,
    *,
    dedup_manager: DeduplicationManagerLike | None = None,
    dedup_interval: int = 10,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 10,
) -> list[StepProtocol[ACEStepContext]]:
    """Return the standard ACE learning steps."""
    steps: list[StepProtocol[ACEStepContext]] = [
        ReflectStep(reflector),
        UpdateStep(skill_manager),
        ApplyStep(skillbook),
    ]
    if dedup_manager:
        steps.append(DeduplicateStep(dedup_manager, skillbook, interval=dedup_interval))
    if checkpoint_dir:
        steps.append(CheckpointStep(checkpoint_dir, skillbook, interval=checkpoint_interval))
    return steps
```

### TraceAnalyser `from_roles`

```python
@classmethod
def from_roles(cls, *, reflector, skill_manager, skillbook=None,
               dedup_manager=None, dedup_interval=10,
               checkpoint_dir=None, checkpoint_interval=10,
               extra_steps=None):
    skillbook = skillbook or Skillbook()
    steps = learning_tail(
        reflector, skill_manager, skillbook,
        dedup_manager=dedup_manager, dedup_interval=dedup_interval,
        checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
    )
    if extra_steps:
        steps.extend(extra_steps)
    return cls(pipeline=Pipeline(steps), skillbook=skillbook)
```

### ACE `from_roles`

```python
@classmethod
def from_roles(cls, *, agent, reflector, skill_manager, environment=None,
               skillbook=None, dedup_manager=None, dedup_interval=10,
               checkpoint_dir=None, checkpoint_interval=10,
               extra_steps=None):
    skillbook = skillbook or Skillbook()
    steps = [
        AgentStep(agent),
        EvaluateStep(environment),
        *learning_tail(
            reflector, skill_manager, skillbook,
            dedup_manager=dedup_manager, dedup_interval=dedup_interval,
            checkpoint_dir=checkpoint_dir, checkpoint_interval=checkpoint_interval,
        ),
    ]
    if extra_steps:
        steps.extend(extra_steps)
    return cls(pipeline=Pipeline(steps), skillbook=skillbook)
```

---

## Runner Implementations

### ACERunner base

```python
class ACERunner:
    """Shared runner infrastructure for all ACE runners."""

    def __init__(self, pipeline: Pipeline, skillbook: Skillbook) -> None:
        self.pipeline = pipeline
        self.skillbook = skillbook

    def save(self, path: str) -> None:
        self.skillbook.save_to_file(path)

    def wait_for_background(self, timeout: float | None = None) -> None:
        self.pipeline.wait_for_background(timeout)

    @property
    def learning_stats(self) -> dict:
        return self.pipeline.background_stats()
```

### Generic run loop (`_run`)

```python
def _run(self, items, *, epochs, wait=True, **kwargs) -> list[SampleResult]:
    if epochs > 1 and not isinstance(items, Sequence):
        raise ValueError("Multi-epoch requires a Sequence, not a consumed Iterable.")

    results: list[SampleResult] = []
    n = len(items) if isinstance(items, Sequence) else None

    for epoch in range(1, epochs + 1):
        contexts = [
            self._build_context(item, epoch=epoch, total_epochs=epochs,
                                index=idx, total=n,
                                global_sample_index=(epoch - 1) * n + idx if n is not None else idx,
                                **kwargs)
            for idx, item in enumerate(items, start=1)
        ]
        epoch_results = self.pipeline.run(contexts)
        results.extend(epoch_results)

    if wait:
        self.pipeline.wait_for_background()
    return results
```

### TraceAnalyser

```python
class TraceAnalyser(ACERunner):
    """Analyse pre-recorded traces to build a skillbook."""

    @classmethod
    def from_roles(cls, *, reflector, skill_manager, skillbook=None, **kwargs) -> "TraceAnalyser": ...

    def run(self, traces: Sequence[Any], epochs: int = 1, *, wait: bool = True) -> list[SampleResult]:
        return self._run(traces, epochs=epochs, wait=wait)

    def _build_context(self, raw_trace, *, epoch, total_epochs, index, total,
                       global_sample_index) -> ACEStepContext:
        return ACEStepContext(
            skillbook=SkillbookView(self.skillbook),
            trace=raw_trace,
            metadata={...},                         # inferred trace identity for provenance
            epoch=epoch, total_epochs=total_epochs,
            step_index=index, total_steps=total,
            global_sample_index=global_sample_index,
        )
```

### ACE

```python
class ACE(ACERunner):
    """Live adaptive pipeline: Agent → Evaluate → Reflect → Update → Apply."""

    @classmethod
    def from_roles(cls, *, agent, reflector, skill_manager,
                   environment=None, skillbook=None, **kwargs) -> "ACE": ...

    def run(self, samples, epochs=1, *, wait=True) -> list[SampleResult]:
        return self._run(samples, epochs=epochs, wait=wait)

    def _build_context(self, sample, *, epoch, total_epochs, index, total,
                       global_sample_index, **_) -> ACEStepContext:
        return ACEStepContext(
            sample=sample,
            skillbook=SkillbookView(self.skillbook),
            metadata={...},
            epoch=epoch, total_epochs=total_epochs,
            step_index=index, total_steps=total,
            global_sample_index=global_sample_index,
        )
```

### Integration runner pattern

```python
class BrowserUse(ACERunner):
    """Browser-use agent with ACE learning pipeline."""

    @classmethod
    def from_roles(cls, *, browser_llm, reflector, skill_manager,
                   skillbook=None, **kwargs):
        skillbook = skillbook or Skillbook()
        steps = [
            BrowserExecuteStep(browser_llm),
            BrowserToTrace(),
            *learning_tail(reflector, skill_manager, skillbook, **kwargs),
        ]
        return cls(pipeline=Pipeline(steps), skillbook=skillbook)

    @classmethod
    def from_model(cls, browser_llm, *, ace_model="gpt-4o-mini",
                   ace_max_tokens=2048, ace_temperature=0.0, **kwargs) -> BrowserUse:
        return cls.from_roles(
            browser_llm=browser_llm,
            reflector=Reflector(ace_model),
            skill_manager=SkillManager(ace_model),
            **kwargs,
        )

    def run(self, tasks, epochs=1, *, wait=True):
        return self._run(tasks, epochs=epochs, wait=wait)

    def _build_context(self, task, *, epoch, total_epochs, index, total,
                       global_sample_index, **_):
        return ACEStepContext(
            sample=task,    # raw string — not wrapped in Sample
            skillbook=SkillbookView(self.skillbook),
            epoch=epoch, total_epochs=total_epochs,
            step_index=index, total_steps=total,
            global_sample_index=global_sample_index,
        )
```

### ACELiteLLM

```python
class ACELiteLLM:
    def __init__(self, model="gpt-4o-mini", *, skillbook=None, environment=None,
                 reflector=None, skill_manager=None, ...):
        self.agent = Agent(model)
        self.reflector = reflector or Reflector(model)
        self.skill_manager = skill_manager or SkillManager(model)
        self._skillbook = skillbook or Skillbook()
        self.environment = environment
        self._ace: ACE | None = None
        self._analyser: TraceAnalyser | None = None

    @classmethod
    def from_model(cls, model="gpt-4o-mini", *, max_tokens=2048,
                   temperature=0.0, **kwargs) -> ACELiteLLM:
        return cls(model, **kwargs)

    def ask(self, question, context="") -> str:
        """Direct Agent call — no pipeline. Stores interaction for learn_from_feedback()."""
        ...

    def learn(self, samples, environment=None, epochs=1, *, wait=True):
        """Delegate to lazy-init ACE runner."""
        return self._get_ace(environment).run(samples, epochs=epochs, wait=wait)

    def learn_from_traces(self, traces, epochs=1, *, wait=True):
        """Delegate to lazy-init TraceAnalyser."""
        return self._get_analyser().run(traces, epochs=epochs, wait=wait)

    def learn_from_feedback(self, feedback, ground_truth=None) -> bool:
        """Manual single-shot learning from last ask() call."""
        ...

    def load(self, path):
        """Load skillbook — invalidates cached runners (stale refs)."""
        self._skillbook = Skillbook.load_from_file(path)
        self._ace = None
        self._analyser = None
```

---

## Role Implementations

### Agent

Produces answers using the current skillbook. Formats the prompt, calls PydanticAI with `AgentOutput` as the structured result type, extracts cited skill IDs via `extract_cited_skill_ids()`.

```python
agent = Agent("gpt-4o-mini")
output = agent.generate(
    question="What is the capital of France?",
    context="Answer concisely",
    skillbook=skillbook,
)
# output.final_answer == "Paris"
# output.skill_ids == ["geography-00001"]
```

### Reflector

Single-pass analysis. Builds a skillbook excerpt from cited IDs, formats the prompt, calls PydanticAI with `ReflectorOutput`.

```python
reflector = Reflector("gpt-4o-mini")
reflection = reflector.reflect(
    question="What is 2+2?",
    agent_output=agent_output,
    skillbook=skillbook,
    ground_truth="4",
    feedback="Correct!",
)
# reflection.key_insight, reflection.skill_tags
```

### SkillManager

Transforms reflections into skillbook updates. Serializes each `ReflectorOutput` into JSON, calls PydanticAI with `SkillManagerOutput`.

```python
sm = SkillManager("gpt-4o-mini")
output = sm.update_skills(
    reflections=(reflection_output,),
    skillbook=skillbook,
    question_context="Math problem solving",
    progress="5/10 correct",
)
skillbook.apply_update(output.update)
```

### Shared helpers (`implementations/helpers.py`)

| Function | Purpose |
|---|---|
| `extract_cited_skill_ids(text)` | Regex `[section-00001]` → deduplicated list of IDs |
| `format_optional(value)` | Returns `"(none)"` for falsy values |
| `make_skillbook_excerpt(skillbook, skill_ids)` | Builds `[id] content` lines for cited skills |

### Prompt templates (`implementations/prompts.py`)

| Constant | Role |
|---|---|
| `AGENT_PROMPT` | Agent prompt with strategic problem-solving protocol |
| `REFLECTOR_PROMPT` | Reflector prompt with diagnostic analysis protocol |
| `SKILL_MANAGER_PROMPT` | SkillManager prompt with atomic strategy creation |
| `SKILLBOOK_USAGE_INSTRUCTIONS` | Shared text for skillbook usage guidance |

Also exports `wrap_skillbook_for_external_agent(skillbook)` — the canonical function for injecting skillbook context into external agentic systems.

---

## Integration Step Examples

### Execute step pattern

```python
class BrowserExecuteStep:
    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(self, browser_llm, browser=None, **agent_kwargs) -> None:
        self.browser_llm = browser_llm
        self.browser = browser
        self.agent_kwargs = agent_kwargs

    async def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task: str = ctx.sample

        # INJECT — prepend skillbook context
        enhanced_task = self._inject(task, ctx.skillbook)

        # EXECUTE — run browser-use agent
        agent = Agent(task=enhanced_task, llm=self.browser_llm, **self.agent_kwargs)
        history = await agent.run()

        result = BrowserResult(
            task=task, success=True, output=history.final_result(),
            steps_count=history.number_of_steps(),
            chronological_steps=..., raw_history=history,
        )
        return ctx.replace(trace=result)
```

### ToTrace step pattern

```python
class SomeToTrace:
    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        r: SomeResult = ctx.trace
        trace = {
            "question": r.task,
            "reasoning": r.execution_trace,
            "answer": r.output,
            "skill_ids": r.cited_skill_ids,
            "feedback": f"Task {'succeeded' if r.success else 'failed'}",
            "ground_truth": None,
        }
        return ctx.replace(trace=trace)
```

### Trace file pipeline composition

```python
steps = [
    LoadTracesStep(),
    OpenClawToTraceStep(),
    *learning_tail(reflector, skill_manager, skillbook),
]
```

### Custom pipeline with `learning_tail`

```python
from ace.steps import learning_tail

skillbook = Skillbook.load_from_file("expert.json")
steps = [
    MyCustomExecuteStep(my_agent),
    MyValidationStep(),
    *learning_tail(reflector, skill_manager, skillbook, dedup_manager=dedup),
]
runner = ACERunner(Pipeline(steps), skillbook)
```

---

## Provider Resolution

```python
# ace/providers/pydantic_ai.py — resolve_model()
# Routes LiteLLM model strings to PydanticAI:

# 1. PydanticAI-native prefix → pass through
#    "openai:gpt-4o" → "openai:gpt-4o"

# 2. LiteLLM prefix matching native provider → rewrite
#    "bedrock/model" → "bedrock:model"

# 3. Fallback → litellm: prefix
#    "ollama/llama3" → "litellm:ollama/llama3"
```

Mapped prefixes: `anthropic`, `azure`, `azure_ai`, `bedrock`, `cohere`, `deepseek`, `groq`, `mistral`, `openrouter`, `vertex_ai`.

Install native provider extras for faster calls:

```bash
uv add "pydantic-ai-slim[anthropic]"    # uses ANTHROPIC_API_KEY
uv add "pydantic-ai-slim[openai]"       # uses OPENAI_API_KEY
uv add "pydantic-ai-slim[bedrock]"      # uses AWS credentials
uv add "pydantic-ai-slim[anthropic,openai,bedrock]"  # multiple
```

---

## Config types

```python
@dataclass
class ModelConfig:
    """Which model to use for a role. No secrets."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    extra_params: dict[str, Any] | None = None

@dataclass
class ACEModelConfig:
    """Model selection per ACE role."""
    default: ModelConfig
    agent: ModelConfig | None = None
    reflector: ModelConfig | None = None
    skill_manager: ModelConfig | None = None

    def for_role(self, role: str) -> ModelConfig: ...
```

### `ace.toml` example

```toml
[default]
model = "gpt-4o-mini"

[agent]
model = "claude-sonnet-4-20250514"
max_tokens = 4096

[reflector]
model = "gpt-4o-mini"
```

### Registry (`ace/providers/registry.py`)

- `validate_connection(model, api_key?)` — 3-token LLM call to verify auth
- `get_required_key(model)` — returns `(provider, env_var)`
- `search_models(query?, provider?)` — searches LiteLLM's model cost database
- `suggest_models(typo)` — fuzzy match for typos
- `available_providers()` — lists providers with key status

---

## Usage Examples

### TraceAnalyser — learn from browser-use history

```python
from ace import TraceAnalyser, Reflector, SkillManager

traces = [
    {
        "task": "Find the cheapest flight to Tokyo",
        "output": "$450 on ANA, departing March 15",
        "feedback": "Correct price found in 8 steps",
        "reasoning": "Step 1: Navigate to Google Flights...",
    },
    {
        "task": "Book a hotel in Shibuya",
        "output": "Failed: could not find checkout button",
        "feedback": "Task failed after 15 steps — checkout button was behind a cookie modal",
        "reasoning": "Step 1: Navigate to Booking.com...",
    },
]

analyser = TraceAnalyser.from_roles(reflector=Reflector("gpt-4o-mini"), skill_manager=SkillManager("gpt-4o-mini"))
results = analyser.run(traces, epochs=2)
analyser.save("travel_agent.json")
```

### ACE — live Q&A training

```python
from ace import ACE, Sample, SimpleEnvironment, Agent, Reflector, SkillManager

samples = [
    Sample(question="Capital of France?", ground_truth="Paris"),
    Sample(question="Largest ocean?", ground_truth="Pacific"),
]

ace = ACE.from_roles(
    agent=Agent("gpt-4o-mini"),
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
    environment=SimpleEnvironment(),
)
results = ace.run(samples, epochs=3)
ace.save("geography.json")
```

### ACE — without environment

```python
ace = ACE.from_roles(
    agent=Agent("gpt-4o-mini"),
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
)
results = ace.run(samples, epochs=3)
```

### ACE — with checkpoints and deduplication

```python
from ace import ACE, Agent, Reflector, SkillManager, SimpleEnvironment
from ace.deduplication import DeduplicationManager
from ace.protocols.deduplication import DeduplicationConfig

ace = ACE.from_roles(
    agent=Agent("gpt-4o-mini"),
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
    environment=SimpleEnvironment(),
    dedup_manager=DeduplicationManager(DeduplicationConfig(similarity_threshold=0.85)),
    checkpoint_dir="./checkpoints",
    checkpoint_interval=10,
)
# Pipeline: Agent → Evaluate → Reflect → Update → Apply → Deduplicate → Checkpoint
results = ace.run(samples, epochs=3)
```

### Integration — browser-use runner

```python
from ace import BrowserUse, Reflector, SkillManager
from langchain_openai import ChatOpenAI

browser_llm = ChatOpenAI(model="gpt-4o")

# Explicit construction
runner = BrowserUse.from_roles(
    browser_llm=browser_llm,
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
)

# Or convenience construction
runner = BrowserUse.from_model(browser_llm, ace_model="gpt-4o-mini")

results = runner.run(["Find top HN post", "Check weather in Tokyo"])
runner.save("browser_expert.json")
```

### Integration — LangChain runner

```python
from ace import LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("Answer: {input}") | ChatOpenAI(model="gpt-4o")

runner = LangChain.from_model(chain, ace_model="gpt-4o-mini")
results = runner.run([{"input": "What is ACE?"}, {"input": "Explain skillbooks"}])
runner.save("chain_expert.json")
```

### Integration — Claude Code runner

```python
from ace import ClaudeCode

runner = ClaudeCode.from_model(working_dir="./my_project", ace_model="gpt-4o-mini")
results = runner.run(["Add unit tests for utils.py", "Refactor the auth module"])
runner.save("code_expert.json")
```

### ACELiteLLM — conversational agent with learning

```python
from ace import ACELiteLLM, SimpleEnvironment, Sample

ace = ACELiteLLM.from_model("gpt-4o-mini")

# Direct Q&A (no pipeline)
answer = ace.ask("What is the capital of France?")

# Batch learning
samples = [
    Sample(question="Capital of France?", ground_truth="Paris"),
    Sample(question="Largest ocean?", ground_truth="Pacific"),
]
ace.learn(samples, environment=SimpleEnvironment(), epochs=3)

# Manual feedback learning from last ask()
ace.ask("What is 2+2?")
ace.learn_from_feedback("The answer should be 4", ground_truth="4")

ace.save("learned.json")

# With Recursive Reflector
from ace import RRStep, RRConfig
rr = RRStep("gpt-4o-mini", config=RRConfig(max_requests=20))
ace = ACELiteLLM("gpt-4o-mini", reflector=rr)
```

### Fire-and-forget — results while learning continues

```python
ace = ACE.from_roles(
    agent=Agent("gpt-4o-mini"),
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
)

# wait=False: returns after foreground steps (Agent + Evaluate)
results = ace.run(samples, epochs=1, wait=False)

# Use agent outputs immediately
for r in results:
    print(r.output.agent_output.final_answer)

# Check learning progress
print(ace.learning_stats)
# {"active": 3, "completed": 12}

# Block when you need the skillbook finalised
ace.wait_for_background(timeout=60.0)
ace.save("learned.json")
```

### Mixed workflow — batch then live

```python
from ace import TraceAnalyser, ACE, Skillbook
from ace.implementations import Agent, Reflector, SkillManager

reflector = Reflector("gpt-4o-mini")
skill_manager = SkillManager("gpt-4o-mini")

# Phase 1: build skillbook from historical traces
skillbook = Skillbook()
analyser = TraceAnalyser.from_roles(
    reflector=reflector, skill_manager=skill_manager, skillbook=skillbook,
)
analyser.run(historical_traces, epochs=3)

# Phase 2: deploy with live learning (reuse the evolved skillbook)
ace = ACE.from_roles(
    agent=Agent("gpt-4o-mini"),
    reflector=reflector, skill_manager=skill_manager, skillbook=skillbook,
)
ace.run(live_samples, epochs=1)
ace.save("production.json")
```

### Offline learning from integration traces

```python
# Record browser executions
histories = [await agent.run(task) for task in tasks]

# Feed raw histories directly — Reflector analyses them as-is
analyser = TraceAnalyser.from_roles(
    reflector=Reflector("gpt-4o-mini"),
    skill_manager=SkillManager("gpt-4o-mini"),
)
analyser.run(histories, epochs=2)
analyser.save("browser_expert.json")
```
