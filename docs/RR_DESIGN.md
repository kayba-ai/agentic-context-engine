# Recursive Reflector (RR) Design

Design document for the Recursive Reflector module (`ace_next/rr/`). The RR is a REPL-based trace analyser that iteratively calls an LLM to generate Python code, executes it in a sandbox, and builds structured reflections from agent execution traces.

---

## Overview

The Recursive Reflector replaces the single-pass `Reflector` with an iterative code-execution loop. Instead of asking the LLM for a one-shot analysis, RR gives the LLM a Python REPL with pre-loaded trace data and lets it explore, query a sub-agent, and submit findings when ready.

**Key properties:**

- Satisfies both `StepProtocol` and `ReflectorLike` — usable as a pipeline step or a drop-in reflector replacement.
- Extends `SubRunner` (from `ace_next/core/sub_runner.py`) — runs an inner `Pipeline` in a loop.
- Single shared `CallBudget` enforces combined LLM call limit across main calls and sub-agent calls.
- Produces `ReflectorOutput` with an enriched `raw["rr_trace"]` dict for downstream observability.

```python
from ace_next.rr import RRStep, RRConfig

# Drop-in replacement for Reflector
ace = ACELiteLLM(llm, reflector=RRStep(llm, config=RRConfig(max_iterations=10)))

# Or as a pipeline step
pipe = Pipeline([..., RRStep(llm), ...])
```

---

## Architecture

### REPL Loop

Each invocation of `RRStep` runs an iterative loop:

```
┌─────────────────────────────────────────────────────────┐
│  RRStep.run_loop()                                      │
│                                                         │
│  for each iteration (up to max_iterations):             │
│    ┌──────────┐   ┌──────────────┐   ┌──────────────┐  │
│    │LLMCallStep│ → │ExtractCodeStep│ → │SandboxExecStep│ │
│    └──────────┘   └──────────────┘   └──────────────┘  │
│         │                                    │          │
│         │              ┌──────────────┐      │          │
│         └──────────────│CheckResultStep│←─────┘          │
│                        └──────────────┘                 │
│                              │                          │
│                    ┌─────────┴──────────┐               │
│                    │                    │               │
│              FINAL() called?      Build feedback        │
│              ↓ yes                ↓ no                  │
│         Return result      Next iteration               │
└─────────────────────────────────────────────────────────┘
```

### Inner Pipeline Steps

Each iteration runs four steps sequentially:

| Step | Requires | Provides | Description |
|------|----------|----------|-------------|
| `LLMCallStep` | `messages` | `llm_response` | Trims message history, calls LLM (respects shared budget) |
| `ExtractCodeStep` | `llm_response` | `code`, `direct_response` | Extracts Python from response (3-layer fallback) |
| `SandboxExecStep` | `code` | `exec_result` | Executes code in `TraceSandbox` with timeout |
| `CheckResultStep` | `exec_result`, `messages`, `llm_response` | `terminated`, `reflection`, `feedback_messages` | Validates result, parses FINAL(), builds feedback |

### Dual Protocol Support

`RRStep` satisfies two protocols simultaneously:

```python
class RRStep(SubRunner):
    # StepProtocol — place in any Pipeline
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext: ...

    # ReflectorLike — use as drop-in reflector in runners
    def reflect(self, *, question, agent_output, skillbook, ...) -> ReflectorOutput: ...
```

---

## RRStep

### Constructor

```python
RRStep(
    llm: Any,                              # LLM client (must have complete_messages)
    config: Optional[RRConfig] = None,     # Configuration (defaults to RRConfig())
    prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,  # Customisable prompt
    subagent_llm: Any = None,              # Optional separate LLM for sub-agent
)
```

| Parameter | Description |
|-----------|-------------|
| `llm` | Main LLM client. Must expose `complete_messages(messages) -> response` where `response.text` is the text. |
| `config` | `RRConfig` instance controlling iteration limits, timeouts, budgets, and sub-agent settings. |
| `prompt_template` | The initial prompt sent to the LLM. Must contain 12 format variables (see [Prompt Template Variables](#prompt-template-variables)). Default is v5.6. |
| `subagent_llm` | Optional separate LLM for `ask_llm()` sub-agent calls. If `None`, uses the main `llm`. Useful for routing sub-agent calls to a smaller/faster model. |

### SubRunner Template Methods

`RRStep` extends `SubRunner` and overrides these template methods:

| Method | Description |
|--------|-------------|
| `_build_inner_pipeline(**kwargs)` | Creates `Pipeline([LLMCallStep, ExtractCodeStep, SandboxExecStep, CheckResultStep])`. Fresh pipeline per `run_loop()` call (steps hold mutable state). |
| `_build_initial_context(**kwargs)` | Creates `RRIterationContext(messages=(initial_prompt,), iteration=0)`. |
| `_is_done(ctx)` | Returns `ctx.terminated` (set by `CheckResultStep` when `FINAL()` is accepted). |
| `_extract_result(ctx)` | Returns `ctx.reflection` (the parsed `ReflectorOutput`). |
| `_accumulate(ctx)` | Appends feedback messages to history, increments iteration counter. |
| `_on_timeout(last_ctx, iteration, **kwargs)` | Builds a fallback `ReflectorOutput`. Optionally attempts fallback synthesis (see [Fallback Synthesis](#fallback-synthesis)). |
| `run_loop(**kwargs)` | Overrides base to collect per-iteration data into `iteration_log` for observability. |

### Prompt Template Variables

The `prompt_template` is formatted with these variables:

| Variable | Type | Description |
|----------|------|-------------|
| `{question_length}` | `int` | Character count of the question |
| `{question_preview}` | `str` | Truncated preview (150 chars max) |
| `{reasoning_length}` | `int` | Character count of agent reasoning |
| `{reasoning_preview}` | `str` | Truncated preview |
| `{answer_length}` | `int` | Character count of agent answer |
| `{answer_preview}` | `str` | Truncated preview |
| `{ground_truth_length}` | `int` | Character count of ground truth |
| `{ground_truth_preview}` | `str` | Truncated preview |
| `{feedback_length}` | `int` | Character count of feedback |
| `{feedback_preview}` | `str` | Truncated preview |
| `{skillbook_length}` | `int` | Character count of skillbook text |
| `{step_count}` | `int` | Number of trace steps |

---

## RRConfig

Exported as `RRConfig` (aliased from `RecursiveConfig`).

```python
from ace_next.rr import RRConfig

config = RRConfig(
    max_iterations=20,           # Max REPL iterations before timeout
    timeout=30.0,                # Per-execution timeout in seconds (Unix only)
    enable_llm_query=True,       # Enable llm_query() in sandbox
    max_llm_calls=30,            # Combined budget for main LLM + sub-agent calls
    max_context_chars=50_000,    # Message history trim threshold
    max_output_chars=20_000,     # Per-execution output truncation limit
    enable_subagent=True,        # Enable ask_llm() sub-agent function
    subagent_model=None,         # Sub-agent model (None = same as main)
    subagent_max_tokens=8192,    # Max tokens for sub-agent responses
    subagent_temperature=0.3,    # Temperature for sub-agent responses
    subagent_system_prompt=None, # Custom sub-agent system prompt (None = default)
    enable_fallback_synthesis=True,  # Attempt LLM synthesis on timeout
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `20` | Maximum REPL loop iterations. When reached, `_on_timeout` fires. |
| `timeout` | `30.0` | Seconds per sandbox `execute()` call. Uses `signal.SIGALRM` on Unix; not enforced on Windows or non-main threads. |
| `enable_llm_query` | `True` | Whether `llm_query()` is available in the sandbox. |
| `max_llm_calls` | `30` | Single shared budget across main LLM calls and sub-agent calls. Prevents effective budget from being 2x the configured value. |
| `max_context_chars` | `50_000` | When message history exceeds this, low-value iterations are trimmed (see [Message Trimming](#message-trimming)). |
| `max_output_chars` | `20_000` | Per-execution stdout/stderr is truncated at this limit with a `[TRUNCATED: N chars remaining]` suffix. |
| `enable_subagent` | `True` | Whether `ask_llm()` is available in the sandbox. When `False`, `ask_llm()` returns a stub message. |
| `subagent_model` | `None` | Model for sub-agent calls. `None` means use the main reflector's model. |
| `subagent_max_tokens` | `8192` | Max tokens for sub-agent responses. |
| `subagent_temperature` | `0.3` | Temperature for sub-agent responses. |
| `subagent_system_prompt` | `None` | Custom system prompt for sub-agent. `None` uses the default analysis prompt. |
| `enable_fallback_synthesis` | `True` | When `True` and max iterations is reached, attempts one more LLM call to synthesise a FINAL() from the conversation history. |

---

## RRIterationContext

Frozen dataclass carrying state through the four inner steps of each REPL iteration. Extends `StepContext`.

```python
@dataclass(frozen=True)
class RRIterationContext(StepContext):
    # Input for this iteration
    messages: tuple[dict[str, str], ...] = ()
    iteration: int = 0

    # LLMCallStep output
    llm_response: str | None = None

    # ExtractCodeStep output
    code: str | None = None
    direct_response: str | None = None

    # SandboxExecStep output
    exec_result: Any | None = None  # ExecutionResult

    # CheckResultStep output
    terminated: bool = False
    reflection: Any | None = None  # ReflectorOutput when FINAL() accepted
    feedback_messages: tuple[dict[str, str], ...] = ()
```

Each iteration creates a fresh context via `.replace()`. The `_accumulate` method appends `feedback_messages` to `messages` for the next iteration.

---

## TraceSandbox

Lightweight `exec()`-based sandbox for running LLM-generated Python code. Located in `ace_next/rr/sandbox.py`.

**Not a security sandbox.** Restricts builtins as defence-in-depth but relies on trusting the LLM not to generate malicious code. Do not use for untrusted code.

### Pre-loaded Namespace

| Variable | Type | Description |
|----------|------|-------------|
| `trace` | `TraceContext \| None` | The agent execution trace |
| `traces` | `dict` | Canonical traces dict (question, ground_truth, feedback, steps) |
| `skillbook` | `str` | Skillbook text via `as_prompt()` |
| `ask_llm` | `Callable` | Sub-agent query function (see [Sub-Agent](#sub-agent)) |
| `llm_query` | `Callable` | Alias for `ask_llm(prompt, "")` (backward compat) |
| `FINAL` | `Callable` | Submit final result dict |
| `FINAL_VAR` | `Callable` | Submit a named variable as final result |
| `SHOW_VARS` | `Callable` | Print available variables (debugging) |
| `json` | module | `json` standard library |
| `re` | module | `re` standard library |
| `collections` | module | `collections` standard library |
| `datetime` | class | `datetime.datetime` |
| `timedelta` | class | `datetime.timedelta` |
| `date` | class | `datetime.date` |
| `time` | class | `datetime.time` |
| `timezone` | class | `datetime.timezone` |

### Blocked Builtins

`open`, `__import__`, `eval`, `exec`, `compile`, `input`, `globals`, `locals`, `breakpoint`, `memoryview` — all set to `None`.

### safe_getattr

The builtin `getattr` is replaced with a safe version that blocks access to names starting with `_`:

```python
def safe_getattr(obj, name, *default):
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' blocked")
    return getattr(obj, name, *default)
```

Available as both the builtin `getattr` and `safe_getattr` in the namespace.

### FINAL(value)

Submits the analysis result. `value` should be a dict matching `ReflectorOutput` fields:

```python
FINAL({
    "reasoning": "...",
    "error_identification": "...",
    "root_cause_analysis": "...",
    "correct_approach": "...",
    "key_insight": "...",
    "extracted_learnings": [
        {"learning": "...", "atomicity_score": 0.8, "evidence": "..."},
    ],
    "skill_tags": [
        {"id": "section-00001", "tag": "helpful"},
    ],
})
```

Raises `StopIteration` internally to exit the `exec()` call. `CheckResultStep` catches this and parses the value into a `ReflectorOutput`.

### FINAL_VAR(name)

Convenience function to submit a pre-built variable:

```python
result = {"reasoning": "...", "extracted_learnings": [...]}
# ... build result across multiple code blocks ...
FINAL_VAR("result")  # equivalent to FINAL(result)
```

Raises `ValueError` if the variable doesn't exist in the namespace.

### SHOW_VARS()

Debug function that prints available user variables (excludes builtins, modules, and internal names).

### ExecutionResult

Return type of `sandbox.execute()`:

```python
@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    final_value: Any = None
    exception: Optional[Exception] = None

    @property
    def success(self) -> bool:
        return self.exception is None
```

### Timeout Behaviour

- **Unix (main thread):** Uses `signal.SIGALRM`. Raises `ExecutionTimeoutError` after `config.timeout` seconds.
- **Windows / non-main thread:** No timeout enforcement. Code runs to completion.

### inject(name, value)

Add or override a variable in the sandbox namespace after construction.

### reset()

Clear `final_value` and `final_called` state. Used by `CheckResultStep` when rejecting premature or errored FINAL() calls.

---

## Sub-Agent

The sub-agent system provides an LLM-callable function (`ask_llm`) inside the sandbox, enabling the main reflector's code to delegate semantic analysis to a secondary LLM call.

### ask_llm(question, context="", mode="analysis")

Available in the sandbox when `config.enable_subagent=True`. Calls the sub-agent LLM with a formatted prompt.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The question to ask |
| `context` | `str` | `""` | Data to analyse (trace excerpt, code output, etc.) |
| `mode` | `str` | `"analysis"` | Prompt protocol: `"analysis"` for survey, `"deep_dive"` for investigation |

When `config.enable_subagent=False`, returns `"(ask_llm disabled - analyze with code)"`.

### llm_query(prompt)

Backward-compatible alias: `llm_query(prompt)` calls `ask_llm(prompt, "")`.

### Modes and System Prompts

| Mode | Prompt | Purpose |
|------|--------|---------|
| `"analysis"` | `SUBAGENT_ANALYSIS_PROMPT` | Survey/categorisation pass — descriptive summaries for downstream categorisation |
| `"deep_dive"` | `SUBAGENT_DEEPDIVE_PROMPT` | Investigation pass — evidence-rich analysis with root cause identification |
| unknown | `config.system_prompt` | Falls back to the configured system prompt |

### CallBudget

Shared budget enforcing a single limit across main LLM calls and sub-agent calls:

```python
budget = CallBudget(max_calls=30)
budget.consume()    # True (29 remaining)
budget.count        # 1
budget.exhausted    # False
```

When the budget is exhausted:
- `LLMCallStep` returns an empty response and logs a warning.
- `ask_llm` returns a limit message: `"(Max N LLM calls exceeded - continue with available data)"`.

The budget is shared — `config.max_llm_calls=30` means 30 total calls, not 30 main + 30 sub-agent.

### SubAgentConfig

```python
@dataclass
class SubAgentConfig:
    model: Optional[str] = None       # None = same model as main reflector
    max_tokens: int = 8192
    temperature: float = 0.3
    system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT
```

### SubAgentLLM

Wrapper class that tracks call history and provides the `ask()` method:

```python
subagent = SubAgentLLM(llm, config=SubAgentConfig(), subagent_llm=separate_llm)
subagent.ask("What pattern do you see?", context="...", mode="deep_dive")
subagent.call_count     # 1
subagent.call_history   # [{"call_number": 1, "question": "...", ...}]
subagent.reset()        # Clear count and history
```

### create_ask_llm_function

Factory that creates the bounded `ask_llm` callable injected into the sandbox:

```python
ask_llm_fn = create_ask_llm_function(
    llm=llm,                    # Main LLM client
    config=SubAgentConfig(),    # Sub-agent configuration
    subagent_llm=None,          # Optional separate LLM
    max_calls=20,               # Standalone limit (when no budget)
    budget=CallBudget(30),      # Shared budget (overrides max_calls)
)
```

When `budget` is provided, it takes precedence over `max_calls`. The returned callable has `.subagent` and `.max_calls` attributes for introspection.

---

## TraceContext

Structured trace wrapper for programmatic exploration in the sandbox. Located in `ace_next/rr/trace_context.py`.

### TraceStep

```python
@dataclass
class TraceStep:
    index: int
    action: str               # e.g. "reasoning", "tool_call:search", "user_message"
    thought: str              # Main content (reasoning, user text, tool args)
    observation: str          # Tool result or answer
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

| Method/Property | Description |
|-----------------|-------------|
| `content` | Combined `thought + observation` |
| `preview(max_len=300)` | Truncated preview with char count |
| `__repr__()` | Short format: `TraceStep(0: reasoning...)` |
| `__str__()` | Detailed multi-line format |

### TraceContext Methods

| Method | Description |
|--------|-------------|
| `steps` | Property returning all `TraceStep` objects |
| `raw_reasoning` | Property returning the raw reasoning text |
| `get_step(index)` | Get step by index (returns `None` if out of bounds) |
| `find_steps(pattern, case_sensitive=False)` | Find steps matching a string pattern |
| `find_steps_regex(pattern, flags=0)` | Find steps matching a regex pattern |
| `get_errors()` | Find steps containing error indicators (`error`, `exception`, `failed`, `traceback`) |
| `get_actions(action_type)` | Get steps with a specific action type |
| `summary()` | Brief summary string |
| `to_markdown()` | Render as markdown conversation trace |
| `search_raw(pattern)` | Search steps, return matching indices |
| `search_raw_text(pattern)` | Search raw reasoning, return matched substrings |
| `__len__()`, `__iter__()`, `__getitem__()` | Standard container protocol |

### Factory Methods

| Method | Input | Description |
|--------|-------|-------------|
| `from_agent_output(agent_output)` | `AgentOutput` | Auto-detects `[assistant]/[user]` markers for multi-step traces |
| `from_reasoning_string(reasoning)` | `str` | Parses numbered steps or falls back to single-step |
| `from_browser_use(history)` | browser-use `AgentHistory` | Converts browser automation history |
| `from_langchain(intermediate_steps)` | `list[tuple]` | Converts LangChain `(AgentAction, observation)` tuples |
| `from_conversation_history(messages, max_text_len=1000)` | `list[dict]` | Parses `{"role": ..., "content": ...}` message lists |
| `from_tau_simulation(messages, system_prompt="")` | TAU-bench messages | Handles `AssistantMessage`, `ToolMessage` with tool calls |
| `combine(traces)` | `list[TraceContext]` | Merge multiple traces with re-indexing |

---

## Code Extraction

Three-layer fallback chain for extracting Python code from LLM responses. Located in `ace_next/rr/code_extraction.py`.

| Layer | Function | Strategy |
|-------|----------|----------|
| 1. Fenced | `extract_fenced_blocks()` | `` ```python ``, `` ~~~python ``, bare `` ``` `` (validated) |
| 2. Indented | `extract_indented_block()` | 4-space or tab indentation |
| 3. FINAL | `extract_final_call()` | Balanced parenthesis extraction of `FINAL(...)` |

### Batch Mode

When the first fenced block starts with `# BATCH`, all fenced blocks in the response are concatenated into a single script:

```python
# In LLM response:
# ```python
# # BATCH
# result_a = analyze_part_a()
# ```
# ```python
# result_b = analyze_part_b()
# ```
# → Both blocks execute as one script
```

### Validation

`looks_like_python(code)` checks for Python indicators (`def `, `import `, `print(`, `FINAL(`, etc.) to filter false positives from bare code fences.

---

## Message Trimming

Semantic importance-based trimming of REPL message history. Located in `ace_next/rr/message_trimming.py`.

When message history exceeds `config.max_context_chars`, iterations are scored by importance and the lowest-value ones are dropped:

| Signal | Score | Rationale |
|--------|-------|-----------|
| Error indicators (Error, Exception, Traceback, stderr:) | +3.0 | Debugging context is high value |
| Finding indicators (found, pattern, insight, discovered) | +2.0 | Analysis progress is valuable |
| FINAL() in assistant message | +2.0 | Near-final attempts are important |
| ask_llm/llm_query in assistant message | +1.0 | Sub-agent calls carry insights |
| Long output (>500 chars) | +1.0 | Substantive output worth keeping |
| "(no output)" in user message | -1.0 | Empty output is low value |

**Behaviour:**
- The first message (initial prompt) is always kept.
- Dropped iterations are summarised: `[N earlier iterations omitted: M error(s), K exploration(s)]`.
- Kept iterations maintain chronological order.

---

## Guard Logic

`CheckResultStep` implements several guards:

### Premature FINAL (Iteration 0)

If `FINAL()` is called on the first iteration, it is rejected. The sandbox is reset and the LLM receives feedback:

> "You called FINAL() before exploring the data. Read the actual variables first, then call FINAL() with evidence-based analysis."

### FINAL After Error

If `FINAL()` is called but the code execution had an error (`result.success == False`), it is rejected:

> "Your code had an error. Fix the bug and try again. Do NOT call FINAL() until your code executes successfully."

### Direct Response Fallback

When no code block is extracted, `CheckResultStep` attempts to parse the LLM response as direct JSON (stripping `` ```json `` fences). If valid, it's treated as a FINAL() value. If not, the LLM receives feedback requesting code.

### Iteration Progress Header

Each feedback message includes an iteration counter: `[Iteration N/max]`. When approaching the limit (within 2 iterations), an urgency suffix is added: `(approaching limit — finalize soon)`.

---

## Fallback Synthesis

When `config.enable_fallback_synthesis=True` and `max_iterations` is reached:

1. A synthesis prompt is appended to the conversation history asking the LLM to call `FINAL()` with its best assessment.
2. The response is parsed for code containing `FINAL()` and executed in a fresh sandbox.
3. If no code is found, direct JSON parsing is attempted.
4. If synthesis fails, a basic timeout `ReflectorOutput` is returned with `raw["timeout"] = True`.

This is a recovery mechanism — it often salvages partial analysis that would otherwise be lost.

---

## Traces Dict

The canonical data structure passed to the sandbox as the `traces` variable:

```python
{
    "question": str,              # The question/task
    "ground_truth": str | None,   # Expected answer
    "feedback": str | None,       # Environment feedback
    "steps": [                    # Agent execution steps
        {
            "role": "agent",
            "reasoning": str,
            "answer": str,
            "skill_ids": list[str],
        }
    ],
}
```

---

## rr_trace Output Schema

After `run_loop()` completes, `RRStep` enriches `ReflectorOutput.raw["rr_trace"]` with execution metadata:

```python
{
    "iterations": [               # Per-iteration log
        {
            "iteration": int,     # 0-indexed
            "code": str | None,   # Code sent to sandbox
            "stdout": str | None, # Captured stdout
            "stderr": str | None, # Captured stderr
            "terminated": bool,   # Whether FINAL() was accepted
        },
        ...
    ],
    "subagent_calls": [           # Sub-agent call history
        {
            "call_number": int,
            "question": str,
            "context_length": int,
            "response_length": int,
            "mode": str,          # "analysis" or "deep_dive"
        },
        ...
    ],
    "total_iterations": int,
    "timed_out": bool,
}
```

This structure is consumed by `RROpikStep` for observability and can be inspected by users for debugging.

---

## RROpikStep

Side-effect step for logging RR traces to Opik. Located in `ace_next/rr/opik.py`.

```python
from ace_next.rr import RROpikStep

# Place after RRStep in the pipeline
steps = [..., rr_step, RROpikStep(project_name="my-project")]
```

### Step Contract

```python
requires = frozenset({"reflection"})
provides = frozenset()
```

### Behaviour

- Reads `ctx.reflection.raw["rr_trace"]` — the dict populated by `RRStep`.
- Creates one Opik trace per RR invocation with child spans per iteration.
- Gracefully degrades to a no-op when Opik is not installed or `OPIK_DISABLED=true`.
- **Explicit opt-in only** — Opik is never auto-enabled.

### Trace Hierarchy

```
rr_reflect (trace)
├── rr_iteration_0 (span)    ← code, stdout, stderr
├── rr_iteration_1 (span)
└── rr_iteration_2 (span)    ← FINAL called here
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPIK_API_KEY` | API key for Opik authentication |
| `OPIK_WORKSPACE` | Opik workspace name |
| `OPIK_URL_OVERRIDE` | Custom Opik server URL |
| `OPIK_DISABLED=true` | Disable all Opik tracing |
| `OPIK_ENABLED=false` | Alternative disable signal |

### Metadata

Parent trace metadata includes:

```python
{
    "total_iterations": int,
    "subagent_call_count": int,   # Only when sub-agent calls exist
    "subagent_calls": list[dict], # Full call history
}
```

### flush()

Call `flush()` after the pipeline finishes to drain buffered traces before the process exits.

---

## Public API

All exports from `ace_next.rr`:

```python
from ace_next.rr import (
    # Core
    RRStep,                  # Main entry point (SubRunner + StepProtocol + ReflectorLike)
    RRConfig,                # Configuration (alias for RecursiveConfig)
    RRIterationContext,      # Per-iteration frozen context
    RROpikStep,              # Opik observability step (lazy-imported)

    # Inner pipeline steps
    LLMCallStep,
    ExtractCodeStep,
    SandboxExecStep,
    CheckResultStep,

    # Sandbox
    TraceSandbox,
    ExecutionResult,
    ExecutionTimeoutError,

    # Sub-agent
    SubAgentLLM,
    SubAgentConfig,
    CallBudget,
    create_ask_llm_function,

    # Trace
    TraceContext,
    TraceStep,
)
```

`RROpikStep` is lazy-imported via `__getattr__` to avoid pulling in the `opik` package at module load time.
