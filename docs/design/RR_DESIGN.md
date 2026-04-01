# Recursive Reflector (RR) Design

Design document for the Recursive Reflector module (`ace/rr/`). The RR is a PydanticAI-powered trace analyser that uses tool calls to generate Python code, execute it in a sandbox, delegate analysis to sub-agents, and produce structured reflections from agent execution traces.

---

## Overview

The Recursive Reflector replaces the single-pass `Reflector` with an iterative tool-calling agent. Instead of asking the LLM for a one-shot analysis, RR gives the LLM three tools — `execute_code`, `analyze`, and `batch_analyze` — and lets it explore trace data, delegate semantic reasoning to a sub-agent, and submit structured findings as typed output.

**Key properties:**

- Satisfies both `StepProtocol` and `ReflectorLike` — usable as a pipeline step or a drop-in reflector replacement.
- Internally uses a **PydanticAI agent** with typed tools and structured output (`ReflectorOutput`).
- PydanticAI's `UsageLimits` enforces a combined request budget across all LLM calls.
- Produces `ReflectorOutput` with an enriched `raw["rr_trace"]` dict for downstream observability.
- Logfire auto-instruments the PydanticAI agent for observability (replaces the old `RROpikStep`).

```python
from ace.rr import RRStep, RRConfig

# Drop-in replacement for Reflector
ace = ACELiteLLM(llm, reflector=RRStep("gpt-4o-mini", config=RRConfig(max_llm_calls=30)))

# Or as a pipeline step
pipe = Pipeline([..., RRStep("gpt-4o-mini"), ...])
```

---

## Architecture

### PydanticAI Agent Loop

Each invocation of `RRStep` runs a PydanticAI agent that drives the analysis through tool calls:

```

RR accepts raw trace objects directly. For batch mode it recognizes:

- a raw `list[...]` of trace items
- a dict with an `"items"` list
- a dict with a `"tasks"` list
- a legacy combined `"steps"` batch of conversation wrapper dicts

RR does **not** rewrite the inner trace payload into a benchmark-specific schema. Instead it injects generic helper variables such as `batch_items`, `item_ids`, and `survey_items`, plus a runtime helper registry so the model can define reusable accessors for whatever raw structure it sees.
┌───────────────────────────────────────────────────────────────┐
│  RRStep._run_reflection()                                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  PydanticAI Agent (model, output_type=ReflectorOutput)  │  │
│  │                                                         │  │
│  │  Tools:                                                 │  │
│  │  ┌──────────────┐  ┌─────────┐  ┌───────────────┐      │  │
│  │  │ execute_code │  │ analyze │  │ batch_analyze │      │  │
│  │  │  (sandbox)   │  │ (sub-   │  │ (parallel     │      │  │
│  │  │              │  │  agent) │  │  sub-agent)   │      │  │
│  │  └──────┬───────┘  └────┬────┘  └──────┬────────┘      │  │
│  │         │               │              │               │  │
│  │         ▼               ▼              ▼               │  │
│  │    TraceSandbox    PydanticAI      ThreadPool +        │  │
│  │    exec() env      sub-agent      PydanticAI          │  │
│  │                    (text out)     sub-agent            │  │
│  │                                                         │  │
│  │  Output:                                                │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  ReflectorOutput (structured, validated)          │  │  │
│  │  │  + output_validator enforces exploration depth    │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  UsageLimits(request_limit=max_llm_calls)                     │
│  → UsageLimitExceeded triggers timeout fallback               │
└───────────────────────────────────────────────────────────────┘
```

### Tools

The PydanticAI agent has three tools, defined in `ace/rr/agent.py`:

| Tool | Signature | Description |
|------|-----------|-------------|
| `execute_code` | `(code: str) -> str` | Run Python in the `TraceSandbox`. Variables persist across calls. Returns captured stdout/stderr. Raises `ModelRetry` on exceptions. |
| `analyze` | `(question: str, context: str, mode: str) -> str` | Async sub-agent call. `mode="analysis"` for survey, `mode="deep_dive"` for investigation. |
| `batch_analyze` | `(question: str, items: list[str], mode: str) -> list[str]` | Parallel sub-agent analysis via `ThreadPoolExecutor`. Each item analyzed independently. |

### Output Validation

An `output_validator` on the agent enforces that the LLM has used `execute_code` at least twice before producing its final `ReflectorOutput`. If the LLM tries to conclude too early, it receives a `ModelRetry` asking it to explore further.

### Dual Protocol Support

`RRStep` satisfies two protocols simultaneously:

```python
class RRStep:
    # StepProtocol — place in any Pipeline
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext: ...

    # ReflectorLike — use as drop-in reflector in runners
    def reflect(self, *, question, agent_output, skillbook, ...) -> ReflectorOutput: ...
```

---

## RRStep

### Constructor

```python
RRStep(
    model: str,                          # LiteLLM or PydanticAI model string
    config: Optional[RecursiveConfig] = None,
    prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
    model_settings: ModelSettings | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `model` | Model string passed to `resolve_model()`. Supports LiteLLM format (`gpt-4o`, `anthropic/claude-3-5-sonnet`) and PydanticAI format. |
| `config` | `RRConfig` instance controlling timeouts, budgets, and sub-agent settings. |
| `prompt_template` | The user prompt sent to the agent. Must contain format variables (see [Prompt Template Variables](#prompt-template-variables)). Default is the tool-calling variant of v5.6. |
| `model_settings` | PydanticAI `ModelSettings` passed to the agent (temperature, max_tokens, etc.). |

### Internal Components

On construction, `RRStep` creates:

| Component | Description |
|-----------|-------------|
| `_agent` | PydanticAI agent (`Agent[RRDeps, ReflectorOutput]`) with `execute_code`, `analyze`, `batch_analyze` tools and an output validator. Created by `create_rr_agent()`. |
| `_sub_agent` | PydanticAI agent (`Agent[None, str]`) for the `analyze`/`batch_analyze` tools. Simple text-in/text-out agent with no tools. Created by `create_sub_agent()`. Set to `None` when `config.enable_subagent=False`. |

### Prompt Template Variables

The `prompt_template` is formatted with these variables:

| Variable | Type | Description |
|----------|------|-------------|
| `{traces_description}` | `str` | Human-readable summary of the traces (single: schema keys or raw type; batch: container summary) |
| `{step_count}` | `int` | Total number of trace steps/messages seen in previews (summed across batch items in batch mode) |
| `{skillbook_length}` | `int` | Character count of skillbook text |
| `{batch_variables}` | `str` | Extra sandbox variable table row for batch mode (empty string in single-trace mode) |
| `{helper_variables}` | `str` | Extra sandbox variable/function rows for reusable helper registration |
| `{traces_previews}` | `str` | Markdown table of trace previews (single: per-field; batch: per-item with message count) |
| `{trace_size_chars}` | `int` | Total character count of the serialised traces dict |
| `{max_iterations}` | `int` | The `max_llm_calls` budget (shown so the LLM can pace itself) |
| `{task_count}` | `int` | Number of analyzed items (1 for single-trace, N for batch) |

---

## RRConfig

Exported as `RRConfig` (aliased from `RecursiveConfig`).

```python
from ace.rr import RRConfig

config = RRConfig(
    max_iterations=20,           # Legacy: max REPL iterations (kept for compat)
    timeout=30.0,                # Per-execution timeout in seconds (Unix only)
    max_llm_calls=30,            # Primary limit: PydanticAI UsageLimits request_limit
    max_context_chars=50_000,    # Message history trim threshold
    max_output_chars=20_000,     # Per-execution output truncation limit
    enable_subagent=True,        # Enable analyze/batch_analyze tools
    subagent_model=None,         # Sub-agent model (None = same as main)
    subagent_max_tokens=8192,    # Max tokens for sub-agent responses
    subagent_temperature=0.3,    # Temperature for sub-agent responses
    subagent_system_prompt=None, # Custom sub-agent system prompt (None = default)
    enable_fallback_synthesis=True,  # Attempt LLM synthesis on timeout
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | `20` | Legacy field kept for backward compatibility. The primary limit is now `max_llm_calls` via PydanticAI's `UsageLimits`. |
| `timeout` | `30.0` | Seconds per sandbox `execute()` call. Uses `signal.SIGALRM` on Unix; not enforced on Windows or non-main threads. |
| `max_llm_calls` | `30` | **Primary budget.** Passed as `UsageLimits(request_limit=N)` to the PydanticAI agent. Covers all requests — main agent turns plus any requests made by tools. When exhausted, `UsageLimitExceeded` triggers the timeout fallback. |
| `max_context_chars` | `50_000` | Trim threshold for message scoring logic (see [Message Trimming](#message-trimming)). |
| `max_output_chars` | `20_000` | Per-execution stdout/stderr is truncated at this limit with a `[TRUNCATED: N chars remaining]` suffix. |
| `enable_subagent` | `True` | Whether the `analyze`/`batch_analyze` tools are functional. When `False`, the sub-agent is not created and the tools return stub messages. |
| `subagent_model` | `None` | Model for sub-agent. `None` means use the main reflector's model. Useful for routing sub-agent calls to a smaller/faster model. |
| `subagent_max_tokens` | `8192` | Max tokens for sub-agent responses. |
| `subagent_temperature` | `0.3` | Temperature for sub-agent responses. |
| `subagent_system_prompt` | `None` | Custom system prompt for sub-agent. `None` uses the default analysis prompt. |
| `subagent_max_requests` | `15` | Per-sub-agent request budget. Used for both `analyze()` and each item in `batch_analyze()`. |
| `enable_fallback_synthesis` | `True` | Legacy field kept for compat. Timeout now always builds a fallback `ReflectorOutput`. |

---

## RRDeps

Dataclass carrying dependencies injected into every tool call via PydanticAI's `RunContext[RRDeps]`. Defined in `ace/rr/agent.py`.

```python
@dataclass
class RRDeps:
    sandbox: TraceSandbox                     # Sandbox for execute_code
    trace_data: dict[str, Any]                # The canonical traces dict
    skillbook_text: str                       # Skillbook text
    config: RecursiveConfig                   # RR configuration
    iteration: int = 0                        # Incremented by execute_code
    sub_agent: PydanticAgent[None, str] | None = None  # Sub-agent for analyze/batch_analyze
    sub_agent_history: list[dict[str, Any]] = field(default_factory=list)  # Call log
```

The `iteration` counter tracks how many `execute_code` calls have been made. The output validator uses this to reject premature conclusions (fewer than 2 code executions).

---

## TraceSandbox

Lightweight `exec()`-based sandbox for running LLM-generated Python code. Located in `ace/rr/sandbox.py`.

**Not a security sandbox.** Restricts builtins as defence-in-depth but relies on trusting the LLM not to generate malicious code. Do not use for untrusted code.

### Pre-loaded Namespace

| Variable | Type | Description |
|----------|------|-------------|
| `trace` | `TraceContext \| None` | The agent execution trace (when available) |
| `traces` | `Any` | Raw trace payload provided to RR (injected by `RRStep`) |
| `skillbook` | `str` | Skillbook text (injected by `RRStep`) |
| `batch_items` | `list[Any]` | Present in batch mode. Ordered view over the raw batch elements regardless of the original container shape. |
| `item_ids` | `list[str]` | Present in batch mode. Stable identifiers derived from each batch item. |
| `item_id_to_index` | `dict[str, int]` | Present in batch mode. Maps item IDs to `batch_items[i]`. |
| `item_preview_by_id` | `dict[str, dict]` | Present in batch mode. Compact previews for question, feedback, first message, and payload type. |
| `survey_items` | `list[str]` | Present in batch mode. Precomputed `batch_analyze()` items with explicit `batch_items[i]` references. |
| `helper_registry` | `dict[str, dict]` | Metadata for reusable helper functions registered during the run. |
| `register_helper` | `Callable` | Define and persist helper code so later `execute_code` calls and sub-agent snapshots can reuse it. |
| `list_helpers` | `Callable` | Return registered helper names and descriptions. |
| `run_helper` | `Callable` | Invoke a registered helper by name. |
| `get_batch_item` | `Callable` | Convenience accessor for `batch_items[index]`. |
| `SHOW_VARS` | `Callable` | Print available variables (debugging) |
| `json` | module | `json` standard library |
| `re` | module | `re` standard library |
| `math` | module | `math` standard library |
| `collections` | module | `collections` standard library |
| `datetime` | class | `datetime.datetime` |
| `timedelta` | class | `datetime.timedelta` |
| `date` | class | `datetime.date` |
| `time` | class | `datetime.time` |
| `timezone` | class | `datetime.timezone` |

**Note:** `FINAL`, `FINAL_VAR`, `ask_llm`, `llm_query`, and `parallel_map` are still present in the sandbox class for backward compatibility with other callers, but `RRStep` does **not** use them. The PydanticAI agent produces `ReflectorOutput` as structured output (replacing `FINAL`), and uses `analyze`/`batch_analyze` tools (replacing `ask_llm`/`parallel_map`).

### Blocked Builtins

`open`, `eval`, `exec`, `compile`, `input`, `globals`, `locals`, `breakpoint`, `memoryview` — all set to `None`.

`__import__` is replaced with a safe import function that only allows pre-loaded modules (`json`, `re`, `math`, `collections`, `datetime`).

### safe_getattr

The builtin `getattr` is replaced with a safe version that blocks access to names starting with `_`:

```python
def safe_getattr(obj, name, *default):
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' blocked")
    return getattr(obj, name, *default)
```

Available as both the builtin `getattr` and `safe_getattr` in the namespace.

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

### Runtime Helper Registry

The sandbox includes a lightweight helper registry for trace-shape adaptation:

- `register_helper(name, source, description="")` executes helper source code, stores the source, and records metadata.
- Registered helpers persist across later `execute_code` calls in the same RR session.
- `create_readonly_sandbox()` replays the stored helper source in sub-agent snapshots so helpers are rebound to the child namespace rather than the parent sandbox.
- Sub-agent prompts include the registered helper catalog when available, so they can start from those helpers instead of re-discovering the raw schema.

### reset()

Clear `final_value` and `final_called` state.

---

## Sub-Agent

The sub-agent system provides LLM reasoning capabilities to the main reflector agent via the `analyze` and `batch_analyze` PydanticAI tools.

### analyze(question, mode, context="")

PydanticAI tool on the main agent. Calls the sub-agent asynchronously with a formatted prompt.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The question to ask |
| `mode` | `str` | `"analysis"` | Prompt protocol: `"analysis"` for survey, `"deep_dive"` for investigation |
| `context` | `str` | `""` | Optional focus instructions. The sub-agent reads raw trace data directly via `execute_code`. |

The sub-agent receives an isolated readonly sandbox snapshot plus its own
`UsageLimits(request_limit=config.subagent_max_requests)`.

When the sub-agent is not configured (`config.enable_subagent=False`), returns `"(analyze unavailable — sub-agent not configured)"`.

### batch_analyze(question, items, mode)

PydanticAI tool that analyzes multiple items in parallel. Uses `ThreadPoolExecutor` to call `sub_agent.run_sync()` for each item concurrently.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The question to ask about each item |
| `items` | `list[str]` | required | List of focus instructions to analyze |
| `mode` | `str` | `"analysis"` | Prompt protocol |

Each item gets its own readonly sandbox snapshot and request budget.
Concurrency is capped at `min(len(items), 10)` workers.

### Modes and System Prompts

| Mode | Prompt | Purpose |
|------|--------|---------|
| `"analysis"` | `SUBAGENT_ANALYSIS_PROMPT` | Survey/categorisation pass — descriptive summaries for downstream categorisation |
| `"deep_dive"` | `SUBAGENT_DEEPDIVE_PROMPT` | Investigation pass — evidence-rich analysis with root cause identification |

### create_sub_agent

Factory function that creates the sub-agent:

```python
def create_sub_agent(
    model: str,
    *,
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
) -> PydanticAgent[SubAgentDeps, str]:
```

The sub-agent is a text-output PydanticAI agent with its own `execute_code`
tool and isolated sandbox snapshot. Model settings default to
`temperature=config.subagent_temperature` and
`max_tokens=config.subagent_max_tokens`.

### Sub-Agent Call History

Each `analyze` and `batch_analyze` call appends metadata to `RRDeps.sub_agent_history`:

```python
# analyze call
{"question": "...", "context_length": 1234, "response_length": 567, "mode": "analysis", "code_calls": 2}

# batch_analyze call
{"question": "...", "items_count": 5, "mode": "analysis", "batch": True, "code_calls_per_item": [2, 1, 3, 2, 2]}
```

This history is included in the `rr_trace` output for observability.

---

## TraceContext

Structured trace wrapper for programmatic exploration in the sandbox. Located in `ace/rr/trace_context.py`.

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

## Message Trimming

Semantic importance-based trimming of message history. Located in `ace/rr/message_trimming.py`.

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

The PydanticAI agent enforces analysis quality through two mechanisms:

### Output Validator (Premature Conclusion)

If the agent tries to produce `ReflectorOutput` before executing code at least twice (`deps.iteration < 2`), the output validator raises `ModelRetry`:

> "You haven't explored the data enough. Use execute_code to analyze the traces first, then provide your final output."

### execute_code Error Handling

When code execution raises an exception, the `execute_code` tool raises `ModelRetry` with the error message:

> "Code error: {error}\n\nFix the bug and try again."

This causes PydanticAI to retry the tool call (up to 3 retries per tool call). The LLM receives the error feedback and can correct its code.

### Output Truncation

Each `execute_code` call truncates output at `config.max_output_chars` with a `[TRUNCATED: N chars remaining]` suffix.

---

## Timeout / Fallback

When `UsageLimitExceeded` is raised (the `max_llm_calls` budget is exhausted):

1. `RRStep._build_timeout_output()` constructs a basic `ReflectorOutput` with `raw["timeout"] = True`.
2. If `agent_output` and `ground_truth` are available, a simple correct/incorrect assessment is included.
3. The output includes the request limit and iteration count for debugging.

When any other exception occurs during the agent run, a minimal `ReflectorOutput` is returned with `raw["error"]` set.

---

## Batch Mode

When `RRStep.__call__` receives a batch trace (a raw list, a dict with `"items"` or `"tasks"`, or a legacy combined `"steps"` batch), it runs a **single PydanticAI agent session** that analyzes all items. The agent uses `batch_analyze` to explore items concurrently within that session.

### Detection

A trace is treated as a batch when:

- `trace` is a raw `list`
- or `trace["items"]` is a list
- or `trace["tasks"]` is a list
- or `trace["steps"]` looks like a legacy combined batch of conversation wrapper dicts

### Single-Session Design

The full batch trace is passed directly to `_run_reflection` without rewriting the inner payloads. The sandbox exposes:

```python
traces        # raw caller-provided payload
batch_items   # ordered list of raw batch elements
item_ids      # stable ids derived from batch items
survey_items  # precomputed batch_analyze instructions referencing batch_items[i]
```

The RR prompt instructs the LLM to:
1. Inspect minimal structure only once.
2. Register reusable helpers when repeated access would help.
3. Use `batch_analyze` to fan out analysis across items concurrently.
4. Identify cross-item patterns.
5. Return a structured `ReflectorOutput` with a `raw["items"]` list containing per-item results.

`_split_batch_reflection` then parses the single `ReflectorOutput` into per-item outputs. If the output doesn't contain per-item results, the single reflection is duplicated as a fallback.

### Output

Returns `ctx.replace(reflections=(refl_0, refl_1, ..., refl_n))` where `reflections[i]` corresponds to `batch_items[i]`.

### Cost

1 batch = 1 PydanticAI agent session regardless of item count. The session runs up to `max_llm_calls` requests total. Within the session, `batch_analyze` calls run sub-agent analyses concurrently via `ThreadPoolExecutor`.

For a batch of 5 items with `batch_analyze` fanning out sub-agent calls, cost is typically the main agent turns + sub-agent calls, all counted against the single `UsageLimits(request_limit=max_llm_calls)` budget.

### Single-Task Backward Compatibility

When the trace is not recognized as a batch container, `__call__` follows the existing single-trace path and returns a 1-tuple `reflections=(reflection,)`.

---

## Traces Input

The raw data structure passed to the sandbox as the `traces` variable.

### Single-trace mode

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

### Batch mode

RR accepts several batch container shapes:

```python
[raw_item_0, raw_item_1, ...]
```

```python
{"items": [raw_item_0, raw_item_1, ...]}
```

```python
{"tasks": [raw_item_0, raw_item_1, ...]}
```

```python
{
    "steps": [
        {"role": "conversation", "id": "...", "content": raw_item_0},
        {"role": "conversation", "id": "...", "content": raw_item_1},
    ]
}
```

RR leaves each `raw_item_n` untouched and injects a generic `batch_items` view plus runtime helper registration so the model can adapt to the item shape during execution.

---

## rr_trace Output Schema

After the agent run completes, `RRStep` enriches `ReflectorOutput.raw["rr_trace"]` with execution metadata:

```python
{
    "total_iterations": int,       # Number of execute_code calls
    "subagent_calls": [            # Sub-agent call history
        {
            "question": str,
            "context_length": int,
            "response_length": int,
            "mode": str,           # "analysis" or "deep_dive"
        },
        # batch_analyze entries:
        {
            "question": str,
            "items_count": int,
            "mode": str,
            "batch": True,
        },
        ...
    ],
    "timed_out": bool,
}
```

Additionally, `ReflectorOutput.raw["usage"]` contains PydanticAI usage statistics:

```python
{
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "requests": int,
}
```

This structure can be inspected by users for debugging. For full observability, Logfire auto-instruments PydanticAI agents (traces, spans, tool calls) without any extra pipeline steps.

---

## Observability

The old `RROpikStep` pipeline step has been removed. Observability is now handled by **Logfire**, which auto-instruments PydanticAI agents. This provides:

- Per-agent-run traces with spans for each LLM request and tool call
- Token usage tracking
- Latency metrics
- No explicit opt-in step required in the pipeline

The `rr_trace` dict in `ReflectorOutput.raw` still provides programmatic access to iteration counts and sub-agent call history for custom observability needs.

---

## Public API

All exports from `ace.rr`:

```python
from ace.rr import (
    # Core
    RRStep,                  # Main entry point (StepProtocol + ReflectorLike)
    RRConfig,                # Configuration (alias for RecursiveConfig)
    RRDeps,                  # PydanticAI RunContext dependencies

    # Agent factories
    create_rr_agent,         # Build the PydanticAI reflector agent
    create_sub_agent,        # Build the PydanticAI sub-agent

    # Sandbox
    TraceSandbox,
    ExecutionResult,
    ExecutionTimeoutError,

    # Trace
    TraceContext,
    TraceStep,
)
```
