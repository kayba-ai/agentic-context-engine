# Recursive Reflector (RR) Design

Design document for the Recursive Reflector (`ace/steps/rr_step.py`). The RR is a PydanticAI-powered trace analyser that uses tool calls to execute Python code in a sandbox, decompose complex inputs via recursive child sessions, and produce structured reflections from agent execution traces.

---

## Overview

The Recursive Reflector replaces the single-pass `Reflector` with an iterative tool-calling agent. Instead of asking the LLM for a one-shot analysis, RR gives the LLM two tools — `execute_code` and `recurse` — and lets it explore trace data programmatically and decompose large inputs into focused sub-problems.

**Key properties:**

- `RRStep` is a subclass of `RecursiveAgent` (`ace/core/recursive_agent.py`).
- Satisfies both `StepProtocol` and `ReflectorLike` — usable as a pipeline step or a drop-in reflector replacement.
- Uses a **PydanticAI agent** with typed tools and structured output (`ReflectorOutput`).
- Two-tier compaction (microcompaction + full summarization) handles context-window pressure.
- Depth-based recursion via the `recurse` tool decomposes large/complex inputs.
- PydanticAI's `UsageLimits` enforces token and request budgets.
- Produces `ReflectorOutput` with an enriched `raw["rr_trace"]` dict for observability.

```python
from ace.steps.rr_step import RRStep, RRConfig

# Drop-in replacement for Reflector
ace = ACELiteLLM(llm, reflector=RRStep("gpt-4o-mini", config=RRConfig(max_requests=30)))

# Or as a pipeline step
pipe = Pipeline([..., RRStep("gpt-4o-mini"), ...])
```

---

## Architecture

### Inheritance

```
RecursiveAgent (ace/core/recursive_agent.py)
  ├── execute_code tool (generic)
  ├── recurse tool (generic, depth-based)
  ├── Two-tier compaction
  ├── Budget management (UsageLimits)
  ├── create_sandbox() helper
  └── on_compaction() callback

RRStep(RecursiveAgent) (ace/steps/rr_step.py)
  ├── RR-specific prompt building
  ├── Trace/sandbox setup
  ├── output_validator tool (ensure exploration before concluding)
  ├── Timeout/error fallback with ground-truth comparison
  └── Online mode skill evaluation
```

### Agent Loop

```
┌───────────────────────────────────────────────────────────────┐
│  RRStep._run_reflection()                                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  PydanticAI Agent (model, output_type=ReflectorOutput)  │  │
│  │                                                         │  │
│  │  Tools:                                                 │  │
│  │  ┌──────────────┐  ┌──────────┐                         │  │
│  │  │ execute_code │  │ recurse  │                         │  │
│  │  │  (sandbox)   │  │ (child   │                         │  │
│  │  │              │  │  session) │                         │  │
│  │  └──────┬───────┘  └────┬─────┘                         │  │
│  │         │               │                               │  │
│  │         ▼               ▼                               │  │
│  │    TraceSandbox    Child RRStep                          │  │
│  │    exec() env      (own sandbox,                        │  │
│  │                     own budget)                          │  │
│  │                                                         │  │
│  │  Output:                                                │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  ReflectorOutput (structured, validated)          │  │  │
│  │  │  + output_validator enforces exploration depth    │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  UsageLimits(total_tokens_limit, request_limit)               │
│  → compaction on context window pressure                      │
│  → BudgetExhausted when total budget spent                    │
└───────────────────────────────────────────────────────────────┘
```

### Tools

| Tool | Signature | Defined in | Description |
|------|-----------|------------|-------------|
| `execute_code` | `(code: str) -> str` | `RecursiveAgent` | Run Python in the `TraceSandbox`. Variables persist across calls. Returns captured stdout/stderr. Raises `ModelRetry` on exceptions. |
| `recurse` | `(prompt: str, context_code: str) -> str` | `RecursiveAgent` | Spawn a child session with its own sandbox. Child inherits data and helpers. Use `context_code` to prepare the child's data. Not available at max depth. |
| `output_validator` | (on output) | `RRStep` | Ensures the agent has used `execute_code` at least once before producing final output. |

### Dual Protocol Support

```python
class RRStep(RecursiveAgent):
    # StepProtocol — place in any Pipeline
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext: ...

    # ReflectorLike — use as drop-in reflector in runners
    def reflect(self, *, question, agent_output, skillbook, ...) -> ReflectorOutput: ...
```

---

## Configuration

### AgenticConfig (base)

Defined in `ace/core/recursive_agent.py`. All fields inherited by `RRConfig`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | `500_000` | Total token budget per agent run. When exhausted → `BudgetExhausted`. |
| `max_requests` | `50` | Safety cap on LLM requests per agent run. When hit → `BudgetExhausted`. |
| `context_window` | `128_000` | Model context window size. |
| `max_depth` | `2` | Max recursion depth. At max depth, `recurse` tool is not registered. |
| `child_budget_fraction` | `0.5` | Fraction of remaining token budget given to each child session. |
| `max_compactions` | `3` | Safety cap on full summarization rounds per session. |
| `microcompact_keep_recent` | `3` | Number of most recent tool results preserved during microcompaction. |
| `timeout` | `60.0` | Seconds per sandbox `execute()` call. Uses `signal.SIGALRM` on Unix. |
| `max_output_chars` | `20_000` | Per-execution stdout/stderr truncation limit. |

### RRConfig (alias for RecursiveConfig)

Defined in `ace/implementations/rr/config.py`. Extends `AgenticConfig`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_output_chars` | `50_000` | Override: larger limit for trace analysis output. |

All other fields are inherited from `AgenticConfig` with the same defaults.

```python
from ace.steps.rr_step import RRConfig

config = RRConfig(
    max_requests=20,
    max_depth=2,
    timeout=60.0,
    max_output_chars=50_000,
)
```

---

## Dependencies

### AgenticDeps (base)

Defined in `ace/core/recursive_agent.py`.

| Field | Type | Description |
|-------|------|-------------|
| `config` | `AgenticConfig` | Configuration |
| `sandbox` | `Any` | TraceSandbox or compatible (used by `execute_code` and `recurse` tools) |
| `depth` | `int` | Current recursion depth |
| `max_depth` | `int` | Maximum recursion depth |
| `iteration` | `int` | Number of `execute_code` calls (incremented by the tool) |
| `run_session_fn` | `Callable` | Callback for spawning child sessions (wired by `RecursiveAgent.run()`) |
| `parent_usage_tokens` | `int` | Token usage from parent (for child budget computation) |

### RRDeps

Defined in `ace/implementations/rr/tools.py`. Extends `AgenticDeps`.

| Field | Type | Description |
|-------|------|-------------|
| `trace_data` | `dict[str, Any]` | The canonical traces dict |
| `skillbook_text` | `str` | Skillbook text |

---

## TraceSandbox

Lightweight `exec()`-based sandbox for running LLM-generated Python code. Located in `ace/core/sandbox.py`.

**Not a security sandbox.** Restricts builtins as defence-in-depth but relies on trusting the LLM not to generate malicious code.

### Pre-loaded Namespace

| Variable | Type | Description |
|----------|------|-------------|
| `traces` | `Any` | Raw trace payload (injected by `RRStep`) |
| `skillbook` | `str` | Skillbook text (injected by `RRStep`) |
| `helper_registry` | `dict` | Metadata for registered reusable helper functions |
| `register_helper` | `Callable` | Define and persist helper code for later calls and child sessions |
| `list_helpers` | `Callable` | Return registered helper names and descriptions |
| `run_helper` | `Callable` | Invoke a registered helper by name |
| `SHOW_VARS` | `Callable` | Print available variables (debugging) |
| `json`, `re`, `math`, `collections` | module | Standard library modules |
| `datetime`, `timedelta`, `date`, `time`, `timezone` | class | datetime classes |

### Blocked Builtins

`open`, `eval`, `exec`, `compile`, `input`, `globals`, `locals`, `breakpoint`, `memoryview` — all set to `None`. `__import__` is replaced with a safe import that only allows pre-loaded modules.

### ExecutionResult

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
- **Windows / non-main thread:** No timeout enforcement.

### Runtime Helper Registry

- `register_helper(name, source, description)` executes helper source code, stores it, and records metadata.
- Registered helpers persist across `execute_code` calls within the same session.
- Child sessions (via `recurse`) inherit registered helpers automatically.

---

## Compaction

When the agent's context window fills up, two-tier compaction kicks in:

```
agent running
    ↓
PydanticAI: UsageLimitExceeded
    ↓
Budget exhausted? → YES: raise BudgetExhausted → fallback output
                  → NO: context window hit, continue ↓
    ↓
Tier 1: microcompact(messages, keep_recent=3)
    - Clear old execute_code tool results
    - Keep last 3 tool results intact
    - Keep all model messages (reasoning chain)
    ↓
Changed? → YES: retry with compacted history
         → NO: fall through to tier 2 ↓
    ↓
Tier 2: summarize_and_compact()
    - compaction_count++ (cap at max_compactions=3)
    - LLM summarizes progress (1 request from budget)
    - Save pre-compaction context to sandbox `history` variable
    - Replace history with [summary + continuation prompt]
    - Retry with compacted history
```

### Compaction Callback

`RecursiveAgent.on_compaction()` saves compaction metadata to the sandbox's `history` variable so the agent can reference prior context after compaction.

---

## Recursion

The `recurse` tool enables depth-based decomposition:

- Root agent runs at `depth=0` with `recurse` available (if `max_depth > 0`)
- Each `recurse` call spawns a child at `depth + 1` with its own sandbox and budget
- At `depth == max_depth`, `recurse` is not registered — the agent must analyze directly
- Child sandbox inherits all non-internal, non-callable variables from parent
- Registered helpers are rehydrated in child sandboxes
- Child budget: `remaining_tokens * child_budget_fraction`

---

## Timeout / Fallback

When `BudgetExhausted` is raised (token or request budget spent):

1. `RRStep._build_budget_exhausted_output()` constructs a `ReflectorOutput` with `raw["timeout"] = True`.
2. If `agent_output` and `ground_truth` are available, `_build_timeout_output()` includes a simple correct/incorrect assessment.

When any other exception occurs, a minimal `ReflectorOutput` is returned with `raw["error"]`.

---

## Online Mode Skill Evaluation

When `ctx.mode == "online"` and the skillbook is non-empty, `RRStep` appends skill evaluation instructions to the prompt. The agent:

1. Scans trace text for skill ID citations (`[section-NNNNN]`)
2. Verifies each cited ID exists in the skillbook
3. Classifies each as `helpful`, `harmful`, or `neutral`
4. Includes results in the `skill_tags` output field

In offline mode, skill evaluation is skipped (traces may be from external agents with no skill IDs).

---

## Traces Input

The `traces` variable in the sandbox contains the raw data structure:

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

For arbitrary trace inputs, the agent discovers the structure via `execute_code` and decomposes via `recurse` if needed.

---

## rr_trace Output Schema

`RRStep` enriches `ReflectorOutput.raw` with execution metadata:

```python
{
    "rr_trace": {
        "total_iterations": int,   # Number of execute_code calls
        "subagent_calls": list,    # Reserved for future use
        "timed_out": bool,         # Whether budget was exhausted
        "compactions": int,        # Number of compaction rounds
        "depth": int,              # Recursion depth of this session
    },
    "usage": {
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int,
        "requests": int,
    },
}
```

---

## Observability

Logfire auto-instruments PydanticAI agents, providing:

- Per-agent-run traces with spans for each LLM request and tool call
- Token usage tracking
- Latency metrics
- No explicit opt-in step required in the pipeline

The `rr_trace` dict in `ReflectorOutput.raw` provides programmatic access to iteration counts and metadata.

---

## Public API

```python
from ace.steps.rr_step import (
    RRStep,              # Main entry point (RecursiveAgent subclass)
    RRConfig,            # Configuration (alias for RecursiveConfig)
    RRDeps,              # PydanticAI RunContext dependencies
    TraceSandbox,        # Sandbox for code execution
    ExecutionResult,     # Result of sandbox.execute()
    ExecutionTimeoutError,
)
```
