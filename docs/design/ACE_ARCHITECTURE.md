# ACE Architecture

> Architecture for ACE, a pipeline-based framework for building self-improving AI agents. Roles are backed by PydanticAI agents; the pipeline engine handles composition and concurrency.

For full code examples and API reference, see [ACE_REFERENCE.md](ACE_REFERENCE.md).
For design decisions and rejected alternatives, see [ACE_DECISIONS.md](ACE_DECISIONS.md).
For the pipeline engine, see [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md).

---

## Overview

ACE (Agentic Context Engine) builds AI agents that learn from their own executions. It combines:

- A **pipeline engine** (`pipeline/`) with typed step contracts, concurrent execution, and structured error handling
- **Roles** (Agent, Reflector, SkillManager) backed by PydanticAI agents for structured LLM interactions
- A **Skillbook** — an evolving knowledge base of strategies that agents read from and learning loops write to
- **Integration steps** for external frameworks (browser-use, LangChain, Claude Code, Anthropic SDK)
- **Observability** via Logfire auto-instrumentation of all PydanticAI agent calls

The LLM interaction layer uses PydanticAI exclusively. Three legacy hand-rolled LLM clients (LiteLLM, Instructor, ClaudeCode) were replaced — PydanticAI handles structured output, retries with error feedback, and multi-provider support as maintained infrastructure. The pipeline engine and skillbook/learning loop are untouched.

| Kept (core IP) | Replaced (commodity plumbing) |
|---|---|
| Pipeline engine (`requires`/`provides`, `async_boundary`, `max_workers`) | LLM client abstraction (3 implementations → PydanticAI agents) |
| Skillbook & learning loop (Reflect → Update → Apply) | Structured output parsing + retries (→ PydanticAI native validation) |
| Step composition (`learning_tail`, pipeline nesting) | RR iteration loop, code extraction, budget tracking (~2,500 lines → PydanticAI agent + tools) |
| Domain-specific prompts | Sub-agent call management (CallBudget → `UsageLimits`) |

---

## Naming

| Legacy | Current | What it does |
|---|---|---|
| `OfflineACE` | `TraceAnalyser` | Analyse pre-recorded traces → evolve a skillbook |
| `OnlineACE` | `ACE` | Live execution → feedback → learning loop |
| `ACEBase` | `ACERunner` | Shared runner infrastructure (composition, not inheritance from Pipeline) |
| `ACEStepResult` | Removed — use `SampleResult` from the pipeline engine | Unified result type |

---

## Architecture Layers

The framework separates concerns into four layers:

| Layer | Location | Responsibility | Example |
|-------|----------|----------------|---------|
| **Protocols** | `ace/protocols/` | Interface contracts | `ReflectorLike.reflect()` |
| **Roles** | `ace/implementations/` | Business logic (LLM calls) | `Reflector`, `RRStep` |
| **Steps** | `ace/steps/` | Context plumbing (extract → call role → put back) | `ReflectStep` |
| **Runners** | `ace/runners/` | Orchestration (sample loop, epoch management) | `ACELiteLLM` |

**Protocols** define what a role must look like. **Roles** implement the logic. **Steps** adapt between the pipeline's context-based data flow and the role's parameter-based API. **Runners** compose steps into pipelines and iterate over inputs.

Roles are interchangeable anywhere their protocol is expected — both `Reflector` (simple single-pass) and `RRStep` (recursive multi-iteration) satisfy `ReflectorLike`. The runner and pipeline don't know or care which one is in use.

---

## Core Concepts

### Sample

The input unit for ACE. A question with optional context and ground truth:

```python
@dataclass
class Sample:
    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: dict = field(default_factory=dict)
    id: str | None = None
```

### ACESample — protocol for step access

Steps access `ctx.sample.question` uniformly. A `Protocol` makes this duck typing explicit and type-safe. `Sample` satisfies it structurally — no inheritance required.

### SkillbookView — read-only projection

The `Skillbook` is mutable — steps add, update, and remove skills. Placing it directly on a `frozen=True` context would allow mutation through the reference, breaking the immutability guarantee.

`SkillbookView` wraps a `Skillbook` and exposes only read methods (`as_prompt()`, `get_skill()`, `skills()`, `stats()`). Write methods don't exist on the class — calling them raises `AttributeError` at runtime and a type error at check time.

**Enforcement:**
- **Type checker** — mypy/pyright flags `ctx.skillbook.add_skill(...)` because `SkillbookView` has no such method.
- **Runtime** — `AttributeError` if someone calls a write method anyway.
- **Convention** — the underlying `_sb` is underscore-prefixed. Accessing it is a deliberate violation.

Steps that only **read** the skillbook (AgentStep, ReflectStep, UpdateStep) access `ctx.skillbook` — the view. Steps that **write** the skillbook (ApplyStep, DeduplicateStep, CheckpointStep) receive the real `Skillbook` via constructor injection and use `self.skillbook`.

### ACEStepContext — immutable step-to-step data

Subclass of the pipeline engine's `StepContext`. Carries all step-to-step data for the ACE pipeline. The pipeline engine only knows about `sample` and `metadata`; all ACE-specific fields live here.

Key fields:

| Field | Type | Source |
|---|---|---|
| `mode` | `Literal["online", "offline"]` | `"online"` (default) — controls skill evaluation in Reflector |
| `sample` | `ACESample \| None` | Set by runner's `_build_context()` |
| `skillbook` | `SkillbookView \| None` | Read-only projection of the real Skillbook |
| `trace` | `object \| None` | Raw execution record — any type, no enforced schema |
| `agent_output` | `AgentOutput \| None` | Produced by `AgentStep` |
| `reflections` | `tuple[ReflectorOutput, ...]` | Produced by `ReflectStep` / `RRStep` |
| `skill_manager_output` | `UpdateBatch \| None` | Produced by `UpdateStep` |
| `epoch`, `total_epochs` | `int` | Runner bookkeeping |
| `step_index`, `total_steps` | `int` | Runner bookkeeping |
| `global_sample_index` | `int` | Runner bookkeeping (used by interval steps) |

The `trace` field holds the raw execution record from any external system — a browser-use `AgentHistoryList`, a LangChain result dict, a Claude Code transcript, or any arbitrary Python object. The Reflector receives the raw trace and is responsible for making sense of it.

The `reflections` field is a tuple. In single-trace mode, it's a 1-tuple. In batch mode, it holds one `ReflectorOutput` per trace. Downstream steps iterate uniformly — no special-casing.

### Context vs constructor injection

| | On the context | Injected via constructor |
|---|---|---|
| **Nature** | Step-to-step data + read-only dependencies | Mutable shared state |
| **Lifetime** | Per-sample (born in `_build_context`, dies after pipeline) | Per-runner (created once, shared across samples) |
| **Immutable?** | Yes — frozen fields, read-only views | No — mutable by design |
| **Examples** | `agent_output`, `reflections`, `skillbook` (view) | `skillbook` (real), `environment`, `dedup_manager` |
| **Validated by engine?** | Yes — `requires`/`provides` | No — runtime error if missing |

---

## Protocols

Steps depend on protocols, not concrete classes. Each protocol defines the minimal interface a step needs. Concrete implementations satisfy them structurally — no inheritance required.

| Protocol | Method | Used by | Satisfied by |
|---|---|---|---|
| `AgentLike` | `generate(question, context, skillbook, reflection, **kwargs) → AgentOutput` | `AgentStep` | `Agent` |
| `ReflectorLike` | `reflect(question, agent_output, skillbook, ground_truth, feedback, **kwargs) → ReflectorOutput` | `ReflectStep` | `Reflector`, `RRStep` |
| `SkillManagerLike` | `update_skills(reflections, skillbook, question_context, progress, **kwargs) → SkillManagerOutput` | `UpdateStep` | `SkillManager` |
| `DeduplicationManagerLike` | `get_similarity_report(skillbook) → str \| None` | `DeduplicateStep` | `DeduplicationManager` |

Roles take a model string directly (e.g. `Agent("gpt-4o-mini")`). Internally each role creates a PydanticAI agent that handles structured output natively — no separate LLM client protocol is needed.

**Why protocols, not ABC:** Protocols use structural typing (duck typing checked by mypy). A class satisfies a protocol if it has the right methods — no `class Agent(AgentLike)` inheritance needed. Users can pass any object with a matching method, mocks satisfy protocols without ceremony, and steps are decoupled from implementations at the type level.

---

## Roles (Implementations)

Concrete LLM-based implementations of the role protocols. Live in `ace/implementations/` — fully self-contained.

| Class | Protocol | Method | What it does |
|---|---|---|---|
| `Agent` | `AgentLike` | `generate()` | Produces answers using the current skillbook of strategies |
| `Reflector` | `ReflectorLike` | `reflect()` | Single-pass analysis of agent outputs to extract lessons |
| `RRStep` | `ReflectorLike` + `StepProtocol` | `reflect()` / `__call__()` | Recursive multi-iteration reflection via PydanticAI agent with tools |
| `SkillManager` | `SkillManagerLike` | `update_skills()` | Transforms reflections into actionable skillbook updates |

All three share the same constructor pattern: `__init__(self, model: str, *, prompt_template=..., max_retries=3)`. The `model` parameter is resolved via `resolve_model()` to a PydanticAI agent.

`RRStep` is both a `StepProtocol[ACEStepContext]` (composable in any pipeline) and `ReflectorLike` (usable as a drop-in reflector). It is a subclass of `RecursiveAgent` with `execute_code` and `recurse` tools, plus two-tier compaction and depth-based recursion. See [RR_DESIGN.md](RR_DESIGN.md) for the full Recursive Reflector architecture.

---

## Steps

Reusable step implementations in `ace/steps/`. Each satisfies `StepProtocol[ACEStepContext]`. Each step does exactly one thing.

**Design principle: steps are stateless.** A step's `__call__` is a pure function of its constructor arguments and the incoming `ACEStepContext`. No internal counters, no accumulated state between invocations. Run-scoped information (like a global sample index for interval logic) comes from the context.

### Step summary

| Step | Requires | Provides | Side effects | `max_workers` |
|---|---|---|---|---|
| **AgentStep** | `sample`, `skillbook` | `agent_output` | None | 1 |
| **EvaluateStep** | `sample`, `agent_output` | `trace` | None | 1 |
| **ReflectStep** | `trace`, `skillbook` | `reflections` | None | 3; `async_boundary = True` |
| **UpdateStep** | `reflections`, `skillbook` | `skill_manager_output` | None | 1 |
| **ApplyStep** | `skill_manager_output` | — | Applies update batch to skillbook | 1 |
| **DeduplicateStep** | `global_sample_index` | — | Consolidates similar skills | 1 |
| **CheckpointStep** | `global_sample_index` | — | Saves skillbook to disk | 1 |
| **LoadTracesStep** | `sample` | `trace` | None | 1 |
| **PersistStep** | `skillbook` | — | Writes skillbook to external file | 1 |
| **ExportSkillbookMarkdownStep** | `skillbook` | — | Exports skillbook as markdown | 1 |

**Requires vs Injected:** `Requires` lists context fields (validated by the pipeline engine at construction time). The `skillbook` on the context is a `SkillbookView` (read-only). Steps that **write** to the skillbook receive the real `Skillbook` via constructor injection.

**`trace` as the universal learning input:** The learning tail's entry point (ReflectStep) requires only `trace` and `skillbook`. In the standard ACE pipeline, `EvaluateStep` bundles structured fields into a `trace` dict. In TraceAnalyser, `_build_context` places the raw trace directly. In integrations, the execute step provides `trace` from its framework's native output. The learning tail is agnostic to trace format.

Steps with empty `provides` are pure side-effect steps — they mutate shared state (skillbook) or write to external systems (disk) but add no new fields to the context.

---

## Runners

### Class hierarchy

```
ACERunner (shared infrastructure: epoch loop, delegates to Pipeline.run())
├── TraceAnalyser       — [Reflect → Update → Apply]
├── ACE                 — [Agent → Evaluate → Reflect → Update → Apply]
├── BrowserUse          — [BrowserExecute → BrowserToTrace → learning_tail]
├── LangChain           — [LangChainExecute → LangChainToTrace → learning_tail]
├── ClaudeCode          — [ClaudeCodeExecute → ClaudeCodeToTrace → learning_tail]
└── OpenClaw (script)   — [LoadTraces → OpenClawToTrace → learning_tail]

ACELiteLLM (standalone convenience wrapper — not an ACERunner subclass)
├── ask()               — direct Agent call, no pipeline
├── learn()             — delegates to lazy-init ACE runner
├── learn_from_traces() — delegates to lazy-init TraceAnalyser
└── learn_from_feedback()— runs learning_tail from last ask()

RRStep (RecursiveAgent subclass — composable iterative step)
├── __call__()          — StepProtocol entry; usable in any runner's pipeline
├── reflect()           — ReflectorLike entry; drop-in reflector for runners
└── _run_reflection()   — PydanticAI agent with execute_code and recurse tools
```

All runners compose a `Pipeline` rather than extending it.

### ACERunner — shared base

Encapsulates everything runners have in common: the epoch loop and Iterable validation. Per-sample iteration, error handling, background execution, and checkpoints are all delegated to `Pipeline.run()`.

Subclasses only override `run()` (public signature) and `_build_context()` (input mapping).

**Responsibilities:**

| Concern | Owner |
|---|---|
| Epoch loop + Iterable validation | `ACERunner._run()` |
| Per-sample iteration + error isolation | `Pipeline.run()` |
| Foreground/background split | `Pipeline.run()` (via `async_boundary`) |
| Concurrent workers | `Pipeline.run(workers=N)` |
| Checkpoints | `CheckpointStep` (in the pipeline) |
| Background drain | `ACERunner.wait_for_background()` → `Pipeline.wait_for_background()` |
| Skillbook I/O | `save(path)` on the runner |

Each sample is independent — no state persists across samples. The skillbook is the only cross-sample coupling.

**Eventual consistency:** `SkillbookView` is a thin delegation wrapper, not a snapshot — it reads from the live `Skillbook` at call time. When background learning is active, concurrent samples may observe partially-updated skillbook state. This is by design: steps see a best-effort view rather than a point-in-time snapshot. The trade-off is acceptable because (1) the skillbook is LLM prompt context where a few missing or extra skills have negligible impact, (2) serialising reads would eliminate the concurrency benefit, and (3) write steps already run with `max_workers = 1`.

### TraceAnalyser

Analyses pre-recorded traces without executing an agent. Runs the learning tail only. Accepts raw trace objects of any type.

**When to use:** You have execution logs from an external system and want to build or refine a skillbook from historical data. Multi-epoch mode re-processes all traces with the evolving skillbook.

**Pipeline:**

```
[ReflectStep] → [UpdateStep] → [ApplyStep]
```

No AgentStep, no EvaluateStep. The trace already contains the agent's output and the evaluation feedback.

**Multi-epoch semantics:** Each epoch re-processes all traces with the current skillbook. Early epochs extract obvious patterns; later epochs refine and consolidate.

### ACE

The full live adaptive pipeline. An agent executes, the reflector analyses, the skill manager updates. Optionally evaluates against a `TaskEnvironment` for feedback-driven learning.

**When to use:** Building a new agent, or running closed-loop learning where the agent improves in real time.

**Pipeline:**

```
[AgentStep] → [EvaluateStep] → [ReflectStep] → [UpdateStep] → [ApplyStep]
```

A single class handles both single-pass (`epochs=1`) and multi-epoch batch training (`epochs > 1`). The `environment` is optional — when provided, `EvaluateStep` generates feedback. When omitted, the Reflector learns from ground-truth comparison or the agent's reasoning alone.

### ACELiteLLM — standalone convenience wrapper

`ACELiteLLM` is not an `ACERunner` subclass. It wraps two different runners (`ACE` and `TraceAnalyser`) and exposes a fundamentally different API:

| Method | What it does |
|---|---|
| `ask(question, context)` | Direct Agent call — no pipeline. Stores interaction for `learn_from_feedback()` |
| `learn(samples, environment, epochs)` | Delegates to lazy-init ACE runner |
| `learn_from_traces(traces, epochs)` | Delegates to lazy-init TraceAnalyser |
| `learn_from_feedback(feedback, ground_truth)` | Manual single-shot learning from last `ask()` call |

Runners are cached and invalidated on `load()` (new skillbook object means stale references).

### Factory methods

All runners provide a `from_roles` factory that takes pre-built role instances. Integration runners also provide `from_model()` that auto-builds PydanticAI-backed roles from a model string.

**Common parameters on `from_roles`:**

| Parameter | Default | Description |
|---|---|---|
| `skillbook` | `Skillbook()` | Starting skillbook |
| `dedup_manager` | `None` | Appends a `DeduplicateStep` |
| `dedup_interval` | `10` | Deduplication frequency |
| `checkpoint_dir` | `None` | Appends a `CheckpointStep` |
| `checkpoint_interval` | `10` | Checkpoint frequency |
| `extra_steps` | `None` | Additional steps appended after the learning tail |

### `learning_tail()` — reusable learning steps

Every integration assembles the same `[Reflect → Update → Apply]` suffix. `learning_tail()` returns this standard step list, with optional dedup and checkpoint steps. If the provided reflector already exposes `provides = {'reflections'}` (e.g. `RRStep`), it's inserted directly instead of being wrapped in `ReflectStep`.

---

## Integration Pattern

External frameworks integrate via composable pipeline steps in `ace/integrations/`. Each integration provides:

1. **Result type** — an integration-specific dataclass (e.g. `BrowserResult`, `ClaudeCodeResult`)
2. **Execute step** — INJECT skillbook context + EXECUTE the framework, writes to `ctx.trace`
3. **ToTrace step** — converts the integration-specific result into the standardised trace dict

### Execute → Convert → Learn

```
Standard ACE:      [Agent → Evaluate]                          → [Reflect → Update → Apply]
                    ╰── execute (built-in) ──╯                    ╰──────── learn (shared) ──────╯
                         provides: trace (dict) ─────────────────► requires: trace

Browser-use:       [BrowserExecute] → [BrowserToTrace]         → [Reflect → Update → Apply]
                    ╰── execute ────╯   ╰── convert ──╯           ╰──────── learn (shared) ──────╯
                    provides: trace      rewrites trace             requires: trace
                    (BrowserResult)      (BrowserResult → dict)

TraceAnalyser:     [_build_context]                            → [Reflect → Update → Apply]
                    ╰── sets ctx.trace (raw object) ───────╯      ╰──────── learn (shared) ──────╯
```

The standardised trace dict keys match what `ReflectStep` expects: `question`, `reasoning`, `answer`, `skill_ids`, `feedback`, `ground_truth`.

### Result types

| Integration | Result type | Key fields |
|---|---|---|
| Browser-use | `BrowserResult` | `task`, `success`, `output`, `error`, `steps_count`, `duration_seconds`, `cited_skill_ids`, `chronological_steps`, `raw_history` |
| Claude Code | `ClaudeCodeResult` | `task`, `success`, `output`, `execution_trace`, `returncode`, `error` |
| Claude SDK | `ClaudeSDKResult` | `task`, `success`, `output`, `error`, `model`, `stop_reason`, `input_tokens`, `output_tokens`, `tool_calls`, `cited_skill_ids` |
| LangChain | `LangChainResult` | `task`, `output`, `result_type`, `success`, `error`, `intermediate_steps`, `messages`, `raw_result` |

### Why two steps instead of one

Splitting execute from trace conversion gives independent testability, reusability (execute step usable standalone), and separation of concerns (framework interaction vs trace formatting).

### Live vs offline

| | Integration Runner | TraceAnalyser |
|---|---|---|
| When | Live execution | Post-hoc analysis |
| Agent | Framework runs it | Already ran |
| Feedback | Generated live | Baked into trace |
| Use case | Production deployment | Historical batch learning, debugging |

Both update the same skillbook. A common workflow: TraceAnalyser builds an initial skillbook from historical data, then an integration runner refines it during live deployment.

> **MCP Server** is a different pattern. It does not add pipeline steps — it's a thin async layer over `ACELiteLLM` that exposes ACE as an MCP tool provider. See [MCP Server docs](../integrations/mcp.md).

---

## Configuration & Providers

### Principles

1. **API keys never appear in ACE APIs** — no `api_key` parameter anywhere. Keys are resolved from the environment by LiteLLM at call time.
2. **Per-role model selection** — Agent, Reflector, and SkillManager can each use different models.
3. **Validate before running** — `ace setup` and `validate_connection()` make a tiny LLM call to verify auth before writing code.

### Config types

- **`ModelConfig`** — which model to use for a role (model string, temperature, max_tokens). No secrets.
- **`ACEModelConfig`** — model selection per ACE role. Serialises to/from `ace.toml` (committable, no secrets).

### Construction paths

| Constructor | Input | Use case |
|---|---|---|
| `ACELiteLLM.from_setup()` | `ace.toml` + `.env` | Teams, CI, guided setup |
| `ACELiteLLM.from_config(config)` | `ACEModelConfig` object | Per-role model selection in code |
| `ACELiteLLM.from_model("gpt-4o")` | Model string | Quick start, single model |
| `ACELiteLLM("gpt-4o-mini", ...)` | Model string + overrides | Full control |

### CLI

| Command | What it does |
|---|---|
| `ace setup` | Interactive wizard: model name, API key, validate, assign per-role models. Saves `.env` + `ace.toml`. |
| `ace models [query]` | Search LiteLLM's model registry (2,600+ models). Filter by `--provider`. |
| `ace providers` | List providers with env var names and key status. |
| `ace validate <model>` | Test a model connection with a tiny LLM call. |

### File layout

| File | Secrets? | Committable? | Purpose |
|---|---|---|---|
| `.env` | Yes | No (gitignored) | API keys only |
| `ace.toml` | No | Yes | Model names + parameters per role |

### Key resolution flow

```
API Key:  .env → os.environ → LiteLLM reads OPENAI_API_KEY / ANTHROPIC_API_KEY / etc.
Model:    ace.toml → ACEModelConfig.for_role("agent") → resolve_model(model) → PydanticAI agent
```

### Provider resolution

ACE model strings follow LiteLLM convention (`provider/model`). The resolver in `ace/providers/pydantic_ai.py` routes them through three paths:

1. **PydanticAI-native prefix** — strings like `openai:gpt-4o` pass through unchanged
2. **LiteLLM prefix → native provider** — when the first path segment matches a PydanticAI native provider, `/` is rewritten to `:` (e.g. `bedrock/model` → `bedrock:model`)
3. **Fallback** — everything else is prefixed with `litellm:` for the proxy provider

```
LiteLLM string                                  → PydanticAI string
─────────────────────────────────────────────────  ──────────────────────────────────────────
gpt-4o-mini                                      → litellm:gpt-4o-mini
bedrock/eu.anthropic.claude-haiku-4-5-v1:0       → bedrock:eu.anthropic.claude-haiku-4-5-v1:0
groq/llama-3.1-70b-versatile                     → groq:llama-3.1-70b-versatile
openrouter/anthropic/claude-3.5-sonnet           → openrouter:anthropic/claude-3.5-sonnet
anthropic/claude-3-5-sonnet-20241022             → anthropic:claude-3-5-sonnet-20241022
ollama/llama3                                    → litellm:ollama/llama3
together_ai/meta-llama/Llama-3-70b               → litellm:together_ai/meta-llama/Llama-3-70b
```

Mapped LiteLLM prefixes: `anthropic`, `azure`, `azure_ai`, `bedrock`, `cohere`, `deepseek`, `groq`, `mistral`, `openrouter`, `vertex_ai`. All others fall through to `litellm:`.

Native providers are faster (no proxy hop) and use the provider's own API key env vars directly. Install with extras: `uv add "pydantic-ai-slim[anthropic,openai,bedrock]"`.

---

## Deduplication

Skill deduplication subsystem in `ace/deduplication/` — fully self-contained.

| Class | Role |
|---|---|
| `SimilarityDetector` | Computes embeddings, detects similar pairs via cosine similarity |
| `DeduplicationManager` | Coordinates detection and consolidation |

**Embedding providers:** LiteLLM (remote) or sentence-transformers (local, lazy-loaded).

**Consolidation operations:**

| Operation | Effect |
|---|---|
| `MergeOp` | Combine skills — accumulate counters, soft-delete others |
| `DeleteOp` | Soft-delete a redundant skill |
| `KeepOp` | Store a similarity decision so the pair is not flagged again |
| `UpdateOp` | Refine content to differentiate, clear embedding |

**Pipeline integration:** Deduplication runs as a separate `DeduplicateStep` at a configurable interval, not inside the SkillManager. Appended by factory methods when a `DeduplicationManagerLike` is provided.

---

## Observability

PydanticAI has first-class Logfire integration. One call auto-instruments everything:

```python
logfire.configure()
logfire.instrument_pydantic_ai()
```

This automatically captures agent runs, tool calls, model requests, structured output validation, and sub-agent delegation — the entire RR execution appears as a structured trace. No custom span-building code.

**Pipeline integration:** Logfire is OpenTelemetry-based, so pipeline-level spans coexist. Steps that don't use PydanticAI can use `logfire.span()` / `logfire.info()` directly.

**Setup:** `ace/observability/configure_logfire()` auto-instruments all PydanticAI agents. Opt-in via `ACELiteLLM(logfire=True)`. Config is purely env-based (`LOGFIRE_TOKEN`), no changes to `ace.toml`.

---

## Concurrency

Both TraceAnalyser and ACE inherit async capabilities from the pipeline engine. No custom async machinery is needed.

### ReflectStep as async boundary

`ReflectStep.async_boundary = True` means everything before it (Agent, Evaluate) runs in the foreground, and everything from ReflectStep onwards runs in a background thread pool:

```
sample 1:  [AgentStep] [EvaluateStep] ──fire──► [ReflectStep] [UpdateStep] [ApplyStep]
sample 2:  [AgentStep] [EvaluateStep] ──fire──► ...
                                       ↑
                                 async_boundary
```

### Concurrency knobs

| Knob | Where | Effect |
|---|---|---|
| `ReflectStep.max_workers = 3` | Step class attribute | Up to 3 reflections in parallel |
| `UpdateStep.max_workers = 1` | Step class attribute | Serialises skill manager LLM calls |
| `ApplyStep.max_workers = 1` | Step class attribute | Serialises skillbook writes |
| `wait_for_background(timeout)` | Runner method | Blocks until background threads drain |

### Cancellation

ACE inherits cancellation from the pipeline engine. Pass a `CancellationToken` to `run()`. The pipeline checks it before each foreground step. Within a step, PydanticAI's async runtime handles cancellation natively. The token flows via `contextvars.ContextVar` — no parameter changes needed across layers. See [PIPELINE_DESIGN.md § Cancellation](PIPELINE_DESIGN.md#cancellation).

---

## Error Handling

Follows the pipeline engine's error model without additions.

- **Per-sample isolation:** A failing sample does not abort the run. The exception is recorded in `SampleResult.error` and `SampleResult.failed_at`.
- **Background failures:** Captured and attached to `SampleResult` by the pipeline engine.
- **No retry logic in the runner.** Retries are the responsibility of individual steps (e.g., PydanticAI's built-in retry with error feedback).

---

## Directory Structure

```
ace/
  __init__.py               ← Public API re-exports
  core/
    context.py              ← ACEStepContext, SkillbookView, ACESample
    insight_source.py       ← TraceIdentity, TraceReference, InsightSource
    outputs.py              ← AgentOutput, ReflectorOutput, SkillManagerOutput
    skillbook.py            ← Skill, Skillbook, SimilarityDecision
    environments.py         ← Sample, TaskEnvironment, SimpleEnvironment
  protocols/                ← Role protocols (one file per protocol)
    agent.py, reflector.py, skill_manager.py, deduplication.py
  implementations/          ← PydanticAI-backed role implementations
    agent.py, reflector.py, skill_manager.py, helpers.py, prompts.py
  steps/                    ← Pipeline steps (one file per class)
    __init__.py             ← learning_tail() helper
    agent.py, evaluate.py, reflect.py, update.py,
    apply.py, deduplicate.py, checkpoint.py,
    load_traces.py, persist.py, export_markdown.py, observability.py
  runners/                  ← Runner classes
    base.py                 ← ACERunner
    trace_analyser.py, ace.py, browser_use.py, langchain.py,
    claude_code.py, litellm.py
  integrations/             ← Integration steps (execute + result + converter)
    browser_use.py, langchain.py, claude_code.py, claude_sdk.py
    openclaw/               ← OpenClaw trace converter
    mcp/                    ← Optional MCP server
  providers/                ← PydanticAI model resolution
    pydantic_ai.py, config.py, registry.py
  deduplication/            ← Skill deduplication subsystem
    detector.py, manager.py, operations.py, prompts.py
  rr/                       ← Recursive Reflector (PydanticAI agent)
  observability/            ← Logfire configuration
```

### Key modules

| Module | Contents |
|---|---|
| `ace/core/` | `ACEStepContext`, `SkillbookView`, `Skillbook`, `AgentOutput`, `ReflectorOutput`, `InsightSource` |
| `ace/protocols/` | `AgentLike`, `ReflectorLike`, `SkillManagerLike` protocols |
| `ace/implementations/` | PydanticAI-backed `Agent`, `Reflector`, `SkillManager` |
| `ace/steps/` | All pipeline steps + `learning_tail()` |
| `ace/runners/` | `ACERunner`, `TraceAnalyser`, `ACE`, `BrowserUse`, `LangChain`, `ClaudeCode`, `ACELiteLLM` |
| `ace/providers/` | `resolve_model`, `ACEModelConfig`, `validate_connection` |
| `ace/steps/rr_step.py` | `RRStep` (RecursiveAgent subclass), `RRConfig`, `TraceSandbox` |
| `ace/core/recursive_agent.py` | `RecursiveAgent`, `AgenticConfig`, `AgenticDeps`, compaction, recursion |
| `ace/integrations/` | Execute steps, result types, ToTrace converters; MCP server |
| `ace/deduplication/` | Dedup subsystem (detector, manager, operations) |
| `ace/observability/` | Logfire configuration (`configure_logfire()`) |

---

## Future Directions

Issues acknowledged but deferred.

**Streaming / lazy iteration:** `_run()` eagerly materializes the full iterable before passing to `Pipeline.run()`. True streaming would require the pipeline to accept an iterator. Deliberate simplification — revisit if memory pressure from large single-pass runs becomes real.

**Builder API for custom pipelines:** The current API offers two extremes: factory methods that hide the pipeline, and manual construction that requires understanding step contracts. A builder could bridge this gap, but `learning_tail()` covers the most common customisation (custom execute step + standard learning). Worth pursuing when users hit friction with manual wiring.

**Skillbook rollback and versioning:** Currently the skillbook is mutated in place with no undo. A lightweight versioning mechanism (snapshotting at epoch boundaries, `rollback(to_version)`) would enable automatic revert when metrics degrade. Deferred because checkpoints cover the common recovery scenario.

**LiteLLM proxy base URL support:** Users running a LiteLLM proxy may need `api_base` configuration. Deferred because all current users connect directly to providers.
