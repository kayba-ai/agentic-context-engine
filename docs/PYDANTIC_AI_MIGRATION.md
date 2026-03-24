# PydanticAI Migration Plan

> Migrating ace_next's LLM interaction layer to PydanticAI while preserving the pipeline engine and learning loop.

## Motivation

ace_next currently has three hand-rolled LLM client implementations (LiteLLMClient, InstructorClient, ClaudeCodeLLMClient) with inconsistent retry/validation behavior, ~3,500 lines of custom agent-loop plumbing in the Recursive Reflector, and manual code extraction via regex. PydanticAI handles all of this as maintained infrastructure.

### What we keep (our competitive advantage)

- **Pipeline engine** (`pipeline/`) — `requires`/`provides` contracts, `async_boundary`, per-step `max_workers`, `SampleResult` error isolation. No framework offers this combination.
- **Skillbook & learning loop** — Reflect -> Tag -> Update -> Apply -> Deduplicate. This is core IP.
- **Step composition** — `learning_tail()`, pipeline-as-step nesting, `SkillbookView` read/write split.
- **Domain-specific prompts** — tightly coupled to skillbook format and ACE's reflection strategy.

### What we replace (commodity plumbing)

- LLM client abstraction (3 implementations -> PydanticAI agents)
- Structured output parsing + retries (hand-rolled -> PydanticAI native validation with error feedback)
- RR iteration loop, code extraction, budget tracking, message management (~2,500 lines -> PydanticAI agent loop + tools)
- Sub-agent call management (CallBudget -> `UsageLimits` + shared `ctx.usage`)

---

## Architecture Overview

### Before

```
Pipeline Step -> Role (Agent/Reflector/SkillManager) -> LLMClientLike -> LiteLLM/Instructor/ClaudeCode
                                                        ^^^^^^^^^^^^
                                                        3 implementations
                                                        manual JSON extraction
                                                        inconsistent retries

RRStep -> SubRunner -> 4 inner steps -> LLM + Sandbox + CodeExtraction + CheckResult
          ^^^^^^^^     ^^^^^^^^^^^^
          custom loop  3,500 lines
```

### After

```
Pipeline Step -> PydanticAI Agent (with output_type + tools) -> any provider
                 ^^^^^^^^^^^^^^^
                 one abstraction
                 validated retries with error feedback
                 native tool calling
                 multi-provider

RRStep -> PydanticAI Agent (with execute_code + analyze tools) -> sandbox + sub-agent
          ^^^^^^^^^^^^^^^
          native agent loop
          no code extraction
          UsageLimits for budget
          history_processors for trimming
```

The pipeline engine is untouched. PydanticAI lives *inside* the steps.

---

## Phase 1: Simple Roles (Agent, Reflector, SkillManager)

### Current state

Each role is a single-shot `prompt -> llm.complete_structured() -> Pydantic model` call. Retry logic varies:
- LiteLLMClient: blind retry (re-calls without error feedback)
- InstructorClient: Instructor's retry with validation error feedback
- ClaudeCodeLLMClient: manual JSON extraction with 3 retries + error prompt

### Target state

Each role becomes a thin wrapper around a PydanticAI `Agent`:

```python
from pydantic_ai import Agent

class Agent:
    def __init__(self, model: str, **kwargs):
        self._agent = Agent(
            model,
            output_type=AgentOutput,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    async def generate(self, question, context, skillbook, reflection=None, **kwargs) -> AgentOutput:
        result = await self._agent.run(
            self._format_prompt(question, context, skillbook, reflection)
        )
        return result.output
```

Similarly for Reflector (`output_type=ReflectorOutput`) and SkillManager (`output_type=SkillManagerOutput`).

### What changes

| Concern | Before | After |
|---|---|---|
| Structured output | Manual JSON extraction + Pydantic parse | PydanticAI validates via tool-call schema, retries with error feedback |
| Retries | 3 blind retries or Instructor | PydanticAI native (configurable, with error context) |
| Provider support | LiteLLM wrapper | PydanticAI (native support for 15+ providers, wraps LiteLLM internally) |
| Streaming | Manual chunk iteration for cancellation | PydanticAI `run_stream` |

### Protocol compatibility

The existing protocols (`AgentLike`, `ReflectorLike`, `SkillManagerLike`) stay unchanged. The PydanticAI-backed implementations satisfy them. Steps don't know or care what's behind the protocol.

### Files affected

- `ace_next/implementations/agent.py` — rewrite internals, keep `generate()` signature
- `ace_next/implementations/reflector.py` — rewrite internals, keep `reflect()` signature
- `ace_next/implementations/skill_manager.py` — rewrite internals, keep `update_skills()` signature
- `ace_next/providers/litellm.py` — deprecate (PydanticAI handles provider abstraction)
- `ace_next/providers/instructor.py` — deprecate (PydanticAI handles structured output)
- `ace_next/protocols/llm.py` — deprecate `LLMClientLike` (roles use PydanticAI agents directly)

### Migration path

1. Implement PydanticAI-backed roles behind existing protocols
2. Update runners to construct PydanticAI-backed roles by default
3. Keep old implementations available behind a flag during transition
4. Remove old implementations once validated

---

## Phase 2: Recursive Reflector (RR) Redesign

### Design principles

1. **Preserve RLM's core insight**: code execution as reasoning medium. The LLM reasons *by writing and running code*, getting ground-truth feedback.
2. **Code-first, tools-assisted**: `execute_code` is the primary tool. Accelerator tools (`analyze`, `batch_analyze`) handle common patterns without requiring code.
3. **PydanticAI-native loop**: no custom SubRunner, no inner pipeline, no code extraction regex.

### Target architecture

```
PydanticAI Agent
  model: configurable
  output_type: ReflectorOutput          <- replaces FINAL()
  system_prompt: REFLECTOR_RECURSIVE_PROMPT (adapted)
  history_processors: [semantic_trim]   <- reuses existing scoring logic
  usage_limits: request_limit=30        <- replaces CallBudget

  Tools:
    execute_code(code: str) -> str       <- PRIMARY: sandbox execution
    analyze(question, context) -> str    <- sub-agent LLM call
    batch_analyze(question, items) -> list[str]  <- parallel sub-agent calls
    inspect_traces(path?) -> str         <- trace schema/overview

  Dependencies (deps_type):
    sandbox: TraceSandbox               <- persists across tool calls
    trace_data: dict                    <- immutable trace data
    skillbook_text: str                 <- skillbook as prompt
    iteration: int                      <- mutable counter
```

### How each RR component maps

#### Iteration loop (runner.py, ~400 lines -> gone)

Current: `SubRunner.run_loop()` manually iterates, builds inner pipeline, checks termination, accumulates messages.

After: PydanticAI's agent loop. The LLM calls tools until it produces `ReflectorOutput`. No custom loop code.

#### Code extraction (code_extraction.py, ~200 lines -> gone)

Current: 200 lines of regex to extract Python from markdown fenced blocks.

After: code arrives as the `code` parameter of `execute_code` tool. The LLM passes code as a typed string argument. Zero extraction logic.

#### Inner pipeline steps (steps.py, ~500 lines -> ~50 lines of tool definitions)

Current: LLMCallStep -> ExtractCodeStep -> SandboxExecStep -> CheckResultStep.

After:
```python
@rr_agent.tool(retries=3)
async def execute_code(ctx: RunContext[RRDeps], code: str) -> str:
    """Execute Python code in the analysis sandbox. Variables persist across calls."""
    ctx.deps.iteration += 1
    result = ctx.deps.sandbox.run(code, timeout=ctx.deps.timeout)
    if result.exception:
        raise ModelRetry(f"Code error: {result.exception}. Fix and retry.")
    return result.stdout[:ctx.deps.max_output_chars]
```

Guard logic (reject premature conclusions) moves to output validation:
```python
@rr_agent.output_validator
async def validate_output(ctx: RunContext[RRDeps], output: ReflectorOutput) -> ReflectorOutput:
    if ctx.deps.iteration < 2:
        raise ModelRetry("You haven't explored the data enough. Use execute_code and analyze first.")
    return output
```

#### Sub-agent / ask_llm (subagent.py, ~200 lines -> ~15 lines)

Current: `SubAgentLLM` class, `CallBudget`, `create_ask_llm_function()` factory.

After:
```python
sub_agent = Agent(sub_model, output_type=str, system_prompt=SUBAGENT_PROMPT)

@rr_agent.tool
async def analyze(ctx: RunContext[RRDeps], question: str, context: str) -> str:
    """Ask a sub-agent to analyze trace data. Use for deep analysis of specific findings."""
    result = await sub_agent.run(
        f"{question}\n\nContext:\n{context}",
        usage=ctx.usage,  # shared budget
    )
    return result.output
```

#### Batch analysis / parallel_map (sandbox.py, ~65 lines -> ~10 lines)

Current: `_parallel_map()` — ThreadPoolExecutor with retries and exponential backoff, injected into sandbox namespace.

After:
```python
@rr_agent.tool
async def batch_analyze(ctx: RunContext[RRDeps], question: str, items: list[str]) -> list[str]:
    """Analyze multiple items in parallel using the sub-agent."""
    tasks = [sub_agent.run(f"{question}\n\nItem:\n{item}", usage=ctx.usage) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r.output if not isinstance(r, Exception) else f"Error: {r}" for r in results]
```

Native async. No thread pool. No retry glue. PydanticAI + asyncio handles concurrency.

#### Message trimming (message_trimming.py, ~150 lines -> ~100 lines)

Current: `trim_messages()` pure function with semantic scoring.

After: reuse the scoring algorithm inside a PydanticAI `history_processor`:
```python
def semantic_trim(ctx: RunContext[RRDeps], messages: list[ModelMessage]) -> list[ModelMessage]:
    # Reuse existing scoring: errors (+3.0) > findings (+2.0) > exploration (+1.0)
    # Trim to fit context budget
    return trimmed
```

The scoring logic is preserved. The plumbing (integration with the iteration loop, manual message accumulation) disappears.

#### Context & state (context.py, sub_runner.py, ~150 lines -> gone)

Current: `RRIterationContext` frozen dataclass, `SubRunner` base class.

After: PydanticAI manages message state internally. Mutable state lives on `ctx.deps`. No custom context type needed.

### Sandbox changes

The sandbox (`sandbox.py`, ~300 lines) is preserved with modifications:

**Kept:**
- Restricted builtins whitelist
- Safe module pre-imports (json, re, collections, datetime)
- Timeout enforcement (signal-based)
- Namespace persistence across calls
- `SHOW_VARS()` debug helper

**Removed from sandbox (now top-level tools):**
- `FINAL()` -> `output_type=ReflectorOutput`
- `FINAL_VAR()` -> gone (LLM produces structured output directly)
- `ask_llm()` -> `analyze` tool
- `parallel_map()` -> `batch_analyze` tool
- `llm_query()` -> `analyze` tool

**Injected via deps:**
- `trace` / `traces` dict — trace data for code exploration
- `skillbook` — skillbook as text

### Prompt adaptation

The 6-phase strategy (Discover -> Adapt -> Survey -> Categorize -> Deep-dive -> Synthesize) is preserved in `system_prompt`. Changes:

- Replace "write Python code in a fenced block" with "call execute_code with your code"
- Replace "call ask_llm()" with "call the analyze tool"
- Replace "call parallel_map(fn, items)" with "call the batch_analyze tool"
- Replace "call FINAL(result)" with "when you have enough evidence, produce your final analysis"
- Keep the verification phase, deep-dive strategy, and batch mode instructions

### Files deleted

| File | Lines | Reason |
|---|---|---|
| `rr/runner.py` | ~400 | PydanticAI agent loop replaces SubRunner |
| `rr/steps.py` | ~500 | Tool definitions replace inner pipeline steps |
| `rr/code_extraction.py` | ~200 | Tool args are pre-parsed, no extraction needed |
| `rr/context.py` | ~50 | PydanticAI manages message state |
| `rr/subagent.py` | ~200 | Delegate agent + `ctx.usage` replaces CallBudget |
| `core/sub_runner.py` | ~100 | PydanticAI is the runner |
| **Total deleted** | **~1,450** | |

### Files modified

| File | Lines | Changes |
|---|---|---|
| `rr/sandbox.py` | ~300 | Remove FINAL, ask_llm, parallel_map injection; keep core sandbox |
| `rr/prompts.py` | ~400 | Adapt for tool-call pattern instead of code-in-markdown |
| `rr/message_trimming.py` | ~150 | Wrap scoring logic in `history_processor` interface |

### Files created

| File | Est. lines | Content |
|---|---|---|
| `rr/agent.py` | ~150 | PydanticAI agent definition, tools, deps, output validator |

### Net result

- Before: ~3,500 lines
- After: ~1,000 lines (sandbox + prompts + trimming + agent definition)
- Deleted: ~2,500 lines of loop/extraction/budget/context plumbing

---

## Phase 3: Provider & Runner Cleanup

### Provider layer: what stays, what goes

The provider layer has two concerns: **model selection/validation** and **LLM calling**. Only the calling part changes.

#### Stays unchanged

- `providers/config.py` — `ModelConfig`, `ACEModelConfig`, ace.toml persistence. This is configuration, not calling. Model strings like `"openrouter/anthropic/claude-3.5-sonnet"` stay the same.
- `providers/registry.py` — `search_models()`, `validate_connection()`, key detection. This is discovery/validation tooling that uses LiteLLM's model registry directly. Independent of how we call the LLM.

#### Replaced

| Provider | Replacement |
|---|---|
| `providers/litellm.py` (LiteLLMClient) | PydanticAI agents inside roles. PydanticAI calls LiteLLM internally via `LiteLLMModel`. |
| `providers/instructor.py` (InstructorClient) | PydanticAI native structured output (validation + retry with error feedback built-in). |
| `providers/claude_code.py` (ClaudeCodeLLMClient) | PydanticAI with Anthropic provider (or keep as niche CLI-based provider). |

#### Multi-provider support (OpenRouter, Groq, Ollama, etc.)

All existing LiteLLM model strings continue to work. `resolve_model()` in `providers/pydantic_ai.py` translates them for PydanticAI using three resolution paths:

1. **PydanticAI-native prefix** — Strings with a `provider:model` prefix matching a PydanticAI provider (e.g. `openai:gpt-4o`, `bedrock:model-id`) pass through unchanged.

2. **LiteLLM prefix → native provider** — When the first path segment of a LiteLLM string matches a PydanticAI native provider, the `/` is rewritten to `:`. This is necessary because PydanticAI's `litellm` provider uses an OpenAI-compatible HTTP client under the hood, which doesn't work for providers with non-OpenAI APIs (Bedrock via SigV4, Anthropic's native API, etc.).

3. **Fallback** — Everything else is prefixed with `litellm:` for the LiteLLM proxy provider (Ollama, Together, Fireworks, etc.).

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

The mapped LiteLLM prefixes are: `anthropic`, `azure`, `azure_ai`, `bedrock`, `cohere`, `deepseek`, `groq`, `mistral`, `openrouter`, `vertex_ai`. All others fall through to the `litellm:` provider.

User-facing API is unchanged:
```python
# ace.toml — same LiteLLM model strings as before
# [default]
# model = "openrouter/anthropic/claude-3.5-sonnet"
# temperature = 0.0

# Python — same as before
ace = ACELiteLLM.from_model("openrouter/anthropic/claude-3.5-sonnet")
ace = ACELiteLLM.from_model("bedrock/eu.anthropic.claude-haiku-4-5-v1:0")
ace = ACELiteLLM.from_setup(config_dir=".")
```

### Protocol changes

- `LLMClientLike` protocol is deprecated. Roles use PydanticAI agents internally.
- `AgentLike`, `ReflectorLike`, `SkillManagerLike` protocols stay unchanged. Steps are unaffected.

### Runner updates

- `ACELiteLLM.from_model(model)` still works — internally creates PydanticAI-backed roles
- `ACE.from_roles(agent, reflector, skill_manager)` still works — accepts any protocol-satisfying object
- `TraceAnalyser` still works — uses PydanticAI-backed reflector

### Config changes

- `ModelConfig` / `ACEModelConfig` adapted to pass model strings to PydanticAI agents
- Per-role model selection preserved (`agent_model`, `reflector_model`, `skill_manager_model`)

---

## Phase 4: Observability — Pydantic Logfire

### Current state

Custom observability via two hand-rolled steps:
- `steps/observability.py` (~38 lines) — logs pipeline metrics to Opik
- `rr/opik.py` (~318 lines) — builds hierarchical Opik traces with child spans per RR iteration and sub-agent call, manually aggregates token/cost totals

This is ~356 lines of plumbing that manually extracts `llm_metadata`, `usage`, and `cost` from internal data structures and reshapes them into Opik spans.

### Target state

PydanticAI has first-class Logfire integration. One call auto-instruments everything:

```python
import logfire

logfire.configure()  # reads LOGFIRE_TOKEN from env
logfire.instrument_pydantic_ai()  # auto-traces all PydanticAI agents
```

This single setup call automatically captures:
- **Agent runs** — input prompt, output, duration, success/failure
- **Tool calls** — name, arguments, return value, retries, `ModelRetry` errors
- **Model requests** — provider, model, request/response, token usage, cost
- **Structured output validation** — validation errors, retry feedback
- **Sub-agent delegation** — nested spans with shared usage tracking

The entire RR execution (execute_code calls, analyze/batch_analyze tool calls, output validation retries) appears as a structured trace automatically — no custom span-building code.

### What Logfire replaces

| Current | Logfire |
|---|---|
| `RROpikStep` (318 lines of manual span building) | Auto-instrumented by `logfire.instrument_pydantic_ai()` |
| `ObservabilityStep` (38 lines of metric logging) | Logfire captures metrics from agent runs natively |
| Manual `llm_metadata` / `usage` extraction | PydanticAI exposes usage on `result.usage()` — Logfire captures it automatically |
| Manual token/cost aggregation across iterations | Logfire aggregates across the full trace |
| Opik client setup, flush, error handling | `logfire.configure()` — one line |

### Integration with pipeline

Logfire is OpenTelemetry-based, so pipeline-level instrumentation can coexist:

```python
import logfire

# Instrument PydanticAI agents (covers all roles + RR)
logfire.instrument_pydantic_ai()

# Optional: add pipeline-level spans
with logfire.span("ace_pipeline", sample_id=ctx.metadata.get("sample_id")):
    result = await pipeline.run(ctx)
```

Pipeline steps that don't use PydanticAI (e.g., skillbook dedup, tag extraction) can use `logfire.span()` / `logfire.info()` directly for lightweight tracing without custom step classes.

### Opik coexistence

Logfire doesn't require dropping Opik. During transition:
1. Keep `RROpikStep` available as an optional step for users who already use Opik
2. Add `logfire.instrument_pydantic_ai()` as the default (zero-config) observability
3. Eventually deprecate `RROpikStep` once Logfire is validated

### Files affected

| File | Action |
|---|---|
| `rr/opik.py` (~318 lines) | Deprecate — Logfire auto-instruments RR agent |
| `steps/observability.py` (~38 lines) | Deprecate — Logfire captures pipeline metrics |
| `steps/opik.py` | Deprecate |
| New: setup in runner/config | ~5 lines — `logfire.configure()` + `logfire.instrument_pydantic_ai()` |

### Environment setup

```bash
# .env (gitignored)
LOGFIRE_TOKEN="your-logfire-token"

# Optional: disable in CI/local dev
LOGFIRE_SEND_TO_LOGFIRE=false
```

No changes to `ace.toml` — Logfire config is purely env-based.

---

## Migration Strategy

### Order of operations

1. ~~**Add PydanticAI dependency** to `pyproject.toml`~~ **DONE**
2. ~~**Phase 1**: Implement PydanticAI-backed Agent, Reflector, SkillManager behind existing protocols. Run existing tests to verify.~~ **DONE** — 1098 tests passing
3. ~~**Phase 2**: Implement PydanticAI-based RR agent. Run RR tests + integration tests.~~ **DONE** — RRStep now uses PydanticAI agent with `execute_code`, `analyze`, `batch_analyze` tools. Old inner pipeline (SubRunner, LLMCallStep, ExtractCodeStep, etc.) deprecated. 1093 tests passing.
4. ~~**Phase 3**: Update runners and config. Deprecate old providers.~~ **DONE** — Deprecation warnings added to `LiteLLMClient`, `InstructorClient`, `ClaudeCodeLLMClient`, `LangChainLiteLLMClient`, and `LLMClientLike`.
5. ~~**Phase 4**: Add Logfire observability (auto-instruments all PydanticAI agents).~~ **DONE** — `ace_next.observability.configure_logfire()` auto-instruments all PydanticAI agents. Opt-in via `ACELiteLLM(logfire=True)`.
6. **Cleanup**: Remove deprecated code, update docs.

### Backward compatibility

- All pipeline steps continue to work unchanged (they depend on protocols, not implementations)
- `ACELiteLLM.from_model()` API unchanged
- `ACE.from_roles()` API unchanged
- Skillbook format unchanged
- Step `requires`/`provides` contracts unchanged

### Risk mitigation

- Existing tests must pass at each phase before proceeding
- Old implementations kept behind a flag until PydanticAI equivalents are validated
- RR prompt adaptation validated with existing benchmarks before deleting old code

---

## Dependencies

### New

- `pydantic-ai-slim[litellm]` (v0.0.36+, installed: 1.70.0) — core agent framework with LiteLLM provider
- `logfire[pydantic-ai]` (optional, v3.0.0+) — observability

### Potentially removable

- `instructor` — replaced by PydanticAI's native structured output
- Direct `litellm` usage in providers — PydanticAI wraps LiteLLM internally (note: LiteLLM may still be needed for provider-specific config or features not exposed by PydanticAI)

---

## Success Criteria

- All existing tests pass
- RR benchmark scores equal or better than current implementation
- Net code reduction of ~2,500+ lines
- Single consistent retry/validation behavior across all roles
- No changes to pipeline engine or step contracts
