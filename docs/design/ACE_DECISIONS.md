# Design Decisions

> What was considered and rejected for ACE and the PydanticAI migration — and why.

For architecture and concepts, see [ACE_ARCHITECTURE.md](ACE_ARCHITECTURE.md).
For code reference and examples, see [ACE_REFERENCE.md](ACE_REFERENCE.md).

---

## PydanticAI Migration

### What we replaced and why

ACE had three hand-rolled LLM client implementations (LiteLLMClient, InstructorClient, ClaudeCodeLLMClient) with inconsistent retry/validation behavior, ~3,500 lines of custom agent-loop plumbing in the Recursive Reflector, and manual code extraction via regex. PydanticAI handles all of this as maintained infrastructure.

| Before | After |
|---|---|
| 3 LLM client implementations | PydanticAI agents inside roles |
| Manual JSON extraction + Pydantic parse | PydanticAI validates via tool-call schema, retries with error feedback |
| 3 blind retries or Instructor | PydanticAI native (configurable, with error context) |
| LiteLLM wrapper for provider support | PydanticAI (native support for 15+ providers, wraps LiteLLM internally) |
| Custom SubRunner loop (~400 lines) | PydanticAI's agent loop — LLM calls tools until it produces structured output |
| 200 lines of regex code extraction | Tool args are pre-parsed (code arrives as `execute_code` parameter) |
| Inner pipeline steps (~500 lines) | ~50 lines of tool definitions |
| CallBudget + SubAgentLLM (~200 lines) | `ctx.usage` shared budget + delegate agent |
| Custom `RRIterationContext` | PydanticAI manages message state internally |
| Manual Opik span building (~356 lines) | `logfire.instrument_pydantic_ai()` auto-instruments everything |

**Net result for RR:** ~3,500 lines → ~1,000 lines (sandbox + prompts + trimming + agent definition). ~2,500 lines of loop/extraction/budget/context plumbing deleted.

### What we kept

- **Pipeline engine** (`pipeline/`) — `requires`/`provides` contracts, `async_boundary`, per-step `max_workers`, `SampleResult` error isolation. No framework offers this combination.
- **Skillbook & learning loop** — Reflect → Update → Apply → Deduplicate. This is core IP.
- **Step composition** — `learning_tail()`, pipeline-as-step nesting, `SkillbookView` read/write split.
- **Domain-specific prompts** — tightly coupled to skillbook format and ACE's reflection strategy.
- **All pipeline steps** — they depend on protocols, not implementations. Completely unchanged.

### Provider resolution design

The resolver routes LiteLLM model strings to PydanticAI through three paths:

1. **PydanticAI-native prefix** — pass through unchanged
2. **LiteLLM prefix → native provider** — rewrite `/` to `:` when the prefix matches a native provider. This is necessary because PydanticAI's `litellm` provider uses an OpenAI-compatible HTTP client under the hood, which doesn't work for providers with non-OpenAI APIs (Bedrock via SigV4, Anthropic's native API, etc.).
3. **Fallback** — prefix with `litellm:` for the proxy provider

User-facing API is unchanged — same LiteLLM model strings as before.

---

## ACE Architecture Decisions

**Runner extends Pipeline:**
Making TraceAnalyser and ACE subclasses of `Pipeline` was considered. Rejected — the runner is not a pipeline. It owns the epoch loop. Composition (`self.pipeline`) keeps responsibilities separate.

**Cross-sample state (reflection window):**
A rolling window of recent reflections that persists across samples was considered, with variants: on the runner, on `StepContext`, on step instances, via a shared mediator object. All rejected — each sample should be independent. The only cross-sample coupling is the skillbook itself. Adding a reflection window complicates the model (reset between epochs, eventual consistency with background steps, ordering issues with concurrent workers) for marginal benefit.

**Separate Online and Offline classes:**
Keeping two runner classes for single-pass and multi-epoch was considered. Rejected — the only difference is `epochs=1` vs `epochs > 1`, which is a parameter, not a class distinction. ACE handles both. TraceAnalyser is a separate class because its input type is fundamentally different (raw traces vs `Sample + Environment`).

**Structured Trace dataclass:**
A `@dataclass Trace` with typed fields (`task`, `output`, `feedback`, `reasoning`, etc.) was considered. Rejected — it imposes a schema on trace data that doesn't match reality. External frameworks produce wildly different trace shapes (browser-use `AgentHistoryList`, LangChain result dicts, Claude Code transcripts). Forcing them through a common dataclass means either losing information or adding catch-all `metadata` buckets that defeat the purpose of typing. Instead, `ctx.trace` is `object | None` and the Reflector makes sense of whatever it receives.

**Steps that accept both traces and samples:**
Making ReflectStep and UpdateStep polymorphic over input type was considered. Rejected — steps always receive `StepContext` with the same named fields. The runner (`_build_context`) is responsible for building the context correctly.

**Observability in the runner:**
Keeping observability logic in `ACERunner._track_observability_data()` was considered. Rejected — it mixes concerns. Observability is handled by Logfire auto-instrumentation.

**Custom AsyncLearningPipeline:**
The legacy `ace/async_learning.py` implements a manual thread pool with reflector and skill manager queues. Rejected — the pipeline engine's `async_boundary` and `max_workers` provide the same functionality with less code and consistent semantics.

**Per-integration pipeline classes:**
Having each integration define its own pipeline class was considered. Rejected — every integration pipeline has the same learning tail; only the execute step differs. Instead, integrations provide execute steps that compose into an `ACERunner` subclass, reusing the shared `_run()` loop.

**Checkpoints in the runner:**
Having the runner own checkpoint logic (via `run()` parameters) was considered. Rejected — a `CheckpointStep` at the end of the pipeline tail keeps checkpointing within the pipeline formalism. Configuration belongs at construction time (factory methods), not at call time (`run()`).

**Mutable Skillbook directly on the context:**
Storing the real `Skillbook` as a field on `ACEStepContext` was the initial design. Rejected — `StepContext` is frozen, but `Skillbook` is mutable. Placing it on the context creates the illusion of immutability while allowing any step to mutate shared state through the reference. Instead, the context carries a `SkillbookView` (read-only projection). Write steps receive the real `Skillbook` via constructor injection.

**Separate TagStep (removed):**
A separate TagStep for incrementing helpful/harmful counters on skills was planned but never implemented. Tagging is now the Reflector's responsibility — when running in online mode, the Reflector evaluates which cited skills helped or harmed and populates `skill_tags` on `ReflectorOutput`. The SkillManager receives these tags and can UPDATE or REMOVE skills accordingly. This avoids a redundant pipeline step and keeps tagging close to the analysis that produces it.

**Instructor auto-wrapping in implementations:**
The old `ace/roles.py` auto-wrapped LLM clients with Instructor if `complete_structured` was missing. Rejected — PydanticAI handles structured output natively via its `result_type` parameter.

**Recursive Reflector (initial rejection, now implemented):**
The old `ace/reflector/` subsystem supports recursive mode. Initially rejected for `ace` due to complexity. Now implemented as `RRStep` — a PydanticAI agent-based step that runs an iterative REPL loop. Satisfies both `StepProtocol[ACEStepContext]` and `ReflectorLike`.

**Observability decorator on implementations:**
The old `ace/roles.py` uses `@maybe_track()` decorators for Opik tracing on every role method. Rejected — Logfire auto-instrumentation handles observability. Per-method decorators would double-count and create coupling.

**Deduplication inside SkillManager:**
The old `ace/roles.py` SkillManager integrates with `DeduplicationManager` directly. Rejected — deduplication is now a separate `DeduplicateStep` in the pipeline. Cleaner separation: the SkillManager only produces output, deduplication runs at a configurable interval.

**Shared `ace/features.py` module:**
A centralized feature detection module was considered. Rejected — the only code that needs it is `deduplication/detector.py`, which uses a local `_has(module)` helper. A shared module would add a file for a single 4-line function.

**Separate wrapper classes for integration runners:**
Separate convenience classes (`ACEAgent`, `ACELangChain`, `ACEClaudeCode`) wrapping the runners were the initial design. Rejected — the wrappers only added `from_model()` and a few lifecycle helpers, which fit naturally on the runner class itself. Two classes for one concept forces users to choose between them. Exception: `ACELiteLLM`, which wraps two runners and exposes a fundamentally different API.

