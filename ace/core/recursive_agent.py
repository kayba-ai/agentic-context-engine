"""Recursive agent — reusable agentic step with compaction and recursion.

A ``RecursiveAgent`` wraps a PydanticAI agent with:
- Two-tier compaction (microcompaction + full summarization)
- Depth-based recursion via a ``recurse`` tool
- Budget management (token + request limits)
- Sync/async execution

Both the RR (Recursive Reflector) and the agentic SkillManager build
on this. Callers provide their own tools, output type, and prompts.

Usage::

    from ace.core.recursive_agent import RecursiveAgent, AgenticConfig

    agent = RecursiveAgent(
        model="gpt-4o-mini",
        output_type=MyOutput,
        system_prompt="You are a ...",
        config=AgenticConfig(max_requests=20),
        tools=[my_tool_registrar],        # list of (agent) -> None functions
        tool_names_to_compact=("my_tool",),
    )
    output, metadata = agent.run(prompt="Analyze this", deps=my_deps)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, Type

from pydantic_ai import Agent as PydanticAgent

try:
    import logfire

    _logfire: Any = logfire
except ImportError:
    _logfire = None


def _rr_span(name: str, **attrs: Any):
    """Open a logfire span if logfire is installed, else a no-op context."""
    if _logfire is not None:
        return _logfire.span(name, **attrs)
    return nullcontext()


from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model as PydanticModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

from pydantic_ai import ModelRetry, RunContext

from .metered_model import MeteredModel
from .sandbox import TraceSandbox
from ..providers.pydantic_ai import resolve_model

UsageCallback = Callable[[RequestUsage, str], None]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Default tools
# ------------------------------------------------------------------


def register_execute_code(agent: PydanticAgent[AgenticDeps, Any]) -> None:
    """Register the generic ``execute_code`` tool.

    Expects ``deps.sandbox`` (a :class:`TraceSandbox` or compatible)
    and ``deps.config.timeout`` / ``deps.config.max_output_chars``.
    """

    @agent.tool(retries=3)
    def execute_code(ctx: RunContext[AgenticDeps], code: str) -> str:
        """Execute Python code in the sandbox.

        Variables persist across calls. Pre-loaded modules:
        ``json``, ``re``, ``collections``, ``datetime``.

        Built-in helper: ``register_helper(name, source, description)``
        defines a reusable Python function in this sandbox AND auto-injects
        it into every child you later spawn via ``recurse`` — register
        extraction/scoring logic once, reuse it across children.

        Args:
            code: Python code to execute.

        Returns:
            Captured stdout/stderr from execution.
        """
        ctx.deps.iteration += 1
        if ctx.deps.sandbox is None:
            return "(no sandbox configured)"

        sandbox = ctx.deps.sandbox
        timeout = ctx.deps.config.timeout
        max_output = ctx.deps.config.max_output_chars

        result = sandbox.execute(code, timeout=timeout)

        if result.exception:
            error_msg = f"{type(result.exception).__name__}: {result.exception}"
            stdout_ctx = ""
            if result.stdout:
                stdout_ctx = f"stdout before error:\n{result.stdout[:max_output]}\n\n"
            raise ModelRetry(
                f"{stdout_ctx}Code error:\n{error_msg}\n\nFix the bug and try again."
            )

        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"stderr: {result.stderr}")

        output = "\n".join(parts) if parts else "(no output)"

        if len(output) > max_output:
            remaining = len(output) - max_output
            output = (
                f"{output[:max_output]}\n" f"[TRUNCATED: {remaining} chars remaining]"
            )

        return output


def register_recurse(agent: PydanticAgent[AgenticDeps, Any]) -> None:
    """Register the generic ``recurse`` tool for depth-based decomposition.

    Expects ``deps.run_session_fn`` to be set (done by
    :meth:`RecursiveAgent.run`).
    """

    @agent.tool
    async def recurse(
        ctx: RunContext[AgenticDeps],
        prompt: str,
        context_code: str = "",
    ) -> str:
        """Spawn a child session to investigate a sub-problem in isolation.

        Use this to keep your context lean: the child works through
        bulky data in its own context window and returns only a text
        summary. The child inherits a copy of your sandbox variables
        and any helpers you've registered via `register_helper`. It
        does NOT see your conversation, so `prompt` must be self-contained.
        Calling `recurse` multiple times in a single assistant turn
        dispatches the children in parallel.

        Args:
            prompt: Self-contained instructions. Name the sandbox
                variables to inspect and say what to return.
            context_code: Optional Python run once in the child's
                sandbox before it starts (e.g. ``chunk = traces[5:10]``).

        Returns:
            Text summary of the child's structured output.
        """
        deps = ctx.deps
        if deps.run_session_fn is None:
            return "(recurse unavailable — no session runner configured)"

        if deps.sandbox is None:
            return "(recurse unavailable — no sandbox on deps)"

        sandbox = deps.sandbox

        # Create child sandbox inheriting parent's injected data
        child_sandbox = TraceSandbox(trace=None)
        for key, value in sandbox.namespace.items():
            if not key.startswith("_") and not callable(value):
                child_sandbox.inject(key, value)

        # Inherit registered helpers
        parent_registry = sandbox.namespace.get("helper_registry", {})
        if isinstance(parent_registry, dict):
            timeout = deps.config.timeout
            for hname, meta in parent_registry.items():
                if isinstance(meta, dict) and isinstance(meta.get("source"), str):
                    try:
                        child_sandbox.execute(meta["source"], timeout=timeout)
                        child_registry = child_sandbox.namespace.setdefault(
                            "helper_registry", {}
                        )
                        child_registry[hname] = {
                            "description": meta.get("description", ""),
                            "source": meta["source"],
                        }
                    except Exception:
                        pass

        # Run optional context_code
        if context_code.strip():
            result = child_sandbox.execute(context_code, timeout=deps.config.timeout)
            if result.exception:
                raise ModelRetry(
                    f"context_code failed: {result.exception}\n"
                    "Fix the code and try again."
                )

        # Compute child budget
        cfg = deps.config
        remaining = max(0, cfg.max_tokens - deps.parent_usage_tokens)
        child_token_budget = max(10_000, int(remaining * cfg.child_budget_fraction))

        # Build child deps (same type as parent)
        child_deps = deps.__class__(
            **{
                **{
                    f.name: getattr(deps, f.name)
                    for f in deps.__dataclass_fields__.values()
                },
                "sandbox": child_sandbox,
                "depth": deps.depth + 1,
                "iteration": 0,
                "parent_usage_tokens": 0,
            }
        )

        # Mutable state in dependency dataclasses must not be shared by
        # sibling sessions.  In particular, SkillManager dependencies carry
        # a Skillbook (which owns a thread lock) and an audit list.  Clone the
        # serializable Skillbook representation instead of deepcopying it so
        # the lock is recreated for the child.
        skillbook = getattr(deps, "skillbook", None)
        if skillbook is not None and hasattr(skillbook, "dumps"):
            child_deps.skillbook = skillbook.__class__.loads(skillbook.dumps())
        if hasattr(child_deps, "operations"):
            child_deps.operations = []

        try:
            output, _ = await deps.run_session_fn(
                deps=child_deps,
                prompt=prompt,
                depth=deps.depth + 1,
            )

            # Serialize child output to text
            if hasattr(output, "model_dump"):
                d = output.model_dump(exclude={"raw"}, exclude_defaults=True)
                parts = [f"{k}: {v}" for k, v in d.items() if v]
                return "\n".join(parts) if parts else "(empty output)"
            return str(output) if output else "(empty output)"

        except Exception as e:
            return f"(child session failed: {e})"


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_COMPACTION_SUMMARY_PROMPT = """\
Summarize your progress so far. Structure your response with these sections:

1. **What you've done**: Steps completed, tools used, key decisions made.
2. **Findings so far**: Concrete results, computed values, identified patterns.
3. **Remaining work**: What hasn't been done yet.
4. **Current direction**: What you were investigating when this summary was requested.

Be concise but preserve all concrete results and variable names."""


@dataclass
class AgenticConfig:
    """Base configuration for agentic steps with compaction and recursion.

    Subclass to add step-specific fields (e.g. sandbox timeout).
    """

    # Budget (wired to PydanticAI UsageLimits)
    max_tokens: int = 500_000
    max_requests: int = 50
    context_window: int = 128_000
    # Recursion
    max_depth: int = 2
    child_budget_fraction: float = 0.5
    # Compaction
    max_compactions: int = 3
    microcompact_keep_recent: int = 3
    # Sandbox execution
    timeout: float = 60.0
    max_output_chars: int = 50_000
    # Metering — fired once per completed pydantic-ai model request
    # (orchestrator turn, child session, compaction summary). Exceptions
    # inside the callback are swallowed by MeteredModel so a broken
    # meter never crashes a run.
    usage_callback: UsageCallback | None = None

    def build_usage_limits(self, remaining_tokens: int | None = None) -> UsageLimits:
        base = remaining_tokens or self.max_tokens
        return UsageLimits(
            total_tokens_limit=base,
            request_limit=self.max_requests,
        )


# ------------------------------------------------------------------
# Dependency container
# ------------------------------------------------------------------


@dataclass
class AgenticDeps:
    """Base dependencies for agentic steps.

    Subclass to add step-specific deps (trace data, etc.).
    """

    config: AgenticConfig
    sandbox: Any = None  # TraceSandbox or compatible
    depth: int = 0
    max_depth: int = 2
    iteration: int = 0
    run_session_fn: Callable[..., Awaitable[tuple[Any, Any]]] | None = None
    parent_usage_tokens: int = 0


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class BudgetExhausted(Exception):
    """Raised when the agent's token or request budget is fully spent."""

    def __init__(self, compaction_count: int = 0, usage: Any = None) -> None:
        self.compaction_count = compaction_count
        self.usage = usage
        super().__init__("Agent budget exhausted")


# ------------------------------------------------------------------
# Compaction utilities
# ------------------------------------------------------------------


def cost_equivalent_tokens(usage: Any) -> int:
    """Cost-equivalent token count for budget purposes.

    Anthropic Bedrock pricing (input side):
      - fresh:        1.0x base
      - cache_write:  1.25x base
      - cache_read:   0.10x base

    PydanticAI's Bedrock wrapper sets ``input_tokens = fresh + cache_write +
    cache_read``. We rebuild the cost-equivalent: subtract 0.90x of cache_read
    (since it should weigh 0.10 not 1.0) and add 0.25x of cache_write (since
    it should weigh 1.25 not 1.0). Output is counted at 1.0x.
    """
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_tokens", 0) or 0
    cache_write = getattr(usage, "cache_write_tokens", 0) or 0
    output = getattr(usage, "output_tokens", 0) or 0
    cost_input = input_tokens - 0.90 * cache_read + 0.25 * cache_write
    return int(cost_input + output)


def is_budget_exhausted(
    limits: UsageLimits,
    usage: Any,
    cost_budget: int | None = None,
) -> bool:
    """True if cost-equivalent or request budget is spent.

    When ``cost_budget`` is provided, the *cost-equivalent* token count
    (``cost_equivalent_tokens``) is checked against it. The gross
    ``total_tokens_limit`` on ``limits`` is treated as a coarse outer cap and
    is normally inflated relative to the real budget, so it should rarely
    trip first when caching is in play.
    """
    if cost_budget is not None and cost_equivalent_tokens(usage) >= cost_budget:
        return True
    if limits.total_tokens_limit and usage.total_tokens >= limits.total_tokens_limit:
        return True
    if limits.request_limit and usage.requests >= limits.request_limit:
        return True
    return False


def microcompact(
    messages: list,
    keep_recent: int,
    tool_names: tuple[str, ...],
    placeholder: str = "[cleared — use tools to re-inspect if needed]",
) -> list:
    """Tier 1: Clear old tool results from message history.

    Returns the **same list object** if nothing was cleared — caller
    uses identity check to detect whether compaction did anything.
    """
    tool_result_positions = []
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            for part_idx, part in enumerate(msg.parts):
                if isinstance(part, ToolReturnPart) and part.tool_name in tool_names:
                    tool_result_positions.append((msg_idx, part_idx))

    if len(tool_result_positions) <= keep_recent:
        return messages

    to_clear = (
        tool_result_positions[:-keep_recent]
        if keep_recent > 0
        else tool_result_positions
    )

    compacted = copy.deepcopy(messages)
    for msg_idx, part_idx in to_clear:
        compacted[msg_idx].parts[part_idx].content = placeholder
    return compacted


async def summarize_and_compact(
    agent: PydanticAgent[AgenticDeps, Any],
    messages: list,
    deps: Any,
    compaction_count: int,
    summary_prompt: str = DEFAULT_COMPACTION_SUMMARY_PROMPT,
    continuation_message: str = "",
) -> list:
    """Tier 2: Full summarization — LLM summarizes, history pruned."""
    summary_result = await agent.run(
        summary_prompt,
        message_history=messages,
        deps=deps,
        output_type=str,
    )
    summary = summary_result.output

    if not continuation_message:
        continuation_message = (
            f"Your conversation was compacted ({compaction_count} time(s)). "
            "Do NOT repeat work already completed. Continue."
        )

    return [
        ModelResponse(
            parts=[
                TextPart(content=f"[Compaction summary #{compaction_count}]\n{summary}")
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content=continuation_message)]),
    ]


# ------------------------------------------------------------------
# Async runner
# ------------------------------------------------------------------


async def run_agent_with_compaction(
    agent: PydanticAgent[AgenticDeps, Any],
    *,
    deps: AgenticDeps,
    prompt: str,
    usage_limits: UsageLimits,
    config: AgenticConfig,
    tool_names_to_compact: tuple[str, ...] = (),
    compaction_summary_prompt: str = DEFAULT_COMPACTION_SUMMARY_PROMPT,
    compaction_continuation: str = "",
    microcompact_placeholder: str = "[cleared — use tools to re-inspect if needed]",
    on_compaction: Callable[[AgenticDeps, int, list], None] | None = None,
    span_label: str = "rr",
) -> tuple[Any, dict]:
    """Run a PydanticAI agent with two-tier compaction.

    Returns ``(output, metadata)``.
    Raises :class:`BudgetExhausted` when budget is fully spent.
    """
    message_history = None
    compaction_count = 0
    user_prompt = prompt
    cumulative_usage = None
    last_run: Any = None

    span_name = (
        f"{span_label}.session" if deps.depth == 0 else f"{span_label}.session.child"
    )
    with _rr_span(span_name, depth=deps.depth):
        while True:
            try:
                async with agent.iter(
                    user_prompt,
                    deps=deps,
                    message_history=message_history,
                    usage_limits=usage_limits,
                    usage=cumulative_usage,
                ) as agent_run:
                    last_run = agent_run
                    async for _node in agent_run:
                        deps.parent_usage_tokens = agent_run.usage().total_tokens or 0

                    assert agent_run.result is not None
                    output = agent_run.result.output
                    usage = agent_run.result.usage()

                    metadata = {
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": usage.total_tokens,
                            "requests": usage.requests,
                            "cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
                            "cache_write_tokens": getattr(
                                usage, "cache_write_tokens", 0
                            ),
                        },
                        "compactions": compaction_count,
                        "depth": deps.depth,
                        "iterations": deps.iteration,
                        "timed_out": False,
                    }
                    return output, metadata

            except UsageLimitExceeded:
                messages = last_run.all_messages()
                cumulative_usage = last_run.usage()

                if is_budget_exhausted(
                    usage_limits,
                    cumulative_usage,
                    cost_budget=config.max_tokens,
                ):
                    raise BudgetExhausted(
                        compaction_count=compaction_count,
                        usage=cumulative_usage,
                    )

                compacted = microcompact(
                    messages,
                    config.microcompact_keep_recent,
                    tool_names_to_compact,
                    placeholder=microcompact_placeholder,
                )

                if compacted is messages:
                    compaction_count += 1
                    if compaction_count > config.max_compactions:
                        raise BudgetExhausted(
                            compaction_count=compaction_count,
                            usage=cumulative_usage,
                        )

                    if on_compaction:
                        on_compaction(deps, compaction_count, messages)

                    compacted = await summarize_and_compact(
                        agent,
                        messages,
                        deps,
                        compaction_count,
                        summary_prompt=compaction_summary_prompt,
                        continuation_message=compaction_continuation,
                    )

                message_history = compacted
                user_prompt = "Continue your analysis."


# ------------------------------------------------------------------
# Sync wrapper
# ------------------------------------------------------------------


def run_agent_sync(
    agent: PydanticAgent[AgenticDeps, Any], **kwargs: Any
) -> tuple[Any, dict]:
    """Synchronous wrapper around :func:`run_agent_with_compaction`."""
    coro = run_agent_with_compaction(agent, **kwargs)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ------------------------------------------------------------------
# RecursiveAgent — high-level API
# ------------------------------------------------------------------


ToolRegistrar = Callable[..., None]


class RecursiveAgent:
    """A PydanticAI agent with compaction, recursion, and budget management.

    This is the high-level API. Callers provide:
    - ``output_type``: The structured output schema
    - ``tools``: List of tool registrar functions ``(agent) -> None``
    - ``system_prompt``: The system prompt
    - ``config``: Budget, compaction, and recursion settings

    The agent handles compaction and child session spawning automatically.

    Example::

        agent = RecursiveAgent(
            model="gpt-4o-mini",
            output_type=ReflectorOutput,
            system_prompt="You are a trace analyst...",
            tools=[register_execute_code, register_analysis_tools],
            tool_names_to_compact=("execute_code", "analyze"),
        )
        output, metadata = agent.run(prompt="Analyze...", deps=my_deps)
    """

    def __init__(
        self,
        model: str | PydanticModel,
        *,
        output_type: Type,
        system_prompt: str,
        config: AgenticConfig | None = None,
        model_settings: ModelSettings | None = None,
        tools: Sequence[ToolRegistrar] = (),
        tool_names_to_compact: tuple[str, ...] = (),
        compaction_summary_prompt: str = DEFAULT_COMPACTION_SUMMARY_PROMPT,
        compaction_continuation: str = "",
        microcompact_placeholder: str = "[cleared — use tools to re-inspect if needed]",
        on_compaction: Callable[[AgenticDeps, int, list], None] | None = None,
        span_label: str = "rr",
    ) -> None:
        self.config = config or AgenticConfig()
        self._model = model
        self._model_settings = model_settings
        self._output_type = output_type
        self._system_prompt = system_prompt
        self._tools = list(tools)
        self._tool_names_to_compact = tool_names_to_compact
        self._compaction_summary_prompt = compaction_summary_prompt
        self._compaction_continuation = compaction_continuation
        self._microcompact_placeholder = microcompact_placeholder
        self._on_compaction = on_compaction
        self._span_label = span_label

        # Build root agent (depth=0)
        self._agent = self._create_agent(depth=0)

    def _create_agent(self, depth: int = 0) -> PydanticAgent[AgenticDeps, Any]:
        """Create a PydanticAI agent for the given recursion depth.

        The root (depth 0) uses the configured ``output_type`` (typically
        a structured Pydantic model). Children return free-form text:
        they exist to investigate one sub-problem and report a focused
        answer, not to produce a full reflection.
        """
        if isinstance(self._model, PydanticModel):
            resolved = self._model
        else:
            resolved = resolve_model(self._model)

        if self.config.usage_callback is not None:
            resolved = MeteredModel(resolved, self.config.usage_callback)

        output_type = self._output_type if depth == 0 else str

        agent: PydanticAgent[AgenticDeps, Any] = PydanticAgent(
            resolved,
            output_type=output_type,
            system_prompt=self._system_prompt,
            retries=3,
            model_settings=self._model_settings,
            defer_model_check=True,
        )

        # Default tools: execute_code + recurse (if not at max depth)
        register_execute_code(agent)
        if depth < self.config.max_depth:
            register_recurse(agent)

        # Additional caller-provided tools
        for registrar in self._tools:
            registrar(agent)

        return agent

    async def _run_child_session(
        self,
        *,
        deps: AgenticDeps,
        prompt: str,
        depth: int = 0,
    ) -> tuple[Any, AgenticDeps]:
        """Run a child session with its own agent and budget."""
        child_agent = self._create_agent(depth=depth)

        remaining = getattr(deps, "_remaining_tokens", None)
        try:
            output, metadata = await run_agent_with_compaction(
                child_agent,
                deps=deps,
                prompt=prompt,
                usage_limits=self.config.build_usage_limits(remaining_tokens=remaining),
                config=self.config,
                tool_names_to_compact=self._tool_names_to_compact,
                compaction_summary_prompt=self._compaction_summary_prompt,
                compaction_continuation=self._compaction_continuation,
                microcompact_placeholder=self._microcompact_placeholder,
                on_compaction=self._on_compaction,
                span_label=self._span_label,
            )
            return output, deps
        except BudgetExhausted:
            return None, deps

    def run(
        self,
        *,
        deps: AgenticDeps,
        prompt: str,
        remaining_tokens: int | None = None,
    ) -> tuple[Any, dict]:
        """Run the agent synchronously with compaction.

        Args:
            deps: Agent dependencies.
            prompt: Initial prompt.
            remaining_tokens: Override token budget (for child sessions).

        Returns:
            Tuple of (output, metadata_dict).

        Raises:
            BudgetExhausted: When budget is fully spent.
        """
        # Wire up child session runner
        deps.run_session_fn = self._run_child_session

        return run_agent_sync(
            self._agent,
            deps=deps,
            prompt=prompt,
            usage_limits=self.config.build_usage_limits(
                remaining_tokens=remaining_tokens
            ),
            config=self.config,
            tool_names_to_compact=self._tool_names_to_compact,
            compaction_summary_prompt=self._compaction_summary_prompt,
            compaction_continuation=self._compaction_continuation,
            microcompact_placeholder=self._microcompact_placeholder,
            on_compaction=self._on_compaction,
            span_label=self._span_label,
        )

    # ------------------------------------------------------------------
    # Sandbox helpers
    # ------------------------------------------------------------------

    def create_sandbox(
        self,
        *,
        trace: Any = None,
        variables: dict[str, Any] | None = None,
    ) -> TraceSandbox:
        """Create a sandbox and inject variables.

        Args:
            trace: Optional trace object passed to TraceSandbox constructor.
            variables: Dict of ``{name: value}`` to inject into the sandbox
                namespace.

        Returns:
            A ready-to-use :class:`TraceSandbox`.
        """
        sandbox = TraceSandbox(trace=trace, llm_query_fn=None)
        if variables:
            for name, value in variables.items():
                sandbox.inject(name, value)
        return sandbox

    @staticmethod
    def on_compaction(deps: AgenticDeps, compaction_count: int, messages: list) -> None:
        """Default compaction callback — save metadata to sandbox history.

        Subclasses can override or pass a different callback via
        ``on_compaction`` in ``__init__``.
        """
        sandbox = getattr(deps, "sandbox", None)
        if sandbox is not None:
            history = sandbox.namespace.get("history", [])
            history.append(
                {
                    "compaction_round": compaction_count,
                    "message_count": len(messages),
                }
            )
            sandbox.namespace["history"] = history
