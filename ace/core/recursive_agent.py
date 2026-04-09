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
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, Type

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from pydantic_ai import ModelRetry, RunContext

from ..providers.pydantic_ai import resolve_model

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Default tools
# ------------------------------------------------------------------


def register_execute_code(agent: PydanticAgent) -> None:
    """Register the generic ``execute_code`` tool.

    Expects ``deps.sandbox`` (a :class:`TraceSandbox` or compatible)
    and ``deps.config.timeout`` / ``deps.config.max_output_chars``.
    """

    @agent.tool(retries=3)
    def execute_code(ctx: RunContext[AgenticDeps], code: str) -> str:
        """Execute Python code in the sandbox.

        Variables persist across calls. Pre-loaded modules:
        ``json``, ``re``, ``collections``, ``datetime``.

        Args:
            code: Python code to execute.

        Returns:
            Captured stdout/stderr from execution.
        """
        ctx.deps.iteration += 1
        sandbox = getattr(ctx.deps, "sandbox", None)
        if sandbox is None:
            return "(no sandbox configured)"

        cfg = ctx.deps.config
        timeout = getattr(cfg, "timeout", 30.0)
        max_output = getattr(cfg, "max_output_chars", 20_000)

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
                f"{output[:max_output]}\n"
                f"[TRUNCATED: {remaining} chars remaining]"
            )

        return output


def register_recurse(agent: PydanticAgent) -> None:
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
        """Spawn a child session to handle a sub-problem.

        The child gets its own sandbox (inheriting data and helpers from
        the parent) and a fraction of the remaining token budget.

        Args:
            prompt: Instructions for the child session.
            context_code: Optional Python code to prepare the child's
                sandbox before the session starts.

        Returns:
            Text summary of the child session's output.
        """
        deps = ctx.deps
        if deps.run_session_fn is None:
            return "(recurse unavailable — no session runner configured)"

        sandbox = getattr(deps, "sandbox", None)
        if sandbox is None:
            return "(recurse unavailable — no sandbox on deps)"

        # Import here to avoid circular imports
        from .sandbox import TraceSandbox

        # Create child sandbox inheriting parent data
        child_sandbox = TraceSandbox(trace=None)
        # Copy all injected variables from parent
        for key in ("traces", "skillbook"):
            if key in sandbox.namespace:
                child_sandbox.inject(key, sandbox.namespace[key])

        # Inherit registered helpers
        parent_registry = sandbox.namespace.get("helper_registry", {})
        if isinstance(parent_registry, dict):
            timeout = getattr(deps.config, "timeout", 30.0)
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
            timeout = getattr(deps.config, "timeout", 30.0)
            result = child_sandbox.execute(context_code, timeout=timeout)
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
                **{f.name: getattr(deps, f.name) for f in deps.__dataclass_fields__.values()},
                "sandbox": child_sandbox,
                "depth": deps.depth + 1,
                "iteration": 0,
                "parent_usage_tokens": 0,
            }
        )

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

    def build_usage_limits(self, remaining_tokens: int | None = None) -> UsageLimits:
        """Build PydanticAI UsageLimits from this config."""
        return UsageLimits(
            total_tokens_limit=remaining_tokens or self.max_tokens,
            request_limit=self.max_requests,
        )


# ------------------------------------------------------------------
# Dependency container
# ------------------------------------------------------------------


@dataclass
class AgenticDeps:
    """Base dependencies for agentic steps.

    Subclass to add step-specific deps (sandbox, trace data, etc.).
    """

    config: AgenticConfig
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


def is_budget_exhausted(limits: UsageLimits, usage: Any) -> bool:
    """True if total token/request budget is spent."""
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
    agent: PydanticAgent,
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
            parts=[TextPart(content=f"[Compaction summary #{compaction_count}]\n{summary}")]
        ),
        ModelRequest(parts=[UserPromptPart(content=continuation_message)]),
    ]


# ------------------------------------------------------------------
# Async runner
# ------------------------------------------------------------------


async def run_agent_with_compaction(
    agent: PydanticAgent,
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

                output = agent_run.result.output
                usage = agent_run.result.usage()

                metadata = {
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "requests": usage.requests,
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

            if is_budget_exhausted(usage_limits, cumulative_usage):
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


def run_agent_sync(agent: PydanticAgent, **kwargs: Any) -> tuple[Any, dict]:
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


ToolRegistrar = Callable[[PydanticAgent], None]


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
        model: str,
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

        # Build root agent (depth=0)
        self._agent = self._create_agent(depth=0)

    def _create_agent(self, depth: int = 0) -> PydanticAgent:
        """Create a PydanticAI agent for the given recursion depth."""
        resolved = resolve_model(self._model)

        agent = PydanticAgent(
            resolved,
            output_type=self._output_type,
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
            usage_limits=self.config.build_usage_limits(remaining_tokens=remaining_tokens),
            config=self.config,
            tool_names_to_compact=self._tool_names_to_compact,
            compaction_summary_prompt=self._compaction_summary_prompt,
            compaction_continuation=self._compaction_continuation,
            microcompact_placeholder=self._microcompact_placeholder,
            on_compaction=self._on_compaction,
        )
