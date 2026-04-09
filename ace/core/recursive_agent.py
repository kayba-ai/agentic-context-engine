"""Reusable infrastructure for agentic (tool-calling, iterative) steps.

Provides base config, deps, and a compaction-aware runner that any
PydanticAI agent can use. The RR and agentic SkillManager both build
on these primitives.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits

logger = logging.getLogger(__name__)

T = TypeVar("T")

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

    Subclass this to add step-specific fields (e.g. sandbox timeout,
    output char limits).
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
    The compaction runner only accesses fields defined here.
    """

    config: AgenticConfig
    depth: int = 0
    max_depth: int = 2
    iteration: int = 0
    run_session_fn: Callable[..., Awaitable[tuple[Any, Any]]] | None = None
    parent_usage_tokens: int = 0


# ------------------------------------------------------------------
# Compaction utilities
# ------------------------------------------------------------------


def is_budget_exhausted(limits: UsageLimits, usage: Any) -> bool:
    """True if total token/request budget is spent.

    Distinguishes budget exhaustion (stop) from context-window overflow
    (compact and retry).
    """
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

    Walks the message list and replaces tool result content for the
    specified tool names with a placeholder. Keeps the most recent
    ``keep_recent`` tool results intact.

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
        return messages  # identity signals "no change"

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
    """Tier 2: Full summarization — LLM summarizes progress, history pruned.

    Asks the agent to summarize the conversation so far, then replaces
    the full message history with [summary + continuation prompt].
    """
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

    compacted = [
        ModelResponse(
            parts=[TextPart(content=f"[Compaction summary #{compaction_count}]\n{summary}")]
        ),
        ModelRequest(parts=[UserPromptPart(content=continuation_message)]),
    ]
    return compacted


# ------------------------------------------------------------------
# Agent runner with compaction
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

    Generic compaction loop that works with any PydanticAI agent and
    output type. Returns ``(output, metadata)`` where metadata contains
    usage and compaction info.

    Args:
        agent: PydanticAI agent to run.
        deps: Agent dependencies (must be AgenticDeps or subclass).
        prompt: Initial user prompt.
        usage_limits: PydanticAI usage limits.
        config: Agentic config for compaction thresholds.
        tool_names_to_compact: Tool names whose results can be cleared
            during microcompaction.
        compaction_summary_prompt: Prompt for tier-2 full summarization.
        compaction_continuation: Message appended after compaction.
        microcompact_placeholder: Replacement text for cleared tool results.
        on_compaction: Optional callback ``(deps, count, messages)`` called
            before tier-2 summarization (e.g. to save history to sandbox).

    Returns:
        Tuple of (agent_output, metadata_dict).

    Raises:
        BudgetExhausted: When token or request budget is fully spent.
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

                # Agent finished successfully
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

            # Tier 1: microcompaction
            compacted = microcompact(
                messages,
                config.microcompact_keep_recent,
                tool_names_to_compact,
                placeholder=microcompact_placeholder,
            )

            if compacted is messages:
                # Tier 2: full summarization
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


class BudgetExhausted(Exception):
    """Raised when the agent's token or request budget is fully spent."""

    def __init__(self, compaction_count: int = 0, usage: Any = None) -> None:
        self.compaction_count = compaction_count
        self.usage = usage
        super().__init__("Agent budget exhausted")


# ------------------------------------------------------------------
# Sync wrapper
# ------------------------------------------------------------------


def run_agent_sync(
    agent: PydanticAgent,
    **kwargs: Any,
) -> tuple[Any, dict]:
    """Synchronous wrapper around :func:`run_agent_with_compaction`.

    Detects whether an event loop is already running and handles
    accordingly.
    """
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
