"""RR agent tool registrars.

Each ``_register_*`` function attaches one or more PydanticAI tools to an
agent instance.  Keeping tool definitions separate from agent factories
makes both easier to read and test independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic_ai import ModelRetry, RunContext

from ace.core.outputs import ReflectorOutput

from .config import RecursiveConfig
from .sandbox import TraceSandbox

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dependency containers
# ------------------------------------------------------------------


@dataclass
class RRDeps:
    """Dependencies injected into RR tool calls via ``RunContext``."""

    sandbox: TraceSandbox
    trace_data: dict[str, Any]
    skillbook_text: str
    config: RecursiveConfig
    iteration: int = 0

    # Recursion state
    depth: int = 0
    max_depth: int = 2
    run_session_fn: Callable[..., Awaitable[tuple[Any, "RRDeps"]]] | None = None
    # Updated by _run_with_compaction so recurse tool can compute child budget
    parent_usage_tokens: int = 0


# ------------------------------------------------------------------
# Tool registrars
# ------------------------------------------------------------------


def register_execute_code(agent: PydanticAgent[RRDeps, Any]) -> None:
    """Register the ``execute_code`` tool on any RR agent."""

    @agent.tool(retries=3)
    def execute_code(ctx: RunContext[RRDeps], code: str) -> str:
        """Execute Python code in the analysis sandbox.

        Use for data preparation: building task lists, formatting batch
        keys, computing summaries.  Variables persist across calls.
        Pre-loaded: ``traces``, ``skillbook``, ``json``, ``re``,
        ``collections``, ``datetime``.

        Args:
            code: Python code to execute.

        Returns:
            Captured stdout/stderr from execution.
        """
        ctx.deps.iteration += 1
        max_output = ctx.deps.config.max_output_chars

        result = ctx.deps.sandbox.execute(code, timeout=ctx.deps.config.timeout)

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


def register_output_validator(agent: PydanticAgent[RRDeps, Any]) -> None:
    """Register the standard output validator on any RR agent."""

    @agent.output_validator
    def validate_output(
        ctx: RunContext[RRDeps], output: ReflectorOutput
    ) -> ReflectorOutput:
        """Ensure the agent explored data before concluding."""
        if ctx.deps.iteration < 1:
            raise ModelRetry(
                "You haven't explored the data enough. "
                "Use execute_code first, "
                "then provide your final output."
            )
        return output


def register_recurse_tool(
    agent: PydanticAgent[RRDeps, Any],
    *,
    create_agent_fn: Callable,
) -> None:
    """Register the ``recurse`` tool for depth-based recursive decomposition."""

    @agent.tool
    async def recurse(
        ctx: RunContext[RRDeps],
        prompt: str,
        context_code: str = "",
    ) -> str:
        """Spawn a child RR session to handle a sub-problem.

        The child gets its own sandbox with the same trace data and
        inherited helpers. Use ``context_code`` to prepare/filter data
        in the child sandbox before the session starts.

        Args:
            prompt: Instructions for the child session — what to analyze and what output to produce.
            context_code: Optional Python code executed in the child sandbox before the session starts.
                         Use this to slice data, set variables, or prepare focused subsets.

        Returns:
            Text summary of the child session's analysis output.
        """
        if ctx.deps.run_session_fn is None:
            return "(recurse unavailable — no session runner configured)"

        # Create child sandbox inheriting parent's trace data and helpers
        child_sandbox = TraceSandbox(
            trace=None,
        )
        child_sandbox.inject("traces", ctx.deps.trace_data)
        child_sandbox.inject("skillbook", ctx.deps.skillbook_text)

        # Inherit registered helpers from parent
        parent_registry = ctx.deps.sandbox.namespace.get("helper_registry", {})
        if isinstance(parent_registry, dict):
            for hname, meta in parent_registry.items():
                if isinstance(meta, dict) and isinstance(meta.get("source"), str):
                    try:
                        child_sandbox.execute(meta["source"], timeout=ctx.deps.config.timeout)
                        child_registry = child_sandbox.namespace.setdefault("helper_registry", {})
                        child_registry[hname] = {
                            "description": meta.get("description", ""),
                            "source": meta["source"],
                        }
                    except Exception:
                        pass  # skip broken helpers

        # Run optional context_code to prepare child data
        if context_code.strip():
            result = child_sandbox.execute(context_code, timeout=ctx.deps.config.timeout)
            if result.exception:
                raise ModelRetry(
                    f"context_code failed: {result.exception}\n"
                    "Fix the code and try again."
                )

        child_depth = ctx.deps.depth + 1

        # Compute child token budget from remaining parent budget
        cfg = ctx.deps.config
        remaining = max(0, cfg.max_tokens - ctx.deps.parent_usage_tokens)
        child_token_budget = max(
            10_000,  # minimum viable budget
            int(remaining * cfg.child_budget_fraction),
        )

        child_deps = RRDeps(
            sandbox=child_sandbox,
            trace_data={**ctx.deps.trace_data, "_remaining_tokens": child_token_budget},
            skillbook_text=ctx.deps.skillbook_text,
            config=ctx.deps.config,
            depth=child_depth,
            max_depth=ctx.deps.max_depth,
            run_session_fn=ctx.deps.run_session_fn,
        )

        try:
            output, _ = await ctx.deps.run_session_fn(
                deps=child_deps,
                prompt=prompt,
                depth=child_depth,
            )

            # Serialize child output as text for parent
            parts = []
            if output.reasoning:
                parts.append(f"Reasoning: {output.reasoning}")
            if output.key_insight:
                parts.append(f"Key insight: {output.key_insight}")
            if output.root_cause_analysis:
                parts.append(f"Root cause: {output.root_cause_analysis}")
            if output.correct_approach:
                parts.append(f"Correct approach: {output.correct_approach}")
            if output.raw.get("items"):
                parts.append(f"Per-item results: {len(output.raw['items'])} items analyzed")
            return "\n".join(parts) if parts else "(child session produced empty output)"

        except Exception as e:
            return f"(child session failed: {e})"
