"""PydanticAI-based Recursive Reflector agent.

Replaces the SubRunner REPL loop with a PydanticAI agent that has three
tools:

- ``execute_code`` — run Python in the analysis sandbox
- ``analyze`` — ask a sub-agent for targeted analysis
- ``batch_analyze`` — parallel sub-agent analysis of multiple items

The agent produces ``ReflectorOutput`` as structured output when it has
gathered enough evidence.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext
from pydantic_ai.settings import ModelSettings

from ace_next.core.outputs import ReflectorOutput
from ace_next.providers.pydantic_ai import resolve_model

from .config import RecursiveConfig
from .sandbox import TraceSandbox
from .subagent import SUBAGENT_ANALYSIS_PROMPT, SUBAGENT_DEEPDIVE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RRDeps:
    """Dependencies injected into RR tool calls via ``RunContext``."""

    sandbox: TraceSandbox
    trace_data: dict[str, Any]
    skillbook_text: str
    config: RecursiveConfig
    iteration: int = 0
    sub_agent: PydanticAgent[None, str] | None = None
    sub_agent_history: list[dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------
# Agent + tool definitions
# ------------------------------------------------------------------

def create_rr_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create the PydanticAI agent for recursive reflection.

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the reflector.
        config: RR configuration (timeouts, limits).
        model_settings: PydanticAI model settings.

    Returns:
        Configured PydanticAI agent with tools.
    """
    cfg = config or RecursiveConfig()
    resolved = resolve_model(model)

    agent: PydanticAgent[RRDeps, ReflectorOutput] = PydanticAgent(
        resolved,
        output_type=ReflectorOutput,
        system_prompt=system_prompt or (
            "You are a trace analyst with tools. "
            "Analyze agent execution traces and extract learnings. "
            "Use execute_code to explore data, analyze for LLM reasoning, "
            "then produce your final structured output."
        ),
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    # -- Tool: execute_code ------------------------------------------

    @agent.tool(retries=3)
    def execute_code(ctx: RunContext[RRDeps], code: str) -> str:
        """Execute Python code in the analysis sandbox.

        Variables persist across calls.  Pre-loaded in namespace:
        ``traces``, ``skillbook``, ``json``, ``re``, ``collections``,
        ``datetime``.

        Args:
            code: Python code to execute.

        Returns:
            Captured stdout/stderr from execution.
        """
        ctx.deps.iteration += 1
        max_output = ctx.deps.config.max_output_chars

        result = ctx.deps.sandbox.execute(
            code, timeout=ctx.deps.config.timeout
        )

        if result.exception:
            error_msg = f"{type(result.exception).__name__}: {result.exception}"
            stdout_ctx = ""
            if result.stdout:
                stdout_ctx = f"stdout before error:\n{result.stdout[:max_output]}\n\n"
            raise ModelRetry(
                f"{stdout_ctx}Code error:\n{error_msg}\n\n"
                "Fix the bug and try again."
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

    # -- Tool: analyze -----------------------------------------------

    @agent.tool
    async def analyze(
        ctx: RunContext[RRDeps],
        question: str,
        context: str,
        mode: str = "analysis",
    ) -> str:
        """Ask a sub-agent to analyze trace data.

        Use for deep analysis of specific findings.  The sub-agent can
        handle large context.

        Args:
            question: What to analyze.
            context: Data to analyze (trace excerpt, code output, etc.).
            mode: ``"analysis"`` for survey, ``"deep_dive"`` for investigation.

        Returns:
            The sub-agent's analysis.
        """
        if ctx.deps.sub_agent is None:
            return "(analyze unavailable — sub-agent not configured)"

        # Select mode-specific system prompt
        sys_prompt = (
            SUBAGENT_DEEPDIVE_PROMPT if mode == "deep_dive"
            else SUBAGENT_ANALYSIS_PROMPT
        )

        prompt = (
            f"{sys_prompt}\n\n"
            f"## Question\n{question}\n\n"
            f"## Context\n{context}\n\n"
            f"## Your Analysis"
        )

        try:
            result = await ctx.deps.sub_agent.run(prompt)
            response = result.output

            ctx.deps.sub_agent_history.append({
                "question": question,
                "context_length": len(context),
                "response_length": len(response),
                "mode": mode,
            })

            return response
        except Exception as e:
            return f"(Sub-agent error: {e})"

    # -- Tool: batch_analyze -----------------------------------------

    @agent.tool
    def batch_analyze(
        ctx: RunContext[RRDeps],
        question: str,
        items: list[str],
        mode: str = "analysis",
    ) -> list[str]:
        """Analyze multiple items in parallel using the sub-agent.

        Each item is analyzed independently with the same question.
        Use for survey batches and independent deep-dives.

        Args:
            question: What to analyze about each item.
            items: List of data items to analyze.
            mode: ``"analysis"`` for survey, ``"deep_dive"`` for investigation.

        Returns:
            Ordered list of analysis results.
        """
        if ctx.deps.sub_agent is None:
            return [
                "(batch_analyze unavailable — sub-agent not configured)"
            ] * len(items)

        if not items:
            return []

        sys_prompt = (
            SUBAGENT_DEEPDIVE_PROMPT if mode == "deep_dive"
            else SUBAGENT_ANALYSIS_PROMPT
        )

        def _analyze_one(item: str) -> str:
            prompt = (
                f"{sys_prompt}\n\n"
                f"## Question\n{question}\n\n"
                f"## Item\n{item}\n\n"
                f"## Your Analysis"
            )
            try:
                result = ctx.deps.sub_agent.run_sync(prompt)
                return result.output
            except Exception as e:
                return f"(Error: {e})"

        pool_size = min(len(items), 10)
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            results = list(pool.map(_analyze_one, items))

        ctx.deps.sub_agent_history.append({
            "question": question,
            "items_count": len(items),
            "mode": mode,
            "batch": True,
        })

        return results

    # -- Output validator --------------------------------------------

    @agent.output_validator
    def validate_output(
        ctx: RunContext[RRDeps], output: ReflectorOutput
    ) -> ReflectorOutput:
        """Ensure the agent explored data before concluding."""
        if ctx.deps.iteration < 2:
            raise ModelRetry(
                "You haven't explored the data enough. "
                "Use execute_code to analyze the traces first, "
                "then provide your final output."
            )
        return output

    return agent


# ------------------------------------------------------------------
# Sub-agent factory
# ------------------------------------------------------------------

def create_sub_agent(
    model: str,
    *,
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
) -> PydanticAgent[None, str]:
    """Create the sub-agent for ``analyze`` / ``batch_analyze`` tools.

    The sub-agent is a simple prompt-in / text-out agent with no tools.

    Args:
        model: LiteLLM or PydanticAI model string.
        config: RR configuration for sub-agent settings.
        model_settings: Override model settings.

    Returns:
        PydanticAI agent producing plain text output.
    """
    cfg = config or RecursiveConfig()
    resolved = resolve_model(model)

    settings = model_settings or ModelSettings(
        temperature=cfg.subagent_temperature,
        max_tokens=cfg.subagent_max_tokens,
    )

    return PydanticAgent(
        resolved,
        output_type=str,
        model_settings=settings,
        defer_model_check=True,
    )
