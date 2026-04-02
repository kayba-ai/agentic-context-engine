"""PydanticAI-based Recursive Reflector agent.

Replaces the SubRunner REPL loop with a PydanticAI agent that has three
tools:

- ``execute_code`` — run Python in the analysis sandbox
- ``analyze`` — ask a sub-agent for targeted analysis
- ``batch_analyze`` — parallel sub-agent analysis of multiple items

Sub-agents have their own ``execute_code`` tool backed by an isolated
sandbox snapshot, so they can explore trace data directly without the
main agent having to serialize data into tool parameters.

The agent produces ``ReflectorOutput`` as structured output when it has
gathered enough evidence.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ace.core.outputs import ReflectorOutput
from ace.providers.pydantic_ai import resolve_model

from .config import RecursiveConfig
from .sandbox import TraceSandbox, create_readonly_sandbox

# --- Sub-agent prompt protocols ---

SUBAGENT_ANALYSIS_PROMPT = """\
You are a trace reader for a multi-phase analysis pipeline. A downstream agent will use your output to categorize traces and decide which ones deserve deep investigation. It will not read the raw traces itself — your summary is its only view into the data.

For each trace or conversation in the context:
1. **Task** — what was requested or attempted (brief).
2. **Approach** — the agent's key steps, tools used, and the overall sequence of actions.
3. **Decision points** — where the agent chose between alternatives. What did it choose and what were the other options?
4. **Mistakes** — errors, wrong turns, retries, wasted steps. Describe what went wrong factually — do not analyze root causes.
5. **What stood out** — anything non-obvious: clever recoveries, unusual tool usage, unexpected results, or signs of a pattern.
6. **Evaluation criteria** — if evaluation criteria, rules, or a checklist are provided in the context, actively evaluate every applicable criterion for every trace — even successful ones. Cite evidence for any violations.

Cite step numbers or message excerpts as evidence. Be thorough — the downstream agent cannot go back to the raw data."""

SUBAGENT_DEEPDIVE_PROMPT = """\
You are an investigator analyzing agent execution traces. A downstream agent has already surveyed these traces and selected them for deeper analysis. Your job is to answer the specific question asked, providing the evidence and reasoning the downstream agent needs to formulate learnings.

Approach:
- **Verify before analyzing.** Before investigating causes, check whether the agent's claims and conclusions accurately reflect the data it received. "Confident but wrong" — where the agent proceeds without hesitation based on incorrect reasoning — is a high-value finding that behavioral analysis alone misses.
- **Check against rules.** If agent operating rules or policy are provided, verify that the agent's actions comply with them. Rule violations are high-value findings even when the agent appeared to succeed — they often look "normal" because many traces share the same violation.
- **Causes, not symptoms.** When something went wrong, identify the root decision or assumption that led to it. What should the agent have done instead — concretely?
- **Contrast directly.** When given multiple traces, find the specific point where they diverged. Do not describe each trace separately — compare them.
- **Cite everything.** Every claim must reference specific evidence (step number, message content, tool output). If something is ambiguous, say so — do not speculate.
- **Suggest alternatives.** For mistakes, describe the concrete action the agent should have taken instead."""

SUBAGENT_SYSTEM = (
    "You are a trace analyst with code execution. "
    "Use execute_code to extract evidence from trace data, then reason about it. "
    "Pre-loaded: traces, skillbook, json, re, collections, datetime, "
    "plus any helper variables injected by the runner. "
    "If helper_registry is populated, prefer those registered helpers before "
    "re-discovering the trace schema. "
    "Treat item/context strings as navigation instructions, not dict keys. "
    "Keep code calls minimal (2-3 max)."
)

logger = logging.getLogger(__name__)


def _format_registered_helpers(sandbox: TraceSandbox) -> str:
    """Render registered helper metadata for sub-agent prompts."""
    registry = sandbox.namespace.get("helper_registry", {})
    if not isinstance(registry, dict) or not registry:
        return ""

    lines = [
        "## Registered Helpers",
        "These helpers are already available inside `execute_code`. Prefer them before inspecting the raw schema again.",
    ]
    for name, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        description = (
            str(meta.get("description", "")).strip() or "No description provided."
        )
        lines.append(f"- `{name}`: {description}")
    lines.append(
        "Use `print(list_helpers())` for the full catalog, call helpers directly in code, or use `run_helper(name, ...)`."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Dependency containers
# ------------------------------------------------------------------


@dataclass
class SubAgentDeps:
    """Dependencies for sub-agent tool calls."""

    sandbox: TraceSandbox
    config: RecursiveConfig
    iteration: int = 0


@dataclass
class RRDeps:
    """Dependencies injected into RR tool calls via ``RunContext``."""

    sandbox: TraceSandbox
    trace_data: dict[str, Any]
    skillbook_text: str
    config: RecursiveConfig
    iteration: int = 0
    sub_agent: PydanticAgent[SubAgentDeps, str] | None = None
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
        system_prompt=system_prompt
        or (
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
                f"{stdout_ctx}Code error:\n{error_msg}\n\n" "Fix the bug and try again."
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

    # -- Tool: analyze -----------------------------------------------

    @agent.tool
    async def analyze(
        ctx: RunContext[RRDeps],
        question: str,
        mode: str = "analysis",
        context: str = "",
    ) -> str:
        """Delegate analysis to a sub-agent that can explore trace data.

        The sub-agent has its own ``execute_code`` tool with access to
        all trace data.  Pass optional ``context`` for focus — do NOT
        serialize large data, the sub-agent reads it directly.

        Args:
            question: What to analyze.
            mode: ``"analysis"`` for survey, ``"deep_dive"`` for investigation.
            context: Optional focus instructions (brief, not data dumps).

        Returns:
            The sub-agent's analysis.
        """
        if ctx.deps.sub_agent is None:
            return "(analyze unavailable — sub-agent not configured)"

        sys_prompt = (
            SUBAGENT_DEEPDIVE_PROMPT
            if mode == "deep_dive"
            else SUBAGENT_ANALYSIS_PROMPT
        )

        prompt_parts = [
            sys_prompt,
            (
                "Treat any context string as navigation instructions for the "
                "trace data. Do not try to look it up as a literal dict key "
                "unless it explicitly names a keyed field."
            ),
            f"## Question\n{question}",
        ]
        if context:
            prompt_parts.append(f"## Additional Context\n{context}")
        helper_prompt = _format_registered_helpers(ctx.deps.sandbox)
        if helper_prompt:
            prompt_parts.append(helper_prompt)
        prompt_parts.append("## Your Analysis")
        prompt = "\n\n".join(prompt_parts)

        # Isolated sandbox snapshot — sub-agent can explore data via code
        snapshot = create_readonly_sandbox(ctx.deps.sandbox)
        sub_deps = SubAgentDeps(sandbox=snapshot, config=ctx.deps.config)
        usage_limits = UsageLimits(
            request_limit=ctx.deps.config.subagent_max_requests,
        )

        try:
            result = await ctx.deps.sub_agent.run(
                prompt,
                deps=sub_deps,
                usage_limits=usage_limits,
            )
            response = result.output

            ctx.deps.sub_agent_history.append(
                {
                    "question": question,
                    "context_length": len(context),
                    "response_length": len(response),
                    "mode": mode,
                    "code_calls": sub_deps.iteration,
                }
            )

            return response

        except UsageLimitExceeded:
            return (
                "(Sub-agent reached request limit. "
                "Partial analysis may be incomplete.)"
            )
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
        """Analyze multiple items in parallel using sub-agents.

        Each item is analyzed by an independent sub-agent with its own
        ``execute_code`` access to the full trace data.  Items should be
        focus instructions (e.g., task IDs or specific patterns to
        investigate), not serialized data.

        Args:
            question: What to analyze about each item.
            items: List of focus instructions for each sub-agent.
            mode: ``"analysis"`` for survey, ``"deep_dive"`` for investigation.

        Returns:
            Ordered list of analysis results.
        """
        if ctx.deps.sub_agent is None:
            return ["(batch_analyze unavailable — sub-agent not configured)"] * len(
                items
            )

        if not items:
            return []

        sub = ctx.deps.sub_agent
        cfg = ctx.deps.config
        parent_sandbox = ctx.deps.sandbox

        sys_prompt = (
            SUBAGENT_DEEPDIVE_PROMPT
            if mode == "deep_dive"
            else SUBAGENT_ANALYSIS_PROMPT
        )

        def _analyze_one(item: str) -> tuple[str, int]:
            # Each sub-agent gets its own sandbox snapshot
            snapshot = create_readonly_sandbox(parent_sandbox)
            sub_deps = SubAgentDeps(sandbox=snapshot, config=cfg)
            usage_limits = UsageLimits(
                request_limit=cfg.subagent_max_requests,
            )

            prompt = (
                f"{sys_prompt}\n\n"
                "Treat the item below as navigation instructions for the trace "
                "data. Do not look up the raw item string as a dict key unless "
                "it explicitly names a keyed field.\n\n"
                f"{_format_registered_helpers(parent_sandbox)}\n\n"
                f"## Question\n{question}\n\n"
                f"## Item\n{item}\n\n"
                f"## Your Analysis"
            )

            try:
                result = sub.run_sync(
                    prompt,
                    deps=sub_deps,
                    usage_limits=usage_limits,
                )
                return result.output, sub_deps.iteration
            except UsageLimitExceeded:
                return "(Sub-agent reached request limit.)", sub_deps.iteration
            except Exception as e:
                return f"(Error: {e})", sub_deps.iteration

        pool_cap = max(1, cfg.subagent_max_parallel)
        pool_size = min(len(items), pool_cap)
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            raw_results = list(pool.map(_analyze_one, items))

        results = [r[0] for r in raw_results]
        code_calls_per_item = [r[1] for r in raw_results]

        ctx.deps.sub_agent_history.append(
            {
                "question": question,
                "items_count": len(items),
                "mode": mode,
                "batch": True,
                "code_calls_per_item": code_calls_per_item,
            }
        )

        return results

    # -- Output validator --------------------------------------------

    @agent.output_validator
    def validate_output(
        ctx: RunContext[RRDeps], output: ReflectorOutput
    ) -> ReflectorOutput:
        """Ensure the agent explored data before concluding."""
        if ctx.deps.iteration < 1 and not ctx.deps.sub_agent_history:
            raise ModelRetry(
                "You haven't explored the data enough. "
                "Use execute_code or analyze/batch_analyze first, "
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
) -> PydanticAgent[SubAgentDeps, str]:
    """Create the sub-agent for ``analyze`` / ``batch_analyze`` tools.

    The sub-agent has its own ``execute_code`` tool backed by an isolated
    sandbox snapshot.  It can explore trace data directly, so the main
    agent doesn't need to serialize data into tool parameters.

    Args:
        model: LiteLLM or PydanticAI model string.
        config: RR configuration for sub-agent settings.
        model_settings: Override model settings.

    Returns:
        PydanticAI agent with execute_code tool, producing text output.
    """
    cfg = config or RecursiveConfig()
    resolved = resolve_model(model)

    settings = model_settings or ModelSettings(
        temperature=cfg.subagent_temperature,
        max_tokens=cfg.subagent_max_tokens,
    )

    sub_agent: PydanticAgent[SubAgentDeps, str] = PydanticAgent(
        resolved,
        output_type=str,
        system_prompt=SUBAGENT_SYSTEM,
        model_settings=settings,
        defer_model_check=True,
        deps_type=SubAgentDeps,
    )

    @sub_agent.tool(retries=3)
    def execute_code(ctx: RunContext[SubAgentDeps], code: str) -> str:
        """Execute Python code to explore trace data.

        Pre-loaded: ``traces``, ``skillbook``, ``json``, ``re``,
        ``collections``, ``datetime``.

        Args:
            code: Python code to execute.

        Returns:
            Captured stdout/stderr from execution.
        """
        ctx.deps.iteration += 1
        max_output = ctx.deps.config.max_output_chars

        result = ctx.deps.sandbox.execute(
            code,
            timeout=ctx.deps.config.timeout,
        )

        if result.exception:
            error_msg = f"{type(result.exception).__name__}: {result.exception}"
            stdout_ctx = ""
            if result.stdout:
                stdout_ctx = (
                    f"stdout before error:\n" f"{result.stdout[:max_output]}\n\n"
                )
            raise ModelRetry(
                f"{stdout_ctx}Code error:\n{error_msg}\n\n" "Fix the bug and try again."
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

    return sub_agent
