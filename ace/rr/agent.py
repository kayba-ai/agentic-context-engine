"""PydanticAI-based Recursive Reflector agent.

Replaces the SubRunner REPL loop with a PydanticAI agent that has three
tools:

- ``execute_code`` — run Python in the analysis sandbox
- ``analyze`` — ask a sub-agent for targeted analysis
- ``batch_analyze`` — parallel sub-agent analysis of multiple items

Orchestrator agents additionally have:

- ``spawn_analysis`` — delegate a subset of traces to a worker RR session
- ``collect_results`` — collect completed worker results

Sub-agents have their own ``execute_code`` tool backed by an isolated
sandbox snapshot, so they can explore trace data directly without the
main agent having to serialize data into tool parameters.

The agent produces ``ReflectorOutput`` as structured output when it has
gathered enough evidence.
"""

from __future__ import annotations

import logging
import time as _time_mod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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


def _get_subagent_parallel_cap(deps: "RRDeps") -> int:
    """Return the sub-agent concurrency cap for this session."""
    if deps.expected_item_count is not None:
        return max(1, deps.config.worker_subagent_max_parallel)
    return max(1, deps.config.subagent_max_parallel)


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

    # Orchestration state (only populated for orchestrator sessions)
    is_orchestrator: bool = False
    pending_clusters: dict[str, dict[str, Any]] = field(default_factory=dict)
    cluster_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    cluster_pool: ThreadPoolExecutor | None = None

    # Worker validation: expected per-item count (workers only)
    expected_item_count: int | None = None


# ------------------------------------------------------------------
# Shared tool registrars
# ------------------------------------------------------------------


def _register_execute_code(agent: PydanticAgent[RRDeps, Any]) -> None:
    """Register the execute_code tool on any RR agent."""

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


def _register_analysis_tools(agent: PydanticAgent[RRDeps, Any]) -> None:
    """Register analyze and batch_analyze tools on any RR agent."""

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

        pool_cap = _get_subagent_parallel_cap(ctx.deps)
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


def _register_output_validator(agent: PydanticAgent[RRDeps, Any]) -> None:
    """Register the standard output validator on any RR agent."""

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

        # Worker output validation: enforce raw["items"] with correct count
        expected = ctx.deps.expected_item_count
        if expected is not None:
            items = output.raw.get("items")
            if not isinstance(items, list) or len(items) != expected:
                got = len(items) if isinstance(items, list) else 0
                raise ModelRetry(
                    f"Worker output MUST include raw['items'] with exactly "
                    f"{expected} entries (one per assigned trace). "
                    f"Got {got}. Each entry needs: reasoning, "
                    f"error_identification, root_cause_analysis, "
                    f"correct_approach, key_insight, extracted_learnings."
                )

        return output


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
    """Create the PydanticAI agent for recursive reflection (single-trace).

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the reflector.
        config: RR configuration (timeouts, limits).
        model_settings: PydanticAI model settings.

    Returns:
        Configured PydanticAI agent with tools.
    """
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

    _register_execute_code(agent)
    _register_analysis_tools(agent)
    _register_output_validator(agent)

    return agent


def create_orchestrator_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
    spawn_worker_fn: Callable[..., tuple[ReflectorOutput, RRDeps]] | None = None,
    slice_batch_fn: Callable[[Any, list[int]], Any] | None = None,
    get_batch_items_fn: Callable[[Any], list[Any] | None] | None = None,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create the orchestrator agent for batch RR with manager/worker tools.

    In addition to the standard tools (execute_code, analyze, batch_analyze),
    the orchestrator agent has ``spawn_analysis`` and ``collect_results``.

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the orchestrator.
        config: RR configuration.
        model_settings: PydanticAI model settings.
        spawn_worker_fn: Callback to run a worker RR session.
        slice_batch_fn: Callback to slice batch traces by indices.
        get_batch_items_fn: Callback to extract batch items from traces.

    Returns:
        Configured orchestrator agent.
    """
    cfg = config or RecursiveConfig()
    resolved = resolve_model(model)

    agent: PydanticAgent[RRDeps, ReflectorOutput] = PydanticAgent(
        resolved,
        output_type=ReflectorOutput,
        system_prompt=system_prompt,
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    _register_execute_code(agent)
    _register_analysis_tools(agent)

    # -- Orchestration tools -------------------------------------------

    @agent.tool
    def spawn_analysis(
        ctx: RunContext[RRDeps],
        cluster_name: str,
        trace_indices: list[int],
        goal: str,
        success_criteria: str = "",
    ) -> str:
        """Delegate analysis of a trace subset to a focused worker session.

        The worker runs an independent RR session with its own code
        execution and budget. It inherits registered helpers from this
        session.

        Args:
            cluster_name: Stable identifier for this assignment.
            trace_indices: Original indices in the parent batch.
            goal: What the worker should figure out.
            success_criteria: What counts as a usable result.

        Returns:
            Status line confirming the assignment was queued.
        """
        if spawn_worker_fn is None or slice_batch_fn is None:
            raise ModelRetry("Worker spawning is not configured.")

        if get_batch_items_fn is None:
            raise ModelRetry("Batch item extraction is not configured.")

        # Validate cluster name
        if cluster_name in ctx.deps.pending_clusters:
            raise ModelRetry(
                f"Cluster {cluster_name!r} already has a pending assignment. "
                "Use a different name or wait for collect_results()."
            )
        if cluster_name in ctx.deps.cluster_results:
            raise ModelRetry(
                f"Cluster {cluster_name!r} already has collected results. "
                "Use a different name."
            )

        # Validate indices
        batch_items = get_batch_items_fn(ctx.deps.trace_data)
        if batch_items is None:
            raise ModelRetry("No batch items found in trace data.")

        batch_size = len(batch_items)
        for idx in trace_indices:
            if not (0 <= idx < batch_size):
                raise ModelRetry(
                    f"Index {idx} out of range [0, {batch_size}). "
                    f"Valid range: 0..{batch_size - 1}."
                )

        if len(set(trace_indices)) != len(trace_indices):
            raise ModelRetry(
                "Duplicate indices in assignment. Each index must be unique."
            )

        # Check for overlaps with active assignments
        active_indices: set[int] = set()
        for name, info in ctx.deps.pending_clusters.items():
            for idx in info["assignment"]["trace_indices"]:
                active_indices.add(idx)
        overlap = set(trace_indices) & active_indices
        if overlap:
            raise ModelRetry(
                f"Indices {sorted(overlap)} overlap with active pending "
                f"assignments. Wait for collect_results() first."
            )

        # Slice the batch trace
        sub_batch = slice_batch_fn(ctx.deps.trace_data, trace_indices)

        # Get inherited helpers from orchestrator sandbox
        registry = ctx.deps.sandbox.namespace.get("helper_registry", {})
        inherited_helpers: dict[str, dict[str, str]] = {}
        if isinstance(registry, dict):
            for hname, meta in registry.items():
                if isinstance(meta, dict) and isinstance(meta.get("source"), str):
                    inherited_helpers[hname] = {
                        "source": meta["source"],
                        "description": meta.get("description", ""),
                    }

        # Create pool lazily
        if ctx.deps.cluster_pool is None:
            ctx.deps.cluster_pool = ThreadPoolExecutor(
                max_workers=cfg.max_cluster_workers,
            )

        assignment = {
            "cluster_name": cluster_name,
            "trace_indices": trace_indices,
            "goal": goal,
            "success_criteria": success_criteria,
        }

        # Submit worker session
        future = ctx.deps.cluster_pool.submit(
            spawn_worker_fn,
            sub_batch=sub_batch,
            skillbook_text=ctx.deps.skillbook_text,
            assignment=assignment,
            inherited_helpers=inherited_helpers,
            trace_indices=trace_indices,
        )

        ctx.deps.pending_clusters[cluster_name] = {
            "assignment": assignment,
            "future": future,
            "status": "running",
        }

        return (
            f"Queued cluster {cluster_name!r}: {len(trace_indices)} traces, "
            f"goal={goal!r}. Call collect_results() when ready."
        )

    @agent.tool
    def collect_results(ctx: RunContext[RRDeps]) -> str:
        """Collect all pending worker results.

        Blocks until all pending assignments complete or timeout.
        Results are stored in ``cluster_results`` (accessible via
        execute_code) and a summary is returned.

        Returns:
            Summary of collected results with status per cluster.
        """
        if not ctx.deps.pending_clusters:
            return "No pending clusters to collect."

        timeout = cfg.worker_collect_timeout
        lines: list[str] = []

        for name, info in list(ctx.deps.pending_clusters.items()):
            future: Future = info["future"]
            assignment = info["assignment"]
            expected_count = len(assignment["trace_indices"])

            try:
                worker_output, worker_deps = future.result(timeout=timeout)

                # Validate per-item results
                item_results = worker_output.raw.get("items", [])
                issues: list[str] = []

                if not item_results:
                    issues.append("Missing raw['items'] in worker output")
                    status = "invalid"
                elif len(item_results) != expected_count:
                    issues.append(
                        f"Expected {expected_count} items, got {len(item_results)}"
                    )
                    status = "invalid"
                else:
                    status = "completed"

                # Build per-item reflections from worker output
                from ace.core.outputs import ExtractedLearning

                per_item: list[ReflectorOutput] = []
                if status == "completed":
                    rr_trace = worker_output.raw.get("rr_trace", {})
                    for tr in item_results:
                        if not isinstance(tr, dict):
                            issues.append("Non-dict item in raw['items']")
                            status = "invalid"
                            break
                        learnings = (
                            tr.get("extracted_learnings", tr.get("learnings", [])) or []
                        )
                        per_item.append(
                            ReflectorOutput(
                                reasoning=tr.get("reasoning", ""),
                                error_identification=str(
                                    tr.get("error_identification", "")
                                ),
                                root_cause_analysis=tr.get("root_cause_analysis", ""),
                                correct_approach=tr.get("correct_approach", ""),
                                key_insight=tr.get("key_insight", ""),
                                extracted_learnings=[
                                    ExtractedLearning(
                                        learning=l.get("learning", ""),
                                        atomicity_score=float(
                                            l.get("atomicity_score", 0.0)
                                        ),
                                        evidence=l.get("evidence", ""),
                                    )
                                    for l in learnings
                                    if isinstance(l, dict)
                                ],
                                raw={**tr, "rr_trace": rr_trace},
                            )
                        )

                # Extract usage from worker output
                usage = worker_output.raw.get("usage", {})
                rr_trace_data = worker_output.raw.get("rr_trace", {})

                ctx.deps.cluster_results[name] = {
                    "assignment": assignment,
                    "status": status,
                    "issues": issues,
                    "worker_output": worker_output,
                    "per_item_reflections": tuple(per_item),
                    "usage": usage,
                    "rr_trace": rr_trace_data,
                }

                lines.append(
                    f"- {name}: {status} "
                    f"({len(per_item)}/{expected_count} items)"
                    + (f" — issues: {issues}" if issues else "")
                )

            except TimeoutError:
                ctx.deps.cluster_results[name] = {
                    "assignment": assignment,
                    "status": "timed_out",
                    "issues": [f"Worker did not finish within {timeout}s"],
                    "worker_output": None,
                    "per_item_reflections": (),
                    "usage": {},
                    "rr_trace": {},
                }
                lines.append(f"- {name}: timed_out (exceeded {timeout}s)")

            except Exception as exc:
                ctx.deps.cluster_results[name] = {
                    "assignment": assignment,
                    "status": "failed",
                    "issues": [f"{type(exc).__name__}: {exc}"],
                    "worker_output": None,
                    "per_item_reflections": (),
                    "usage": {},
                    "rr_trace": {},
                }
                lines.append(f"- {name}: failed ({type(exc).__name__}: {exc})")

        # Clear pending after collection
        ctx.deps.pending_clusters.clear()

        # Expose cluster_results in sandbox for inspection
        ctx.deps.sandbox.inject("cluster_results", ctx.deps.cluster_results)

        summary = "## Collected Results\n" + "\n".join(lines)
        return summary

    _register_output_validator(agent)

    return agent


def create_worker_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
    enable_subagent: bool = False,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create a worker agent for delegated RR sessions.

    Workers have ``execute_code`` and optionally ``analyze``/``batch_analyze``.
    They never have orchestration tools (spawn_analysis, collect_results).

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the worker.
        config: RR configuration.
        model_settings: PydanticAI model settings.
        enable_subagent: Whether to include analyze/batch_analyze tools.

    Returns:
        Configured worker agent.
    """
    resolved = resolve_model(model)

    agent: PydanticAgent[RRDeps, ReflectorOutput] = PydanticAgent(
        resolved,
        output_type=ReflectorOutput,
        system_prompt=system_prompt,
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    _register_execute_code(agent)

    if enable_subagent:
        _register_analysis_tools(agent)

    _register_output_validator(agent)

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
