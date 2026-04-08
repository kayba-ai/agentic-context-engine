"""RR agent tool registrars.

Each ``_register_*`` function attaches one or more PydanticAI tools to an
agent instance.  Keeping tool definitions separate from agent factories
makes both easier to read and test independently.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from ace.core.outputs import ExtractedLearning, ReflectorOutput

from .config import RecursiveConfig
from .prompts import SUBAGENT_ANALYSIS_PROMPT, SUBAGENT_DEEPDIVE_PROMPT
from .sandbox import TraceSandbox, create_readonly_sandbox

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent

logger = logging.getLogger(__name__)


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
# Helpers
# ------------------------------------------------------------------


def _format_registered_helpers(sandbox: TraceSandbox) -> str:
    """Render registered helper metadata for sub-agent prompts."""
    registry = sandbox.namespace.get("helper_registry", {})
    if not isinstance(registry, dict) or not registry:
        return ""

    lines = [
        "## Registered Helpers",
        "These helpers are already available inside `execute_code`. "
        "Prefer them before inspecting the raw schema again.",
    ]
    for name, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        description = (
            str(meta.get("description", "")).strip() or "No description provided."
        )
        lines.append(f"- `{name}`: {description}")
    lines.append(
        "Use `print(list_helpers())` for the full catalog, call helpers "
        "directly in code, or use `run_helper(name, ...)`."
    )
    return "\n".join(lines)


def _get_subagent_parallel_cap(deps: RRDeps) -> int:
    """Return the sub-agent concurrency cap for this session."""
    if deps.expected_item_count is not None:
        return max(1, deps.config.worker_subagent_max_parallel)
    return max(1, deps.config.subagent_max_parallel)


def _compact_text(value: Any, limit: int = 160) -> str:
    """Render a short, single-line preview for sandbox summaries."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_cluster_results_view(
    cluster_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a compact sandbox-facing view of collected worker results."""
    view: dict[str, dict[str, Any]] = {}

    for name, result in cluster_results.items():
        assignment = result.get("assignment", {})
        worker_output = result.get("worker_output")
        per_item = result.get("per_item_reflections") or ()
        usage = result.get("usage") or {}

        item_preview: list[dict[str, Any]] = []
        for local_idx, reflection in enumerate(tuple(per_item)[:2]):
            item_preview.append(
                {
                    "local_index": local_idx,
                    "key_insight": _compact_text(
                        getattr(reflection, "key_insight", ""), limit=120
                    ),
                    "correct_approach": _compact_text(
                        getattr(reflection, "correct_approach", ""), limit=120
                    ),
                }
            )

        worker_summary = {}
        if worker_output is not None:
            worker_summary = {
                "key_insight": _compact_text(
                    getattr(worker_output, "key_insight", ""), limit=140
                ),
                "correct_approach": _compact_text(
                    getattr(worker_output, "correct_approach", ""), limit=140
                ),
            }

        usage_summary = {}
        for key in ("requests", "request_tokens", "response_tokens", "total_tokens"):
            if key in usage:
                usage_summary[key] = usage[key]

        view[name] = {
            "status": result.get("status"),
            "trace_indices": list(assignment.get("trace_indices", [])),
            "goal": assignment.get("goal", ""),
            "success_criteria": assignment.get("success_criteria", ""),
            "issues": list(result.get("issues", [])),
            "item_count": len(per_item),
            "worker_summary": worker_summary,
            "item_preview": item_preview,
            "usage": usage_summary,
        }

    return view


def validate_worker_assignment_size(
    cfg: RecursiveConfig,
    trace_indices: list[int],
) -> None:
    """Raise if an assignment is too large for a single worker session."""
    worker_max_items = max(1, cfg.worker_max_items)
    if len(trace_indices) > worker_max_items:
        raise ModelRetry(
            f"Assignment has {len(trace_indices)} traces, which exceeds the "
            f"worker limit of {worker_max_items}. Split it into smaller, "
            "semantically coherent groups before spawning."
        )


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


def register_analysis_tools(agent: PydanticAgent[RRDeps, Any]) -> None:
    """Register ``analyze`` and ``batch_analyze`` tools on any RR agent."""

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


def register_output_validator(agent: PydanticAgent[RRDeps, Any]) -> None:
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


def register_orchestration_tools(
    agent: PydanticAgent[RRDeps, Any],
    *,
    cfg: RecursiveConfig,
    spawn_worker_fn: Callable[..., tuple[ReflectorOutput, RRDeps]],
    slice_batch_fn: Callable[[Any, list[int]], Any],
    get_batch_items_fn: Callable[[Any], list[Any] | None],
) -> None:
    """Register ``spawn_analysis`` and ``collect_results`` on an orchestrator agent."""

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

        validate_worker_assignment_size(cfg, trace_indices)

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
                per_item: list[ReflectorOutput] = []
                if status == "completed":
                    rr_trace = worker_output.raw.get("rr_trace", {})
                    for tr in item_results:
                        if not isinstance(tr, dict):
                            issues.append("Non-dict item in raw['items']")
                            status = "invalid"
                            break
                        learnings = (
                            tr.get("extracted_learnings", tr.get("learnings", []))
                            or []
                        )
                        per_item.append(
                            ReflectorOutput(
                                reasoning=tr.get("reasoning", ""),
                                error_identification=str(
                                    tr.get("error_identification", "")
                                ),
                                root_cause_analysis=tr.get(
                                    "root_cause_analysis", ""
                                ),
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
                lines.append(
                    f"- {name}: failed ({type(exc).__name__}: {exc})"
                )

        # Clear pending after collection
        ctx.deps.pending_clusters.clear()

        # Expose a compact cluster summary in sandbox for inspection.
        ctx.deps.sandbox.inject(
            "cluster_results",
            build_cluster_results_view(ctx.deps.cluster_results),
        )

        summary = "## Collected Results\n" + "\n".join(lines)
        return summary


def register_subagent_execute_code(
    sub_agent: PydanticAgent[SubAgentDeps, str],
) -> None:
    """Register ``execute_code`` on a sub-agent (isolated sandbox snapshot)."""

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
                    f"stdout before error:\n{result.stdout[:max_output]}\n\n"
                )
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
