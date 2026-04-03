"""RRStep — Recursive Reflector powered by PydanticAI.

Uses a PydanticAI agent with ``execute_code``, ``analyze``, and
``batch_analyze`` tools.  The agent explores traces via code execution
and sub-agent analysis, then produces ``ReflectorOutput`` as structured
output.

Batch mode uses an orchestrator/worker pattern: the orchestrator agent
can delegate trace subsets to focused worker RR sessions via
``spawn_analysis`` / ``collect_results`` tools.

Satisfies both ``StepProtocol`` (for Pipeline composition) and
``ReflectorLike`` (drop-in replacement for simple Reflector).
"""

from __future__ import annotations

import copy
import json as _json
import logging
from typing import Any, Optional

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ace.core.context import ACEStepContext
from ace.core.outputs import AgentOutput, ExtractedLearning, ReflectorOutput

from .agent import (
    RRDeps,
    create_orchestrator_agent,
    create_rr_agent,
    create_sub_agent,
    create_worker_agent,
)
from .config import RecursiveConfig
from .prompts import (
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_SYSTEM,
    REFLECTOR_RECURSIVE_PROMPT,
    REFLECTOR_RECURSIVE_SYSTEM,
    WORKER_PROMPT,
    WORKER_SYSTEM,
)
from .sandbox import TraceSandbox
from .trace_context import TraceContext

logger = logging.getLogger(__name__)
_USE_DEFAULT_SUB_AGENT = object()


def _preview(text: str | None, max_len: int = 150) -> str:
    """Return a short preview safe for str.format()."""
    if not text:
        return "(empty)"
    snippet = text if len(text) <= max_len else text[:max_len]
    return snippet.replace("{", "{{").replace("}", "}}")


class RRStep:
    """Recursive Reflector as a pipeline step (PydanticAI agent).

    Satisfies **StepProtocol** (place directly in a Pipeline) and
    **ReflectorLike** (use as drop-in reflector in runners).

    Internally uses a PydanticAI agent with tools for code execution
    and sub-agent analysis.  The agent produces ``ReflectorOutput``
    as structured output when it has gathered enough evidence.

    Batch mode uses an orchestrator/worker pattern where the orchestrator
    can delegate subsets of traces to focused worker sessions.

    Args:
        model: LiteLLM or PydanticAI model string.
        config: RR configuration (timeouts, limits, sub-agent settings).
        prompt_template: User prompt template with format placeholders.
        model_settings: Override PydanticAI model settings.
    """

    # StepProtocol
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __init__(
        self,
        model: str,
        config: Optional[RecursiveConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.config = config or RecursiveConfig()
        self.prompt_template = prompt_template
        self._model = model
        self._model_settings = model_settings

        # Build PydanticAI agents (single-trace)
        self._agent = create_rr_agent(
            model,
            system_prompt=REFLECTOR_RECURSIVE_SYSTEM,
            config=self.config,
            model_settings=model_settings,
        )

        # Sub-agent for analyze/batch_analyze tools
        subagent_model = self.config.subagent_model or model
        self._sub_agent = (
            create_sub_agent(subagent_model, config=self.config)
            if self.config.enable_subagent
            else None
        )

        # Batch-only agents — created lazily on first batch use
        self._orchestrator_agent: PydanticAgent[RRDeps, ReflectorOutput] | None = None
        self._worker_agent: PydanticAgent[RRDeps, ReflectorOutput] | None = None

    # ------------------------------------------------------------------
    # Lazy batch agent creation
    # ------------------------------------------------------------------

    def _ensure_batch_agents(self) -> None:
        """Create orchestrator and worker agents on first batch use."""
        if self._orchestrator_agent is not None:
            return

        self._orchestrator_agent = create_orchestrator_agent(
            self._model,
            system_prompt=ORCHESTRATOR_SYSTEM,
            config=self.config,
            model_settings=self._model_settings,
            spawn_worker_fn=self._run_worker_session,
            slice_batch_fn=self._slice_batch_trace,
            get_batch_items_fn=self._get_batch_items,
        )

        self._worker_agent = create_worker_agent(
            self.config.worker_model or self._model,
            system_prompt=WORKER_SYSTEM,
            config=self.config,
            model_settings=self._model_settings,
            enable_subagent=self.config.worker_enable_subagent,
        )

    # ------------------------------------------------------------------
    # StepProtocol entry
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        """Run the Recursive Reflector and attach the reflection(s).

        When the trace is a batch container (a raw list, ``"items"``,
        ``"tasks"``, or a legacy combined ``steps`` batch), the
        orchestrator agent manages the analysis.
        """
        trace = ctx.trace or {}
        if self._get_batch_items(trace) is not None:
            reflections = self._run_batch_reflections(trace, ctx.skillbook)
            return ctx.replace(reflections=reflections)
        elif isinstance(trace, dict):
            reflection = self._run_reflection(
                traces=trace,
                question=trace.get("question", ""),
                ground_truth=trace.get("ground_truth"),
                feedback=trace.get("feedback"),
                skillbook=ctx.skillbook,
            )
        else:
            reflection = self._run_reflection(
                skillbook=ctx.skillbook,
                trace=trace,
            )
        return ctx.replace(reflections=(reflection,))

    # ------------------------------------------------------------------
    # ReflectorLike protocol
    # ------------------------------------------------------------------

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """ReflectorLike — delegates to the PydanticAI agent."""
        return self._run_reflection(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core reflection logic (single-trace)
    # ------------------------------------------------------------------

    def _run_reflection(
        self,
        *,
        question: str = "",
        agent_output: Optional[AgentOutput] = None,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Run the PydanticAI agent and return analysis."""
        trace_obj = kwargs.pop("trace", None)
        if trace_obj is None and agent_output is not None:
            trace_obj = getattr(agent_output, "trace_context", None)
            if trace_obj is None:
                trace_obj = TraceContext.from_agent_output(agent_output)  # type: ignore[arg-type]

        # Build traces dict — canonical data structure for sandbox code
        traces = kwargs.pop("traces", None)
        if traces is None:
            traces = self._build_traces_dict(
                question, agent_output, ground_truth, feedback, trace_obj
            )

        output, deps = self._run_reflection_session(
            traces=traces,
            skillbook=skillbook,
            agent=self._agent,
            max_llm_calls=self.config.max_llm_calls,
            config=self.config,
            trace_obj=trace_obj,
        )

        # Enrich timeout output with ground-truth comparison if available
        if output.raw.get("timeout") and (ground_truth or agent_output):
            output = self._build_timeout_output(
                question, agent_output, ground_truth, feedback, deps
            )

        return output

    # ------------------------------------------------------------------
    # Factored session helper
    # ------------------------------------------------------------------

    def _run_reflection_session(
        self,
        *,
        traces: Any,
        skillbook: Any,
        agent: PydanticAgent[RRDeps, ReflectorOutput],
        max_llm_calls: int,
        config: RecursiveConfig,
        is_orchestrator: bool = False,
        assignment: dict[str, Any] | None = None,
        inherited_helpers: dict[str, dict[str, str]] | None = None,
        trace_obj: Any = None,
        sub_agent: Any = _USE_DEFAULT_SUB_AGENT,
    ) -> tuple[ReflectorOutput, RRDeps]:
        """Run a complete RR session lifecycle.

        Shared by single-trace, orchestrator, and worker paths.

        Args:
            traces: Trace data (single or batch).
            skillbook: Skillbook instance or text.
            agent: PydanticAI agent to run.
            max_llm_calls: LLM call budget for this session.
            config: RR configuration.
            is_orchestrator: Whether this is an orchestrator session.
            assignment: Worker assignment metadata (workers only).
            inherited_helpers: Helper definitions from parent session.
            trace_obj: Optional TraceContext for single-trace sessions.

        Returns:
            Tuple of (ReflectorOutput, RRDeps).
        """
        # Build sandbox
        sandbox = self._create_sandbox(trace_obj, traces, skillbook)

        # Inject inherited helpers into worker sandbox
        if inherited_helpers:
            for hname, meta in inherited_helpers.items():
                source = meta.get("source", "")
                if source.strip():
                    try:
                        sandbox.execute(source, timeout=config.timeout)
                        registry = sandbox.namespace.setdefault("helper_registry", {})
                        registry[hname] = {
                            "description": meta.get("description", ""),
                            "source": source,
                        }
                    except Exception as exc:
                        logger.warning(
                            "Failed to inject helper %s into worker: %s",
                            hname,
                            exc,
                        )

        # Resolve skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)

        # Compute expected item count for worker output validation
        expected_item_count = None
        if assignment is not None:
            expected_item_count = len(assignment.get("trace_indices", []))

        # Build deps
        deps_sub_agent = self._sub_agent
        if sub_agent is not _USE_DEFAULT_SUB_AGENT:
            deps_sub_agent = sub_agent
        deps = RRDeps(
            sandbox=sandbox,
            trace_data=traces,
            skillbook_text=skillbook_text or "(empty skillbook)",
            config=config,
            sub_agent=deps_sub_agent,
            is_orchestrator=is_orchestrator,
            expected_item_count=expected_item_count,
        )

        # Build prompt
        if is_orchestrator:
            initial_prompt = self._build_orchestrator_prompt(traces, skillbook)
        elif assignment is not None:
            initial_prompt = self._build_worker_prompt(traces, skillbook, assignment)
        else:
            initial_prompt = self._build_initial_prompt(traces, skillbook, trace_obj)

        # Run the PydanticAI agent
        usage_limits = UsageLimits(request_limit=max_llm_calls)

        try:
            result = agent.run_sync(
                initial_prompt,
                deps=deps,
                usage_limits=usage_limits,
            )
            output = result.output

            # Merge execution metadata into raw
            usage = result.usage()
            output.raw = {
                **output.raw,
                "max_llm_calls_scope": "main_agent_session",
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "requests": usage.requests,
                },
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": deps.sub_agent_history,
                    "timed_out": False,
                },
            }

            logger.info(
                "RR session completed: %d tool calls, %d sub-agent calls%s",
                deps.iteration,
                len(deps.sub_agent_history),
                " (orchestrator)" if is_orchestrator else "",
            )

            return output, deps

        except UsageLimitExceeded:
            logger.warning(
                "RR usage limit reached (%d requests)%s",
                max_llm_calls,
                " in orchestrator" if is_orchestrator else "",
            )
            output = ReflectorOutput(
                reasoning=(
                    f"Recursive analysis reached usage limit "
                    f"({max_llm_calls} requests)."
                ),
                error_identification="timeout",
                root_cause_analysis="Analysis incomplete due to request limit",
                correct_approach=(
                    "Consider increasing budget or simplifying the analysis"
                ),
                key_insight="Session reached usage limit before completing",
                raw={
                    "timeout": True,
                    "max_llm_calls": max_llm_calls,
                    "max_llm_calls_scope": "main_agent_session",
                    "rr_trace": {
                        "total_iterations": deps.iteration,
                        "subagent_calls": deps.sub_agent_history,
                        "timed_out": True,
                    },
                },
            )
            return output, deps

        except Exception as e:
            logger.error("RR agent failed: %s", e, exc_info=True)
            output = ReflectorOutput(
                reasoning=f"Recursive analysis failed: {e}",
                correct_approach="",
                key_insight="",
                raw={"error": str(e)},
            )
            return output, deps

        finally:
            # Clean up worker pool if orchestrator
            if is_orchestrator:
                self._harvest_pending_clusters(deps)
                if deps.cluster_pool is not None:
                    deps.cluster_pool.shutdown(wait=False)
                    deps.cluster_pool = None

    # ------------------------------------------------------------------
    # Worker session
    # ------------------------------------------------------------------

    def _run_worker_session(
        self,
        *,
        sub_batch: Any,
        skillbook_text: str,
        assignment: dict[str, Any],
        inherited_helpers: dict[str, dict[str, str]],
        trace_indices: list[int],
    ) -> tuple[ReflectorOutput, RRDeps]:
        """Run a worker RR session for a delegated trace subset.

        Called by the orchestrator's ``spawn_analysis`` tool via the
        thread pool.

        Args:
            sub_batch: Sliced batch trace for this worker.
            skillbook_text: Skillbook text from the orchestrator.
            assignment: Assignment metadata (cluster_name, goal, etc.).
            inherited_helpers: Helper definitions from the orchestrator.
            trace_indices: Original indices in the parent batch.

        Returns:
            Tuple of (ReflectorOutput, RRDeps).
        """
        assert self._worker_agent is not None, "Worker agent not initialized"

        # Compute worker budget from assignment size
        item_count = len(trace_indices)
        serialized_size = len(_json.dumps(sub_batch, default=str))

        # Budget heuristic: base + per-item + size factor
        base_budget = 10
        per_item_budget = 3
        size_factor = max(1, serialized_size // 50_000)
        computed_budget = base_budget + (per_item_budget * item_count) + size_factor

        # Cap at main agent budget
        max_budget = self.config.max_llm_calls
        worker_budget = min(computed_budget, max_budget)

        # Worker sub-agent: use separate sub-agent if enabled
        worker_sub_agent = None
        if self.config.worker_enable_subagent:
            subagent_model = (
                self.config.subagent_model
                or self.config.worker_model
                or self._model
            )
            worker_sub_agent = create_sub_agent(subagent_model, config=self.config)

        output, deps = self._run_reflection_session(
            traces=sub_batch,
            skillbook=skillbook_text,
            agent=self._worker_agent,
            max_llm_calls=worker_budget,
            config=self.config,
            assignment=assignment,
            inherited_helpers=inherited_helpers,
            sub_agent=worker_sub_agent,
        )

        return output, deps

    # ------------------------------------------------------------------
    # Batch reflection (orchestrator path)
    # ------------------------------------------------------------------

    def _run_batch_reflections(
        self,
        batch_trace: Any,
        skillbook: Any,
    ) -> tuple[ReflectorOutput, ...]:
        """Run orchestrated batch analysis.

        Uses the orchestrator agent which can delegate to worker sessions.
        The orchestrator decides whether to analyze directly or delegate
        based on batch size and complexity.
        """
        self._ensure_batch_agents()
        assert self._orchestrator_agent is not None

        items = self._get_batch_items(batch_trace) or []

        output, deps = self._run_reflection_session(
            traces=batch_trace,
            skillbook=skillbook,
            agent=self._orchestrator_agent,
            max_llm_calls=self.config.orchestrator_max_llm_calls,
            config=self.config,
            is_orchestrator=True,
        )

        # Determine if workers were used
        has_workers = bool(deps.cluster_results)

        if has_workers:
            # Validate and merge worker results
            reflections = self._merge_cluster_results(deps, items)
            orchestration_summary = self._build_orchestration_summary(output)
            for reflection in reflections:
                reflection.raw["orchestration_summary"] = copy.deepcopy(
                    orchestration_summary
                )
            return reflections
        else:
            # Direct analysis — orchestrator produced raw["items"]
            return self._split_batch_reflection(output, items)

    # ------------------------------------------------------------------
    # Batch slicing
    # ------------------------------------------------------------------

    def _slice_batch_trace(self, traces: Any, indices: list[int]) -> Any:
        """Slice a batch trace to a subset of items, preserving container shape.

        Supports:
        - Raw ``list[...]``
        - Dict with ``"items"`` key
        - Dict with ``"tasks"`` key
        - Legacy combined ``"steps"`` batch
        - Preserves batch-level metadata outside the item list
        """
        if isinstance(traces, list):
            return [traces[i] for i in indices]

        if not isinstance(traces, dict):
            return traces

        # Find the item list key
        for key in ("items", "tasks"):
            item_list = traces.get(key)
            if isinstance(item_list, list):
                sliced = {k: v for k, v in traces.items() if k != key}
                sliced[key] = [item_list[i] for i in indices]
                return sliced

        # Legacy combined steps batch
        if self._looks_like_combined_steps_batch(traces):
            steps = traces["steps"]
            sliced = {k: v for k, v in traces.items() if k != "steps"}
            sliced["steps"] = [steps[i] for i in indices]
            return sliced

        return traces

    # ------------------------------------------------------------------
    # Validation and merge
    # ------------------------------------------------------------------

    def _merge_cluster_results(
        self,
        deps: RRDeps,
        batch_items: list[Any],
    ) -> tuple[ReflectorOutput, ...]:
        """Merge validated cluster results into original item order.

        Raises:
            RuntimeError: If coverage is incomplete, overlapping, or
                any cluster has unresolved issues.
        """
        batch_size = len(batch_items)
        result_slots: list[ReflectorOutput | None] = [None] * batch_size
        coverage: set[int] = set()
        errors: list[str] = []

        for name, cr in deps.cluster_results.items():
            status = cr["status"]
            assignment = cr["assignment"]
            trace_indices: list[int] = assignment["trace_indices"]

            if status == "completed":
                per_item: tuple[ReflectorOutput, ...] = cr["per_item_reflections"]
                if len(per_item) != len(trace_indices):
                    errors.append(
                        f"Cluster {name!r}: expected {len(trace_indices)} "
                        f"items, got {len(per_item)}"
                    )
                    continue

                for local_idx, global_idx in enumerate(trace_indices):
                    if global_idx in coverage:
                        errors.append(
                            f"Cluster {name!r}: index {global_idx} already "
                            f"covered by another cluster"
                        )
                        continue
                    coverage.add(global_idx)
                    item_id = self._get_batch_item_id(
                        batch_items[global_idx], global_idx
                    )
                    r = per_item[local_idx]
                    r.raw["item_id"] = item_id
                    result_slots[global_idx] = r

            elif status in ("failed", "timed_out", "invalid"):
                issues = cr.get("issues", [])
                errors.append(f"Cluster {name!r}: {status} — {issues}")
            else:
                errors.append(f"Cluster {name!r}: unexpected status {status!r}")

        # Check for missing coverage
        missing = set(range(batch_size)) - coverage
        if missing:
            errors.append(f"Missing coverage for trace indices: {sorted(missing)}")

        if errors:
            error_detail = "\n".join(f"  - {e}" for e in errors)
            raise RuntimeError(
                f"Orchestrated batch RR failed validation:\n{error_detail}\n"
                f"Coverage: {len(coverage)}/{batch_size} traces."
            )

        # All slots must be populated
        assert all(s is not None for s in result_slots)
        return tuple(result_slots)  # type: ignore[arg-type]

    def _build_orchestration_summary(
        self,
        output: ReflectorOutput,
    ) -> dict[str, Any]:
        """Preserve the orchestrator's own final output for observability."""
        raw = {k: v for k, v in output.raw.items() if k not in {"items", "tasks"}}
        return {
            "reasoning": output.reasoning,
            "error_identification": output.error_identification,
            "root_cause_analysis": output.root_cause_analysis,
            "correct_approach": output.correct_approach,
            "key_insight": output.key_insight,
            "raw": raw,
        }

    def _record_cluster_result(
        self,
        deps: RRDeps,
        *,
        name: str,
        assignment: dict[str, Any],
        worker_output: ReflectorOutput,
        expected_count: int,
    ) -> None:
        """Validate a worker output and store it in ``cluster_results``."""
        item_results = worker_output.raw.get("items", [])
        issues: list[str] = []

        if not item_results:
            issues.append("Missing raw['items'] in worker output")
            status = "invalid"
        elif len(item_results) != expected_count:
            issues.append(f"Expected {expected_count} items, got {len(item_results)}")
            status = "invalid"
        else:
            status = "completed"

        per_item: list[ReflectorOutput] = []
        if status == "completed":
            rr_trace = worker_output.raw.get("rr_trace", {})
            for tr in item_results:
                if not isinstance(tr, dict):
                    issues.append("Non-dict item in raw['items']")
                    status = "invalid"
                    break
                learnings = tr.get("extracted_learnings", tr.get("learnings", [])) or []
                per_item.append(
                    ReflectorOutput(
                        reasoning=tr.get("reasoning", ""),
                        error_identification=str(tr.get("error_identification", "")),
                        root_cause_analysis=tr.get("root_cause_analysis", ""),
                        correct_approach=tr.get("correct_approach", ""),
                        key_insight=tr.get("key_insight", ""),
                        extracted_learnings=[
                            ExtractedLearning(
                                learning=l.get("learning", ""),
                                atomicity_score=float(l.get("atomicity_score", 0.0)),
                                evidence=l.get("evidence", ""),
                            )
                            for l in learnings
                            if isinstance(l, dict)
                        ],
                        raw={**tr, "rr_trace": rr_trace},
                    )
                )

        deps.cluster_results[name] = {
            "assignment": assignment,
            "status": status,
            "issues": issues,
            "worker_output": worker_output,
            "per_item_reflections": tuple(per_item),
            "usage": worker_output.raw.get("usage", {}),
            "rr_trace": worker_output.raw.get("rr_trace", {}),
        }

    def _record_cluster_failure(
        self,
        deps: RRDeps,
        *,
        name: str,
        assignment: dict[str, Any],
        status: str,
        issues: list[str],
    ) -> None:
        """Record an explicit failed or timed-out cluster result."""
        deps.cluster_results[name] = {
            "assignment": assignment,
            "status": status,
            "issues": issues,
            "worker_output": None,
            "per_item_reflections": (),
            "usage": {},
            "rr_trace": {},
        }

    def _harvest_pending_clusters(self, deps: RRDeps) -> None:
        """Record any uncollected worker results before the session ends."""
        if not deps.pending_clusters:
            return

        for name, info in list(deps.pending_clusters.items()):
            future = info["future"]
            assignment = info["assignment"]
            expected_count = len(assignment["trace_indices"])

            if not future.done():
                self._record_cluster_failure(
                    deps,
                    name=name,
                    assignment=assignment,
                    status="failed",
                    issues=[
                        "Worker assignment was still pending when the "
                        "orchestrator session ended. Call collect_results() "
                        "before finalizing."
                    ],
                )
                continue

            try:
                worker_output, _worker_deps = future.result()
            except Exception as exc:
                self._record_cluster_failure(
                    deps,
                    name=name,
                    assignment=assignment,
                    status="failed",
                    issues=[f"{type(exc).__name__}: {exc}"],
                )
                continue

            self._record_cluster_result(
                deps,
                name=name,
                assignment=assignment,
                worker_output=worker_output,
                expected_count=expected_count,
            )

        deps.pending_clusters.clear()
        deps.sandbox.inject("cluster_results", deps.cluster_results)

    def _split_batch_reflection(
        self,
        reflection: ReflectorOutput,
        items: list[Any],
    ) -> tuple[ReflectorOutput, ...]:
        """Extract per-item ReflectorOutputs from validated batch output."""
        if not items:
            return ()

        item_results = reflection.raw.get("items")
        if item_results is None:
            item_results = reflection.raw.get("tasks")

        if not isinstance(item_results, list):
            raise RuntimeError(
                "Direct batch analysis did not return raw['items'] as a list."
            )
        if len(item_results) != len(items):
            raise RuntimeError(
                "Direct batch analysis returned the wrong number of per-item "
                f"results: expected {len(items)}, got {len(item_results)}."
            )

        reflections: list[ReflectorOutput] = []
        rr_trace = reflection.raw.get("rr_trace", {})
        for i, (item, tr) in enumerate(zip(items, item_results)):
            item_id = self._get_batch_item_id(item, i)
            if not isinstance(tr, dict):
                raise RuntimeError(
                    "Direct batch analysis returned a non-dict entry in "
                    f"raw['items'] at index {i}."
                )
            learnings = tr.get("extracted_learnings", tr.get("learnings", []))
            reflections.append(
                ReflectorOutput(
                    reasoning=tr.get("reasoning", reflection.reasoning),
                    error_identification=str(tr.get("error_identification", "")),
                    root_cause_analysis=tr.get("root_cause_analysis", ""),
                    correct_approach=tr.get(
                        "correct_approach", reflection.correct_approach
                    ),
                    key_insight=tr.get("key_insight", reflection.key_insight),
                    extracted_learnings=[
                        ExtractedLearning(
                            learning=l.get("learning", ""),
                            atomicity_score=float(l.get("atomicity_score", 0.0)),
                            evidence=l.get("evidence", ""),
                        )
                        for l in learnings
                        if isinstance(l, dict)
                    ],
                    raw={
                        **tr,
                        "item_id": item_id,
                        "rr_trace": rr_trace,
                    },
                )
            )
        return tuple(reflections)

    # ------------------------------------------------------------------
    # Orchestrator prompt
    # ------------------------------------------------------------------

    def _build_orchestrator_prompt(
        self,
        traces: Any,
        skillbook: Any,
    ) -> str:
        """Build the slim batch manager prompt for the orchestrator."""
        batch_items = self._get_batch_items(traces) or []
        batch_count = len(batch_items)

        # Compute totals
        total_size_chars = len(_json.dumps(traces, default=str))
        total_message_count = sum(
            len(self._extract_batch_messages(item)) for item in batch_items
        )

        # Pass/fail breakdown
        pass_count = 0
        fail_count = 0
        unknown_count = 0
        for item in batch_items:
            feedback = self._extract_batch_field(item, "feedback")
            if "reward=1.0" in str(feedback) or "PASSED" in str(feedback).upper():
                pass_count += 1
            elif "reward=0.0" in str(feedback) or "FAILED" in str(feedback).upper():
                fail_count += 1
            else:
                unknown_count += 1

        # Capped exemplar IDs
        exemplar_ids = []
        cap = min(10, batch_count)
        for i in range(cap):
            exemplar_ids.append(self._get_batch_item_id(batch_items[i], i))
        exemplar_section = ""
        if exemplar_ids:
            exemplar_section = f"- **Exemplar IDs (first {cap}):** " + ", ".join(
                f"`{eid}`" for eid in exemplar_ids
            )
            if batch_count > cap:
                exemplar_section += f" ... and {batch_count - cap} more"

        # Survey group count
        survey_items = self._build_survey_items(batch_items)
        survey_group_section = f"- **Precomputed survey groups:** {len(survey_items)}"

        # Skillbook length
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        return ORCHESTRATOR_PROMPT.format(
            batch_count=batch_count,
            total_size_chars=total_size_chars,
            total_message_count=total_message_count,
            pass_count=pass_count,
            fail_count=fail_count,
            unknown_count=unknown_count,
            exemplar_section=exemplar_section,
            survey_group_section=survey_group_section,
            skillbook_length=len(skillbook_text),
            max_iterations=self.config.orchestrator_max_llm_calls,
        )

    # ------------------------------------------------------------------
    # Worker prompt
    # ------------------------------------------------------------------

    def _build_worker_prompt(
        self,
        traces: Any,
        skillbook: Any,
        assignment: dict[str, Any],
    ) -> str:
        """Build the worker prompt for a delegated trace subset."""
        batch_items = self._get_batch_items(traces) or []
        trace_count = len(batch_items)

        skillbook_text = ""
        if skillbook is not None:
            if isinstance(skillbook, str):
                skillbook_text = skillbook
            elif hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)

        # Helper section
        sandbox = self._create_sandbox(None, traces, skillbook)
        helper_prompt = ""
        registry = sandbox.namespace.get("helper_registry", {})
        if isinstance(registry, dict) and registry:
            helper_lines = [
                "| `helper_registry` | Inherited helpers from parent session | dynamic |"
            ]
            for hname, meta in registry.items():
                if isinstance(meta, dict):
                    desc = meta.get("description", "no description")
                    helper_lines.append(f"| `{hname}` | {desc} | callable |")
            helper_prompt = "\n".join(helper_lines)

        # Analysis tools section
        analysis_tools = ""
        if self.config.worker_enable_subagent:
            analysis_tools = (
                "| `analyze(question, mode, context?)` | Sub-agent analysis. |\n"
                "| `batch_analyze(question, items, mode)` | Parallel sub-agent analysis. |\n"
            )

        # Analysis strategy
        if self.config.worker_enable_subagent:
            analysis_strategy = (
                "Use analyze/batch_analyze for deep investigation, "
                "execute_code for data preparation."
            )
        else:
            analysis_strategy = (
                "Use execute_code to explore and analyze traces directly. "
                "Build your analysis from code-extracted evidence."
            )

        # Compute worker budget
        item_count = trace_count
        serialized_size = len(_json.dumps(traces, default=str))
        base_budget = 10
        per_item_budget = 3
        size_factor = max(1, serialized_size // 50_000)
        worker_budget = min(
            base_budget + (per_item_budget * item_count) + size_factor,
            self.config.max_llm_calls,
        )

        return WORKER_PROMPT.format(
            cluster_name=assignment.get("cluster_name", "unknown"),
            goal=assignment.get("goal", "Analyze assigned traces"),
            success_criteria=assignment.get(
                "success_criteria", "Per-item reflections with evidence"
            ),
            trace_count=trace_count,
            trace_indices=assignment.get("trace_indices", []),
            skillbook_length=len(skillbook_text),
            helper_section=helper_prompt,
            analysis_tools_section=analysis_tools,
            analysis_strategy=analysis_strategy,
            max_iterations=worker_budget,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_traces_dict(
        self,
        question: str,
        agent_output: Optional[AgentOutput],
        ground_truth: Optional[str],
        feedback: Optional[str],
        trace_obj: Any,
    ) -> dict[str, Any]:
        """Build the canonical ``traces`` dict from individual parameters."""
        ao = agent_output
        return {
            "question": question,
            "ground_truth": ground_truth,
            "feedback": feedback,
            "steps": [
                {
                    "role": "agent",
                    "reasoning": ao.reasoning if ao else "",
                    "answer": ao.final_answer if ao else "",
                    "skill_ids": ao.skill_ids if ao else [],
                }
            ],
        }

    def _create_sandbox(
        self,
        trace_obj: Any,
        traces: Any,
        skillbook: Any,
    ) -> TraceSandbox:
        """Create a simplified sandbox for code execution."""
        sandbox = TraceSandbox(
            trace=trace_obj,
            llm_query_fn=None,
            parallel_max_concurrency=self.config.local_parallel_max_concurrency,
            parallel_timeout=self.config.local_parallel_timeout,
        )

        # Skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if isinstance(skillbook, str):
                skillbook_text = skillbook
            elif hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)
        sandbox.inject("skillbook", skillbook_text or "(empty skillbook)")

        # Traces data
        sandbox.inject("traces", traces)
        self._inject_helper_variables(sandbox, traces)

        return sandbox

    def _looks_like_combined_steps_batch(self, traces: dict[str, Any]) -> bool:
        """Return True for legacy combined batches encoded inside ``steps``."""
        steps = traces.get("steps")
        if not isinstance(steps, list) or not steps:
            return False

        return all(
            isinstance(step, dict)
            and step.get("role") == "conversation"
            and isinstance(step.get("content"), dict)
            for step in steps
        )

    def _get_batch_items(self, traces: Any) -> list[Any] | None:
        """Return batch items from a raw list or common batch container shapes."""
        if isinstance(traces, list):
            return traces
        if not isinstance(traces, dict):
            return None
        for key in ("items", "tasks"):
            items = traces.get(key)
            if isinstance(items, list):
                return items
        if self._looks_like_combined_steps_batch(traces):
            return traces.get("steps")
        return None

    def _extract_batch_item_payload(self, item: Any) -> Any:
        """Return the analysis payload for a batch item without rewriting it."""
        if (
            isinstance(item, dict)
            and item.get("role") == "conversation"
            and isinstance(item.get("content"), dict)
        ):
            return item["content"]
        return item

    def _get_batch_item_id(self, item: Any, index: int) -> str:
        """Choose a stable identifier for a batch item."""
        payload = self._extract_batch_item_payload(item)
        if isinstance(item, dict):
            for key in ("item_id", "task_id", "id"):
                value = item.get(key)
                if value is not None:
                    return str(value)
        if isinstance(payload, dict):
            for key in ("item_id", "task_id", "id"):
                value = payload.get(key)
                if value is not None:
                    return str(value)
        return f"item_{index}"

    def _extract_batch_messages(self, item: Any) -> list[Any]:
        """Return a best-effort message/step list for previews and summaries."""
        payload = self._extract_batch_item_payload(item)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            trace_value = payload.get("trace")
            if isinstance(trace_value, list):
                return trace_value
            if isinstance(trace_value, dict):
                for key in ("messages", "steps", "trace"):
                    nested = trace_value.get(key)
                    if isinstance(nested, list):
                        return nested
            for key in ("steps", "messages"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _extract_batch_field(self, item: Any, field: str) -> str:
        """Extract a string field from a batch item payload when present."""
        payload = self._extract_batch_item_payload(item)
        if isinstance(payload, dict):
            value = payload.get(field)
            if value is not None:
                return str(value)
        return ""

    def _inject_helper_variables(
        self,
        sandbox: TraceSandbox,
        traces: Any,
    ) -> None:
        """Inject generic helper data for raw batch traces."""
        batch_items = self._get_batch_items(traces)
        if batch_items is None:
            return

        item_ids = [
            self._get_batch_item_id(item, index)
            for index, item in enumerate(batch_items)
        ]
        item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
        item_preview_by_id = {
            item_id: self._build_item_preview(item)
            for item_id, item in zip(item_ids, batch_items)
        }
        survey_items = self._build_survey_items(batch_items)

        sandbox.inject("batch_items", batch_items)
        sandbox.inject("item_ids", item_ids)
        sandbox.inject("item_id_to_index", item_id_to_index)
        sandbox.inject("item_preview_by_id", item_preview_by_id)
        sandbox.inject("survey_items", survey_items)

    def _build_item_preview(self, item: Any) -> dict[str, Any]:
        """Build a compact preview row for a generic batch item."""
        item_trace = self._extract_batch_messages(item)
        first_message = ""
        if item_trace and isinstance(item_trace[0], dict):
            first_message = str(item_trace[0].get("content", ""))
        elif item_trace:
            first_message = str(item_trace[0])

        feedback = self._extract_batch_field(item, "feedback")
        question = self._extract_batch_field(item, "question")
        payload_type = type(self._extract_batch_item_payload(item)).__name__

        return {
            "question_preview": _preview(question, 120),
            "feedback_preview": _preview(feedback, 120),
            "first_message_preview": _preview(first_message, 120),
            "message_count": len(item_trace) if isinstance(item_trace, list) else 0,
            "payload_type": payload_type,
        }

    def _build_survey_items(self, batch_items: list[Any]) -> list[str]:
        """Precompute explicit batch_analyze items for grouped item survey."""
        survey_items: list[str] = []
        group_size = 3
        for start in range(0, len(batch_items), group_size):
            stop = min(start + group_size, len(batch_items))
            refs = []
            for index in range(start, stop):
                item_id = self._get_batch_item_id(batch_items[index], index)
                refs.append(f"batch_items[{index}] (item_id='{item_id}')")
            survey_items.append("Inspect " + ", ".join(refs))
        return survey_items

    def _build_data_summary(self, traces: Any) -> str:
        """Pre-compute a data summary so the agent doesn't waste calls exploring structure."""
        batch_items = self._get_batch_items(traces)
        is_batch = batch_items is not None

        if is_batch:
            assert batch_items is not None
            survey_items = self._build_survey_items(batch_items)
            # Compute pass/fail breakdown
            pass_count = 0
            fail_count = 0
            item_summaries = []
            for index, item in enumerate(batch_items):
                item_id = self._get_batch_item_id(item, index)
                trace = self._extract_batch_messages(item)
                feedback = self._extract_batch_field(item, "feedback")
                reward_str = ""
                if "reward=1.0" in str(feedback) or "PASSED" in str(feedback).upper():
                    pass_count += 1
                    reward_str = "PASS"
                elif "reward=0.0" in str(feedback) or "FAILED" in str(feedback).upper():
                    fail_count += 1
                    reward_str = "FAIL"
                item_summaries.append(
                    f"  {item_id}: {reward_str}, {len(trace)} messages"
                )

            lines = [
                "### Data Summary (pre-computed)",
                f"- **{len(batch_items)} batch items**: {pass_count} PASS, {fail_count} FAIL",
                f"- Item list:",
            ]
            # Show all tasks compactly
            lines.extend(item_summaries[:50])  # cap at 50
            if len(item_summaries) > 50:
                lines.append(f"  ... and {len(item_summaries) - 50} more")
            if survey_items:
                lines.append(
                    "- **Precomputed survey_items**: pass these strings directly "
                    "to `batch_analyze`; they already reference `batch_items[i]`."
                )
                lines.extend(f"  - {item}" for item in survey_items[:10])
                if len(survey_items) > 10:
                    lines.append(f"  - ... and {len(survey_items) - 10} more groups")

            # Check for policy/rules in the first batch item
            if batch_items:
                first_trace = self._extract_batch_messages(batch_items[0])
                for msg in first_trace[:3]:
                    content = (
                        str(msg.get("content", ""))
                        if isinstance(msg, dict)
                        else str(msg)
                    )
                    if len(content) > 500 and any(
                        kw in content.lower()
                        for kw in [
                            "policy",
                            "rule",
                            "instruction",
                            "you must",
                            "you should",
                        ]
                    ):
                        lines.append(
                            "- **Potential embedded policy/rules** detected in early "
                            f"content ({len(content)} chars) — inspect via helpers "
                            "or a small `execute_code` call."
                        )
                        break

            return "\n".join(lines)

        else:
            # Single trace
            if not isinstance(traces, dict):
                return "\n".join(
                    [
                        "### Data Summary (pre-computed)",
                        f"- **Trace type**: {type(traces).__name__}",
                        f'- **Preview**: "{_preview(str(traces), 200)}"',
                    ]
                )

            steps = traces.get("steps", [])
            question = traces.get("question", "")
            feedback = traces.get("feedback", "")
            ground_truth = traces.get("ground_truth", "")

            lines = ["### Data Summary (pre-computed)"]
            if feedback:
                lines.append(f"- **Feedback**: {_preview(feedback, 200)}")
            if ground_truth:
                lines.append(f"- **Ground truth**: {_preview(ground_truth, 200)}")
            lines.append(f"- **Steps**: {len(steps)}")
            if question:
                lines.append(f"- **Task**: {_preview(question, 200)}")

            # Check for messages in trace
            messages = traces.get("messages", [])
            if messages:
                lines.append(f"- **Messages**: {len(messages)} conversation turns")
                # Count tool calls
                tool_calls = sum(
                    1 for m in messages if isinstance(m, dict) and m.get("tool_calls")
                )
                if tool_calls:
                    lines.append(f"- **Tool calls**: {tool_calls}")

            return "\n".join(lines)

    def _build_initial_prompt(
        self,
        traces: Any,
        skillbook: Any,
        trace_obj: Any,
    ) -> str:
        """Format the prompt template with previews and metadata."""
        batch_items = self._get_batch_items(traces)
        is_batch = batch_items is not None
        t_steps = (
            traces.get("steps", []) if isinstance(traces, dict) and not is_batch else []
        )

        trace_size_chars = len(_json.dumps(traces, default=str))

        skillbook_text = ""
        if skillbook is not None:
            if isinstance(skillbook, str):
                skillbook_text = skillbook
            elif hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        if is_batch:
            assert batch_items is not None
            total_steps = sum(
                len(self._extract_batch_messages(item)) for item in batch_items
            )
            preview_rows = []
            for index, item in enumerate(batch_items):
                item_id = self._get_batch_item_id(item, index)
                tr = self._extract_batch_messages(item)
                first_msg = ""
                if tr and isinstance(tr[0], dict):
                    first_msg = tr[0].get("content", "")
                elif tr:
                    first_msg = str(tr[0])
                preview_rows.append(
                    f"| `{item_id}` | {len(tr)} messages | "
                    f'"{_preview(first_msg, 80)}" |'
                )

            if isinstance(traces, list):
                traces_description = f"List of {len(batch_items)} raw trace items"
            elif isinstance(traces, dict):
                traces_description = (
                    f"Batch container with keys {sorted(traces.keys())}. "
                    "Use the injected `batch_items` helper list for iteration."
                )
            else:
                traces_description = (
                    f"Batch-like trace container of type {type(traces).__name__}"
                )

            fmt_kwargs = dict(
                traces_description=traces_description,
                batch_variables=(
                    f"| `batch_items` | Ordered batch view over the raw trace items | "
                    f"{len(batch_items)} items |\n"
                ),
                helper_variables=(
                    "| `helper_registry` | Registered reusable helper functions and descriptions | dynamic |\n"
                    "| `register_helper` | Persist helper code for this run and sub-agent snapshots | callable |\n"
                    "| `list_helpers()` | List registered helper names/descriptions | callable |\n"
                    "| `run_helper(name, *args, **kwargs)` | Invoke a registered helper by name | callable |\n"
                    "| `get_batch_item(index)` | Convenience accessor for `batch_items[index]` | callable |\n"
                    "| `get_item_payload(item_or_index)` | Normalize a batch item or index to its payload dict/list | callable |\n"
                    "| `get_item_messages(item_or_index)` | Return the best-effort message list for a batch item | callable |\n"
                    "| `get_item_question(item_or_index)` | Return the question string for a batch item | callable |\n"
                    "| `get_item_feedback(item_or_index)` | Return the feedback string for a batch item | callable |\n"
                    "| `get_item_id(item_or_index)` | Return the stable item identifier | callable |\n"
                    "| `get_message_text(message)` | Safely render message content/tool metadata as text | callable |\n"
                    "| `preview_item(item_or_index)` | Compact summary for one batch item | callable |\n"
                    f"| `item_ids` | Ordered item ids for this batch | {len(batch_items)} ids |\n"
                    f"| `item_id_to_index` | Maps item id to `batch_items[i]` | {len(batch_items)} entries |\n"
                    f"| `item_preview_by_id` | Compact previews per batch item | {len(batch_items)} entries |\n"
                    f"| `survey_items` | Precomputed `batch_analyze` items with explicit item references | {len(self._build_survey_items(batch_items))} items |\n"
                ),
                traces_previews=(
                    f"| Item | Steps | First message |\n"
                    f"|------|-------|---------------|\n" + "\n".join(preview_rows)
                ),
                step_count=total_steps,
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.config.max_llm_calls,
                task_count=len(batch_items),
                data_summary=self._build_data_summary(traces),
            )
        else:
            single_trace_type = type(traces).__name__
            if not isinstance(traces, dict):
                fmt_kwargs = dict(
                    traces_description=f"Raw trace object of type {single_trace_type}",
                    batch_variables="",
                    helper_variables=(
                        "| `helper_registry` | Registered reusable helper functions and descriptions | dynamic |\n"
                        "| `register_helper` | Persist helper code for this run and sub-agent snapshots | callable |\n"
                        "| `list_helpers()` | List registered helper names/descriptions | callable |\n"
                        "| `run_helper(name, *args, **kwargs)` | Invoke a registered helper by name | callable |\n"
                        "| `get_message_text(message)` | Safely render message content/tool metadata as text | callable |\n"
                    ),
                    traces_previews=(
                        "| Field | Preview | Size |\n"
                        "|-------|---------|------|\n"
                        f'| raw trace | "{_preview(str(traces), 120)}" | {len(str(traces))} chars |'
                    ),
                    step_count=0,
                    skillbook_length=len(skillbook_text),
                    trace_size_chars=trace_size_chars,
                    max_iterations=self.config.max_llm_calls,
                    task_count=1,
                    data_summary=self._build_data_summary(traces),
                )
                return self.prompt_template.format(**fmt_kwargs)

            t_question = traces.get("question", "")
            t_ground_truth = traces.get("ground_truth")
            t_feedback = traces.get("feedback")
            first_agent: dict[str, str] = next(
                (s for s in t_steps if s.get("role") == "agent"), {}
            )
            t_reasoning = first_agent.get("reasoning", "")

            fmt_kwargs = dict(
                traces_description=(
                    "Dict with keys: question, ground_truth, feedback, "
                    "steps (List[Dict])"
                ),
                batch_variables="",
                helper_variables=(
                    "| `helper_registry` | Registered reusable helper functions and descriptions | dynamic |\n"
                    "| `register_helper` | Persist helper code for this run and sub-agent snapshots | callable |\n"
                    "| `list_helpers()` | List registered helper names/descriptions | callable |\n"
                    "| `run_helper(name, *args, **kwargs)` | Invoke a registered helper by name | callable |\n"
                ),
                traces_previews=(
                    f"| Field | Preview | Size |\n"
                    f"|-------|---------|------|\n"
                    f'| `traces["question"]` | "{_preview(t_question)}" '
                    f"| {len(t_question)} chars |\n"
                    f'| first step | "{_preview(t_reasoning)}..." '
                    f"| {len(t_reasoning) if t_reasoning else 0} chars |\n"
                    f'| `traces["ground_truth"]` | "{_preview(t_ground_truth)}" '
                    f"| {len(t_ground_truth) if t_ground_truth else 0} chars |\n"
                    f'| `traces["feedback"]` | "{_preview(t_feedback)}..." '
                    f"| {len(t_feedback) if t_feedback else 0} chars |"
                ),
                step_count=(
                    len(t_steps) if t_steps else (len(trace_obj) if trace_obj else 0)
                ),
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.config.max_llm_calls,
                task_count=1,
                data_summary=self._build_data_summary(traces),
            )

        return self.prompt_template.format(**fmt_kwargs)

    # ------------------------------------------------------------------
    # Timeout / error fallback (single-trace only)
    # ------------------------------------------------------------------

    def _build_timeout_output(
        self,
        question: str,
        agent_output: Optional[AgentOutput],
        ground_truth: Optional[str],
        feedback: Optional[str],
        deps: RRDeps,
    ) -> ReflectorOutput:
        """Build a ReflectorOutput when usage limits are reached."""
        is_correct = False
        if ground_truth and agent_output:
            is_correct = (
                agent_output.final_answer.strip().lower()
                == ground_truth.strip().lower()
            )

        return ReflectorOutput(
            reasoning=(
                f"Recursive analysis reached the main-agent usage limit "
                f"({self.config.max_llm_calls} requests). "
                f"Basic analysis: Answer was "
                f"{'correct' if is_correct else 'incorrect'}."
            ),
            error_identification="timeout" if not is_correct else "none",
            root_cause_analysis="Analysis incomplete due to request limit",
            correct_approach=(
                "Consider increasing max_llm_calls or simplifying the analysis"
            ),
            key_insight=(
                "Complex traces may require more requests for thorough analysis"
            ),
            extracted_learnings=[
                ExtractedLearning(
                    learning="Usage limit reached during recursive analysis",
                    atomicity_score=0.5,
                )
            ],
            skill_tags=[],
            raw={
                "timeout": True,
                "max_llm_calls": self.config.max_llm_calls,
                "max_llm_calls_scope": "main_agent_session",
                "question": question,
                "feedback": feedback,
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": deps.sub_agent_history,
                    "timed_out": True,
                },
            },
        )
