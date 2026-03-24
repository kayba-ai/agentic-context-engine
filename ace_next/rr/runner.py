"""RRStep — Recursive Reflector powered by PydanticAI.

Uses a PydanticAI agent with ``execute_code``, ``analyze``, and
``batch_analyze`` tools.  The agent explores traces via code execution
and sub-agent analysis, then produces ``ReflectorOutput`` as structured
output.

Satisfies both ``StepProtocol`` (for Pipeline composition) and
``ReflectorLike`` (drop-in replacement for simple Reflector).
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Optional

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ace_next.core.context import ACEStepContext
from ace_next.core.outputs import AgentOutput, ExtractedLearning, ReflectorOutput

from .agent import RRDeps, create_rr_agent, create_sub_agent
from .config import RecursiveConfig
from .prompts import REFLECTOR_RECURSIVE_PROMPT, REFLECTOR_RECURSIVE_SYSTEM
from .sandbox import TraceSandbox
from .trace_context import TraceContext

logger = logging.getLogger(__name__)


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

        # Build PydanticAI agents
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

    # ------------------------------------------------------------------
    # StepProtocol entry
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        """Run the Recursive Reflector and attach the reflection(s).

        When the trace is a batch dict (has a ``"tasks"`` key), a single
        session analyzes all tasks.  Per-task results are parsed from
        the structured output.
        """
        trace = ctx.trace or {}
        if isinstance(trace, dict) and "tasks" in trace:
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
        """ReflectorLike — delegates to the PydanticAI agent.

        Allows RRStep to be used as a drop-in replacement for Reflector
        in any runner or learning_tail pipeline.
        """
        return self._run_reflection(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core reflection logic
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

        # Build sandbox with trace data (no ask_llm/FINAL injection)
        sandbox = self._create_sandbox(trace_obj, traces, skillbook)

        # Resolve skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)

        # Build deps for PydanticAI agent
        deps = RRDeps(
            sandbox=sandbox,
            trace_data=traces,
            skillbook_text=skillbook_text or "(empty skillbook)",
            config=self.config,
            sub_agent=self._sub_agent,
        )

        # Build user prompt
        initial_prompt = self._build_initial_prompt(traces, skillbook, trace_obj)

        # Run the PydanticAI agent with usage limits
        usage_limits = UsageLimits(request_limit=self.config.max_llm_calls)

        try:
            result = self._agent.run_sync(
                initial_prompt,
                deps=deps,
                usage_limits=usage_limits,
            )
            output = result.output

            # Merge execution metadata into raw (preserve LLM-populated fields)
            usage = result.usage()
            output.raw = {
                **output.raw,
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
                "RR completed: %d tool calls, %d sub-agent calls",
                deps.iteration,
                len(deps.sub_agent_history),
            )

            return output

        except UsageLimitExceeded:
            logger.warning(
                "RR usage limit reached (%d requests)",
                self.config.max_llm_calls,
            )
            return self._build_timeout_output(
                question, agent_output, ground_truth, feedback, deps
            )
        except Exception as e:
            logger.error("RR agent failed: %s", e, exc_info=True)
            return ReflectorOutput(
                reasoning=f"Recursive analysis failed: {e}",
                correct_approach="",
                key_insight="",
                raw={"error": str(e)},
            )

    # ------------------------------------------------------------------
    # Batch reflection
    # ------------------------------------------------------------------

    def _run_batch_reflections(
        self,
        batch_trace: dict[str, Any],
        skillbook: Any,
    ) -> tuple[ReflectorOutput, ...]:
        """Run a single session for all tasks in a batch."""
        tasks = batch_trace["tasks"]

        reflection = self._run_reflection(
            traces=batch_trace,
            skillbook=skillbook,
        )

        return self._split_batch_reflection(reflection, tasks)

    def _split_batch_reflection(
        self,
        reflection: ReflectorOutput,
        tasks: list[dict[str, Any]],
    ) -> tuple[ReflectorOutput, ...]:
        """Extract per-task ReflectorOutputs from batch output."""
        task_results = reflection.raw.get("tasks", [])

        if not task_results or len(task_results) != len(tasks):
            logger.warning(
                "Batch missing per-task results (got %d, expected %d); "
                "duplicating single reflection",
                len(task_results) if task_results else 0,
                len(tasks),
            )
            fallback = []
            for i, task in enumerate(tasks):
                r = reflection.model_copy(deep=True)
                r.raw["task_id"] = task.get("task_id", f"task_{i}")
                fallback.append(r)
            return tuple(fallback)

        reflections: list[ReflectorOutput] = []
        rr_trace = reflection.raw.get("rr_trace", {})
        for i, (task, tr) in enumerate(zip(tasks, task_results)):
            task_id = task.get("task_id", f"task_{i}")
            if not isinstance(tr, dict):
                tr = {}
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
                        "task_id": task_id,
                        "rr_trace": rr_trace,
                    },
                )
            )
        return tuple(reflections)

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
        traces: dict[str, Any],
        skillbook: Any,
    ) -> TraceSandbox:
        """Create a simplified sandbox for code execution.

        Unlike the old RRStep, does NOT inject ask_llm, FINAL, or
        parallel_map — those are now PydanticAI tools.
        """
        sandbox = TraceSandbox(trace=trace_obj, llm_query_fn=None)

        # Skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)
        sandbox.inject("skillbook", skillbook_text or "(empty skillbook)")

        # Traces data
        sandbox.inject("traces", traces)

        return sandbox

    def _build_initial_prompt(
        self,
        traces: dict[str, Any],
        skillbook: Any,
        trace_obj: Any,
    ) -> str:
        """Format the prompt template with previews and metadata."""
        is_batch = isinstance(traces, dict) and "tasks" in traces
        t_steps = traces.get("steps", []) if not is_batch else []

        trace_size_chars = len(_json.dumps(traces, default=str))

        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        if is_batch:
            tasks = traces["tasks"]
            total_steps = sum(len(t.get("trace", [])) for t in tasks)
            preview_rows = []
            for t in tasks:
                tid = t.get("task_id", "?")
                tr = t.get("trace", [])
                first_msg = ""
                if tr and isinstance(tr[0], dict):
                    first_msg = tr[0].get("content", "")
                preview_rows.append(
                    f"| `{tid}` | {len(tr)} messages | "
                    f'"{_preview(first_msg, 80)}" |'
                )

            fmt_kwargs = dict(
                traces_description=(
                    f"Batch of {len(tasks)} tasks. "
                    f"Keys: tasks (list of {{task_id, trace}})"
                ),
                batch_variables=(
                    f'| `traces["tasks"]` | All tasks in this batch '
                    f"(list of {{task_id, trace}}) | "
                    f"{len(tasks)} tasks |\n"
                ),
                traces_previews=(
                    f"| Task | Steps | First message |\n"
                    f"|------|-------|---------------|\n"
                    + "\n".join(preview_rows)
                ),
                step_count=total_steps,
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.config.max_llm_calls,
                task_count=len(tasks),
            )
        else:
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
                    len(t_steps) if t_steps
                    else (len(trace_obj) if trace_obj else 0)
                ),
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.config.max_llm_calls,
                task_count=1,
            )

        return self.prompt_template.format(**fmt_kwargs)

    # ------------------------------------------------------------------
    # Timeout / error fallback
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
                f"Recursive analysis reached usage limit "
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
                "question": question,
                "feedback": feedback,
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": deps.sub_agent_history,
                    "timed_out": True,
                },
            },
        )
