"""RRStep — Recursive Reflector pipeline step.

Subclass of :class:`RecursiveAgent` that satisfies both ``StepProtocol``
and ``ReflectorLike``.  Adds RR-specific trace setup, prompt building,
and timeout handling on top of the generic recursive agent infrastructure.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Optional

from pydantic_ai.settings import ModelSettings

from ace.core.context import ACEStepContext
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.recursive_agent import (
    BudgetExhausted,
    RecursiveAgent,
)
from ace.core.sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox
from ace.implementations.rr.config import RecursiveConfig as RRConfig
from ace.implementations.rr.prompts import (
    COMPACTION_SUMMARY_PROMPT,
    REFLECTOR_RECURSIVE_PROMPT,
    REFLECTOR_RECURSIVE_SYSTEM,
    RR_SKILL_EVAL_SECTION,
)
from ace.implementations.rr.tools import (
    RRDeps,
    register_output_validator,
)

logger = logging.getLogger(__name__)


def _preview(text: str | None, max_len: int = 150) -> str:
    """Return a short preview safe for str.format()."""
    if not text:
        return "(empty)"
    snippet = text if len(text) <= max_len else text[:max_len]
    return snippet.replace("{", "{{").replace("}", "}}")


class RRStep(RecursiveAgent):
    """Recursive Reflector as a pipeline step.

    Satisfies **StepProtocol** (``requires``/``provides``) and
    **ReflectorLike** (``reflect`` method).

    Subclass of :class:`RecursiveAgent` — inherits compaction,
    recursion, and budget management.

    Args:
        model: LiteLLM or PydanticAI model string.
        config: RR configuration (timeouts, limits, sub-agent settings).
        prompt_template: User prompt template with format placeholders.
        model_settings: Override PydanticAI model settings.
    """

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __init__(
        self,
        model: str,
        config: Optional[RRConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.prompt_template = prompt_template

        super().__init__(
            model,
            output_type=ReflectorOutput,
            system_prompt=REFLECTOR_RECURSIVE_SYSTEM,
            config=config or RRConfig(),
            model_settings=model_settings,
            tools=[register_output_validator],
            tool_names_to_compact=("execute_code",),
            compaction_summary_prompt=COMPACTION_SUMMARY_PROMPT,
            compaction_continuation=(
                "Your conversation was compacted. "
                "All sandbox variables persist — use execute_code to re-inspect data. "
                "Do NOT repeat work already completed. Continue your analysis."
            ),
            microcompact_placeholder=(
                "[cleared — data still in sandbox variables, "
                "use execute_code to re-inspect]"
            ),
            on_compaction=RecursiveAgent.on_compaction,
        )

    # ------------------------------------------------------------------
    # StepProtocol
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        """Run the Recursive Reflector and attach the reflection."""
        trace = ctx.trace or {}
        reflection = self._run_reflection(
            traces=trace if isinstance(trace, dict) else None,
            question=trace.get("question", "") if isinstance(trace, dict) else "",
            ground_truth=trace.get("ground_truth") if isinstance(trace, dict) else None,
            feedback=trace.get("feedback") if isinstance(trace, dict) else None,
            skillbook=ctx.skillbook,
            trace=trace if not isinstance(trace, dict) else None,
            mode=ctx.mode,
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
        mode: str = "online",
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Run the PydanticAI agent and return analysis."""
        trace_obj = kwargs.pop("trace", None)
        if trace_obj is None and agent_output is not None:
            trace_obj = getattr(agent_output, "trace_context", None)

        traces = kwargs.pop("traces", None)
        if traces is None:
            traces = self._build_traces_dict(
                question, agent_output, ground_truth, feedback, trace_obj
            )

        sandbox = self._create_sandbox(trace_obj, traces, skillbook)

        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)

        deps = RRDeps(
            sandbox=sandbox,
            trace_data=traces,
            skillbook_text=skillbook_text or "(empty skillbook)",
            config=self.config,
            depth=0,
            max_depth=self.config.max_depth,
        )

        initial_prompt = self._build_initial_prompt(traces, skillbook)

        if mode == "online" and skillbook_text and skillbook_text != "(empty skillbook)":
            initial_prompt += "\n\n" + RR_SKILL_EVAL_SECTION

        remaining = (
            traces.get("_remaining_tokens")
            if isinstance(traces, dict)
            else None
        )
        try:
            output, metadata = self.run(
                deps=deps,
                prompt=initial_prompt,
                remaining_tokens=remaining,
            )
            output.raw = {
                **output.raw,
                **metadata,
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": [],
                    "timed_out": False,
                    "compactions": metadata.get("compactions", 0),
                    "depth": 0,
                },
            }
        except BudgetExhausted as exc:
            output = self._build_budget_exhausted_output(
                deps, exc.compaction_count, depth=0
            )
        except Exception as e:
            logger.error("RR agent failed: %s", e, exc_info=True)
            output = ReflectorOutput(
                reasoning=f"Recursive analysis failed: {e}",
                correct_approach="",
                key_insight="",
                raw={"error": str(e)},
            )

        if output.raw.get("timeout") and (ground_truth or agent_output):
            output = self._build_timeout_output(
                question, agent_output, ground_truth, feedback, deps
            )

        return output

    def _build_budget_exhausted_output(
        self, deps: RRDeps, compaction_count: int, depth: int
    ) -> ReflectorOutput:
        return ReflectorOutput(
            reasoning="Analysis reached budget limit.",
            error_identification="budget_exhausted",
            root_cause_analysis="Analysis incomplete due to token/request budget",
            correct_approach="Consider increasing budget or simplifying the analysis",
            key_insight="Session reached budget limit before completing",
            raw={
                "timeout": True,
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": [],
                    "timed_out": True,
                    "compactions": compaction_count,
                    "depth": depth,
                },
            },
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

    def _create_sandbox(self, trace_obj: Any, traces: Any, skillbook: Any):
        skillbook_text = ""
        if skillbook is not None:
            if isinstance(skillbook, str):
                skillbook_text = skillbook
            elif hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)

        return self.create_sandbox(
            trace=trace_obj,
            variables={
                "traces": traces,
                "skillbook": skillbook_text or "(empty skillbook)",
            },
        )

    def _build_data_summary(self, traces: Any) -> str:
        if not isinstance(traces, dict):
            return (
                f"### Data Summary\n"
                f"- **Trace type**: {type(traces).__name__}\n"
                f'- **Preview**: "{_preview(str(traces), 200)}"'
            )

        steps = traces.get("steps", [])
        question = traces.get("question", "")
        feedback = traces.get("feedback", "")
        ground_truth = traces.get("ground_truth", "")

        lines = ["### Data Summary"]
        if feedback:
            lines.append(f"- **Feedback**: {_preview(feedback, 200)}")
        if ground_truth:
            lines.append(f"- **Ground truth**: {_preview(ground_truth, 200)}")
        lines.append(f"- **Steps**: {len(steps)}")
        if question:
            lines.append(f"- **Task**: {_preview(question, 200)}")

        messages = traces.get("messages", [])
        if messages:
            lines.append(f"- **Messages**: {len(messages)} conversation turns")
            tool_calls = sum(
                1 for m in messages if isinstance(m, dict) and m.get("tool_calls")
            )
            if tool_calls:
                lines.append(f"- **Tool calls**: {tool_calls}")

        return "\n".join(lines)

    def _build_initial_prompt(self, traces: Any, skillbook: Any) -> str:
        trace_size_chars = len(_json.dumps(traces, default=str))

        skillbook_text = ""
        if skillbook is not None:
            if isinstance(skillbook, str):
                skillbook_text = skillbook
            elif hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        if isinstance(traces, dict):
            traces_description = (
                f"Dict with keys: {', '.join(sorted(traces.keys()))}"
            )
        elif isinstance(traces, list):
            traces_description = f"List of {len(traces)} items"
        else:
            traces_description = f"Object of type {type(traces).__name__}"

        return self.prompt_template.format(
            traces_description=traces_description,
            trace_size_chars=trace_size_chars,
            skillbook_length=len(skillbook_text),
            max_iterations=self.config.max_requests,
            data_summary=self._build_data_summary(traces),
        )

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
        is_correct = False
        if ground_truth and agent_output:
            is_correct = (
                agent_output.final_answer.strip().lower()
                == ground_truth.strip().lower()
            )

        return ReflectorOutput(
            reasoning=(
                f"Recursive analysis reached budget limit. "
                f"Basic analysis: Answer was "
                f"{'correct' if is_correct else 'incorrect'}."
            ),
            error_identification="timeout" if not is_correct else "none",
            root_cause_analysis="Analysis incomplete due to budget limit",
            correct_approach=(
                "Consider increasing budget or simplifying the analysis"
            ),
            key_insight=(
                "Complex traces may require more budget for thorough analysis"
            ),
            skill_tags=[],
            raw={
                "timeout": True,
                "question": question,
                "feedback": feedback,
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": [],
                    "timed_out": True,
                },
            },
        )


__all__ = [
    "RRConfig",
    "RRDeps",
    "RRStep",
    "ExecutionResult",
    "ExecutionTimeoutError",
    "TraceSandbox",
]
