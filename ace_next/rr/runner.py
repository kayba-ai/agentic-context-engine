"""RRStep — Recursive Reflector as a pipeline step (SubRunner pattern).

Extends :class:`~ace_next.core.sub_runner.SubRunner` to implement the
iterative REPL loop.  Uses a Pipeline internally for per-iteration step
execution and exposes itself as a StepProtocol step for seamless
composition into the ACE pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from pipeline import Pipeline
from pipeline.context import StepContext

from ace_next.core.sub_runner import SubRunner

from ace.reflector.config import RecursiveConfig
from ace.reflector.prompts_rr_v3 import REFLECTOR_RECURSIVE_V3_PROMPT
from ace.reflector.sandbox import TraceSandbox
from ace.reflector.subagent import CallBudget, SubAgentConfig, create_ask_llm_function
from ace.reflector.trace_context import TraceContext

from ace_next.core.context import ACEStepContext
from ace_next.core.outputs import ExtractedLearning, ReflectorOutput

from .context import RRIterationContext
from .steps import LLMCallStep, ExtractCodeStep, SandboxExecStep, CheckResultStep

if TYPE_CHECKING:
    from ace_next.core.outputs import AgentOutput

logger = logging.getLogger(__name__)


def _preview(text: str | None, max_len: int = 150) -> str:
    """Return a short preview safe for str.format()."""
    if not text:
        return "(empty)"
    snippet = text if len(text) <= max_len else text[:max_len]
    return snippet.replace("{", "{{").replace("}", "}}")


class RRStep(SubRunner):
    """Recursive Reflector as a pipeline step (SubRunner pattern).

    Satisfies **StepProtocol** — can be placed directly in a Pipeline.
    Also satisfies **ReflectorLike** — can be passed to ``ReflectStep``
    or used standalone via ``reflect()``.

    Internally builds a ``Pipeline([LLMCallStep, ExtractCodeStep,
    SandboxExecStep, CheckResultStep])`` and calls it once per REPL
    iteration until the LLM calls ``FINAL()`` or the iteration budget
    is exhausted.
    """

    # StepProtocol
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    def __init__(
        self,
        llm: Any,
        config: Optional[RecursiveConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_V3_PROMPT,
        subagent_llm: Any = None,
    ) -> None:
        cfg = config or RecursiveConfig()
        super().__init__(max_iterations=cfg.max_iterations)
        self.llm = llm
        self.config = cfg
        self.prompt_template = prompt_template
        self.subagent_llm = subagent_llm

    # ------------------------------------------------------------------
    # SubRunner template methods
    # ------------------------------------------------------------------

    def _build_inner_pipeline(self, **kwargs: Any) -> Pipeline:
        """Build a fresh inner Pipeline for this call.

        Steps hold mutable state (sandbox, budget), so a new pipeline
        is created per ``run_loop`` invocation.
        """
        sandbox: TraceSandbox = kwargs["sandbox"]
        budget: CallBudget = kwargs["budget"]
        return Pipeline(
            [
                LLMCallStep(self.llm, self.config, budget),
                ExtractCodeStep(),
                SandboxExecStep(sandbox, self.config),
                CheckResultStep(sandbox, self.config),
            ]
        )

    def _build_initial_context(self, **kwargs: Any) -> StepContext:
        initial_prompt: str = kwargs["initial_prompt"]
        messages: tuple[dict[str, str], ...] = (
            {"role": "user", "content": initial_prompt},
        )
        return RRIterationContext(messages=messages, iteration=0)

    def _is_done(self, ctx: StepContext) -> bool:
        return getattr(ctx, "terminated", False)

    def _extract_result(self, ctx: StepContext) -> Any:
        return getattr(ctx, "reflection", None)

    def _accumulate(self, ctx: StepContext) -> StepContext:
        rr_ctx: RRIterationContext = ctx  # type: ignore[assignment]
        messages = rr_ctx.messages + rr_ctx.feedback_messages
        return RRIterationContext(messages=messages, iteration=rr_ctx.iteration + 1)

    def _on_timeout(self, last_ctx: StepContext, iteration: int, **kwargs: Any) -> Any:
        """Build a fallback ReflectorOutput when max iterations is reached.

        Timeout args are passed through ``run_loop(**kwargs)`` — no
        instance-level stashing required, eliminating the race condition
        when ``run_loop`` is called directly or concurrently.
        """
        logger.warning("Max iterations (%d) reached", self.max_iterations)
        args = kwargs.get("timeout_args", {})
        return self._build_timeout_output(
            args.get("question", ""),
            args.get("agent_output"),
            args.get("ground_truth"),
            args.get("feedback"),
            args.get("messages"),
        )

    # ------------------------------------------------------------------
    # StepProtocol entry
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:  # type: ignore[override]
        """Run the Recursive Reflector and attach the reflection to *ctx*."""
        reflection = self.reflect(
            trace=ctx.trace,
            skillbook=ctx.skillbook,
        )
        return ctx.replace(reflection=reflection)

    # ------------------------------------------------------------------
    # ReflectorLike entry (also usable standalone)
    # ------------------------------------------------------------------

    def reflect(
        self,
        *,
        question: str = "",
        agent_output: Optional[AgentOutput] = None,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Run the recursive REPL loop and return analysis.

        Accepts the same signature as ``ReflectorLike.reflect()`` so it
        can be used as a drop-in replacement.  When called from
        ``__call__`` (StepProtocol), only *trace* and *skillbook* are
        provided via kwargs.
        """
        # Allow passing trace/skillbook via kwargs (from __call__)
        trace_obj = kwargs.pop("trace", None)
        if trace_obj is None and agent_output is not None:
            trace_obj = getattr(agent_output, "trace_context", None)
            if trace_obj is None:
                trace_obj = TraceContext.from_agent_output(agent_output)  # type: ignore[arg-type]

        # Build traces dict — the canonical data structure for sandbox code
        traces = kwargs.pop("traces", None)
        if traces is None:
            traces = self._build_traces_dict(
                question, agent_output, ground_truth, feedback, trace_obj
            )

        # Setup
        sandbox = self._create_sandbox(trace_obj, traces, skillbook, **kwargs)
        budget = CallBudget(self.config.max_llm_calls)
        initial_prompt = self._build_initial_prompt(traces, skillbook, trace_obj)

        timeout_args = {
            "question": question,
            "agent_output": agent_output,
            "ground_truth": ground_truth,
            "feedback": feedback,
        }

        result = self.run_loop(
            sandbox=sandbox,
            budget=budget,
            initial_prompt=initial_prompt,
            timeout_args=timeout_args,
        )
        return result  # type: ignore[return-value]

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
        **kwargs: Any,
    ) -> TraceSandbox:
        """Create and configure the TraceSandbox."""
        sandbox = TraceSandbox(trace=trace_obj, llm_query_fn=None)

        # ask_llm sub-agent
        if self.config.enable_subagent:
            subagent_config = SubAgentConfig(
                model=self.config.subagent_model,
                max_tokens=self.config.subagent_max_tokens,
                temperature=self.config.subagent_temperature,
                system_prompt=self.config.subagent_system_prompt
                or SubAgentConfig.system_prompt,
            )
            ask_llm_fn = create_ask_llm_function(
                llm=self.llm,
                config=subagent_config,
                subagent_llm=self.subagent_llm,
                budget=CallBudget(self.config.max_llm_calls),
            )
        else:

            def ask_llm_fn(question: str, context: str = "") -> str:
                return "(ask_llm disabled - analyze with code)"

        sandbox.inject("ask_llm", ask_llm_fn)
        sandbox.inject("llm_query", lambda prompt: ask_llm_fn(prompt, ""))

        # Skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)
        sandbox.inject("skillbook", skillbook_text or "(empty skillbook)")

        # Traces
        sandbox.inject("traces", traces)

        return sandbox

    def _build_initial_prompt(
        self,
        traces: dict[str, Any],
        skillbook: Any,
        trace_obj: Any,
    ) -> str:
        """Format the prompt template with previews and metadata."""
        t_question = traces.get("question", "")
        t_ground_truth = traces.get("ground_truth")
        t_feedback = traces.get("feedback")
        t_steps = traces.get("steps", [])
        first_agent: dict[str, str] = next(
            (s for s in t_steps if s.get("role") == "agent"), {}
        )
        t_reasoning = first_agent.get("reasoning", "")
        t_answer = first_agent.get("answer", "")

        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        return self.prompt_template.format(
            question_length=len(t_question),
            question_preview=_preview(t_question),
            reasoning_length=len(t_reasoning) if t_reasoning else 0,
            reasoning_preview=_preview(t_reasoning),
            answer_length=len(t_answer) if t_answer else 0,
            answer_preview=_preview(t_answer),
            ground_truth_length=len(t_ground_truth) if t_ground_truth else 0,
            ground_truth_preview=_preview(t_ground_truth),
            feedback_length=len(t_feedback) if t_feedback else 0,
            feedback_preview=_preview(t_feedback),
            skillbook_length=len(skillbook_text),
            step_count=(
                len(t_steps) if t_steps else (len(trace_obj) if trace_obj else 0)
            ),
        )

    # ------------------------------------------------------------------
    # Timeout fallback
    # ------------------------------------------------------------------

    def _build_timeout_output(
        self,
        question: str,
        agent_output: Optional[AgentOutput],
        ground_truth: Optional[str],
        feedback: Optional[str],
        messages: Optional[list[dict[str, str]]] = None,
    ) -> ReflectorOutput:
        """Build a ReflectorOutput when max iterations is reached.

        Optionally attempts fallback synthesis if enabled.
        """
        if self.config.enable_fallback_synthesis and messages and len(messages) > 1:
            try:
                synthesized = self._attempt_fallback_synthesis(messages)
                if synthesized is not None:
                    logger.info("Fallback synthesis succeeded after timeout")
                    return synthesized
            except Exception as e:
                logger.warning("Fallback synthesis failed: %s", e)

        is_correct = False
        if ground_truth and agent_output:
            is_correct = (
                agent_output.final_answer.strip().lower()
                == ground_truth.strip().lower()
            )

        return ReflectorOutput(
            reasoning=(
                f"Recursive analysis reached max iterations "
                f"({self.config.max_iterations}). "
                f"Basic analysis: Answer was "
                f"{'correct' if is_correct else 'incorrect'}."
            ),
            error_identification="timeout" if not is_correct else "none",
            root_cause_analysis="Analysis incomplete due to iteration limit",
            correct_approach="Consider increasing max_iterations or simplifying the analysis",
            key_insight="Complex traces may require more iterations for thorough analysis",
            extracted_learnings=[
                ExtractedLearning(
                    learning="Timeout occurred during recursive analysis",
                    atomicity_score=0.5,
                )
            ],
            skill_tags=[],
            raw={
                "timeout": True,
                "max_iterations": self.config.max_iterations,
                "question": question,
                "feedback": feedback,
            },
        )

    def _attempt_fallback_synthesis(
        self, messages: list[dict[str, str]]
    ) -> Optional[ReflectorOutput]:
        """Attempt to synthesize a final answer from conversation history."""
        from .steps import _parse_final_value, _parse_direct_response
        from .code_extraction import extract_code

        synthesis_prompt = (
            "Your analysis timed out before calling FINAL().\n"
            "Based on your exploration so far, provide your final structured output now.\n\n"
            "Call FINAL() with your best assessment using the evidence you gathered.\n"
            "Include any learnings from the patterns you observed, even if the analysis "
            "was incomplete.\n\n"
            "If you found no significant insights, call FINAL() with empty "
            "extracted_learnings and a brief summary."
        )

        synthesis_messages = messages.copy()
        synthesis_messages.append({"role": "user", "content": synthesis_prompt})

        response = self.llm.complete_messages(synthesis_messages)
        response_text = response.text

        code = extract_code(response_text)
        if code and "FINAL(" in code:
            sandbox = TraceSandbox(trace=None)
            sandbox.execute(code, timeout=10.0)
            if sandbox.final_called:
                return _parse_final_value(sandbox.final_value)

        try:
            return _parse_direct_response(response_text)
        except Exception:
            pass

        return None
