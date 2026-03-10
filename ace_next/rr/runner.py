"""RRStep — Recursive Reflector as a pipeline step (SubRunner pattern).

Extends :class:`~ace_next.core.sub_runner.SubRunner` to implement the
iterative REPL loop.  Uses a Pipeline internally for per-iteration step
execution and exposes itself as a StepProtocol step for seamless
composition into the ACE pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pipeline import Pipeline
from pipeline.context import StepContext

from ace_next.core.sub_runner import SubRunner

from .config import RecursiveConfig
from .prompts import REFLECTOR_RECURSIVE_PROMPT
from .sandbox import TraceSandbox
from .subagent import CallBudget, SubAgentConfig, create_ask_llm_function
from .trace_context import TraceContext

from ace_next.core.context import ACEStepContext
from ace_next.core.outputs import AgentOutput, ExtractedLearning, ReflectorOutput

from .context import RRIterationContext
from .steps import LLMCallStep, ExtractCodeStep, SandboxExecStep, CheckResultStep

logger = logging.getLogger(__name__)


def _preview(text: str | None, max_len: int = 150) -> str:
    """Return a short preview safe for str.format()."""
    if not text:
        return "(empty)"
    snippet = text if len(text) <= max_len else text[:max_len]
    return snippet.replace("{", "{{").replace("}", "}}")


def _format_iteration_log(
    iteration: int,
    max_iterations: int,
    status: str,
    code: str | None,
    stdout: str | None,
    stderr: str | None,
) -> str:
    """Build a clean, multi-line log message for an RR iteration."""
    is_final = status == "FINAL"
    icon = "✓" if is_final else "→"
    label = f"FINAL" if is_final else f"iter"

    header = f"[RR {label} {iteration}/{max_iterations}] {icon}"

    lines = [header]

    # Code preview — show first 2 meaningful lines
    if code:
        code_lines = [l for l in code.strip().splitlines() if l.strip()][:2]
        if code_lines:
            lines.append(f"  code  │ {code_lines[0][:100]}")
            for cl in code_lines[1:]:
                lines.append(f"        │ {cl[:100]}")
            total_code_lines = len(code.strip().splitlines())
            if total_code_lines > 2:
                lines.append(f"        │ … ({total_code_lines - 2} more lines)")

    # Stdout preview — show first 3 lines
    if stdout:
        out_lines = [l for l in stdout.strip().splitlines() if l.strip()][:3]
        if out_lines:
            lines.append(f"  out   │ {out_lines[0][:120]}")
            for ol in out_lines[1:]:
                lines.append(f"        │ {ol[:120]}")
            total_out_lines = len(stdout.strip().splitlines())
            if total_out_lines > 3:
                lines.append(f"        │ … ({total_out_lines - 3} more lines)")

    # Stderr — only if present
    if stderr:
        err_lines = stderr.strip().splitlines()[:2]
        lines.append(f"  err   │ {err_lines[0][:120]}")
        for el in err_lines[1:]:
            lines.append(f"        │ {el[:120]}")

    return "\n".join(lines)


class RRStep(SubRunner[ACEStepContext]):
    """Recursive Reflector as a pipeline step (SubRunner pattern).

    Satisfies **StepProtocol** (place directly in a Pipeline) and
    **ReflectorLike** (use as drop-in reflector in runners via ReflectStep).

    Internally builds a ``Pipeline([LLMCallStep, ExtractCodeStep,
    SandboxExecStep, CheckResultStep])`` and calls it once per REPL
    iteration until the LLM calls ``FINAL()`` or the iteration budget
    is exhausted.
    """

    # StepProtocol
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __init__(
        self,
        llm: Any,
        config: Optional[RecursiveConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
        subagent_llm: Any = None,
    ) -> None:
        cfg = config or RecursiveConfig()
        super().__init__(max_iterations=cfg.max_iterations)
        self.llm = llm
        self.config = cfg
        self.prompt_template = prompt_template
        self.subagent_llm = subagent_llm

    # ------------------------------------------------------------------
    # Loop override — collect iteration data for observability
    # ------------------------------------------------------------------

    def run_loop(self, **kwargs: Any) -> Any:
        """Execute the iterative loop, collecting per-iteration data.

        Iteration logs are appended to ``kwargs["iteration_log"]`` (a
        mutable list supplied by :meth:`reflect`).  The data is later
        attached to ``ReflectorOutput.raw["rr_trace"]`` so that a
        downstream ``RROpikStep`` can create hierarchical Opik traces
        without embedding observability logic here.
        """
        pipe = self._build_inner_pipeline(**kwargs)
        ctx = self._build_initial_context(**kwargs)
        iteration_log: list[dict[str, Any]] = kwargs.get("iteration_log", [])

        for i in range(self.max_iterations):
            ctx = pipe(ctx)
            rr_ctx: RRIterationContext = ctx  # type: ignore[assignment]

            exec_result = rr_ctx.exec_result
            stdout = getattr(exec_result, "stdout", None) if exec_result else None
            stderr = getattr(exec_result, "stderr", None) if exec_result else None
            iteration_log.append(
                {
                    "iteration": i,
                    "code": rr_ctx.code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "terminated": rr_ctx.terminated,
                    "llm_metadata": rr_ctx.llm_metadata,
                }
            )

            # Structured iteration logging
            status = "FINAL" if rr_ctx.terminated else "continue"
            log_msg = _format_iteration_log(
                iteration=i + 1,
                max_iterations=self.max_iterations,
                status=status,
                code=rr_ctx.code,
                stdout=stdout,
                stderr=stderr,
            )
            logger.info("\n%s", log_msg)

            if self._is_done(ctx):
                return self._extract_result(ctx)
            ctx = self._accumulate(ctx)

        return self._on_timeout(ctx, self.max_iterations, **kwargs)

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

        The message history is extracted from *last_ctx* so that fallback
        synthesis has access to the full REPL conversation.
        """
        logger.warning("Max iterations (%d) reached", self.max_iterations)
        args = kwargs.get("timeout_args", {})
        # Extract messages from the last iteration context for fallback synthesis
        messages = list(getattr(last_ctx, "messages", ()) or ())
        return self._build_timeout_output(
            args.get("question", ""),
            args.get("agent_output"),
            args.get("ground_truth"),
            args.get("feedback"),
            messages or None,
        )

    # ------------------------------------------------------------------
    # StepProtocol entry
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        """Run the Recursive Reflector and attach the reflection(s) to *ctx*.

        When the trace is a batch dict (has a ``"tasks"`` key), a single
        REPL session analyzes all tasks.  The RR uses ``parallel_map`` +
        ``ask_llm`` to explore tasks concurrently.  Per-task results are
        parsed from the ``FINAL()`` output into individual
        ``ReflectorOutput`` objects.
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
    # Batch reflection
    # ------------------------------------------------------------------

    def _run_batch_reflections(
        self,
        batch_trace: dict[str, Any],
        skillbook: Any,
    ) -> tuple[ReflectorOutput, ...]:
        """Run a single REPL session for all tasks in a batch.

        The RR sees the full batch (all tasks) in one session and uses
        ``ask_llm`` + ``parallel_map`` to analyze them concurrently.
        Returns per-task ``ReflectorOutput`` objects parsed from the
        single ``FINAL()`` output.
        """
        tasks = batch_trace["tasks"]

        # Single REPL session with the full batch trace
        reflection = self._run_reflection(
            traces=batch_trace,
            skillbook=skillbook,
        )

        # Split the batch result into per-task ReflectorOutputs
        return self._split_batch_reflection(reflection, tasks)

    def _split_batch_reflection(
        self,
        reflection: ReflectorOutput,
        tasks: list[dict[str, Any]],
    ) -> tuple[ReflectorOutput, ...]:
        """Extract per-task ReflectorOutputs from a batch FINAL result.

        The FINAL dict should contain a ``"tasks"`` key with per-task
        results.  If missing, the single reflection is duplicated for
        all tasks with task_id metadata.
        """
        task_results = reflection.raw.get("tasks", [])

        if not task_results or len(task_results) != len(tasks):
            # Fallback: duplicate the single reflection for each task
            logger.warning(
                "Batch FINAL missing per-task results (got %d, expected %d); "
                "duplicating single reflection",
                len(task_results) if task_results else 0,
                len(tasks),
            )
            reflections = []
            for i, task in enumerate(tasks):
                r = reflection.model_copy(deep=True)
                r.raw["task_id"] = task.get("task_id", f"task_{i}")
                reflections.append(r)
            return tuple(reflections)

        # Build a ReflectorOutput per task from the batch results
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
                    error_identification=str(
                        tr.get("error_identification", "")
                    ),
                    root_cause_analysis=tr.get(
                        "root_cause_analysis", ""
                    ),
                    correct_approach=tr.get(
                        "correct_approach", reflection.correct_approach
                    ),
                    key_insight=tr.get(
                        "key_insight", reflection.key_insight
                    ),
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
                    raw={
                        **tr,
                        "task_id": task_id,
                        "rr_trace": rr_trace,
                    },
                )
            )
        return tuple(reflections)

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
        """ReflectorLike — delegates to the internal REPL loop.

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
        """Run the recursive REPL loop and return analysis."""
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

        # Setup — single shared budget for main LLM + sub-agent calls
        budget = CallBudget(self.config.max_llm_calls)
        sandbox = self._create_sandbox(
            trace_obj, traces, skillbook, budget=budget, **kwargs
        )
        initial_prompt = self._build_initial_prompt(traces, skillbook, trace_obj)
        iteration_log: list[dict[str, Any]] = []

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
            iteration_log=iteration_log,
        )

        # Guarantee we always return a ReflectorOutput
        if not isinstance(result, ReflectorOutput):
            logger.warning(
                "run_loop returned %s instead of ReflectorOutput, wrapping",
                type(result).__name__,
            )
            result = ReflectorOutput(
                reasoning=str(result) if result else "No analysis produced",
                correct_approach="",
                key_insight="",
                raw={"original_result": result},
            )

        # Enrich result with RR execution metadata for downstream
        # observability steps (e.g. RROpikStep).
        subagent_calls = self._get_subagent_history(sandbox)
        result.raw["rr_trace"] = {
            "iterations": iteration_log,
            "subagent_calls": subagent_calls,
            "total_iterations": len(iteration_log),
            "timed_out": result.raw.get("timeout", False),
        }

        return result

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
        *,
        budget: CallBudget,
        **kwargs: Any,
    ) -> TraceSandbox:
        """Create and configure the TraceSandbox."""
        sandbox = TraceSandbox(trace=trace_obj, llm_query_fn=None)

        # ask_llm sub-agent
        if self.config.enable_subagent:
            subagent_system = (
                self.config.subagent_system_prompt
                if self.config.subagent_system_prompt is not None
                else SubAgentConfig.system_prompt
            )
            subagent_config = SubAgentConfig(
                model=self.config.subagent_model,
                max_tokens=self.config.subagent_max_tokens,
                temperature=self.config.subagent_temperature,
                system_prompt=subagent_system,
            )
            ask_llm_fn = create_ask_llm_function(
                llm=self.llm,
                config=subagent_config,
                subagent_llm=self.subagent_llm,
                budget=budget,
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

    @staticmethod
    def _get_subagent_history(sandbox: TraceSandbox) -> list[dict[str, Any]]:
        """Extract sub-agent call history from the sandbox's ask_llm function."""
        ask_llm_fn = sandbox.namespace.get("ask_llm")
        subagent = getattr(ask_llm_fn, "subagent", None)
        if subagent is not None:
            return list(subagent.call_history)
        return []

    def _build_initial_prompt(
        self,
        traces: dict[str, Any],
        skillbook: Any,
        trace_obj: Any,
    ) -> str:
        """Format the prompt template with previews and metadata."""
        import json as _json

        is_batch = isinstance(traces, dict) and "tasks" in traces

        t_steps = traces.get("steps", []) if not is_batch else []

        # Compute total trace size in chars for size-adaptive prompts
        trace_size_chars = len(_json.dumps(traces, default=str))

        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        if is_batch:
            # Batch mode: all tasks in one REPL session
            tasks = traces["tasks"]
            total_steps = sum(
                len(t.get("trace", [])) for t in tasks
            )
            # Build per-task preview rows
            preview_rows = []
            for t in tasks:
                tid = t.get("task_id", "?")
                tr = t.get("trace", [])
                first_msg = ""
                if tr and isinstance(tr[0], dict):
                    first_msg = tr[0].get("content", "")
                preview_rows.append(
                    f"| `{tid}` | {len(tr)} messages | "
                    f"\"{_preview(first_msg, 80)}\" |"
                )

            fmt_kwargs = dict(
                traces_description=(
                    f"Batch of {len(tasks)} tasks. "
                    f"Keys: tasks (list of {{task_id, trace}})"
                ),
                batch_variables=(
                    f"| `traces[\"tasks\"]` | All tasks in this batch "
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
                max_iterations=self.max_iterations,
                task_count=len(tasks),
            )
        else:
            # Single-trace mode: original format
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
                    f"| `traces[\"question\"]` | \"{_preview(t_question)}\" | {len(t_question)} chars |\n"
                    f"| first step | \"{_preview(t_reasoning)}...\" | {len(t_reasoning) if t_reasoning else 0} chars |\n"
                    f"| `traces[\"ground_truth\"]` | \"{_preview(t_ground_truth)}\" | {len(t_ground_truth) if t_ground_truth else 0} chars |\n"
                    f"| `traces[\"feedback\"]` | \"{_preview(t_feedback)}...\" | {len(t_feedback) if t_feedback else 0} chars |"
                ),
                step_count=(
                    len(t_steps) if t_steps else (len(trace_obj) if trace_obj else 0)
                ),
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.max_iterations,
                task_count=1,
            )

        return self.prompt_template.format(**fmt_kwargs)

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
