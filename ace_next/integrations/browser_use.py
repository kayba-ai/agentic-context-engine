"""Browser-use integration — execute step, result type, and trace converter."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.context import ACEStepContext
from ..implementations.prompts import wrap_skillbook_for_external_agent

logger = logging.getLogger(__name__)

try:
    from browser_use import Agent, Browser

    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    Agent = None  # type: ignore[misc,assignment]
    Browser = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------


@dataclass
class BrowserResult:
    """Output from a browser-use execution.

    This is the integration-specific result — not yet in ACE trace format.
    Use ``BrowserToTrace`` to convert to a standardised trace dict.
    """

    task: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    steps_count: int = 0
    duration_seconds: Optional[float] = None
    cited_skill_ids: List[str] = field(default_factory=list)
    chronological_steps: List[dict] = field(default_factory=list)
    raw_history: Any = None


# ---------------------------------------------------------------------------
# Execute step
# ---------------------------------------------------------------------------


class BrowserExecuteStep:
    """INJECT skillbook context and EXECUTE via browser-use Agent.

    Reads a task string from ``ctx.sample``, writes a ``BrowserResult``
    to ``ctx.trace``.

    This is an **async** step — ``__call__`` is a coroutine because
    browser-use is an async framework.
    """

    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(
        self, browser_llm: Any, browser: Any = None, **agent_kwargs: Any
    ) -> None:
        if not BROWSER_USE_AVAILABLE:
            raise ImportError(
                "browser-use is not installed. Install with: " "pip install browser-use"
            )
        self.browser_llm = browser_llm
        self.browser = browser
        self.agent_kwargs = agent_kwargs

    async def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task: str = ctx.sample

        # -- INJECT --
        enhanced_task = self._inject(task, ctx.skillbook)

        # -- EXECUTE --
        agent_params: dict[str, Any] = {
            **self.agent_kwargs,
            "task": enhanced_task,
            "llm": self.browser_llm,
        }
        if self.browser is not None:
            agent_params["browser"] = self.browser

        success = False
        error: Optional[str] = None
        history: Any = None
        try:
            agent = Agent(**agent_params)
            history = await agent.run()
            success = True
        except Exception as exc:
            error = str(exc)

        result = self._build_result(task, history, success, error)
        return ctx.replace(trace=result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inject(task: str, skillbook: Any) -> str:
        if skillbook is None:
            return task
        context = wrap_skillbook_for_external_agent(skillbook)
        if not context:
            return task
        return f"{task}\n\n{context}"

    @staticmethod
    def _build_result(
        task: str,
        history: Any,
        success: bool,
        error: Optional[str],
    ) -> BrowserResult:
        if history is None:
            return BrowserResult(task=task, success=success, error=error)

        # Extract basic info
        try:
            output = (
                history.final_result() if hasattr(history, "final_result") else ""
            ) or ""
        except Exception:
            output = ""

        try:
            steps_count = (
                history.number_of_steps() if hasattr(history, "number_of_steps") else 0
            )
        except Exception:
            steps_count = 0

        duration: Optional[float] = None
        try:
            if hasattr(history, "total_duration_seconds"):
                duration = round(history.total_duration_seconds(), 2)
        except Exception:
            pass

        # Extract chronological step data
        chronological: list[dict] = []
        try:
            if hasattr(history, "history"):
                for step_idx, step in enumerate(history.history, 1):
                    step_data: dict[str, Any] = {"step_number": step_idx}

                    if step.model_output:
                        step_data["thought"] = {
                            "thinking": step.model_output.thinking,
                            "evaluation": step.model_output.evaluation_previous_goal,
                            "memory": step.model_output.memory,
                            "next_goal": step.model_output.next_goal,
                        }
                        if step.model_output.action:
                            step_data["actions"] = [
                                {k: v for k, v in a.model_dump().items()}
                                for a in step.model_output.action
                            ]

                    if step.result:
                        step_data["results"] = [
                            {
                                "is_done": r.is_done,
                                "success": r.success,
                                "error": r.error,
                                "extracted_content": r.extracted_content,
                            }
                            for r in step.result
                        ]

                    if step.state:
                        step_data["url"] = step.state.url

                    chronological.append(step_data)
        except Exception as exc:
            logger.debug("Trace extraction error: %s", exc)

        # Extract cited skill IDs from agent thoughts
        cited_ids: list[str] = []
        try:
            if hasattr(history, "model_thoughts"):
                thoughts = history.model_thoughts()
                thoughts_text = "\n".join(
                    t.thinking
                    for t in thoughts
                    if hasattr(t, "thinking") and t.thinking
                )
                from ..implementations.helpers import extract_cited_skill_ids

                cited_ids = extract_cited_skill_ids(thoughts_text)
        except Exception:
            pass

        return BrowserResult(
            task=task,
            success=success,
            output=output,
            error=error,
            steps_count=steps_count,
            duration_seconds=duration,
            cited_skill_ids=cited_ids,
            chronological_steps=chronological,
            raw_history=history,
        )


# ---------------------------------------------------------------------------
# Convert step — BrowserResult → standardised trace dict
# ---------------------------------------------------------------------------


class BrowserToTrace:
    """Convert a ``BrowserResult`` on ``ctx.trace`` to the standardised
    trace dict that the learning tail (``ReflectStep``) expects.
    """

    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        r: BrowserResult = ctx.trace  # type: ignore[assignment]

        # Build human-readable reasoning from chronological steps
        parts: list[str] = []
        status = "succeeded" if r.success else "failed"
        parts.append(f"Browser task {status} in {r.steps_count} steps")
        if r.duration_seconds is not None:
            parts.append(f"Duration: {r.duration_seconds}s")
        if r.output:
            preview = r.output[:150] + ("..." if len(r.output) > 150 else "")
            parts.append(f"\nFinal output: {preview}")
        if r.error:
            parts.append(f"\nFailure reason: {r.error}")

        if r.chronological_steps:
            parts.append("\n\n=== BROWSER EXECUTION TRACE (Chronological) ===")
            for step in r.chronological_steps:
                step_num = step["step_number"]
                parts.append(f"\n--- Step {step_num} ---")
                if "thought" in step:
                    thought = step["thought"]
                    if thought.get("thinking"):
                        parts.append(f"Thinking: {thought['thinking']}")
                    if thought.get("evaluation"):
                        parts.append(f"   Evaluation: {thought['evaluation']}")
                    if thought.get("next_goal"):
                        parts.append(f"   Next Goal: {thought['next_goal']}")
                if "actions" in step:
                    for action in step["actions"]:
                        name = next(iter(action), "unknown")
                        parts.append(f"Action: {name}({action.get(name, {})})")
                if "results" in step:
                    for res in step["results"]:
                        res_parts = []
                        if res.get("success") is not None:
                            res_parts.append(f"success={res['success']}")
                        if res.get("error"):
                            res_parts.append(f"error={res['error']}")
                        if res.get("extracted_content"):
                            res_parts.append(
                                f"content={str(res['extracted_content'])[:200]}"
                            )
                        parts.append(f"Result: {', '.join(res_parts)}")
                if "url" in step:
                    parts.append(f"URL: {step['url']}")
            parts.append("\n=== END EXECUTION TRACE ===")

        reasoning = "\n".join(parts)

        feedback = f"Browser task {status} in {r.steps_count} steps"
        if r.duration_seconds is not None:
            feedback += f" ({r.duration_seconds}s)"
        if r.error:
            feedback += f"\nError: {r.error}"

        trace: dict = {
            "question": r.task,
            "reasoning": reasoning,
            "answer": r.output,
            "skill_ids": r.cited_skill_ids,
            "feedback": feedback,
            "ground_truth": None,
        }
        return ctx.replace(trace=trace)


__all__ = [
    "BrowserExecuteStep",
    "BrowserResult",
    "BrowserToTrace",
]
