"""Inner pipeline steps for a single Recursive Reflector REPL iteration.

Each step handles one concern within an iteration. Mutable state (sandbox,
budget) is injected via the constructor — same pattern as ApplyStep(skillbook).
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from .code_extraction import extract_code
from .context import RRIterationContext
from .message_trimming import trim_messages

if TYPE_CHECKING:
    from ace.reflector.config import RecursiveConfig
    from ace.reflector.sandbox import TraceSandbox
    from ace.reflector.subagent import CallBudget

logger = logging.getLogger(__name__)


def _truncate_output(output: str, max_chars: int = 20_000) -> str:
    """Truncate output with metadata suffix showing how much was cut."""
    if not output or len(output) <= max_chars:
        return output
    truncated = output[:max_chars]
    remaining = len(output) - max_chars
    return f"{truncated}\n... + [{remaining} chars truncated]"


# ---------------------------------------------------------------------------
# LLMCallStep
# ---------------------------------------------------------------------------


class LLMCallStep:
    """Trim messages and call the LLM.

    Mutable state: *budget* is shared across iterations and consumed on
    each call.
    """

    requires = frozenset({"messages"})
    provides = frozenset({"llm_response"})

    def __init__(
        self,
        llm: Any,
        config: RecursiveConfig,
        budget: CallBudget,
    ) -> None:
        self.llm = llm
        self.config = config
        self.budget = budget

    def __call__(self, ctx: RRIterationContext) -> RRIterationContext:
        if self.budget.exhausted:
            logger.warning("Budget exhausted, returning empty response")
            return ctx.replace(llm_response="")

        trimmed = trim_messages(list(ctx.messages), self.config.max_context_chars)
        response = self.llm.complete_messages(trimmed)
        response_text: str = response.text or ""
        self.budget.consume()

        if not response_text:
            logger.warning(
                "LLM returned empty response on iteration %d", ctx.iteration + 1
            )

        return ctx.replace(llm_response=response_text)


# ---------------------------------------------------------------------------
# ExtractCodeStep
# ---------------------------------------------------------------------------


class ExtractCodeStep:
    """Extract Python code from the LLM response.

    Pure — no mutable state.  Sets ``code`` when a code block is found, or
    ``direct_response`` when the LLM responds with plain text (for fallback
    JSON parsing by CheckResultStep).
    """

    requires = frozenset({"llm_response"})
    provides = frozenset({"code", "direct_response"})

    def __call__(self, ctx: RRIterationContext) -> RRIterationContext:
        response = ctx.llm_response or ""
        code = extract_code(response)
        if code:
            return ctx.replace(code=code)
        # No code — pass the raw response for direct JSON parsing attempt
        return ctx.replace(direct_response=response)


# ---------------------------------------------------------------------------
# SandboxExecStep
# ---------------------------------------------------------------------------


class SandboxExecStep:
    """Execute extracted code in the sandbox.

    If ``code`` is None (no code extracted), passes through with
    ``exec_result = None``.  The sandbox is mutable shared state held via
    the constructor.
    """

    requires = frozenset({"code"})
    provides = frozenset({"exec_result"})

    def __init__(self, sandbox: TraceSandbox, config: RecursiveConfig) -> None:
        self.sandbox = sandbox
        self.config = config

    def __call__(self, ctx: RRIterationContext) -> RRIterationContext:
        if ctx.code is None:
            return ctx.replace(exec_result=None)

        logger.debug("Executing code:\n%s...", ctx.code[:200])
        result = self.sandbox.execute(ctx.code, timeout=self.config.timeout)
        return ctx.replace(exec_result=result)


# ---------------------------------------------------------------------------
# CheckResultStep
# ---------------------------------------------------------------------------


class CheckResultStep:
    """Inspect execution result and determine next action.

    Implements all guard logic:
    - Reject FINAL on iteration 0 (premature)
    - Reject FINAL when execution had errors
    - Parse FINAL value into ReflectorOutput
    - Parse direct JSON response (no-code fallback)
    - Build feedback messages for the next iteration
    """

    requires = frozenset({"exec_result", "messages", "llm_response"})
    provides = frozenset({"terminated", "reflection", "feedback_messages"})

    def __init__(self, sandbox: TraceSandbox, config: RecursiveConfig) -> None:
        self.sandbox = sandbox
        self.config = config

    def __call__(self, ctx: RRIterationContext) -> RRIterationContext:
        response_text = ctx.llm_response or ""

        # --- No code path: try direct JSON parse ---
        if ctx.code is None and ctx.direct_response is not None:
            return self._handle_direct_response(ctx)

        # --- Empty LLM response ---
        if not response_text:
            return ctx.replace(
                feedback_messages=(
                    {"role": "assistant", "content": ""},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was empty. "
                            "Please write Python code to analyze the trace."
                        ),
                    },
                ),
            )

        result = ctx.exec_result

        # --- Premature FINAL (iteration 0) ---
        if self.sandbox.final_called and ctx.iteration == 0:
            self.sandbox.reset()
            stdout = result.stdout if result else ""
            return ctx.replace(
                feedback_messages=(
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            f"Output:\n{stdout}\n\n"
                            "You called FINAL() before exploring the data. "
                            "Read the actual variables first, then call FINAL() "
                            "with evidence-based analysis."
                        ),
                    },
                ),
            )

        # --- FINAL after error ---
        if self.sandbox.final_called and result and not result.success:
            logger.warning("Rejecting FINAL() called after execution error")
            self.sandbox.reset()
            return ctx.replace(
                feedback_messages=(
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            f"Output:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n\n"
                            "Your code had an error. Fix the bug and try again. "
                            "Do NOT call FINAL() until your code executes successfully."
                        ),
                    },
                ),
            )

        # --- Successful FINAL ---
        if self.sandbox.final_called:
            logger.debug("FINAL() called, parsing result")
            reflection = _parse_final_value(self.sandbox.final_value)
            return ctx.replace(terminated=True, reflection=reflection)

        # --- Normal continuation: feed output back ---
        max_output = self.config.max_output_chars
        output_parts: list[str] = []

        # Iteration progress header
        next_iter = ctx.iteration + 1
        iter_header = f"[Iteration {next_iter}/{self.config.max_iterations}]"
        if next_iter >= self.config.max_iterations - 2:
            iter_header += " (approaching limit — finalize soon)"

        if result and result.stdout:
            output_parts.append(
                f"stdout:\n{_truncate_output(result.stdout, max_output)}"
            )
        if result and result.stderr:
            output_parts.append(
                f"stderr:\n{_truncate_output(result.stderr, max_output)}"
            )

        output_message = "\n".join(output_parts) if output_parts else "(no output)"

        return ctx.replace(
            feedback_messages=(
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": f"{iter_header}\nOutput:\n{output_message}"},
            ),
        )

    # -- helpers --

    def _handle_direct_response(self, ctx: RRIterationContext) -> RRIterationContext:
        """Try to parse a direct JSON response when no code was extracted."""
        response = ctx.direct_response or ""
        logger.debug("No code block found, attempting to parse direct response")
        try:
            reflection = _parse_direct_response(response)
            return ctx.replace(terminated=True, reflection=reflection)
        except Exception as e:
            logger.warning("Failed to parse direct response: %s", e)
            return ctx.replace(
                feedback_messages=(
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": (
                            "Please write Python code to analyze the trace "
                            "and call FINAL() with your analysis."
                        ),
                    },
                ),
            )


# ---------------------------------------------------------------------------
# Shared parsing helpers (ported from RecursiveReflector)
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_final_value(value: Any) -> Any:
    """Parse the value from FINAL() into a ReflectorOutput."""
    from ace_next.core.outputs import ExtractedLearning, ReflectorOutput, SkillTag

    if not isinstance(value, dict):
        if isinstance(value, ReflectorOutput):
            return value
        value = {"reasoning": str(value)}

    extracted_learnings = []
    for learning in value.get("extracted_learnings", []):
        if isinstance(learning, dict):
            extracted_learnings.append(
                ExtractedLearning(
                    learning=learning.get("learning", ""),
                    atomicity_score=_safe_float(learning.get("atomicity_score", 0.0)),
                    evidence=learning.get("evidence", ""),
                )
            )

    skill_tags = []
    for tag in value.get("skill_tags", []):
        if isinstance(tag, dict):
            skill_tags.append(
                SkillTag(
                    id=tag.get("id", ""),
                    tag=tag.get("tag", "neutral"),
                )
            )

    return ReflectorOutput(
        reasoning=value.get("reasoning", ""),
        error_identification=value.get("error_identification", ""),
        root_cause_analysis=value.get("root_cause_analysis", ""),
        correct_approach=value.get("correct_approach", ""),
        key_insight=value.get("key_insight", ""),
        extracted_learnings=extracted_learnings,
        skill_tags=skill_tags,
        raw=value,
    )


def _parse_direct_response(response: str) -> Any:
    """Try to parse a direct JSON response without code execution.

    Raises ValueError if the response is not valid JSON.
    """
    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()
    data = json.loads(response)
    return _parse_final_value(data)
