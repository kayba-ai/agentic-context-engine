"""LangChain integration — execute step, result type, and trace converter."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.context import ACEStepContext
from ..implementations.prompts import wrap_skillbook_for_external_agent

logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import Runnable

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Runnable = None  # type: ignore

try:
    from langchain.agents import AgentExecutor

    AGENT_EXECUTOR_AVAILABLE = True
except ImportError:
    AGENT_EXECUTOR_AVAILABLE = False
    AgentExecutor = None  # type: ignore

try:
    from langgraph.graph.state import CompiledStateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    CompiledStateGraph = None  # type: ignore


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------


@dataclass
class LangChainResult:
    """Output from a LangChain Runnable execution.

    This is the integration-specific result — not yet in ACE trace format.
    Use ``LangChainToTrace`` to convert to a standardised trace dict.

    ``result_type`` indicates the source variant:
    - ``"simple"``    — basic chain (prompt | llm)
    - ``"agent"``     — AgentExecutor with intermediate_steps
    - ``"langgraph"`` — LangGraph CompiledStateGraph with messages
    - ``"error"``     — execution failed
    """

    task: str
    output: str = ""
    result_type: str = "simple"
    success: bool = True
    error: Optional[str] = None
    intermediate_steps: List[Tuple[Any, Any]] = field(default_factory=list)
    messages: List[Any] = field(default_factory=list)
    raw_result: Any = None


# ---------------------------------------------------------------------------
# Execute step
# ---------------------------------------------------------------------------


class LangChainExecuteStep:
    """INJECT skillbook context and EXECUTE a LangChain Runnable.

    Reads input from ``ctx.sample`` (string, dict, or message list),
    writes a ``LangChainResult`` to ``ctx.trace``.

    Handles three Runnable variants automatically:
    - Simple chains (prompt | llm)
    - AgentExecutor (intermediate_steps tracing)
    - LangGraph CompiledStateGraph (message-based I/O)
    """

    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(
        self,
        runnable: Any,
        output_parser: Optional[Callable[[Any], str]] = None,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: "
                "pip install langchain-core"
            )
        self.runnable = runnable
        self.output_parser = output_parser or self._default_output_parser

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task = self._get_task_str(ctx.sample)

        # -- INJECT --
        enhanced_input = self._inject_context(ctx.sample, ctx.skillbook)

        # -- EXECUTE --
        is_agent = self._is_agent_executor()
        is_langgraph = self._is_langgraph()

        original_setting = False
        if is_agent:
            original_setting = getattr(
                self.runnable, "return_intermediate_steps", False
            )
            self.runnable.return_intermediate_steps = True

        try:
            raw = self.runnable.invoke(enhanced_input)
        except Exception as exc:
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting
            result = LangChainResult(
                task=task,
                output=f"Failed: {exc}",
                result_type="error",
                success=False,
                error=str(exc),
            )
            return ctx.replace(trace=result)
        finally:
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting

        # -- BUILD RESULT --
        if is_agent and isinstance(raw, dict) and "intermediate_steps" in raw:
            result = self._build_agent_result(task, raw)
        elif is_langgraph and isinstance(raw, dict) and "messages" in raw:
            result = self._build_langgraph_result(task, raw)
        else:
            result = self._build_simple_result(task, raw)

        return ctx.replace(trace=result)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_context(original_input: Any, skillbook: Any) -> Any:
        if skillbook is None:
            return original_input
        context = wrap_skillbook_for_external_agent(skillbook)
        if not context:
            return original_input

        if isinstance(original_input, str):
            return f"{original_input}\n\n{context}"

        if isinstance(original_input, dict) and "messages" in original_input:
            messages = original_input["messages"]
            if messages and hasattr(messages[0], "content"):
                enhanced = list(messages)
                first = enhanced[0]
                enhanced[0] = type(first)(content=f"{context}\n\n{first.content}")
                return {
                    "messages": enhanced,
                    **{k: v for k, v in original_input.items() if k != "messages"},
                }
            return original_input

        if isinstance(original_input, dict) and "input" in original_input:
            enhanced = original_input.copy()
            enhanced["input"] = f"{original_input['input']}\n\n{context}"
            return enhanced

        if isinstance(original_input, dict):
            enhanced = original_input.copy()
            enhanced["skillbook_context"] = context
            return enhanced

        return original_input

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_simple_result(self, task: str, raw: Any) -> LangChainResult:
        return LangChainResult(
            task=task,
            output=self.output_parser(raw),
            result_type="simple",
            raw_result=raw,
        )

    def _build_agent_result(self, task: str, raw: Dict[str, Any]) -> LangChainResult:
        output = raw.get("output", "")
        steps = raw.get("intermediate_steps", [])

        intermediate: List[Tuple[Any, Any]] = []
        for step_tuple in steps:
            if len(step_tuple) == 2:
                intermediate.append(tuple(step_tuple))  # type: ignore[arg-type]

        return LangChainResult(
            task=task,
            output=str(output),
            result_type="agent",
            intermediate_steps=intermediate,
            raw_result=raw,
        )

    def _build_langgraph_result(
        self, task: str, raw: Dict[str, Any]
    ) -> LangChainResult:
        messages = raw.get("messages", [])
        output = self._extract_langgraph_output(raw)
        intermediate = self._extract_langgraph_steps(raw)

        return LangChainResult(
            task=task,
            output=output,
            result_type="langgraph",
            intermediate_steps=intermediate,
            messages=list(messages),
            raw_result=raw,
        )

    # ------------------------------------------------------------------
    # LangGraph helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_langgraph_output(result: Dict[str, Any]) -> str:
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                msg_type = getattr(msg, "type", msg.__class__.__name__.lower())
                if msg_type != "tool":
                    return str(msg.content)
        return ""

    @staticmethod
    def _extract_langgraph_steps(
        result: Dict[str, Any],
    ) -> List[Tuple[Any, Any]]:
        intermediate: List[Tuple[Any, Any]] = []
        for msg in result.get("messages", []):
            msg_type = getattr(msg, "type", msg.__class__.__name__.lower())
            content = getattr(msg, "content", str(msg))
            if msg_type == "ai":
                for tc in getattr(msg, "tool_calls", []):
                    intermediate.append((tc, None))
            elif msg_type == "tool":
                for i in range(len(intermediate) - 1, -1, -1):
                    if intermediate[i][1] is None:
                        intermediate[i] = (intermediate[i][0], content)
                        break
        return intermediate

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_agent_executor(self) -> bool:
        if not AGENT_EXECUTOR_AVAILABLE or AgentExecutor is None:
            return False
        return isinstance(self.runnable, AgentExecutor)

    def _is_langgraph(self) -> bool:
        if not LANGGRAPH_AVAILABLE or CompiledStateGraph is None:
            return False
        return isinstance(self.runnable, CompiledStateGraph)

    @staticmethod
    def _get_task_str(original_input: Any) -> str:
        if isinstance(original_input, str):
            return original_input
        if isinstance(original_input, dict):
            if "messages" in original_input:
                messages = original_input["messages"]
                if messages and hasattr(messages[0], "content"):
                    return str(messages[0].content)
            return (
                original_input.get("input")
                or original_input.get("question")
                or original_input.get("query")
                or str(original_input)
            )
        return str(original_input)

    @staticmethod
    def _default_output_parser(result: Any) -> str:
        if isinstance(result, str):
            return result
        if hasattr(result, "content"):
            return str(result.content)
        if isinstance(result, dict):
            for key in ("output", "answer", "result", "text"):
                if key in result:
                    return str(result[key])
            return str(result)
        return str(result)


# ---------------------------------------------------------------------------
# Convert step — LangChainResult → standardised trace dict
# ---------------------------------------------------------------------------


class LangChainToTrace:
    """Convert a ``LangChainResult`` on ``ctx.trace`` to the standardised
    trace dict that the learning tail (``ReflectStep``) expects.
    """

    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        r: LangChainResult = ctx.trace  # type: ignore[assignment]

        reasoning = self._build_reasoning(r)

        if r.success:
            feedback = self._build_feedback(r)
        else:
            feedback = f"Chain execution failed. Error: {r.error}"

        trace: dict = {
            "question": r.task,
            "reasoning": reasoning,
            "answer": r.output,
            "skill_ids": [],
            "feedback": feedback,
            "ground_truth": None,
        }
        return ctx.replace(trace=trace)

    # ------------------------------------------------------------------
    # Reasoning formatters per result type
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(r: LangChainResult) -> str:
        if r.result_type == "error":
            return (
                f"Question/Task: {r.task}\n\n"
                f"Execution Result: FAILED\nError: {r.error}"
            )

        if r.result_type == "agent":
            parts = [f"Question/Task: {r.task}", ""]
            parts.append(
                f"=== AGENT EXECUTION TRACE ({len(r.intermediate_steps)} steps) ==="
            )
            for i, (action, observation) in enumerate(r.intermediate_steps, 1):
                parts.append(f"\n--- Step {i} ---")
                if hasattr(action, "log") and action.log:
                    parts.append(f"Thought: {action.log}")
                if hasattr(action, "tool"):
                    parts.append(f"Action: {action.tool}")
                    parts.append(f"Action Input: {str(action.tool_input)[:300]}")
                parts.append(f"Observation: {str(observation)[:300]}")
            parts.append("\n=== END TRACE ===")
            parts.append(f"\nFinal Answer: {r.output}")
            return "\n".join(parts)

        if r.result_type == "langgraph":
            msg_parts: list[str] = []
            for msg in r.messages:
                msg_type = getattr(msg, "type", msg.__class__.__name__.lower())
                content = getattr(msg, "content", str(msg))
                if msg_type == "human":
                    msg_parts.append(f"Human: {str(content)[:300]}")
                elif msg_type == "ai":
                    if content:
                        msg_parts.append(f"Assistant: {str(content)[:300]}")
                    for tc in getattr(msg, "tool_calls", []):
                        name = (
                            tc.get("name", "unknown")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "unknown")
                        )
                        msg_parts.append(f"  Tool Call: {name}")
                elif msg_type == "tool":
                    msg_parts.append(f"Tool Result: {str(content)[:300]}")

            trace_str = "\n".join(msg_parts)
            return (
                f"Question/Task: {r.task}\n\n"
                f"=== LANGGRAPH EXECUTION TRACE ({len(r.messages)} messages) ===\n"
                f"{trace_str}\n"
                f"=== END TRACE ===\n\n"
                f"Final Answer: {r.output}"
            )

        # simple
        return (
            f"Question/Task: {r.task}\n\n"
            f"Chain Output: {r.output}\n\n"
            f"Note: External LangChain chain execution."
        )

    @staticmethod
    def _build_feedback(r: LangChainResult) -> str:
        if r.result_type == "agent":
            return f"Agent completed task in {len(r.intermediate_steps)} steps"
        if r.result_type == "langgraph":
            return f"LangGraph agent completed in {len(r.messages)} messages"
        return f"External chain completed for task: {r.task[:200]}"


__all__ = [
    "LangChainExecuteStep",
    "LangChainResult",
    "LangChainToTrace",
]
