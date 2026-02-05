"""Recursive reflector with code execution for trace analysis."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import RecursiveConfig
from .prompts import REFLECTOR_RECURSIVE_PROMPT
from .sandbox import TraceSandbox
from .subagent import CallBudget, SubAgentConfig, create_ask_llm_function
from .trace_context import TraceContext
from ..observability.tracers import maybe_track

if TYPE_CHECKING:
    from ..llm import LLMClient
    from ..roles import AgentOutput, ReflectorOutput
    from ..skillbook import Skillbook

logger = logging.getLogger(__name__)


def _preview(text: str | None, max_len: int = 150) -> str:
    """Return a short preview of text for prompt grounding."""
    if not text:
        return "(empty)"
    if len(text) <= max_len:
        return text
    return text[:max_len]


class RecursiveReflector:
    """Recursive reflector with code execution for trace analysis.

    This reflector uses a REPL loop where an LLM generates Python code
    to analyze agent traces. The code is executed in a restricted sandbox,
    and the output is fed back to the LLM until it calls FINAL() with
    the analysis result.

    This enables more sophisticated analysis than single-pass reflection:
    - Programmatic exploration of long traces
    - Sub-LLM queries for complex reasoning
    - Iterative refinement of analysis
    - Pattern matching and search in traces

    Example:
        >>> from ace.reflector import RecursiveReflector, RecursiveConfig
        >>> from ace.llm_providers.litellm_client import LiteLLMClient
        >>>
        >>> llm = LiteLLMClient(model="gpt-4o-mini")
        >>> reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))
        >>>
        >>> output = reflector.reflect(
        ...     question="What is 2+2?",
        ...     agent_output=agent_output,
        ...     skillbook=skillbook,
        ...     ground_truth="4",
        ...     feedback="Correct!",
        ... )
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: Optional[RecursiveConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
        subagent_llm: Optional["LLMClient"] = None,
    ) -> None:
        """Initialize the RecursiveReflector.

        Args:
            llm: The LLM client to use for code generation
            config: Configuration for the recursive reflector
            prompt_template: Custom prompt template (uses default if not provided)
            subagent_llm: Optional separate LLM for sub-agent calls (ask_llm).
                          If provided, ask_llm() uses this model for exploration.
                          If not provided, uses the main llm (or subagent_model from config).
        """
        self.llm = llm
        self.config = config or RecursiveConfig()
        self.prompt_template = prompt_template
        self.subagent_llm = subagent_llm

    @maybe_track(name="recursive_reflector", tags=["reflector", "recursive"])
    def reflect(
        self,
        *,
        question: str,
        agent_output: "AgentOutput",
        skillbook: "Skillbook",
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> "ReflectorOutput":
        """Perform recursive reflection using code execution.

        Args:
            question: The original task/question
            agent_output: The agent's output containing reasoning and answer
            skillbook: The current skillbook of strategies
            ground_truth: Expected correct answer (if available)
            feedback: Execution feedback

        Returns:
            ReflectorOutput with analysis and skill classifications
        """
        # Build trace context from agent output
        trace = TraceContext.from_agent_output(agent_output)

        # Create shared call budget
        budget = CallBudget(self.config.max_llm_calls)

        # Create ask_llm function with shared budget
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
                budget=budget,
            )
        else:

            def _disabled_ask_llm(question: str, context: str = "") -> str:
                return "(ask_llm disabled - analyze with code)"

            ask_llm_fn = _disabled_ask_llm

        # Create sandbox with trace
        sandbox = TraceSandbox(trace=trace, llm_query_fn=None)

        # Inject ask_llm as primary, llm_query as legacy alias
        sandbox.inject("ask_llm", ask_llm_fn)
        sandbox.inject("llm_query", lambda prompt: ask_llm_fn(prompt, ""))

        # Inject context variables
        sandbox.inject("question", question)
        sandbox.inject("reasoning", agent_output.reasoning)
        sandbox.inject("final_answer", agent_output.final_answer)
        sandbox.inject("ground_truth", ground_truth)
        sandbox.inject("feedback", feedback)
        sandbox.inject("skillbook", skillbook.as_prompt() or "(empty skillbook)")

        # Build initial prompt with previews and metadata
        # Full data is injected into sandbox - previews provide grounding
        skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
        initial_prompt = self.prompt_template.format(
            question_length=len(question),
            question_preview=_preview(question),
            reasoning_length=len(agent_output.reasoning),
            reasoning_preview=_preview(agent_output.reasoning),
            answer_length=len(agent_output.final_answer),
            answer_preview=_preview(agent_output.final_answer),
            ground_truth_length=len(ground_truth) if ground_truth else 0,
            ground_truth_preview=_preview(ground_truth),
            feedback_length=len(feedback) if feedback else 0,
            feedback_preview=_preview(feedback),
            skillbook_length=len(skillbook_text),
            step_count=len(trace) if trace else 0,
        )

        # REPL loop
        messages: List[Dict[str, str]] = [{"role": "user", "content": initial_prompt}]

        for iteration in range(self.config.max_iterations):
            output = self._execute_iteration(
                iteration, messages, sandbox, budget, **kwargs
            )
            if output is not None:
                # Log FINAL() output to parent span
                try:
                    from opik import opik_context

                    opik_context.update_current_span(
                        metadata={
                            "final_output": {
                                "key_insight": output.key_insight,
                                "learnings_count": len(output.extracted_learnings),
                                "learnings": [
                                    {
                                        "learning": l.learning,
                                        "score": l.atomicity_score,
                                    }
                                    for l in output.extracted_learnings
                                ],
                                "skill_tags": [
                                    {"id": t.id, "tag": t.tag}
                                    for t in output.skill_tags
                                ],
                            },
                            "total_iterations": iteration + 1,
                            "total_llm_calls": budget.count,
                        }
                    )
                except Exception:
                    pass
                return output

        # Max iterations reached - build timeout output
        logger.warning(f"Max iterations ({self.config.max_iterations}) reached")
        return self._build_timeout_output(
            question, agent_output, ground_truth, feedback
        )

    @maybe_track(name="repl_iteration", tags=["reflector", "iteration"])
    def _execute_iteration(
        self,
        iteration: int,
        messages: List[Dict[str, str]],
        sandbox: TraceSandbox,
        budget: CallBudget,
        **kwargs: Any,
    ) -> Optional["ReflectorOutput"]:
        """Execute a single REPL iteration.

        Args:
            iteration: Current iteration index
            messages: Message history (modified in place)
            sandbox: The trace sandbox
            budget: Shared call budget
            **kwargs: Additional LLM kwargs

        Returns:
            ReflectorOutput if FINAL() was called or direct response parsed, None to continue
        """
        logger.debug(f"Recursive reflector iteration {iteration + 1}")

        # Trim messages to fit context budget
        trimmed = self._trim_messages(messages)

        # Get code from LLM
        response = self.llm.complete_messages(trimmed, **kwargs)
        response_text = response.text

        # Extract code blocks
        code = self._extract_code(response_text)

        if not code:
            # No code in response - try to parse as final answer
            logger.debug("No code block found, attempting to parse direct response")
            try:
                return self._parse_direct_response(response_text)
            except Exception as e:
                logger.warning(f"Failed to parse direct response: {e}")
                # Ask LLM to output code
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please write Python code to analyze the trace and call FINAL() with your analysis.",
                    }
                )
                return None

        # Execute code in sandbox with timeout
        logger.debug(f"Executing code:\n{code[:200]}...")
        result = sandbox.execute(code, timeout=self.config.timeout)

        # Reject premature FINAL() on first iteration - force the model
        # to see actual data before finalizing to prevent hallucination
        if sandbox.final_called and iteration == 0:
            sandbox._final_called = False
            sandbox._final_value = None
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Output:\n{result.stdout}\n\n"
                    "You called FINAL() before exploring the data. "
                    "Read the actual variables first, then call FINAL() with evidence-based analysis.",
                }
            )
            return None  # continue to next iteration

        # Check if FINAL() was called
        if sandbox.final_called:
            logger.debug("FINAL() called, parsing result")
            # Update iteration span metadata
            try:
                from opik import opik_context

                opik_context.update_current_span(
                    metadata={
                        "iteration_number": iteration + 1,
                        "code_generated": code[:2000] if code else "(no code)",
                        "stdout": result.stdout[:2000],
                        "stderr": result.stderr[:2000],
                        "execution_success": result.success,
                        "final_called": True,
                    }
                )
            except Exception:
                pass
            return self._parse_final_value(sandbox.final_value)

        # Build output message
        output_parts = []
        if result.stdout:
            output_parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"stderr:\n{result.stderr}")

        output_message = "\n".join(output_parts) if output_parts else "(no output)"

        # Feed output back to LLM
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": f"Output:\n{output_message}"})

        # Update iteration span metadata
        try:
            from opik import opik_context

            opik_context.update_current_span(
                metadata={
                    "iteration_number": iteration + 1,
                    "code_generated": code[:2000] if code else "(no code)",
                    "stdout": result.stdout[:2000],
                    "stderr": result.stderr[:2000],
                    "execution_success": result.success,
                    "final_called": False,
                }
            )
        except Exception:
            pass

        return None

    def _trim_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Trim messages to fit within context budget.

        Keeps the first message (instructions) and the most recent messages
        that fit within max_context_chars. Inserts a summary marker for
        dropped messages.

        Args:
            messages: Full message history

        Returns:
            Trimmed message list
        """
        max_chars = self.config.max_context_chars
        total = sum(len(m["content"]) for m in messages)
        if total <= max_chars:
            return messages

        # Always keep the first message (instructions)
        first = messages[0]
        remaining_budget = max_chars - len(first["content"])

        # Add messages from the end until budget is exhausted
        kept: List[Dict[str, str]] = []
        for msg in reversed(messages[1:]):
            msg_len = len(msg["content"])
            if remaining_budget - msg_len < 0:
                break
            kept.insert(0, msg)
            remaining_budget -= msg_len

        dropped_count = len(messages) - 1 - len(kept)
        if dropped_count > 0:
            summary = {
                "role": "user",
                "content": f"[{dropped_count} earlier iterations omitted]",
            }
            return [first, summary] + kept
        return [first] + kept

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format message list into a single prompt string.

        For LLM clients that use simple string prompts, we concatenate
        the messages with role prefixes.
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Previous response]\n{content}")
        return "\n\n".join(parts)

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response.

        Args:
            response: The LLM response text

        Returns:
            Extracted Python code or None if no code block found
        """
        # Match ```python ... ``` blocks
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return only the first code block to force multi-turn iteration
            return matches[0].strip()

        # Also try ``` ... ``` without language specifier
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Filter to likely Python code (has common Python keywords/syntax)
            python_indicators = [
                "def ",
                "import ",
                "print(",
                "FINAL(",
                "=",
                "if ",
                "for ",
            ]
            for match in matches:
                if any(indicator in match for indicator in python_indicators):
                    return match.strip()

        return None

    def _parse_final_value(self, value: Any) -> "ReflectorOutput":
        """Parse the value from FINAL() into ReflectorOutput.

        Args:
            value: The value passed to FINAL() (should be a dict)

        Returns:
            ReflectorOutput constructed from the value
        """
        from ..roles import ReflectorOutput, ExtractedLearning, SkillTag

        if not isinstance(value, dict):
            # Try to use it as-is if it's already the right type
            if isinstance(value, ReflectorOutput):
                return value
            # Convert to dict representation
            value = {"reasoning": str(value)}

        # Extract learnings
        extracted_learnings = []
        for learning in value.get("extracted_learnings", []):
            if isinstance(learning, dict):
                extracted_learnings.append(
                    ExtractedLearning(
                        learning=learning.get("learning", ""),
                        atomicity_score=float(learning.get("atomicity_score", 0.0)),
                        evidence=learning.get("evidence", ""),
                    )
                )

        # Extract skill tags
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

    def _parse_direct_response(self, response: str) -> "ReflectorOutput":
        """Try to parse a direct JSON response without code execution.

        Args:
            response: The LLM response text

        Returns:
            ReflectorOutput parsed from JSON

        Raises:
            ValueError: If response is not valid JSON
        """
        import json

        # Try to extract JSON from the response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # Parse JSON
        data = json.loads(response)
        return self._parse_final_value(data)

    def _build_timeout_output(
        self,
        question: str,
        agent_output: "AgentOutput",
        ground_truth: Optional[str],
        feedback: Optional[str],
    ) -> "ReflectorOutput":
        """Build a ReflectorOutput when max iterations is reached.

        Args:
            question: The original question
            agent_output: The agent's output
            ground_truth: Expected answer
            feedback: Execution feedback

        Returns:
            ReflectorOutput with timeout analysis
        """
        from ..roles import ReflectorOutput, ExtractedLearning

        is_correct = False
        if ground_truth:
            is_correct = (
                agent_output.final_answer.strip().lower()
                == ground_truth.strip().lower()
            )

        return ReflectorOutput(
            reasoning=f"Recursive analysis reached max iterations ({self.config.max_iterations}). "
            f"Basic analysis: Answer was {'correct' if is_correct else 'incorrect'}.",
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
