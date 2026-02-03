"""Recursive reflector with code execution for trace analysis."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import RecursiveConfig
from .prompts import REFLECTOR_RECURSIVE_PROMPT
from .sandbox import TraceSandbox
from .subagent import SubAgentConfig, create_ask_llm_function
from .trace_context import TraceContext

if TYPE_CHECKING:
    from ..llm import LLMClient
    from ..roles import AgentOutput, ReflectorOutput
    from ..skillbook import Skillbook

logger = logging.getLogger(__name__)


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

        # Create bounded LLM query function for sub-analyses
        llm_call_count = 0
        max_llm_calls = self.config.max_llm_calls

        def bounded_llm_query(prompt: str) -> str:
            """Spawn a sub-LLM query for complex analysis (with call limit)."""
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count > max_llm_calls:
                return f"(Max {max_llm_calls} LLM calls exceeded - analyze with available data)"
            response = self.llm.complete(prompt, **kwargs)
            return response.text

        # Create sandbox with trace and utilities
        sandbox = TraceSandbox(
            trace=trace,
            llm_query_fn=bounded_llm_query if self.config.enable_llm_query else None,
        )

        # Create and inject the sub-agent function if enabled
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
                max_calls=self.config.max_llm_calls,
            )
            sandbox.inject("ask_llm", ask_llm_fn)
        else:
            # Provide a stub that explains the feature is disabled
            sandbox.inject(
                "ask_llm",
                lambda question, context="": "(ask_llm disabled - analyze with code)",
            )

        # Inject context variables
        sandbox.inject("question", question)
        sandbox.inject("reasoning", agent_output.reasoning)
        sandbox.inject("final_answer", agent_output.final_answer)
        sandbox.inject("ground_truth", ground_truth)
        sandbox.inject("feedback", feedback)
        sandbox.inject("skillbook", skillbook.as_prompt() or "(empty skillbook)")

        # Build initial prompt with ONLY metadata (not full content)
        # Full data is injected into sandbox - this is the key RLM benefit
        skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
        initial_prompt = self.prompt_template.format(
            question_length=len(question),
            reasoning_length=len(agent_output.reasoning),
            answer_length=len(agent_output.final_answer),
            ground_truth_length=len(ground_truth) if ground_truth else 0,
            feedback_length=len(feedback) if feedback else 0,
            skillbook_length=len(skillbook_text),
            step_count=len(trace) if trace else 0,
        )

        # REPL loop
        messages: List[Dict[str, str]] = [{"role": "user", "content": initial_prompt}]

        for iteration in range(self.config.max_iterations):
            logger.debug(f"Recursive reflector iteration {iteration + 1}")

            # Get code from LLM
            response = self.llm.complete(
                self._format_messages(messages),
                **kwargs,
            )
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
                    continue

            # Execute code in sandbox with timeout
            logger.debug(f"Executing code:\n{code[:200]}...")
            result = sandbox.execute(code, timeout=self.config.timeout)

            # Check if FINAL() was called
            if sandbox.final_called:
                logger.debug("FINAL() called, parsing result")
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

        # Max iterations reached - build timeout output
        logger.warning(f"Max iterations ({self.config.max_iterations}) reached")
        return self._build_timeout_output(
            question, agent_output, ground_truth, feedback
        )

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
            # Return all code blocks concatenated
            return "\n\n".join(match.strip() for match in matches)

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
