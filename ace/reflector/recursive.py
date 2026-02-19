"""Recursive reflector with code execution for trace analysis."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import RecursiveConfig
from .prompts_rr_v3 import REFLECTOR_RECURSIVE_V3_PROMPT
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
    """Return a short preview of text for prompt grounding.

    Escapes curly braces so the result is safe for str.format().
    """
    if not text:
        return "(empty)"
    snippet = text if len(text) <= max_len else text[:max_len]
    return snippet.replace("{", "{{").replace("}", "}}")


def _truncate_output(output: str, max_chars: int = 20000) -> str:
    """Truncate output with metadata suffix showing how much was cut.

    Args:
        output: The output string to potentially truncate
        max_chars: Maximum characters before truncation

    Returns:
        Original output if under limit, otherwise truncated with metadata
    """
    if not output or len(output) <= max_chars:
        return output
    truncated = output[:max_chars]
    remaining = len(output) - max_chars
    return f"{truncated}\n... + [{remaining} chars truncated]"


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
        prompt_template: str = REFLECTOR_RECURSIVE_V3_PROMPT,
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
        # Build trace context from agent output (use pre-built trace if provided)
        trace = agent_output.trace_context or TraceContext.from_agent_output(
            agent_output
        )

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

        # Max iterations reached - build timeout output with optional synthesis
        logger.warning(f"Max iterations ({self.config.max_iterations}) reached")
        return self._build_timeout_output(
            question, agent_output, ground_truth, feedback, messages
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

        # Guard against None/empty response (e.g. Gemini returning None for oversized prompts)
        if not response_text:
            logger.warning(f"LLM returned empty response on iteration {iteration + 1}")
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous response was empty. Please write Python code to analyze the trace.",
                }
            )
            return None

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
            sandbox.reset()
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

        # Reject FINAL() if execution had errors - prevent hallucinated analysis
        if sandbox.final_called and not result.success:
            logger.warning("Rejecting FINAL() called after execution error")
            sandbox.reset()
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Output:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n\n"
                    "Your code had an error. Fix the bug and try again. "
                    "Do NOT call FINAL() until your code executes successfully.",
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

        # Build output message with per-output truncation
        max_output = self.config.max_output_chars
        output_parts = []
        if result.stdout:
            truncated_stdout = _truncate_output(result.stdout, max_output)
            output_parts.append(f"stdout:\n{truncated_stdout}")
        if result.stderr:
            truncated_stderr = _truncate_output(result.stderr, max_output)
            output_parts.append(f"stderr:\n{truncated_stderr}")

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
        """Trim messages to fit within context budget using semantic scoring.

        Scores each iteration by importance and keeps highest-value ones:
        - Errors are high value (debugging context)
        - Findings/insights are high value
        - Substantive output is medium value
        - Empty output is low value

        Always keeps the first message (instructions) and ensures chronological
        order of kept messages.

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

        # Group messages into iteration pairs (assistant response + user output)
        # Messages after first: [asst, user, asst, user, ...]
        iterations: List[tuple] = []
        i = 1
        while i < len(messages):
            if i + 1 < len(messages):
                # Full pair: assistant response + execution output
                iterations.append((i, messages[i], messages[i + 1]))
                i += 2
            else:
                # Single trailing message
                iterations.append((i, messages[i], None))
                i += 1

        # Score and sort iterations by importance
        scored = []
        for idx, asst_msg, user_msg in iterations:
            score = self._score_iteration(asst_msg, user_msg)
            pair_size = len(asst_msg["content"])
            if user_msg:
                pair_size += len(user_msg["content"])
            scored.append((score, idx, asst_msg, user_msg, pair_size))

        # Sort by score descending, keep highest-scoring within budget
        scored.sort(key=lambda x: (-x[0], x[1]))

        kept_indices = []
        used_budget = 0
        for score, idx, asst_msg, user_msg, pair_size in scored:
            if used_budget + pair_size <= remaining_budget:
                kept_indices.append((idx, asst_msg, user_msg))
                used_budget += pair_size

        # Sort kept iterations back to chronological order
        kept_indices.sort(key=lambda x: x[0])

        # Build result list
        kept: List[Dict[str, str]] = []
        for _, asst_msg, user_msg in kept_indices:
            kept.append(asst_msg)
            if user_msg:
                kept.append(user_msg)

        dropped_count = len(iterations) - len(kept_indices)
        if dropped_count > 0:
            # Generate semantic summary of dropped iterations
            dropped_summary = self._summarize_dropped_iterations(
                [(asst, user) for _, _, asst, user, _ in scored[len(kept_indices) :]]
            )
            summary = {
                "role": "user",
                "content": f"[{dropped_count} earlier iterations omitted: {dropped_summary}]",
            }
            return [first, summary] + kept
        return [first] + kept

    def _score_iteration(
        self, assistant_msg: Dict[str, str], user_msg: Optional[Dict[str, str]]
    ) -> float:
        """Score iteration importance for retention priority.

        Higher scores indicate more valuable context to keep.

        Args:
            assistant_msg: The assistant's response (code generated)
            user_msg: The user message with execution output (may be None)

        Returns:
            Importance score (higher = more important to keep)
        """
        score = 0.0

        if user_msg:
            content = user_msg["content"]

            # Errors are high value (debugging context)
            error_indicators = ["Error", "Exception", "Traceback", "stderr:"]
            if any(ind in content for ind in error_indicators):
                score += 3.0

            # Findings/insights are high value
            finding_indicators = [
                "found",
                "pattern",
                "insight",
                "discovered",
                "result:",
            ]
            if any(ind.lower() in content.lower() for ind in finding_indicators):
                score += 2.0

            # Substantive output is medium value
            if len(content) > 500:
                score += 1.0

            # Empty output is low value
            if "(no output)" in content:
                score -= 1.0

        # Code with FINAL() call is high value
        if "FINAL(" in assistant_msg["content"]:
            score += 2.0

        # Code with ask_llm/llm_query is medium value
        if (
            "ask_llm(" in assistant_msg["content"]
            or "llm_query(" in assistant_msg["content"]
        ):
            score += 1.0

        return score

    def _summarize_dropped_iterations(self, dropped: List[tuple]) -> str:
        """Generate a brief semantic summary of dropped iterations.

        Args:
            dropped: List of (assistant_msg, user_msg) tuples that were dropped

        Returns:
            Brief summary string
        """
        if not dropped:
            return "no significant findings"

        summaries = []

        # Check for common patterns in dropped content
        error_count = 0
        explore_count = 0

        for asst_msg, user_msg in dropped:
            if user_msg and any(
                ind in user_msg["content"] for ind in ["Error", "Exception", "stderr:"]
            ):
                error_count += 1
            if "print(" in asst_msg["content"]:
                explore_count += 1

        if error_count:
            summaries.append(f"{error_count} error(s)")
        if explore_count:
            summaries.append(f"{explore_count} exploration(s)")

        if summaries:
            return ", ".join(summaries)
        return "exploration iterations"

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

        Uses a layered extraction approach with fallback chain:
        1. Fenced blocks: ```python, ~~~python, bare ```
        2. Indented blocks: 4-space or tab-indented code
        3. FINAL() extraction: extract just the FINAL(...) call

        Supports batch mode: if first block starts with '# BATCH',
        all code blocks are combined for sequential execution.

        Args:
            response: The LLM response text

        Returns:
            Extracted Python code or None if no code block found
        """
        # Layer 1: Fenced blocks - try ```python first
        matches = self._extract_fenced_blocks(response)
        if matches:
            first_block = matches[0].strip()
            # Explicit batch request: combine all blocks
            if first_block.startswith("# BATCH"):
                return "\n\n".join(m.strip() for m in matches)
            # Default: single block (forces multi-turn iteration)
            return first_block

        # Layer 2: Indented blocks (4 spaces or tab)
        indented = self._extract_indented_block(response)
        if indented and self._looks_like_python(indented):
            return indented

        # Layer 3: FINAL() extraction - last resort
        final_call = self._extract_final_call(response)
        if final_call:
            return final_call

        return None

    def _extract_fenced_blocks(self, response: str) -> List[str]:
        """Extract all fenced code blocks from response.

        Tries multiple fence styles in order:
        1. ```python ... ```
        2. ~~~python ... ~~~
        3. ``` ... ``` (validates as Python)

        Args:
            response: The LLM response text

        Returns:
            List of extracted code blocks (may be empty)
        """
        # Try ```python blocks first
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches

        # Try ~~~python blocks
        pattern = r"~~~python\s*(.*?)~~~"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches

        # Try bare ``` blocks, but validate they look like Python
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            python_matches = [m for m in matches if self._looks_like_python(m)]
            if python_matches:
                return python_matches

        return []

    def _extract_indented_block(self, response: str) -> Optional[str]:
        """Extract code from indented block (4 spaces or tab).

        Finds contiguous lines that are indented and returns them
        with indentation removed.

        Args:
            response: The LLM response text

        Returns:
            Extracted code or None
        """
        lines = response.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            # Check for 4-space or tab indentation
            if line.startswith("    ") or line.startswith("\t"):
                in_code = True
                # Remove one level of indentation
                if line.startswith("    "):
                    code_lines.append(line[4:])
                else:
                    code_lines.append(line[1:])
            elif in_code:
                # End of indented block - stop at first non-blank, non-indented line
                if line.strip():
                    break
                # Allow blank lines within code
                code_lines.append("")

        if code_lines:
            # Trim trailing blank lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            return "\n".join(code_lines)

        return None

    def _extract_final_call(self, response: str) -> Optional[str]:
        """Extract FINAL(...) call from response text.

        Last-resort extraction when no code blocks are found.
        Attempts to extract a complete FINAL() call with balanced parentheses.

        Args:
            response: The LLM response text

        Returns:
            Extracted FINAL() call or None
        """
        # Find FINAL( and extract with balanced parentheses
        match = re.search(r"FINAL\s*\(", response)
        if not match:
            return None

        start = match.start()
        # Find balanced closing paren
        depth = 0
        in_string = None
        escape_next = False

        for i, char in enumerate(response[match.end() - 1 :], start=match.end() - 1):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char in "\"'" and not in_string:
                in_string = char
            elif char == in_string:
                in_string = None
            elif not in_string:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        return response[start : i + 1]

        return None

    def _looks_like_python(self, code: str) -> bool:
        """Check if code looks like valid Python.

        Args:
            code: Code string to validate

        Returns:
            True if code contains Python indicators
        """
        indicators = [
            "def ",
            "import ",
            "print(",
            "FINAL(",
            "for ",
            "if ",
            "while ",
            "class ",
            "return ",
            "= ",
            "==",
            "+=",
            "try:",
            "except",
            "with ",
        ]
        return any(ind in code for ind in indicators)

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
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> "ReflectorOutput":
        """Build a ReflectorOutput when max iterations is reached.

        If enable_fallback_synthesis is True and messages are provided, attempts
        to make a final LLM call to synthesize insights from the conversation
        history before falling back to generic output.

        Args:
            question: The original question
            agent_output: The agent's output
            ground_truth: Expected answer
            feedback: Execution feedback
            messages: Conversation history for fallback synthesis

        Returns:
            ReflectorOutput with timeout analysis or synthesized insights
        """
        from ..roles import ReflectorOutput, ExtractedLearning

        # Try fallback synthesis if enabled and we have conversation history
        if self.config.enable_fallback_synthesis and messages and len(messages) > 1:
            try:
                synthesized = self._attempt_fallback_synthesis(messages)
                if synthesized is not None:
                    logger.info("Fallback synthesis succeeded after timeout")
                    return synthesized
            except Exception as e:
                logger.warning(f"Fallback synthesis failed: {e}")

        # Fall back to generic timeout output
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

    def _attempt_fallback_synthesis(
        self, messages: List[Dict[str, str]]
    ) -> Optional["ReflectorOutput"]:
        """Attempt to synthesize a final answer from conversation history.

        Makes one final LLM call asking to synthesize insights from the
        incomplete analysis.

        Args:
            messages: The conversation history

        Returns:
            ReflectorOutput if synthesis succeeded, None otherwise
        """
        synthesis_prompt = """Your analysis timed out before calling FINAL().
Based on your exploration so far, provide your final structured output now.

Call FINAL() with your best assessment using the evidence you gathered.
Include any learnings from the patterns you observed, even if the analysis was incomplete.

If you found no significant insights, call FINAL() with empty extracted_learnings and a brief summary."""

        # Add synthesis request to messages
        synthesis_messages = messages.copy()
        synthesis_messages.append({"role": "user", "content": synthesis_prompt})

        # Make one synthesis attempt
        response = self.llm.complete_messages(synthesis_messages)
        response_text = response.text

        # Try to extract FINAL() call from response
        code = self._extract_code(response_text)
        if code and "FINAL(" in code:
            # Execute just enough to capture FINAL() call
            from .sandbox import TraceSandbox

            sandbox = TraceSandbox(trace=None)
            result = sandbox.execute(code, timeout=10.0)
            if sandbox.final_called:
                return self._parse_final_value(sandbox.final_value)

        # Try direct JSON parsing as fallback
        try:
            return self._parse_direct_response(response_text)
        except Exception:
            pass

        return None
