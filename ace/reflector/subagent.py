"""Sub-agent LLM wrapper for trace exploration in the sandbox.

This module provides an LLM wrapper that can be called from within sandbox code
to perform targeted analysis on partial traces or data. The sub-agent is designed
to be a smaller/faster model that has no tools - it just receives a question and
context, then returns insights.

The key insight is that the main reflector (running code) can programmatically
explore the trace and call the sub-agent to get LLM insights on specific parts,
combining code-based analysis with LLM reasoning.

Example usage in sandbox code:
    # Get insights on a specific error
    error_steps = trace.get_errors()
    if error_steps:
        insight = ask_llm(
            question="What caused this error and how to fix it?",
            context=str(error_steps[0])
        )
        print(insight)

    # Analyze a pattern across multiple steps
    pattern_data = [s for s in trace if "retry" in s.observation.lower()]
    insight = ask_llm(
        question="Why did the agent retry so many times?",
        context=json.dumps([{"step": s.index, "obs": s.observation} for s in pattern_data])
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm import LLMClient


# Default system prompt for the sub-agent
DEFAULT_SUBAGENT_SYSTEM_PROMPT = """You are a trace analysis assistant. Your job is to analyze agent execution traces and provide insights.

You will receive:
1. A question about some aspect of an agent's execution
2. Context data (partial trace, code output, or specific data to analyze)

Your response should be:
- Concise and focused on answering the question
- Based only on the provided context
- Actionable when possible (suggest what to do differently)

You have no tools - just analyze the provided context and answer the question directly."""


@dataclass
class SubAgentConfig:
    """Configuration for the sub-agent LLM.

    Attributes:
        model: Model identifier for the sub-agent (e.g., "gpt-4o-mini", "claude-3-haiku")
        max_tokens: Maximum tokens for sub-agent responses (default: 500)
        temperature: Temperature for sub-agent responses (default: 0.3)
        system_prompt: System prompt for the sub-agent
    """

    model: Optional[str] = None  # None means use same model as main reflector
    max_tokens: int = 500
    temperature: float = 0.3
    system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT


class SubAgentLLM:
    """Wrapper for calling a sub-agent LLM from sandbox code.

    This class provides a simple interface for sandbox code to call an LLM
    for targeted analysis. The sub-agent is designed to be stateless and
    tool-less - it just receives context and returns analysis.

    The main use case is allowing the reflector's code to:
    1. Programmatically extract relevant parts of a trace
    2. Ask the sub-agent specific questions about that data
    3. Combine the insights with other code-based analysis

    Example:
        >>> # In RecursiveReflector setup
        >>> from ace.reflector.subagent import SubAgentLLM, SubAgentConfig
        >>> subagent = SubAgentLLM(llm, config=SubAgentConfig(model="gpt-4o-mini"))
        >>> sandbox.inject("ask_llm", subagent.ask)
        >>>
        >>> # In sandbox code
        >>> insight = ask_llm(
        ...     question="What pattern do you see?",
        ...     context="Step 1: search -> no results\\nStep 2: search again -> no results"
        ... )
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: Optional[SubAgentConfig] = None,
        subagent_llm: Optional["LLMClient"] = None,
    ) -> None:
        """Initialize the sub-agent LLM wrapper.

        Args:
            llm: The main LLM client (used if no subagent_llm provided)
            config: Configuration for the sub-agent
            subagent_llm: Optional separate LLM client for sub-agent calls.
                          If provided, this model is used instead of the main llm.
                          This allows using a smaller/faster model for exploration.
        """
        self.main_llm = llm
        self.subagent_llm = subagent_llm
        self.config = config or SubAgentConfig()
        self._call_count = 0
        self._call_history: List[Dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Return the number of sub-agent calls made."""
        return self._call_count

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Return the history of sub-agent calls."""
        return self._call_history

    def reset(self) -> None:
        """Reset call count and history for a new reflection session."""
        self._call_count = 0
        self._call_history = []

    def ask(
        self,
        question: str,
        context: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Ask the sub-agent a question with context.

        This is the main method called from sandbox code. It formats the
        question and context into a prompt, calls the LLM, and returns
        the response.

        Args:
            question: The question to ask about the context
            context: The context data to analyze (trace excerpt, code output, etc.)
            max_tokens: Override max tokens for this call
            temperature: Override temperature for this call

        Returns:
            The sub-agent's response text

        Example:
            >>> insight = ask_llm(
            ...     question="What went wrong in step 3?",
            ...     context="Step 3: Called API -> Error: timeout after 30s"
            ... )
            >>> print(insight)
            "The API call timed out. Consider increasing the timeout or adding retry logic."
        """
        self._call_count += 1

        # Build the prompt
        prompt = self._build_prompt(question, context)

        # Choose which LLM to use
        llm = self.subagent_llm if self.subagent_llm is not None else self.main_llm

        # Call the LLM
        try:
            response = llm.complete(
                prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
            )
            result = response.text
        except Exception as e:
            result = f"(Sub-agent error: {e})"

        # Record the call
        self._call_history.append(
            {
                "call_number": self._call_count,
                "question": question,
                "context_length": len(context),
                "response_length": len(result),
            }
        )

        return result

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the sub-agent.

        Args:
            question: The question to ask
            context: The context data

        Returns:
            Formatted prompt string
        """
        return f"""{self.config.system_prompt}

## Question
{question}

## Context
{context}

## Your Analysis"""


def create_ask_llm_function(
    llm: "LLMClient",
    config: Optional[SubAgentConfig] = None,
    subagent_llm: Optional["LLMClient"] = None,
    max_calls: int = 20,
) -> Callable[[str, str], str]:
    """Create a bounded ask_llm function for use in the sandbox.

    This factory function creates an ask_llm callable that can be injected
    into the sandbox. It includes call limiting to prevent runaway costs.

    Args:
        llm: The main LLM client
        config: Configuration for the sub-agent
        subagent_llm: Optional separate LLM for sub-agent calls
        max_calls: Maximum number of sub-agent calls allowed

    Returns:
        A callable that takes (question, context) and returns a response string

    Example:
        >>> ask_llm = create_ask_llm_function(llm, max_calls=10)
        >>> sandbox.inject("ask_llm", ask_llm)
    """
    subagent = SubAgentLLM(llm, config=config, subagent_llm=subagent_llm)

    def bounded_ask_llm(question: str, context: str = "") -> str:
        """Ask the sub-agent a question with context (bounded by max_calls).

        Args:
            question: The question to ask
            context: The context data to analyze (default: empty string)

        Returns:
            The sub-agent's response, or a limit message if max calls exceeded
        """
        if subagent.call_count >= max_calls:
            return f"(Max {max_calls} sub-agent calls exceeded - continue with available data)"
        return subagent.ask(question, context)

    # Attach metadata for introspection
    bounded_ask_llm.subagent = subagent  # type: ignore
    bounded_ask_llm.max_calls = max_calls  # type: ignore

    return bounded_ask_llm
