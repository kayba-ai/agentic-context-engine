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
from typing import Any, Callable, Dict, List, Optional


# --- Mode-specific subagent prompts ---
# Analysis mode: descriptive survey — enables the main agent to categorize and pick deep-dive targets
SUBAGENT_ANALYSIS_PROMPT = """\
You are a trace reader for a multi-phase analysis pipeline. A downstream agent will use your output to categorize traces and decide which ones deserve deep investigation. It will not read the raw traces itself — your summary is its only view into the data.

For each trace or conversation in the context:
1. **Task** — what was requested or attempted (brief).
2. **Approach** — the agent's key steps, tools used, and the overall sequence of actions.
3. **Decision points** — where the agent chose between alternatives. What did it choose and what were the other options?
4. **Mistakes** — errors, wrong turns, retries, wasted steps. Describe what went wrong factually — do not analyze root causes.
5. **What stood out** — anything non-obvious: clever recoveries, unusual tool usage, unexpected results, or signs of a pattern.
6. **Evaluation criteria** — if evaluation criteria, rules, or a checklist are provided in the context, actively evaluate every applicable criterion for every trace — even successful ones. Cite evidence for any violations.

Cite step numbers or message excerpts as evidence. Be thorough — the downstream agent cannot go back to the raw data."""

# Deep-dive mode: analytical — provides evidence-rich material for synthesis
SUBAGENT_DEEPDIVE_PROMPT = """\
You are an investigator analyzing agent execution traces. A downstream agent has already surveyed these traces and selected them for deeper analysis. Your job is to answer the specific question asked, providing the evidence and reasoning the downstream agent needs to formulate learnings.

Approach:
- **Verify before analyzing.** Before investigating causes, check whether the agent's claims and conclusions accurately reflect the data it received. "Confident but wrong" — where the agent proceeds without hesitation based on incorrect reasoning — is a high-value finding that behavioral analysis alone misses.
- **Check against rules.** If agent operating rules or policy are provided, verify that the agent's actions comply with them. Rule violations are high-value findings even when the agent appeared to succeed — they often look "normal" because many traces share the same violation.
- **Causes, not symptoms.** When something went wrong, identify the root decision or assumption that led to it. What should the agent have done instead — concretely?
- **Contrast directly.** When given multiple traces, find the specific point where they diverged. Do not describe each trace separately — compare them.
- **Cite everything.** Every claim must reference specific evidence (step number, message content, tool output). If something is ambiguous, say so — do not speculate.
- **Suggest alternatives.** For mistakes, describe the concrete action the agent should have taken instead."""

# Backward-compat alias
DEFAULT_SUBAGENT_SYSTEM_PROMPT = SUBAGENT_ANALYSIS_PROMPT

# Mode → prompt mapping for _build_prompt
_MODE_PROMPTS = {
    "analysis": SUBAGENT_ANALYSIS_PROMPT,
    "deep_dive": SUBAGENT_DEEPDIVE_PROMPT,
}


class CallBudget:
    """Shared budget for tracking LLM calls across functions.

    Used to enforce a single limit across llm_query and ask_llm,
    preventing the effective budget from being 2x the configured value.
    """

    def __init__(self, max_calls: int) -> None:
        self._max_calls = max_calls
        self._count = 0

    def consume(self) -> bool:
        """Consume one call. Returns False if budget is exhausted."""
        if self._count >= self._max_calls:
            return False
        self._count += 1
        return True

    @property
    def count(self) -> int:
        """Number of calls consumed so far."""
        return self._count

    @property
    def exhausted(self) -> bool:
        """Whether the budget is exhausted."""
        return self._count >= self._max_calls


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
    max_tokens: int = 8192
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
        >>> from ace_next.rr.subagent import SubAgentLLM, SubAgentConfig
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
        llm: Any,
        config: Optional[SubAgentConfig] = None,
        subagent_llm: Optional[Any] = None,
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
        mode: str = "analysis",
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
            mode: Prompt protocol — "analysis" for survey, "deep_dive" for
                  investigation. Unknown modes fall back to config.system_prompt.
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
        prompt = self._build_prompt(question, context, mode=mode)

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
                "mode": mode,
            }
        )

        # Update span metadata (Opik, if available)
        try:
            from opik import opik_context

            opik_context.update_current_span(
                metadata={
                    "question_preview": question[:200],
                    "context_length": len(context),
                    "response_length": len(result),
                    "call_number": self._call_count,
                }
            )
        except Exception:
            pass

        return result

    def _build_prompt(self, question: str, context: str, *, mode: str = "analysis") -> str:
        """Build the prompt for the sub-agent.

        Args:
            question: The question to ask
            context: The context data
            mode: Prompt protocol — known modes use protocol prompts,
                  unknown modes fall back to config.system_prompt.

        Returns:
            Formatted prompt string
        """
        system = _MODE_PROMPTS.get(mode, self.config.system_prompt)
        return f"""{system}

## Question
{question}

## Context
{context}

## Your Analysis"""


def create_ask_llm_function(
    llm: Any,
    config: Optional[SubAgentConfig] = None,
    subagent_llm: Optional[Any] = None,
    max_calls: int = 20,
    budget: Optional[CallBudget] = None,
) -> Callable[[str, str], str]:
    """Create a bounded ask_llm function for use in the sandbox.

    This factory function creates an ask_llm callable that can be injected
    into the sandbox. It includes call limiting to prevent runaway costs.

    Args:
        llm: The main LLM client
        config: Configuration for the sub-agent
        subagent_llm: Optional separate LLM for sub-agent calls
        max_calls: Maximum number of sub-agent calls allowed (standalone limit)
        budget: Optional shared CallBudget (overrides max_calls when provided)

    Returns:
        A callable that takes (question, context) and returns a response string

    Example:
        >>> ask_llm = create_ask_llm_function(llm, max_calls=10)
        >>> sandbox.inject("ask_llm", ask_llm)
    """
    subagent = SubAgentLLM(llm, config=config, subagent_llm=subagent_llm)

    def bounded_ask_llm(question: str, context: str = "", mode: str = "analysis") -> str:
        """Ask the sub-agent a question with context (bounded by budget/max_calls).

        Args:
            question: The question to ask
            context: The context data to analyze (default: empty string)
            mode: Prompt protocol — "analysis" for survey, "deep_dive" for
                  investigation (default: "analysis")

        Returns:
            The sub-agent's response, or a limit message if max calls exceeded
        """
        if budget is not None:
            if not budget.consume():
                return f"(Max {budget._max_calls} LLM calls exceeded - continue with available data)"
        elif subagent.call_count >= max_calls:
            return f"(Max {max_calls} sub-agent calls exceeded - continue with available data)"
        return subagent.ask(question, context, mode=mode)

    # Attach metadata for introspection
    bounded_ask_llm.subagent = subagent  # type: ignore
    bounded_ask_llm.max_calls = max_calls  # type: ignore

    return bounded_ask_llm
