"""Structured trace context for programmatic exploration in recursive reflector."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..roles import AgentOutput


@dataclass
class TraceStep:
    """A single step in an agent's execution trace.

    Represents one action/thought/observation cycle in agent reasoning.
    """

    index: int
    action: str
    thought: str
    observation: str
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"TraceStep({self.index}: {self.action[:30]}...)"

    def __str__(self) -> str:
        """Return a detailed string representation with content preview."""
        parts = [f"Step {self.index} [{self.action}]"]
        if self.thought:
            preview = (
                self.thought[:200] + "..." if len(self.thought) > 200 else self.thought
            )
            parts.append(f"  Thought: {preview}")
        if self.observation:
            preview = (
                self.observation[:200] + "..."
                if len(self.observation) > 200
                else self.observation
            )
            parts.append(f"  Observation: {preview}")
        return "\n".join(parts)

    @property
    def content(self) -> str:
        """Return the main content (thought + observation)."""
        parts = []
        if self.thought:
            parts.append(self.thought)
        if self.observation:
            parts.append(self.observation)
        return "\n".join(parts)

    def preview(self, max_len: int = 300) -> str:
        """Return a truncated preview of the step content.

        Args:
            max_len: Maximum length of the preview (default: 300)

        Returns:
            Truncated content with character count if truncated
        """
        content = self.content
        if len(content) <= max_len:
            return content
        return content[:max_len] + f"... ({len(content) - max_len} more chars)"


class TraceContext:
    """Structured trace for programmatic exploration by the recursive reflector.

    This class wraps agent reasoning traces and provides utility methods
    for the LLM-generated code to explore and analyze execution patterns.

    Example:
        >>> trace = TraceContext.from_agent_output(agent_output)
        >>> errors = trace.get_errors()
        >>> first_step = trace.get_step(0)
        >>> search_steps = trace.find_steps("search")
    """

    def __init__(self, steps: List[TraceStep], raw_reasoning: str = "") -> None:
        """Initialize TraceContext with a list of steps.

        Args:
            steps: List of TraceStep objects representing the execution trace
            raw_reasoning: The raw reasoning string (for fallback text search)
        """
        self._steps = steps
        self._raw_reasoning = raw_reasoning

    @property
    def steps(self) -> List[TraceStep]:
        """Return all steps in the trace."""
        return self._steps

    @property
    def raw_reasoning(self) -> str:
        """Return the raw reasoning text."""
        return self._raw_reasoning

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def __getitem__(self, index: int) -> TraceStep:
        return self._steps[index]

    def get_step(self, index: int) -> Optional[TraceStep]:
        """Get a step by index.

        Args:
            index: The step index (0-based)

        Returns:
            The TraceStep at the given index, or None if out of bounds
        """
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    def find_steps(self, pattern: str, case_sensitive: bool = False) -> List[TraceStep]:
        """Find steps matching a pattern in action, thought, or observation.

        Args:
            pattern: String pattern to search for
            case_sensitive: Whether the search is case-sensitive

        Returns:
            List of TraceStep objects that contain the pattern
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(re.escape(pattern), flags)

        return [
            s
            for s in self._steps
            if regex.search(s.action)
            or regex.search(s.thought)
            or regex.search(s.observation)
        ]

    def find_steps_regex(self, pattern: str, flags: int = 0) -> List[TraceStep]:
        """Find steps matching a regex pattern.

        Args:
            pattern: Regex pattern to search for
            flags: Regex flags (e.g., re.IGNORECASE)

        Returns:
            List of TraceStep objects matching the regex
        """
        regex = re.compile(pattern, flags)
        return [
            s
            for s in self._steps
            if regex.search(s.action)
            or regex.search(s.thought)
            or regex.search(s.observation)
        ]

    def get_errors(self) -> List[TraceStep]:
        """Find steps that contain error indicators.

        Returns:
            List of TraceStep objects that appear to contain errors
        """
        error_patterns = ["error", "exception", "failed", "failure", "traceback"]
        error_regex = re.compile("|".join(error_patterns), re.IGNORECASE)

        return [
            s
            for s in self._steps
            if error_regex.search(s.action)
            or error_regex.search(s.thought)
            or error_regex.search(s.observation)
        ]

    def get_actions(self, action_type: str) -> List[TraceStep]:
        """Get all steps with a specific action type.

        Args:
            action_type: The action type to filter by

        Returns:
            List of TraceStep objects with matching action type
        """
        return [s for s in self._steps if action_type.lower() in s.action.lower()]

    def summary(self) -> str:
        """Generate a brief summary of the trace.

        Returns:
            A string summary of the trace
        """
        if not self._steps:
            return "Empty trace (no structured steps)"

        return (
            f"Trace with {len(self._steps)} steps: "
            f"{self._steps[0].action[:20]}... -> {self._steps[-1].action[:20]}..."
        )

    def search_raw(self, pattern: str) -> List[int]:
        """Search steps for a pattern and return matching indices.

        Searches action, thought, and observation fields of each step.

        Args:
            pattern: Regex pattern to search for

        Returns:
            List of step indices where pattern was found
        """
        matching_indices = []
        regex = re.compile(pattern, re.IGNORECASE)
        for i, step in enumerate(self._steps):
            content = f"{step.action} {step.thought} {step.observation}"
            if regex.search(content):
                matching_indices.append(i)
        return matching_indices

    def search_raw_text(self, pattern: str) -> List[str]:
        """Search raw reasoning text and return matched substrings.

        Useful when you need the actual matched text rather than step indices.

        Args:
            pattern: Regex pattern to search for

        Returns:
            List of matching substrings from raw reasoning
        """
        return re.findall(pattern, self._raw_reasoning, re.IGNORECASE)

    # -------------------------------------------------------------------------
    # Static helper methods (defined before factory methods that use them)
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_conversation_markers(text: str) -> List[Dict[str, Any]]:
        """Parse [assistant]/[user] markers into message list.

        Handles conversation history stored in reasoning field with markers like:
            [assistant] I'll help you with that...
            [user] Thanks, can you also...
            [assistant] Sure, here's the updated code...

        Args:
            text: Raw text containing [assistant]/[user] markers

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages: List[Dict[str, Any]] = []

        # Pattern matches [role] followed by content until next marker or end
        # Uses non-greedy match and lookahead
        pattern = r"\[(assistant|user)\]\s*(.*?)(?=\[(?:assistant|user)\]|$)"

        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            role = match.group(1).lower()
            content = match.group(2).strip()

            if content:  # Only add non-empty messages
                messages.append({"role": role, "content": content})

        return messages

    @staticmethod
    def _parse_reasoning_steps(reasoning: str) -> List[TraceStep]:
        """Parse reasoning string into structured steps.

        Attempts to find numbered steps or thought/action/observation patterns.
        """
        steps = []

        # Try to parse numbered steps (1., 2., etc.)
        step_pattern = re.compile(r"(\d+)\.\s*(.+?)(?=\d+\.|$)", re.DOTALL)
        matches = step_pattern.findall(reasoning)

        if matches:
            for idx, content in matches:
                steps.append(
                    TraceStep(
                        index=int(idx) - 1,
                        action="step",
                        thought=content.strip(),
                        observation="",
                    )
                )
        else:
            # Fallback: single step with entire reasoning
            steps.append(
                TraceStep(
                    index=0,
                    action="reasoning",
                    thought=reasoning.strip(),
                    observation="",
                )
            )

        return steps

    # -------------------------------------------------------------------------
    # Factory class methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_agent_output(cls, agent_output: "AgentOutput") -> "TraceContext":
        """Create a TraceContext from an AgentOutput.

        Auto-detects conversation format with [assistant]/[user] markers and
        creates multiple steps for proper trace exploration. Falls back to
        single-step trace for simple reasoning strings.

        Args:
            agent_output: The AgentOutput to convert

        Returns:
            A TraceContext representing the agent's reasoning
        """
        reasoning = agent_output.reasoning

        # Auto-detect conversation markers [assistant]/[user]
        if "[assistant]" in reasoning or "[user]" in reasoning:
            messages = cls._parse_conversation_markers(reasoning)
            if messages:
                # Use from_conversation_history for proper multi-step trace
                trace = cls.from_conversation_history(messages)
                # Append final answer as last observation if available
                if trace._steps and agent_output.final_answer:
                    trace._steps[-1] = TraceStep(
                        index=trace._steps[-1].index,
                        action=trace._steps[-1].action,
                        thought=trace._steps[-1].thought,
                        observation=f"Final Answer: {agent_output.final_answer}",
                    )
                return trace

        # Fallback: single-step trace for simple reasoning
        steps = [
            TraceStep(
                index=0,
                action="reasoning",
                thought=agent_output.reasoning,
                observation=f"Answer: {agent_output.final_answer}",
            )
        ]
        return cls(steps=steps, raw_reasoning=agent_output.reasoning)

    @classmethod
    def from_reasoning_string(cls, reasoning: str) -> "TraceContext":
        """Create a TraceContext from a raw reasoning string.

        Attempts to parse structured steps from the reasoning, falling back
        to a single-step trace if parsing fails.

        Args:
            reasoning: The raw reasoning string

        Returns:
            A TraceContext parsed from the reasoning
        """
        steps = cls._parse_reasoning_steps(reasoning)
        return cls(steps=steps, raw_reasoning=reasoning)

    @classmethod
    def from_browser_use(cls, history: Any) -> "TraceContext":
        """Create a TraceContext from browser-use AgentHistory.

        Args:
            history: A browser-use AgentHistory object

        Returns:
            A TraceContext representing the browser automation trace
        """
        steps = []
        if hasattr(history, "history"):
            for i, item in enumerate(history.history):
                action = getattr(item, "action", "") or str(type(item).__name__)
                thought = getattr(item, "thought", "") or ""
                observation = getattr(item, "result", "") or ""

                steps.append(
                    TraceStep(
                        index=i,
                        action=str(action),
                        thought=str(thought),
                        observation=str(observation),
                    )
                )

        raw = str(history) if history else ""
        return cls(steps=steps, raw_reasoning=raw)

    @classmethod
    def from_langchain(cls, intermediate_steps: List[Any]) -> "TraceContext":
        """Create a TraceContext from LangChain intermediate_steps.

        Args:
            intermediate_steps: List of (AgentAction, observation) tuples

        Returns:
            A TraceContext representing the LangChain agent trace
        """
        steps = []
        for i, step in enumerate(intermediate_steps):
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step[0], step[1]
                # LangChain AgentAction has tool and tool_input
                action_str = getattr(action, "tool", str(action))
                thought = getattr(action, "log", "") or ""
                observation_str = str(observation)

                steps.append(
                    TraceStep(
                        index=i,
                        action=str(action_str),
                        thought=thought,
                        observation=observation_str,
                    )
                )

        return cls(steps=steps, raw_reasoning=str(intermediate_steps))

    @classmethod
    def from_conversation_history(
        cls, messages: List[Dict[str, Any]], max_text_len: int = 1000
    ) -> "TraceContext":
        """Create TraceContext from conversation message history.

        Parses a list of messages (e.g., from Bloom/Claude conversations) into
        structured trace steps for analysis by the recursive reflector.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Content can be a string or list of content blocks.
            max_text_len: Maximum length for text content (truncated for memory)

        Returns:
            A TraceContext representing the conversation flow
        """
        steps = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", [])

            # Handle content as string or list of blocks
            if isinstance(content, str):
                text = content[:max_text_len]
                tool_use = None
            else:
                # Extract text and tool_use from content blocks
                text_parts = []
                tool_use = None
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_use = block.get("name")
                text = " ".join(text_parts)[:max_text_len]

            # Create action label
            action = f"{role}:{tool_use}" if tool_use else role

            steps.append(
                TraceStep(
                    index=i,
                    action=action,
                    thought=text,
                    observation="",  # Observations come from tool results
                )
            )

        # Build raw reasoning from conversation
        raw_parts = []
        for msg in messages[-20:]:  # Last 20 messages for context
            role = msg.get("role", "")
            content = msg.get("content", [])
            if isinstance(content, str):
                raw_parts.append(f"[{role}] {content[:500]}")
            else:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        raw_parts.append(f"[{role}] {block.get('text', '')[:500]}")

        return cls(steps=steps, raw_reasoning="\n\n".join(raw_parts))

    @classmethod
    def from_tau_simulation(cls, messages: list) -> "TraceContext":
        """Create TraceContext from TAU-bench SimulationRun messages.

        Builds structured steps directly from TAU message objects using
        duck-typing (no TAU imports required). Handles AssistantMessage
        (with tool_calls or content) and ToolMessage objects.

        Args:
            messages: List of TAU-bench message objects with .tool_calls,
                      .content, and .error attributes.

        Returns:
            A TraceContext with structured tool-call steps.
        """
        steps: List[TraceStep] = []
        raw_parts: List[str] = []
        step_idx = 0

        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            content = getattr(msg, "content", None)
            error = getattr(msg, "error", None)

            if tool_calls:
                # AssistantMessage with tool calls
                for tc in tool_calls:
                    name = getattr(tc, "name", "unknown_tool")
                    args = str(getattr(tc, "arguments", {}))
                    if len(args) > 500:
                        args = args[:500] + "..."
                    steps.append(
                        TraceStep(
                            index=step_idx,
                            action=f"tool_call:{name}",
                            thought=args,
                            observation="",
                        )
                    )
                    raw_parts.append(f"Agent: [TOOL] {name}({args})")
                    step_idx += 1
            elif content and not hasattr(msg, "tool_call_id"):
                # AssistantMessage with text content (not a ToolMessage)
                text = content[:1000] if len(content) > 1000 else content
                steps.append(
                    TraceStep(
                        index=step_idx,
                        action="agent_response",
                        thought=text,
                        observation="",
                    )
                )
                raw_parts.append(f"Agent: {text}")
                step_idx += 1
            elif content is not None:
                # ToolMessage — attach as observation to previous step
                text = content[:1000] if len(content) > 1000 else content
                status = "[ERROR]" if error else "[OK]"
                obs = f"{status} {text}"
                if steps:
                    prev = steps[-1]
                    steps[-1] = TraceStep(
                        index=prev.index,
                        action=prev.action,
                        thought=prev.thought,
                        observation=obs,
                    )
                else:
                    steps.append(
                        TraceStep(
                            index=step_idx,
                            action="tool_result",
                            thought="",
                            observation=obs,
                        )
                    )
                    step_idx += 1
                raw_parts.append(f"Tool {status}: {text}")

        return cls(steps=steps, raw_reasoning="\n".join(raw_parts))
