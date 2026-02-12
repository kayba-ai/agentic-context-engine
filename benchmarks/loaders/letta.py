"""
Letta benchmark loader for memory and conversation evaluation.

Requires: pip install letta-evals
"""

from __future__ import annotations

from typing import Dict, Iterator, Any, Optional

from ..base import DataLoader


class LettaLoader(DataLoader):
    """
    Data loader for Letta benchmark (formerly MemGPT evaluations).

    Supports multiple evaluation types:
    - memory: Tests long-term memory recall
    - conversation: Tests multi-turn conversation quality
    - tool_use: Tests tool selection and usage

    Requirements:
        pip install letta-evals
    """

    def __init__(self):
        self._letta = None

    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        return source == "letta"

    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load Letta benchmark tasks.

        Args:
            split: Dataset split ("train", "test", "dev")
            limit: Maximum number of tasks
            task_type: Type of evaluation ("memory", "conversation", "tool_use")

        Yields:
            Task dictionaries with conversation history, query, and expected response
        """
        try:
            from letta_evals import LettaBenchmark, TaskType
        except ImportError:
            raise ImportError(
                "letta-evals package is required. Install with: pip install letta-evals"
            )

        split = kwargs.get("split", "test")
        limit = kwargs.get("limit")
        task_type_str = kwargs.get("task_type", "memory")

        # Map string to TaskType enum
        task_type_map = {
            "memory": TaskType.MEMORY_RECALL,
            "conversation": TaskType.CONVERSATION,
            "tool_use": TaskType.TOOL_USE,
        }
        task_type = task_type_map.get(task_type_str, TaskType.MEMORY_RECALL)

        # Initialize benchmark
        if self._letta is None:
            self._letta = LettaBenchmark()

        # Get tasks
        tasks = self._letta.get_tasks(split=split, task_type=task_type)

        if limit:
            tasks = tasks[:limit]

        for task in tasks:
            yield {
                "task_id": getattr(task, "task_id", ""),
                "conversation_history": self._format_history(
                    getattr(task, "conversation_history", [])
                ),
                "query": getattr(task, "current_query", ""),
                "expected_response": getattr(task, "expected_response", ""),
                "relevant_memories": getattr(task, "relevant_memories", []),
                "task_type": task_type_str,
                "metadata": {
                    "task_id": getattr(task, "task_id", ""),
                    "difficulty": getattr(task, "difficulty", "unknown"),
                    "memory_depth": getattr(task, "memory_depth", 0),
                    "conversation_turns": len(
                        getattr(task, "conversation_history", [])
                    ),
                },
            }

    def _format_history(self, history: list) -> str:
        """Format conversation history for the prompt."""
        formatted = []
        for turn in history:
            if isinstance(turn, dict):
                role = turn.get("role", "user")
                content = turn.get("content", "")
            else:
                # Handle non-dict turns (e.g., objects with attributes)
                role = getattr(turn, "role", "user")
                content = getattr(turn, "content", str(turn))
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)
