"""
AppWorld benchmark loader with Docker-based execution environment.

Requires: pip install appworld
"""

from __future__ import annotations

from typing import Dict, Iterator, Any, Optional
import os

from ..base import DataLoader


class AppWorldLoader(DataLoader):
    """
    Data loader for AppWorld benchmark.

    AppWorld provides a simulated environment for autonomous agent execution
    with realistic API interactions.

    Requirements:
        pip install appworld
        Docker must be running for execution environment

    Example:
        >>> loader = AppWorldLoader()
        >>> tasks = loader.load(split="test")
        >>> for task in tasks:
        ...     print(task["instruction"])
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize AppWorld loader.

        Args:
            data_dir: Directory for AppWorld data. Defaults to ~/.appworld
        """
        self.data_dir = data_dir or os.path.expanduser("~/.appworld")
        self._world = None

    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        return source == "appworld"

    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load AppWorld tasks.

        Args:
            split: Dataset split ("train", "test", "dev")
            limit: Maximum number of tasks to load
            difficulty: Filter by difficulty ("easy", "medium", "hard")
            category: Filter by task category

        Yields:
            Task dictionaries with instruction, api_docs, and metadata
        """
        try:
            from appworld import AppWorld
        except ImportError:
            raise ImportError(
                "appworld package is required. Install with: pip install appworld\n"
                "Note: AppWorld also requires Docker to be running."
            )

        split = kwargs.get("split", "test")
        limit = kwargs.get("limit")
        difficulty = kwargs.get("difficulty")
        category = kwargs.get("category")

        # Initialize AppWorld
        if self._world is None:
            self._world = AppWorld(data_dir=self.data_dir)

        # Get tasks for the split
        tasks = self._world.get_tasks(split=split)

        # Apply filters
        if difficulty:
            tasks = [t for t in tasks if getattr(t, "difficulty", None) == difficulty]
        if category:
            tasks = [t for t in tasks if getattr(t, "category", None) == category]

        # Apply limit
        if limit:
            tasks = tasks[:limit]

        # Yield task data
        for task in tasks:
            yield {
                "task_id": getattr(task, "task_id", ""),
                "instruction": getattr(task, "instruction", ""),
                "api_docs": self._format_api_docs(getattr(task, "available_apis", [])),
                "difficulty": getattr(task, "difficulty", "unknown"),
                "category": getattr(task, "category", "unknown"),
                "expected_apis": getattr(task, "expected_apis", []),
                "metadata": {
                    "task_id": getattr(task, "task_id", ""),
                    "difficulty": getattr(task, "difficulty", "unknown"),
                    "category": getattr(task, "category", "unknown"),
                    "timeout": getattr(task, "timeout", 300),
                },
            }

    def _format_api_docs(self, apis: list) -> str:
        """Format API documentation for the prompt."""
        docs = []
        for api in apis:
            name = getattr(api, "name", str(api))
            description = getattr(api, "description", "")
            doc = f"- {name}: {description}"

            parameters = getattr(api, "parameters", None)
            if parameters:
                params = ", ".join(
                    f"{getattr(p, 'name', str(p))}: {getattr(p, 'type', 'any')}"
                    for p in parameters
                )
                doc += f"\n  Parameters: {params}"

            returns = getattr(api, "returns", None)
            if returns:
                doc += f"\n  Returns: {returns}"

            docs.append(doc)
        return "\n".join(docs)

    def execute_task(self, task_id: str, agent_actions: list) -> Dict[str, Any]:
        """
        Execute a task with agent-provided actions.

        This method interfaces with AppWorld's Docker execution environment.

        Args:
            task_id: Task identifier
            agent_actions: List of API calls to execute

        Returns:
            Execution results including success status and API call results
        """
        if self._world is None:
            raise RuntimeError("AppWorld not initialized. Call load() first.")

        try:
            result = self._world.execute(task_id, agent_actions)
            return {
                "success": getattr(result, "success", False),
                "api_calls": [
                    {
                        "api": getattr(call, "api_name", ""),
                        "params": getattr(call, "parameters", {}),
                        "success": getattr(call, "success", False),
                        "result": getattr(call, "result", None),
                        "error": getattr(call, "error", None),
                    }
                    for call in getattr(result, "api_calls", [])
                ],
                "final_state": getattr(result, "final_state", None),
                "error": (
                    getattr(result, "error", None)
                    if not getattr(result, "success", False)
                    else None
                ),
            }
        except Exception as e:
            return {
                "success": False,
                "api_calls": [],
                "error": str(e),
            }

    def validate_docker(self) -> bool:
        """Check if Docker is available and running."""
        import subprocess

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific task."""
        if self._world is None:
            raise RuntimeError("AppWorld not initialized. Call load() first.")

        task = self._world.get_task(task_id)
        return {
            "task_id": getattr(task, "task_id", ""),
            "instruction": getattr(task, "instruction", ""),
            "difficulty": getattr(task, "difficulty", "unknown"),
            "category": getattr(task, "category", "unknown"),
            "expected_apis": getattr(task, "expected_apis", []),
            "timeout": getattr(task, "timeout", 300),
        }
