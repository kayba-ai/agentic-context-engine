"""
TAU2-bench data loader for tool-calling agent evaluation.

This module provides data loading from tau2-bench, a benchmark for evaluating
tool-calling agents in customer service domains (airline, retail, telecom).

Setup Requirements:
    1. Install tau2: pip install ace-framework[tau-bench]
    2. Set TAU2_DATA_DIR environment variable to point to tau2 data directory
    3. Download data from: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List

from ..base import DataLoader

logger = logging.getLogger(__name__)


class Tau2Loader(DataLoader):
    """
    Data loader for TAU2-bench (τ²-bench) tasks.

    TAU2-bench evaluates tool-calling agents in customer service domains:
    - airline: Flight bookings, cancellations, seat changes
    - retail: Order management, returns, product inquiries
    - telecom: Account management, plan changes, billing

    Example:
        >>> loader = Tau2Loader()
        >>> for task in loader.load(domain="airline", task_split="base", limit=10):
        ...     print(task["task_id"], task["instruction"])

    Setup:
        1. Install: pip install ace-framework[tau-bench]
        2. Clone data: git clone https://github.com/sierra-research/tau2-bench
        3. Set environment: export TAU2_DATA_DIR=/path/to/tau2-bench/data
    """

    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        return source == "tau2"

    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load TAU2-bench tasks for a specific domain.

        Args:
            domain: Domain to load tasks from (airline, retail, telecom)
            task_split: Task split to use (base, human, gpt4o) - for airline/retail
            limit: Maximum number of tasks to load
            **kwargs: Additional arguments (unused)

        Yields:
            Dict containing task data:
                - task_id: Unique task identifier
                - instruction: Initial user instruction
                - tools: List of available tool definitions
                - user_llm: LLM model for user simulation
                - domain: Domain name
                - task_split: Split name
                - metadata: Additional task metadata

        Raises:
            ImportError: If tau2 is not installed
            ValueError: If data directory not configured or tasks cannot be loaded
        """
        try:
            from tau2.registry import registry
        except ImportError:
            raise ImportError(
                "tau2 is required for TAU2 loader. "
                "Install with: pip install ace-framework[tau-bench]"
            )

        # Check if data directory is configured
        data_dir = os.environ.get("TAU2_DATA_DIR")
        if not data_dir:
            raise ValueError(
                "TAU2_DATA_DIR environment variable not set. "
                "Please set it to point to the tau2-bench data directory. "
                "Clone data from: https://github.com/sierra-research/tau2-bench"
            )

        domain = kwargs.get("domain", "airline")
        task_split = kwargs.get("task_split", "base")
        limit = kwargs.get("limit")
        user_llm = kwargs.get("user_llm", "gpt-4o-mini")

        # Get tasks for the domain using the registry
        try:
            tasks = self._get_tasks_for_domain(registry, domain, task_split)
        except FileNotFoundError as e:
            raise ValueError(
                f"Failed to load tasks for {domain}/{task_split}. "
                f"Ensure TAU2_DATA_DIR points to valid tau2 data directory. "
                f"Error: {e}"
            )
        except Exception as e:
            raise ValueError(f"Failed to get tasks for {domain}/{task_split}: {e}")

        if not tasks:
            logger.warning(f"No tasks found for {domain}/{task_split}")
            return

        # Apply limit if specified
        if limit:
            tasks = tasks[:limit]

        # Yield each task
        for task in tasks:
            try:
                task_id = getattr(task, "id", str(id(task)))

                # Extract instruction from user_scenario
                instruction = self._extract_instruction(task)

                # Get tools from the domain environment
                tools = self._get_domain_tools(registry, domain)

                yield {
                    "task_id": task_id,
                    "instruction": instruction,
                    "tools": tools,
                    "user_llm": user_llm,
                    "domain": domain,
                    "task_split": task_split,
                    "task": task,  # Store the full task object for gym
                    "metadata": {
                        "task_id": task_id,
                        "domain": domain,
                        "task_split": task_split,
                        "max_steps": 30,
                    },
                }
            except Exception as e:
                logger.warning(f"Failed to process task: {e}")
                continue

    def _extract_instruction(self, task) -> str:
        """Extract instruction text from a tau2 Task object."""
        # Try user_scenario.instructions.reason_for_call first
        if hasattr(task, "user_scenario"):
            scenario = task.user_scenario
            if hasattr(scenario, "instructions"):
                instr = scenario.instructions
                if hasattr(instr, "reason_for_call") and instr.reason_for_call:
                    return str(instr.reason_for_call)

        # Fallback to description
        if hasattr(task, "description") and task.description:
            return str(task.description)

        return ""

    def _get_tasks_for_domain(
        self, registry, domain: str, task_split: str
    ) -> List[Any]:
        """Get tasks for a domain using the appropriate registry method."""
        # Get the task loader function for this domain
        tasks_loader = registry.get_tasks_loader(domain)

        # Load tasks with optional split
        if task_split and task_split != "base":
            # Check if domain supports splits
            splits_loader = registry.get_task_splits_loader(domain)
            if splits_loader:
                splits = splits_loader()
                if task_split in splits:
                    # Filter tasks by split
                    all_tasks = tasks_loader()
                    split_ids = set(splits[task_split])
                    return [t for t in all_tasks if t.id in split_ids]

        # Default: load all tasks for domain
        return tasks_loader()

    def _get_domain_tools(self, registry, domain: str) -> List[Dict[str, Any]]:
        """Get available tools for a domain."""
        try:
            env_constructor = registry.get_env_constructor(domain)
            env = env_constructor()

            # get_tools() returns list of Tool objects
            if hasattr(env, "get_tools"):
                tools = env.get_tools()
                if isinstance(tools, list):
                    # Convert Tool objects to dicts
                    return [
                        {
                            "name": getattr(t, "name", str(t)),
                            "description": getattr(t, "long_desc", ""),
                        }
                        for t in tools
                    ]

        except Exception as e:
            logger.debug(f"Could not get tools for {domain}: {e}")
        return []

    def get_domains(self) -> List[str]:
        """Get list of available domains."""
        return ["airline", "retail", "telecom"]

    def get_task_splits(self) -> List[str]:
        """Get list of available task splits."""
        return ["base", "human", "gpt4o"]

    def get_task_count(self, domain: str, task_split: str = "base") -> int:
        """Get number of tasks available for a domain/split combination."""
        try:
            from tau2.registry import registry

            tasks = self._get_tasks_for_domain(registry, domain, task_split)
            return len(tasks)
        except ImportError:
            return 0
        except Exception:
            return 0

    def check_data_available(self) -> bool:
        """Check if tau2 data is available and configured."""
        data_dir = os.environ.get("TAU2_DATA_DIR")
        if not data_dir:
            return False
        return os.path.isdir(data_dir)
