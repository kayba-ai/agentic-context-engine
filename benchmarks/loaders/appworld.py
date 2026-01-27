"""
AppWorld data loader.

AppWorld benchmarks are run via HAL harness which handles data loading
and execution environment setup. This loader provides a minimal interface
for compatibility with the benchmark framework.

For running AppWorld:
    hal-eval --benchmark appworld_test_normal \
        --agent_dir agents/ace_agent \
        --agent_function main.run
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

from ..base import DataLoader


class AppWorldLoader(DataLoader):
    """
    Data loader for AppWorld benchmark.

    AppWorld is run via HAL harness which provides:
    - Isolated Docker execution
    - Proper pydantic v1 dependencies
    - Task loading and evaluation

    This loader provides a minimal interface for framework compatibility.
    Actual execution happens through HAL.
    """

    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        return source == "appworld"

    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Yield placeholder task metadata.

        Note: AppWorld tasks are loaded by HAL harness.
        This method yields minimal metadata for framework compatibility.

        Args:
            dataset: Dataset split (train, dev, test_normal, test_challenge)
            limit: Maximum number of tasks (approximate)

        Yields:
            Placeholder task metadata
        """
        dataset = kwargs.get("dataset", "test_normal")
        limit = kwargs.get("limit", 10)

        # Yield placeholder tasks - actual loading happens in HAL
        for i in range(limit):
            yield {
                "task_id": f"{dataset}_{i}",
                "dataset": dataset,
                "hal_benchmark": f"appworld_{dataset}",
            }

    def get_available_datasets(self) -> List[str]:
        """Get list of available AppWorld datasets."""
        return ["train", "dev", "test_normal", "test_challenge"]
