"""
SWE-bench benchmark loader for software engineering tasks.

Supports both HuggingFace dataset loading and local SWE-bench harness integration.
Requires: pip install swebench (for evaluation harness)
"""

from __future__ import annotations

from typing import Dict, Iterator, Any, Optional
import subprocess
import os

from ..base import DataLoader


class SWEBenchLoader(DataLoader):
    """
    Data loader for SWE-bench benchmark.

    SWE-bench provides real GitHub issues from popular Python repositories
    with corresponding patches. The loader supports:
    - Loading from HuggingFace (princeton-nlp/SWE-bench_Lite)
    - Docker-based evaluation using the SWE-bench harness

    Requirements:
        pip install swebench (for evaluation)
        Docker must be running for execution environment

    Example:
        >>> loader = SWEBenchLoader()
        >>> tasks = loader.load(split="test", limit=50)
        >>> for task in tasks:
        ...     print(task["instance_id"], task["repo"])
    """

    def __init__(self):
        self._docker_available = None
        self._harness_available = None

    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        return source == "swebench"

    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load SWE-bench tasks from HuggingFace.

        Args:
            split: Dataset split ("dev", "test")
            limit: Maximum number of tasks to load
            difficulty: Filter by difficulty if available
            repo: Filter by repository name

        Yields:
            Task dictionaries with problem statement, repo info, and expected patch
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for SWE-bench loader. "
                "Install with: pip install datasets"
            )

        split = kwargs.get("split", "test")
        limit = kwargs.get("limit")
        repo_filter = kwargs.get("repo")
        dataset_path = kwargs.get("dataset_path", "princeton-nlp/SWE-bench_Lite")

        # Load from HuggingFace
        dataset = load_dataset(dataset_path, split=split)

        count = 0
        for item in dataset:
            # Apply repo filter if specified
            if repo_filter and item.get("repo") != repo_filter:
                continue

            yield {
                "instance_id": item.get("instance_id", ""),
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement", ""),
                "hints_text": item.get("hints_text", ""),
                "created_at": item.get("created_at", ""),
                "patch": item.get("patch", ""),  # Ground truth
                "test_patch": item.get("test_patch", ""),
                "version": item.get("version", ""),
                "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
                "environment_setup_commit": item.get("environment_setup_commit", ""),
                "metadata": {
                    "instance_id": item.get("instance_id", ""),
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "test_patch": item.get("test_patch", ""),
                },
            }

            count += 1
            if limit and count >= limit:
                break

    def validate_docker(self) -> bool:
        """Check if Docker is available and running."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            self._docker_available = result.returncode == 0
        except Exception:
            self._docker_available = False

        return self._docker_available

    def validate_harness(self) -> bool:
        """Check if SWE-bench harness is available."""
        if self._harness_available is not None:
            return self._harness_available

        try:
            import swebench
            self._harness_available = True
        except ImportError:
            self._harness_available = False

        return self._harness_available

    def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific SWE-bench instance."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required")

        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

        for item in dataset:
            if item.get("instance_id") == instance_id:
                return dict(item)

        raise ValueError(f"Instance not found: {instance_id}")

    def list_repositories(self) -> list:
        """List all unique repositories in the benchmark."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required")

        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        repos = set()

        for item in dataset:
            if item.get("repo"):
                repos.add(item["repo"])

        return sorted(repos)
