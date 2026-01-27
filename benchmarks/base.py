"""
Base classes and interfaces for benchmark integration.

This module provides the foundation for configuration-driven benchmark
evaluation that follows production patterns from lm-evaluation-harness.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ace import Sample, TaskEnvironment, EnvironmentResult


__all__ = [
    "BenchmarkConfig",
    "BenchmarkSample",
    "DataLoader",
    "BenchmarkEnvironment",
    "get_cache_dir",
    "get_data_dir",
]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark task loaded from YAML."""

    task: str
    version: str
    data: Dict[str, Any]
    preprocessing: Dict[str, str]
    metrics: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create BenchmarkConfig from loaded YAML dictionary."""
        return cls(
            task=config_dict["task"],
            version=config_dict["version"],
            data=config_dict["data"],
            preprocessing=config_dict["preprocessing"],
            metrics=config_dict["metrics"],
            metadata=config_dict.get("metadata"),
        )

    @property
    def execution_mode(self) -> str:
        """Get execution mode for this benchmark.

        Execution modes determine how the benchmark runner processes samples:
        - 'standard': Single-turn Q&A evaluation (default)
        - 'iterative': Multi-step agent execution loop (AppWorld, SWE-bench)
        - 'sandbox': Docker-isolated execution with test harness

        Returns:
            Execution mode string from metadata, defaults to 'standard'.
        """
        if self.metadata:
            return self.metadata.get("execution_mode", "standard")
        return "standard"

    @property
    def requires_docker(self) -> bool:
        """Check if this benchmark requires Docker for execution."""
        if self.metadata:
            return self.metadata.get("requires_docker", False)
        return False

    @property
    def max_steps(self) -> int:
        """Get maximum execution steps for iterative benchmarks."""
        if self.metadata:
            return self.metadata.get("max_steps", 50)
        return 50


# Note: BenchmarkSample is now just an alias for Sample for simplicity
# Legacy code can still use BenchmarkSample, but new code should use Sample directly
BenchmarkSample = Sample


class DataLoader(ABC):
    """Abstract base class for loading benchmark data from different sources."""

    @abstractmethod
    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Load benchmark data and yield individual samples."""
        pass

    @abstractmethod
    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        pass


class BenchmarkEnvironment(TaskEnvironment):
    """Base class for benchmark evaluation environments."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_config = {m["name"]: m for m in config.metrics}

    @abstractmethod
    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate agent output against benchmark criteria."""
        pass

    def _compute_metrics(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Compute configured metrics for the benchmark."""
        metrics = {}

        for metric_config in self.config.metrics:
            metric_name = metric_config["name"]

            if metric_name == "exact_match":
                metrics[metric_name] = float(prediction.strip() == ground_truth.strip())
            elif metric_name == "accuracy":
                metrics[metric_name] = float(prediction.strip() == ground_truth.strip())
            elif metric_name == "f1":
                # Simplified F1 - can be extended for token-level F1
                metrics[metric_name] = self._compute_f1(prediction, ground_truth)

        return metrics

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score with proper frequency handling.

        Uses Counter to correctly handle repeated tokens, unlike set-based
        approaches which only count unique token matches.

        Args:
            prediction: The predicted text.
            ground_truth: The expected ground truth text.

        Returns:
            F1 score between 0.0 and 1.0.
        """
        pred_tokens = Counter(prediction.lower().split())
        gt_tokens = Counter(ground_truth.lower().split())

        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0

        # Count common tokens considering frequency (min of both counts)
        common = sum((pred_tokens & gt_tokens).values())
        pred_total = sum(pred_tokens.values())
        gt_total = sum(gt_tokens.values())

        if common == 0:
            return 0.0

        precision = common / pred_total if pred_total else 0.0
        recall = common / gt_total

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)


def get_cache_dir(benchmark_name: str) -> Path:
    """Get cache directory for a benchmark, respecting environment variables."""

    # Check for benchmark-specific cache dir
    cache_dir = os.getenv("BENCHMARK_CACHE_DIR")
    if not cache_dir:
        # Fall back to HuggingFace default location
        cache_dir = os.getenv(
            "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
        )

    cache_path = Path(cache_dir) / "benchmarks" / benchmark_name
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_data_dir(benchmark_name: str) -> Path:
    """Get data directory for benchmarks requiring local storage."""

    data_dir = os.getenv("BENCHMARK_DATA_DIR", "/tmp/benchmark_data")
    data_path = Path(data_dir) / benchmark_name
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path
