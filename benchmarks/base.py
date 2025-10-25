"""
Base classes and interfaces for benchmark integration.

This module provides the foundation for configuration-driven benchmark
evaluation that follows production patterns from lm-evaluation-harness.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from ace import Sample, TaskEnvironment, EnvironmentResult


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
            metadata=config_dict.get("metadata")
        )


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
    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        """Evaluate generator output against benchmark criteria."""
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
        """Compute F1 score between prediction and ground truth."""
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())

        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0

        intersection = pred_tokens & gt_tokens
        if not intersection:
            return 0.0

        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(gt_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)


def get_cache_dir(benchmark_name: str) -> Path:
    """Get cache directory for a benchmark, respecting environment variables."""

    # Check for benchmark-specific cache dir
    cache_dir = os.getenv("BENCHMARK_CACHE_DIR")
    if not cache_dir:
        # Fall back to HuggingFace default location
        cache_dir = os.getenv("HF_DATASETS_CACHE",
                             os.path.expanduser("~/.cache/huggingface/datasets"))

    cache_path = Path(cache_dir) / "benchmarks" / benchmark_name
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_data_dir(benchmark_name: str) -> Path:
    """Get data directory for benchmarks requiring local storage."""

    data_dir = os.getenv("BENCHMARK_DATA_DIR", "/tmp/benchmark_data")
    data_path = Path(data_dir) / benchmark_name
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path