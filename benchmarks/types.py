"""Type definitions for benchmark results.

This module provides structured dataclasses for benchmark evaluation results,
enabling type-safe handling of evaluation outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SampleResult:
    """Result for a single sample evaluation."""

    sample_id: str
    question: str
    prediction: str
    ground_truth: str
    metrics: Dict[str, float]
    feedback: str
    split: str  # "train", "test", "baseline", "online"
    step: Optional[int] = None


@dataclass
class SummaryMetrics:
    """Aggregated metrics across samples."""

    metrics: Dict[str, float] = field(default_factory=dict)

    def add_metric(self, name: str, mean: float, min_val: float, max_val: float):
        """Add a metric with its statistics."""
        self.metrics[f"{name}_mean"] = mean
        self.metrics[f"{name}_min"] = min_val
        self.metrics[f"{name}_max"] = max_val

    def get(self, key: str, default: float = 0.0) -> float:
        """Get a metric value by key."""
        return self.metrics.get(key, default)

    def items(self):
        """Return items for iteration."""
        return self.metrics.items()

    def __iter__(self):
        """Allow iteration over metric keys."""
        return iter(self.metrics)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a benchmark run.

    This dataclass captures all outputs from a benchmark evaluation,
    including per-sample results, summary statistics, and metadata.
    """

    benchmark: str
    model: str
    prompt_version: str
    evaluation_mode: str  # "baseline", "online", "offline_train_test_split"
    samples_evaluated: int
    results: List[SampleResult]
    summary: SummaryMetrics

    # Optional fields for train/test split mode
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    split_ratio: Optional[float] = None
    epochs: Optional[int] = None
    train_summary: Optional[SummaryMetrics] = None
    test_summary: Optional[SummaryMetrics] = None
    overfitting_gap: Optional[Dict[str, float]] = None

    # Timing and tokens (added after evaluation)
    elapsed_seconds: Optional[float] = None
    tokens: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "benchmark": self.benchmark,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "evaluation_mode": self.evaluation_mode,
            "samples_evaluated": self.samples_evaluated,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "prediction": r.prediction,
                    "ground_truth": r.ground_truth,
                    "metrics": r.metrics,
                    "feedback": r.feedback,
                    "split": r.split,
                    "step": r.step,
                }
                for r in self.results
            ],
            "summary": self.summary.metrics,
        }

        # Add optional fields if present
        if self.train_samples is not None:
            result["train_samples"] = self.train_samples
        if self.test_samples is not None:
            result["test_samples"] = self.test_samples
        if self.split_ratio is not None:
            result["split_ratio"] = self.split_ratio
        if self.epochs is not None:
            result["epochs"] = self.epochs
        if self.train_summary is not None:
            result["train_summary"] = self.train_summary.metrics
        if self.test_summary is not None:
            result["test_summary"] = self.test_summary.metrics
        if self.overfitting_gap is not None:
            result["overfitting_gap"] = self.overfitting_gap
        if self.elapsed_seconds is not None:
            result["elapsed_seconds"] = self.elapsed_seconds
        if self.tokens is not None:
            result["tokens"] = self.tokens

        return result


__all__ = [
    "SampleResult",
    "SummaryMetrics",
    "EvaluationResult",
]
