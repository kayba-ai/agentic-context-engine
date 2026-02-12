"""Constants for benchmark evaluation.

This module centralizes magic numbers and threshold values used throughout
the benchmark framework, making them easy to configure and maintain.
"""


class PerformanceThreshold:
    """Thresholds for performance feedback."""

    EXCELLENT = 0.8
    GOOD = 0.6
    MODERATE = 0.5


class ResponseLimits:
    """Limits for response validation."""

    MIN_WORDS = 5
    MAX_WORDS = 500


class Timeouts:
    """Timeout defaults in seconds."""

    DOCKER_CHECK = 10
    HARNESS_EXECUTION = 600
    LLM_REQUEST = 120


class Tolerance:
    """Numerical tolerance values."""

    EXACT_MATCH = 1e-6
    MATH_ANSWER = 0.001
    WITHIN_1_PERCENT = 0.01
    WITHIN_5_PERCENT = 0.05


class OverfittingThreshold:
    """Thresholds for overfitting analysis."""

    SIGNIFICANT = 0.05
    MINOR = 0.02


__all__ = [
    "PerformanceThreshold",
    "ResponseLimits",
    "Timeouts",
    "Tolerance",
    "OverfittingThreshold",
]
