"""
Benchmark integration for ACE — currently TAU-bench only.

Usage:
    >>> from benchmarks.loaders.tau2 import Tau2Loader
    >>> loader = Tau2Loader()
    >>> for task in loader.load(domain="airline"):
    ...     print(task["task_id"])
"""

from .base import DataLoader
from .loaders.tau2 import Tau2Loader

__all__ = [
    "DataLoader",
    "Tau2Loader",
]

__version__ = "0.1.0"
