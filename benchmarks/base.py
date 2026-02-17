"""
Base classes for benchmark data loading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator


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
