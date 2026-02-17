"""Data loaders for benchmark sources."""

from ..base import DataLoader

__all__ = ["DataLoader"]

# Tau2 loader is imported conditionally since tau2-bench might not be installed
try:
    from .tau2 import Tau2Loader

    __all__.append("Tau2Loader")
except ImportError:
    pass
