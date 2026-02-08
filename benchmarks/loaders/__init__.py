"""Data loaders for different benchmark sources."""

from ..base import DataLoader
from .huggingface import HuggingFaceLoader

# Build __all__ list dynamically based on available loaders
__all__ = ["DataLoader", "HuggingFaceLoader"]

# AppWorld loader is imported conditionally since appworld might not be installed
try:
    from .appworld import AppWorldLoader

    __all__.append("AppWorldLoader")
except ImportError:
    pass

# Tau2 loader is imported conditionally since tau2-bench might not be installed
try:
    from .tau2 import Tau2Loader

    __all__.append("Tau2Loader")
except ImportError:
    pass
