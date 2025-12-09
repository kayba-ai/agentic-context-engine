"""Data loaders for different benchmark sources."""

from ..base import DataLoader
from .huggingface import HuggingFaceLoader

__all__ = ["DataLoader", "HuggingFaceLoader"]

# AppWorld loader is imported conditionally since appworld might not be installed
try:
    from .appworld import AppWorldLoader

    __all__.append("AppWorldLoader")
except ImportError:
    pass

# Letta loader is imported conditionally since letta-evals might not be installed
try:
    from .letta import LettaLoader

    __all__.append("LettaLoader")
except ImportError:
    pass
