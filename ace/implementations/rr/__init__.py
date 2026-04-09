"""Recursive Reflector implementation.

Config, prompts, and tools live here. The ``RecursiveReflector`` class
and the pipeline step ``RRStep`` live in ``ace.steps.rr_step``.
"""

from .config import RecursiveConfig as RRConfig
from .tools import RRDeps

__all__ = [
    "RRConfig",
    "RRDeps",
]
