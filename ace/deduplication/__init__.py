"""Skill deduplication module for ACE framework.

This module provides semantic deduplication for skillbook skills using
embeddings and SkillManager-driven consolidation decisions.
"""

from .config import DeduplicationConfig
from .detector import SimilarityDetector
from .manager import DeduplicationManager
from .operations import (
    ConsolidationOperation,
    DeleteOp,
    KeepOp,
    MergeOp,
    UpdateOp,
    apply_consolidation_operations,
)

__all__ = [
    "DeduplicationConfig",
    "SimilarityDetector",
    "DeduplicationManager",
    "ConsolidationOperation",
    "MergeOp",
    "DeleteOp",
    "KeepOp",
    "UpdateOp",
    "apply_consolidation_operations",
]
