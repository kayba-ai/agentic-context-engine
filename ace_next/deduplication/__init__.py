"""Skill deduplication subsystem for ACE framework."""

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
    "DeduplicationManager",
    "SimilarityDetector",
    "ConsolidationOperation",
    "MergeOp",
    "DeleteOp",
    "KeepOp",
    "UpdateOp",
    "apply_consolidation_operations",
]
