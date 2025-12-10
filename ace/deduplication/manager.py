"""Deduplication manager for coordinating similarity detection and operations."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .config import DeduplicationConfig
from .detector import SimilarityDetector
from .operations import (
    ConsolidationOperation,
    DeleteOp,
    KeepOp,
    MergeOp,
    UpdateOp,
    apply_consolidation_operations,
)
from .prompts import format_pair_for_logging, generate_similarity_report

if TYPE_CHECKING:
    from ..skillbook import Skillbook

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """Manages similarity detection and feeds info to SkillManager.

    This class coordinates:
    1. Computing/updating embeddings for skills
    2. Detecting similar skill pairs
    3. Generating similarity reports for the SkillManager prompt
    4. Parsing and applying consolidation operations from SkillManager

    Usage:
        manager = DeduplicationManager(config)
        report = manager.get_similarity_report(skillbook)
        # Include report in SkillManager prompt...
        # After SkillManager responds:
        manager.apply_operations_from_response(skill_manager_response, skillbook)
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.detector = SimilarityDetector(config)

    def get_similarity_report(self, skillbook: "Skillbook") -> Optional[str]:
        """Generate similarity report to include in SkillManager prompt.

        This should be called BEFORE the SkillManager runs.

        Args:
            skillbook: The skillbook to analyze

        Returns:
            Formatted similarity report string, or None if no similar pairs found
            or deduplication is disabled
        """
        if not self.config.enabled:
            return None

        # Ensure all skills have embeddings
        self.detector.ensure_embeddings(skillbook)

        # Detect similar pairs
        similar_pairs = self.detector.detect_similar_pairs(skillbook)

        if len(similar_pairs) < self.config.min_pairs_to_report:
            if similar_pairs:
                logger.debug(
                    f"Found {len(similar_pairs)} similar pairs, "
                    f"below threshold of {self.config.min_pairs_to_report}"
                )
            return None

        # Log found pairs
        logger.info(f"Found {len(similar_pairs)} similar skill pairs")
        for skill_a, skill_b, similarity in similar_pairs:
            logger.debug(format_pair_for_logging(skill_a, skill_b, similarity))

        # Generate report
        return generate_similarity_report(similar_pairs)

    def parse_consolidation_operations(
        self, response_data: Dict[str, Any]
    ) -> List[ConsolidationOperation]:
        """Parse consolidation operations from SkillManager response.

        Args:
            response_data: Parsed JSON response from SkillManager

        Returns:
            List of ConsolidationOperation objects
        """
        operations: List[ConsolidationOperation] = []
        raw_ops = response_data.get("consolidation_operations", [])

        if not isinstance(raw_ops, list):
            logger.warning("consolidation_operations is not a list")
            return operations

        for raw_op in raw_ops:
            if not isinstance(raw_op, dict):
                continue

            op_type = raw_op.get("type", "").upper()

            try:
                if op_type == "MERGE":
                    operations.append(
                        MergeOp(
                            source_ids=raw_op.get("source_ids", []),
                            merged_content=raw_op.get("merged_content", ""),
                            keep_id=raw_op.get("keep_id", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type == "DELETE":
                    operations.append(
                        DeleteOp(
                            skill_id=raw_op.get("skill_id", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type == "KEEP":
                    operations.append(
                        KeepOp(
                            skill_ids=raw_op.get("skill_ids", []),
                            differentiation=raw_op.get("differentiation", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type == "UPDATE":
                    operations.append(
                        UpdateOp(
                            skill_id=raw_op.get("skill_id", ""),
                            new_content=raw_op.get("new_content", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                else:
                    logger.warning(f"Unknown consolidation operation type: {op_type}")
            except Exception as e:
                logger.warning(f"Failed to parse consolidation operation: {e}")

        logger.info(f"Parsed {len(operations)} consolidation operations")
        return operations

    def apply_operations(
        self,
        operations: List[ConsolidationOperation],
        skillbook: "Skillbook",
    ) -> None:
        """Apply consolidation operations to the skillbook.

        Args:
            operations: List of operations to apply
            skillbook: Skillbook to modify
        """
        if not operations:
            return

        logger.info(f"Applying {len(operations)} consolidation operations")
        apply_consolidation_operations(operations, skillbook)

    def apply_operations_from_response(
        self,
        response_data: Dict[str, Any],
        skillbook: "Skillbook",
    ) -> List[ConsolidationOperation]:
        """Parse and apply consolidation operations from SkillManager response.

        Convenience method that combines parse and apply.

        Args:
            response_data: Parsed JSON response from SkillManager
            skillbook: Skillbook to modify

        Returns:
            List of operations that were applied
        """
        operations = self.parse_consolidation_operations(response_data)
        self.apply_operations(operations, skillbook)
        return operations
