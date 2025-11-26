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
    from ..playbook import Playbook

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """Manages similarity detection and feeds info to Curator.

    This class coordinates:
    1. Computing/updating embeddings for bullets
    2. Detecting similar bullet pairs
    3. Generating similarity reports for the Curator prompt
    4. Parsing and applying consolidation operations from Curator

    Usage:
        manager = DeduplicationManager(config)
        report = manager.get_similarity_report(playbook)
        # Include report in Curator prompt...
        # After Curator responds:
        manager.apply_operations_from_response(curator_response, playbook)
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.detector = SimilarityDetector(config)

    def get_similarity_report(self, playbook: "Playbook") -> Optional[str]:
        """Generate similarity report to include in Curator prompt.

        This should be called BEFORE the Curator runs.

        Args:
            playbook: The playbook to analyze

        Returns:
            Formatted similarity report string, or None if no similar pairs found
            or deduplication is disabled
        """
        if not self.config.enabled:
            return None

        # Ensure all bullets have embeddings
        self.detector.ensure_embeddings(playbook)

        # Detect similar pairs
        similar_pairs = self.detector.detect_similar_pairs(playbook)

        if len(similar_pairs) < self.config.min_pairs_to_report:
            if similar_pairs:
                logger.debug(
                    f"Found {len(similar_pairs)} similar pairs, "
                    f"below threshold of {self.config.min_pairs_to_report}"
                )
            return None

        # Log found pairs
        logger.info(f"Found {len(similar_pairs)} similar bullet pairs")
        for bullet_a, bullet_b, similarity in similar_pairs:
            logger.debug(format_pair_for_logging(bullet_a, bullet_b, similarity))

        # Generate report
        return generate_similarity_report(similar_pairs)

    def parse_consolidation_operations(
        self, response_data: Dict[str, Any]
    ) -> List[ConsolidationOperation]:
        """Parse consolidation operations from Curator response.

        Args:
            response_data: Parsed JSON response from Curator

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
                            bullet_id=raw_op.get("bullet_id", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type == "KEEP":
                    operations.append(
                        KeepOp(
                            bullet_ids=raw_op.get("bullet_ids", []),
                            differentiation=raw_op.get("differentiation", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type == "UPDATE":
                    operations.append(
                        UpdateOp(
                            bullet_id=raw_op.get("bullet_id", ""),
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
        playbook: "Playbook",
    ) -> None:
        """Apply consolidation operations to the playbook.

        Args:
            operations: List of operations to apply
            playbook: Playbook to modify
        """
        if not operations:
            return

        logger.info(f"Applying {len(operations)} consolidation operations")
        apply_consolidation_operations(operations, playbook)

    def apply_operations_from_response(
        self,
        response_data: Dict[str, Any],
        playbook: "Playbook",
    ) -> List[ConsolidationOperation]:
        """Parse and apply consolidation operations from Curator response.

        Convenience method that combines parse and apply.

        Args:
            response_data: Parsed JSON response from Curator
            playbook: Playbook to modify

        Returns:
            List of operations that were applied
        """
        operations = self.parse_consolidation_operations(response_data)
        self.apply_operations(operations, playbook)
        return operations
