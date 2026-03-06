"""DeduplicationManager — coordinates similarity detection and consolidation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..protocols.deduplication import DeduplicationConfig
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
    from ..core.skillbook import Skillbook

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """Manages similarity detection and feeds info to SkillManager.

    Coordinates:
    1. Computing / updating embeddings for skills
    2. Detecting similar skill pairs
    3. Generating similarity reports for the SkillManager prompt
    4. Parsing and applying consolidation operations

    Satisfies :class:`DeduplicationManagerLike` via ``get_similarity_report``.
    """

    def __init__(self, config: DeduplicationConfig | None = None) -> None:
        self.config = config or DeduplicationConfig()
        self.detector = SimilarityDetector(self.config)

    # ------------------------------------------------------------------
    # DeduplicationManagerLike interface
    # ------------------------------------------------------------------

    def get_similarity_report(self, skillbook: "Skillbook") -> Optional[str]:
        """Generate a similarity report for the SkillManager prompt.

        Should be called **before** the SkillManager runs.

        Returns:
            Formatted report, or ``None`` if no similar pairs found or
            deduplication is disabled.
        """
        if not self.config.enabled:
            return None

        self.detector.ensure_embeddings(skillbook)
        similar_pairs = self.detector.detect_similar_pairs(skillbook)

        if len(similar_pairs) < self.config.min_pairs_to_report:
            if similar_pairs:
                logger.debug(
                    "Found %d similar pairs, below threshold of %d",
                    len(similar_pairs),
                    self.config.min_pairs_to_report,
                )
            return None

        logger.info("Found %d similar skill pairs", len(similar_pairs))
        for skill_a, skill_b, similarity in similar_pairs:
            logger.debug(format_pair_for_logging(skill_a, skill_b, similarity))

        return generate_similarity_report(similar_pairs)

    # ------------------------------------------------------------------
    # Consolidation operation parsing / application
    # ------------------------------------------------------------------

    def parse_consolidation_operations(
        self, response_data: Dict[str, Any]
    ) -> List[ConsolidationOperation]:
        """Parse consolidation operations from SkillManager response data."""
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
                    logger.warning("Unknown consolidation operation type: %s", op_type)
            except Exception as e:
                logger.warning("Failed to parse consolidation operation: %s", e)

        logger.info("Parsed %d consolidation operations", len(operations))
        return operations

    def apply_operations(
        self,
        operations: List[ConsolidationOperation],
        skillbook: "Skillbook",
    ) -> None:
        """Apply consolidation operations to the skillbook."""
        if not operations:
            return
        logger.info("Applying %d consolidation operations", len(operations))
        apply_consolidation_operations(operations, skillbook)

    def apply_operations_from_response(
        self,
        response_data: Dict[str, Any],
        skillbook: "Skillbook",
    ) -> List[ConsolidationOperation]:
        """Parse and apply consolidation operations in one step."""
        operations = self.parse_consolidation_operations(response_data)
        self.apply_operations(operations, skillbook)
        return operations
