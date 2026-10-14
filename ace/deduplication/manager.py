"""DeduplicationManager — coordinates similarity detection and consolidation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from ..protocols.deduplication import DeduplicationConfig
from .detector import SimilarityDetector
from .operations import (
    ConsolidationOperation,
    ConsolidationOpType,
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


class _ConsolidationOperationModel(BaseModel):
    type: Literal["MERGE", "DELETE", "KEEP", "UPDATE"]
    source_ids: List[str] = []
    keep_id: str = ""
    merged_content: str = ""
    skill_id: str = ""
    skill_ids: List[str] = []
    differentiation: str = ""
    new_content: str = ""
    reasoning: str = ""


class _ConsolidationPlan(BaseModel):
    consolidation_operations: List[_ConsolidationOperationModel] = []


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
        self._consolidator: Any = None
        if self.config.consolidation_model is not None:
            from pydantic_ai import Agent as PydanticAgent

            from ..providers.pydantic_ai import resolve_model

            model = self.config.consolidation_model
            self._consolidator = PydanticAgent(
                resolve_model(model) if isinstance(model, str) else model,
                output_type=_ConsolidationPlan,
                defer_model_check=True,
            )

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

    def consolidate(self, skillbook: "Skillbook") -> List[ConsolidationOperation]:
        """Detect similar skills and apply the consolidation the model picks.

        Without ``config.consolidation_model`` this only logs the
        similarity report and returns an empty list.
        """
        report = self.get_similarity_report(skillbook)
        if not report:
            return []
        if self._consolidator is None:
            logger.info(
                "No consolidation_model configured; similarity report only:\n%s",
                report,
            )
            return []

        try:
            plan = self._consolidator.run_sync(report).output
        except Exception as e:
            logger.warning("Skill consolidation failed, skillbook unchanged: %s", e)
            return []
        return self.apply_operations_from_response(
            plan.model_dump(exclude_defaults=True), skillbook
        )

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
            raw_type = raw_op.get("type", "").upper()

            try:
                op_type = ConsolidationOpType(raw_type)
            except ValueError:
                logger.warning("Unknown consolidation operation type: %r", raw_type)
                continue

            try:
                if op_type is ConsolidationOpType.MERGE:
                    operations.append(
                        MergeOp(
                            source_ids=raw_op.get("source_ids", []),
                            merged_content=raw_op.get("merged_content", ""),
                            keep_id=raw_op.get("keep_id", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type is ConsolidationOpType.DELETE:
                    operations.append(
                        DeleteOp(
                            skill_id=raw_op.get("skill_id", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type is ConsolidationOpType.KEEP:
                    operations.append(
                        KeepOp(
                            skill_ids=raw_op.get("skill_ids", []),
                            differentiation=raw_op.get("differentiation", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
                elif op_type is ConsolidationOpType.UPDATE:
                    operations.append(
                        UpdateOp(
                            skill_id=raw_op.get("skill_id", ""),
                            new_content=raw_op.get("new_content", ""),
                            reasoning=raw_op.get("reasoning", ""),
                        )
                    )
            except Exception as e:
                logger.warning(
                    "Failed to parse consolidation operation (%s): %s",
                    type(e).__name__,
                    e,
                )

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
