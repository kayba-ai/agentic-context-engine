"""SkillManager role — transforms reflections into actionable skillbook updates."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..llm import LLMClient
from ..skillbook import Skillbook
from ..updates import UpdateBatch
from ..prompt_manager import PromptManager
from .reflector import ReflectorOutput
from ._helpers import _maybe_wrap_with_instructor, maybe_track

if TYPE_CHECKING:
    from ..deduplication import DeduplicationManager

logger = logging.getLogger(__name__)

# Default prompt (v2.1 with {current_date} filled in)
_prompt_manager = PromptManager(default_version="2.1")
SKILL_MANAGER_PROMPT = _prompt_manager.get_skill_manager_prompt()


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class SkillManagerOutput(BaseModel):
    """Output from the SkillManager role containing skillbook update operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    update: UpdateBatch = Field(
        ..., description="Batch of update operations to apply to skillbook"
    )
    consolidation_operations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Operations to consolidate similar skills (MERGE, DELETE, KEEP, UPDATE)",
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )


# ---------------------------------------------------------------------------
# SkillManager
# ---------------------------------------------------------------------------

class SkillManager:
    """
    Transforms reflections into actionable skillbook updates.

    The SkillManager is the third ACE role. It analyzes the Reflector's output
    and decides how to update the skillbook - adding new strategies, updating
    existing ones, or removing harmful patterns.

    Args:
        llm: The LLM client to use for skill management
        prompt_template: Custom prompt template (uses SKILL_MANAGER_PROMPT by default)
        max_retries: Maximum validation retries via Instructor (default: 3)
        dedup_manager: Optional DeduplicationManager for skill deduplication

    Example:
        >>> from ace import SkillManager, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-4")
        >>> skill_manager = SkillManager(client)
        >>>
        >>> # Process reflection to get update operations
        >>> output = skill_manager.update_skills(
        ...     reflection=reflection_output,
        ...     skillbook=skillbook,
        ...     question_context="Math problem solving",
        ...     progress="5/10 problems solved correctly"
        ... )
        >>> # Apply the update to skillbook
        >>> skillbook.apply_update(output.update)

    With Deduplication:
        >>> from ace.deduplication import DeduplicationManager, DeduplicationConfig
        >>> dedup_manager = DeduplicationManager(DeduplicationConfig())
        >>> skill_manager = SkillManager(client, dedup_manager=dedup_manager)
        >>> # SkillManager will now include similarity reports in prompts
        >>> # and handle MERGE/DELETE/KEEP/UPDATE consolidation operations

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Progress: {progress}
        ... Stats: {stats}
        ... Reflection: {reflection}
        ... Skillbook: {skillbook}
        ... Context: {question_context}
        ... Similarity Report: {similarity_report}
        ... Decide what changes to make. Return JSON with update operations.
        ... '''
        >>> skill_manager = SkillManager(client, prompt_template=custom_prompt)

    The SkillManager emits UpdateOperations:
        - ADD: Add new strategy skills
        - UPDATE: Modify existing skills
        - TAG: Update helpful/harmful counts
        - REMOVE: Delete unhelpful skills

    With deduplication enabled, also handles ConsolidationOperations:
        - MERGE: Combine similar skills
        - DELETE: Soft-delete redundant skills
        - KEEP: Mark similar skills as intentionally separate
        - UPDATE: Refine content to differentiate similar skills
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = SKILL_MANAGER_PROMPT,
        *,
        max_retries: int = 3,
        dedup_manager: Optional["DeduplicationManager"] = None,
    ) -> None:
        self.llm = _maybe_wrap_with_instructor(llm, max_retries)
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.dedup_manager = dedup_manager

    @maybe_track(
        name="skill_manager_update_skills",
        tags=["ace-framework", "role", "skill-manager"],
        project_name="ace-roles",
    )
    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        return self._update_skills_impl(
            reflection=reflection,
            skillbook=skillbook,
            question_context=question_context,
            progress=progress,
            **kwargs,
        )

    def _update_skills_impl(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """
        Generate update operations to modify the skillbook based on reflection.

        If a DeduplicationManager is configured, this method will:
        1. Generate a similarity report for similar skill pairs
        2. Include the report in the prompt for the SkillManager to handle
        3. Parse and apply consolidation operations from the response

        Args:
            reflection: The Reflector's analysis of what went right/wrong
            skillbook: Current skillbook to potentially update
            question_context: Description of the task domain or question type
            progress: Current progress summary (e.g., "5/10 correct")
            **kwargs: Additional arguments passed to the LLM

        Returns:
            SkillManagerOutput containing the update operations to apply

        Raises:
            RuntimeError: If unable to produce valid JSON after max_retries
        """
        # Get similarity report if deduplication is enabled
        similarity_report = None
        if self.dedup_manager is not None:
            similarity_report = self.dedup_manager.get_similarity_report(skillbook)
            if similarity_report:
                logger.info("Including similarity report in SkillManager prompt")

        # Serialize reflection with all meaningful fields (not just empty 'raw')
        reflection_data = {
            "reasoning": reflection.reasoning,
            "error_identification": reflection.error_identification,
            "root_cause_analysis": reflection.root_cause_analysis,
            "correct_approach": reflection.correct_approach,
            "key_insight": reflection.key_insight,
            "extracted_learnings": [
                l.model_dump() for l in reflection.extracted_learnings
            ],
        }

        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflection=json.dumps(reflection_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        # Append similarity report if available
        if similarity_report:
            base_prompt = base_prompt + "\n\n" + similarity_report

        # Filter out non-LLM kwargs (like 'sample' used for ReplayAgent)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Use Instructor for automatic validation (always available - core dependency)
        output = self.llm.complete_structured(
            base_prompt, SkillManagerOutput, **llm_kwargs
        )

        # Apply consolidation operations if deduplication is enabled
        if self.dedup_manager is not None and output.consolidation_operations:
            response_data = {
                "consolidation_operations": output.consolidation_operations
            }
            applied_ops = self.dedup_manager.apply_operations_from_response(
                response_data, skillbook
            )
            if applied_ops:
                logger.info(f"Applied {len(applied_ops)} consolidation operations")

        return output
