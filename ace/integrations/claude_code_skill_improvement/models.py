"""Pydantic models for skill improvement workflow.

This module defines the data models used in the skill improvement pipeline,
primarily the PublishOutput which is the result of the Publisher LLM step.

The Reflector and SkillManager outputs are reused from ace.roles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AcceptedChange(BaseModel):
    """A change that was accepted by the Publisher."""

    change: str = Field(..., description="Description of the change made")
    reason: str = Field(..., description="Why this change was accepted")
    evidence: str = Field(
        default="", description="Evidence from transcript supporting this change"
    )


class RejectedLearning(BaseModel):
    """A learning that was rejected by the Publisher."""

    learning: str = Field(..., description="The learning that was rejected")
    reason: str = Field(..., description="Why this learning was rejected")


class PublishOutput(BaseModel):
    """
    Output from the Publisher step.

    The Publisher decides which of the SkillManager's proposed operations
    to apply to the playbook skill file, producing minimal high-signal edits.

    Attributes:
        updated_skill_text: Full new content for the skill file
        accepted: List of changes that were applied
        rejected: List of learnings that were not applied
    """

    updated_skill_text: str = Field(
        ..., description="Full new content for the skill file"
    )
    accepted: List[AcceptedChange] = Field(
        default_factory=list, description="Changes that were applied"
    )
    rejected: List[RejectedLearning] = Field(
        default_factory=list, description="Learnings that were not applied"
    )


class SkillImprovementResult(BaseModel):
    """
    Result of the skill improvement process.

    Contains the outputs from all three LLM calls plus the computed diff.
    """

    # LLM outputs
    reflector_output: Optional[Dict[str, Any]] = Field(
        default=None, description="Output from Reflector step"
    )
    skill_manager_output: Optional[Dict[str, Any]] = Field(
        default=None, description="Output from SkillManager step"
    )
    publish_output: Optional[PublishOutput] = Field(
        default=None, description="Output from Publisher step"
    )

    # Diff information
    original_content: str = Field(default="", description="Original skill file content")
    updated_content: str = Field(default="", description="Updated skill file content")
    diff: str = Field(default="", description="Unified diff between original and updated")
    has_changes: bool = Field(default=False, description="Whether any changes were made")

    # Metadata
    transcript_path: str = Field(default="", description="Path to the transcript file")
    skill_path: str = Field(default="", description="Path to the skill file")
    backup_path: Optional[str] = Field(
        default=None, description="Path to backup file (if --apply was used)"
    )
