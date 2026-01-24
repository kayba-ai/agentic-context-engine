"""Update operations produced by the ACE SkillManager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, cast


OperationType = Literal["ADD", "UPDATE", "TAG", "REMOVE"]


@dataclass
class UpdateOperation:
    """Single mutation to apply to the skillbook."""

    type: OperationType
    section: str
    content: Optional[str] = None
    skill_id: Optional[str] = None
    metadata: Dict[str, int] = field(default_factory=dict)
    justification: Optional[str] = None
    evidence: Optional[str] = None

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "UpdateOperation":
        # Filter metadata for TAG operations to only include valid tags
        metadata_raw = payload.get("metadata") or {}
        metadata: Dict[str, Any] = (
            cast(Dict[str, Any], metadata_raw) if isinstance(metadata_raw, dict) else {}
        )

        if str(payload["type"]).upper() == "TAG":
            # Only include valid tag names for TAG operations
            valid_tags = {"helpful", "harmful", "neutral"}
            metadata = {k: v for k, v in metadata.items() if str(k) in valid_tags}

        op_type = str(payload["type"]).upper()
        if op_type not in ("ADD", "UPDATE", "TAG", "REMOVE"):
            raise ValueError(f"Invalid operation type: {op_type}")

        return cls(
            type=cast(OperationType, op_type),
            section=str(payload.get("section", "")),
            content=(
                str(payload["content"]) if payload.get("content") is not None else None
            ),
            skill_id=(
                str(payload["skill_id"])
                if payload.get("skill_id") is not None
                else None
            ),
            metadata={str(k): int(v) for k, v in metadata.items()},
            justification=(
                str(payload["justification"])
                if payload.get("justification") is not None
                else None
            ),
            evidence=(
                str(payload["evidence"])
                if payload.get("evidence") is not None
                else None
            ),
        )

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {"type": self.type, "section": self.section}
        if self.content is not None:
            data["content"] = self.content
        if self.skill_id is not None:
            data["skill_id"] = self.skill_id
        if self.metadata:
            data["metadata"] = self.metadata
        if self.justification is not None:
            data["justification"] = self.justification
        if self.evidence is not None:
            data["evidence"] = self.evidence
        return data


@dataclass
class UpdateBatch:
    """Bundle of skill manager reasoning and operations."""

    reasoning: str
    operations: List[UpdateOperation] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "UpdateBatch":
        ops_payload = payload.get("operations")
        operations = []
        if isinstance(ops_payload, Iterable):
            for item in ops_payload:
                if isinstance(item, dict):
                    operations.append(UpdateOperation.from_json(item))
        return cls(reasoning=str(payload.get("reasoning", "")), operations=operations)

    def to_json(self) -> Dict[str, object]:
        return {
            "reasoning": self.reasoning,
            "operations": [op.to_json() for op in self.operations],
        }
