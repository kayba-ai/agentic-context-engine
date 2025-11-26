"""Delta operations produced by the ACE Curator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, cast


OperationType = Literal["ADD", "UPDATE", "TAG", "REMOVE"]


@dataclass
class DeltaOperation:
    """Single mutation to apply to the playbook."""

    type: OperationType
    section: str
    content: Optional[str] = None
    bullet_id: Optional[str] = None
    metadata: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "DeltaOperation":
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
            bullet_id=(
                str(payload["bullet_id"])
                if payload.get("bullet_id") is not None
                else None
            ),
            metadata={str(k): int(v) for k, v in metadata.items()},
        )

    def to_json(self) -> Dict[str, object]:
        data: Dict[str, object] = {"type": self.type, "section": self.section}
        if self.content is not None:
            data["content"] = self.content
        if self.bullet_id is not None:
            data["bullet_id"] = self.bullet_id
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class DeltaBatch:
    """Bundle of curator reasoning and operations."""

    reasoning: str
    operations: List[DeltaOperation] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "DeltaBatch":
        ops_payload = payload.get("operations")
        operations = []
        if isinstance(ops_payload, Iterable):
            for item in ops_payload:
                if isinstance(item, dict):
                    operations.append(DeltaOperation.from_json(item))
        return cls(reasoning=str(payload.get("reasoning", "")), operations=operations)

    def to_json(self) -> Dict[str, object]:
        return {
            "reasoning": self.reasoning,
            "operations": [op.to_json() for op in self.operations],
        }
