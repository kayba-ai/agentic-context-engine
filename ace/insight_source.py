"""Insight source tracing for ACE skills.

Tracks provenance of each skill: which sample triggered the learning,
what error was observed, and where in the trace the insight came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceReference:
    """Points to a specific location in an execution trace.

    Uses structured step indices when a TraceContext is available,
    falls back to text excerpts otherwise.
    """

    step_indices: Optional[List[int]] = None
    action_types: Optional[List[str]] = None
    text_excerpt: Optional[str] = None
    excerpt_location: Optional[str] = None  # "reasoning" | "feedback" | "observation"

    def to_dict(self) -> Dict[str, Any]:
        """Compact serialization - omits None fields."""
        d: Dict[str, Any] = {}
        if self.step_indices is not None:
            d["step_indices"] = self.step_indices
        if self.action_types is not None:
            d["action_types"] = self.action_types
        if self.text_excerpt is not None:
            d["text_excerpt"] = self.text_excerpt
        if self.excerpt_location is not None:
            d["excerpt_location"] = self.excerpt_location
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceReference":
        return cls(
            step_indices=data.get("step_indices"),
            action_types=data.get("action_types"),
            text_excerpt=data.get("text_excerpt"),
            excerpt_location=data.get("excerpt_location"),
        )


@dataclass
class InsightSource:
    """Full provenance metadata for a skill.

    Tracks the sample, epoch/step position, and trace locations
    that produced a particular skill entry.
    """

    sample_question: str
    epoch: int
    step: int
    trace_refs: List[TraceReference] = field(default_factory=list)
    learning_text: Optional[str] = None
    error_identification: Optional[str] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Compact serialization - omits None fields."""
        d: Dict[str, Any] = {
            "sample_question": self.sample_question,
            "epoch": self.epoch,
            "step": self.step,
        }
        if self.trace_refs:
            d["trace_refs"] = [ref.to_dict() for ref in self.trace_refs]
        if self.learning_text is not None:
            d["learning_text"] = self.learning_text
        if self.error_identification is not None:
            d["error_identification"] = self.error_identification
        if self.sample_id is not None:
            d["sample_id"] = self.sample_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsightSource":
        trace_refs = [
            TraceReference.from_dict(ref) for ref in data.get("trace_refs", [])
        ]
        return cls(
            sample_question=data["sample_question"],
            epoch=data["epoch"],
            step=data["step"],
            trace_refs=trace_refs,
            learning_text=data.get("learning_text"),
            error_identification=data.get("error_identification"),
            sample_id=data.get("sample_id"),
        )


# Cap constants
_MAX_QUESTION_LEN = 200
_MAX_EXCERPT_LEN = 200


def build_insight_source(
    *,
    sample_question: str,
    epoch: int,
    step: int,
    error_identification: str,
    agent_output: Any,
    reflection: Any,
    operations: list,
    sample_id: Optional[str] = None,
) -> None:
    """Attach InsightSource metadata to ADD/UPDATE operations in-place.

    Builds trace references from agent_output.trace_context (structured)
    or agent_output.reasoning (text fallback), resolves learning_text via
    each operation's learning_index into reflection.extracted_learnings,
    and assigns insight_source dicts to each operation.

    Args:
        sample_question: The task/question text.
        epoch: Current epoch number.
        step: Current step index.
        error_identification: Reflector's error_identification text.
        agent_output: AgentOutput with optional trace_context.
        reflection: ReflectorOutput with extracted_learnings.
        operations: List of UpdateOperation objects to annotate.
        sample_id: Optional caller-defined identifier (e.g., trace filename).
    """
    # Build trace references
    trace_refs: List[TraceReference] = []
    trace_context = getattr(agent_output, "trace_context", None)

    if trace_context is not None:
        # Structured path: use TraceContext step indices
        error_steps = []
        get_errors = getattr(trace_context, "get_errors", None)
        if callable(get_errors):
            error_steps = get_errors()

        if error_steps:
            indices = []
            action_types = []
            for es in error_steps:
                idx = getattr(es, "index", None) or getattr(es, "step_index", None)
                if idx is not None:
                    indices.append(idx)
                action = getattr(es, "action_type", None) or getattr(es, "type", None)
                if action:
                    action_types.append(str(action))
            trace_refs.append(
                TraceReference(
                    step_indices=indices or None,
                    action_types=action_types or None,
                )
            )
        else:
            # Success path or no errors: reference all steps
            steps = getattr(trace_context, "steps", [])
            if steps:
                indices = []
                for i, s in enumerate(steps):
                    idx = getattr(s, "index", None) or getattr(s, "step_index", None)
                    indices.append(idx if idx is not None else i)
                trace_refs.append(TraceReference(step_indices=indices))
    else:
        # Fallback: text excerpt from reasoning
        reasoning = getattr(agent_output, "reasoning", "")
        if reasoning:
            trace_refs.append(
                TraceReference(
                    text_excerpt=reasoning[:_MAX_EXCERPT_LEN],
                    excerpt_location="reasoning",
                )
            )

    # Get extracted learnings from reflection
    extracted_learnings = getattr(reflection, "extracted_learnings", [])

    # Cap question length
    capped_question = sample_question[:_MAX_QUESTION_LEN]

    # Attach to each ADD/UPDATE operation
    for op in operations:
        op_type = getattr(op, "type", "")
        if op_type not in ("ADD", "UPDATE"):
            continue

        # Resolve learning_text via learning_index
        learning_text: Optional[str] = None
        learning_index = getattr(op, "learning_index", None)
        if learning_index is not None and isinstance(learning_index, int):
            if 0 <= learning_index < len(extracted_learnings):
                learning_text = getattr(
                    extracted_learnings[learning_index], "learning", None
                )

        # Cap error_identification text
        capped_error_id = (
            error_identification[:_MAX_EXCERPT_LEN] if error_identification else None
        )

        source = InsightSource(
            sample_question=capped_question,
            epoch=epoch,
            step=step,
            trace_refs=trace_refs,
            learning_text=learning_text,
            error_identification=capped_error_id,
            sample_id=sample_id,
        )
        op.insight_source = source.to_dict()
