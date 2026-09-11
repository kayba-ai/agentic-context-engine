"""SkillManager tool registrars and dependency container.

The agentic SkillManager operates on the real :class:`Skillbook` via
atomic mutation tools (ADD / UPDATE / REMOVE / TAG) and read-only
inspection tools (search / read). At the top level, tools apply changes
to the real skillbook immediately. A recursive child instead runs
against an isolated copy (see :meth:`SMDeps.for_child`); its mutations
are staged against a fresh snapshot of the parent skillbook, validated,
and atomically committed back under the parent skillbook's lock (see
:meth:`SMDeps.commit_child`). Each mutation appends an
``UpdateOperation`` to ``deps.operations`` so the caller can recover an
audit trail after the run.

Generic tools (``execute_code``, ``recurse``) are provided by
:mod:`ace.core.recursive_agent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, cast

from pydantic_ai import RunContext

from ace.core.insight_source import InsightSource
from ace.core.recursive_agent import AgenticDeps
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent


# ------------------------------------------------------------------
# Dependency container
# ------------------------------------------------------------------


@dataclass
class SMDeps(AgenticDeps):
    """Dependencies injected into SkillManager tool calls via ``RunContext``."""

    skillbook: Optional[Skillbook] = None
    operations: list[UpdateOperation] = field(default_factory=list)
    current_source: Optional[InsightSource] = None

    def for_child(self, *, sandbox: Any) -> "SMDeps":
        child = cast(SMDeps, super().for_child(sandbox=sandbox))

        if self.skillbook is not None:
            child.skillbook = self.skillbook.clone()

        child.operations = []
        return child

    def commit_child(self, child: AgenticDeps) -> dict[str, str]:
        """Atomically merge a successful child's skillbook mutations.

        Returns:
            Mapping of the child's temporary skill IDs (assigned inside
            its isolated copy) to the IDs they were committed under in
            the parent skillbook.
        """
        child = cast(SMDeps, child)

        if self.skillbook is None or child.skillbook is None:
            return {}

        with self.skillbook.lock:
            staged_skillbook = self.skillbook.clone()

            id_map: dict[str, str] = {}
            committed_operations: list[UpdateOperation] = []
            apply_operations: list[UpdateOperation] = []

            for operation in child.operations:
                skill_id = operation.skill_id

                if skill_id is not None:
                    skill_id = id_map.get(skill_id, skill_id)

                if operation.type == "ADD":
                    child_skill_id = operation.skill_id

                    skill = staged_skillbook.add_skill(
                        section=operation.section,
                        issue=operation.issue,
                        keywords=operation.keywords,
                        insight=operation.insight,
                        insight_source=operation.insight_source,
                    )

                    if child_skill_id is not None:
                        id_map[child_skill_id] = skill.id

                    committed_operation = replace(
                        operation,
                        skill_id=skill.id,
                    )
                    committed_operations.append(committed_operation)

                    apply_operations.append(replace(committed_operation, skill_id=None))
                    continue

                committed_operation = replace(
                    operation,
                    skill_id=skill_id,
                )

                staged_skillbook.apply_update(
                    UpdateBatch(
                        reasoning="",
                        operations=[committed_operation],
                    )
                )

                committed_operations.append(committed_operation)
                apply_operations.append(committed_operation)

            self.skillbook.apply_update(
                UpdateBatch(
                    reasoning="",
                    operations=apply_operations,
                )
            )
            self.operations.extend(committed_operations)

        return id_map


def _normalize_keywords(keywords: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        text = str(keyword).strip().lower().replace(" ", "_")
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _derive_operation_source(
    base: InsightSource | None,
    *,
    operation_type: str,
    issue: str | None = None,
    insight: str | None = None,
    reason: str | None = None,
) -> InsightSource | None:
    if base is None:
        return None
    return InsightSource(
        trace_uid=base.trace_uid,
        source_system=base.source_system,
        trace_id=base.trace_id,
        display_name=base.display_name,
        relation=base.relation,
        sample_question=base.sample_question,
        epoch=base.epoch,
        operation_type=operation_type,
        error_identification=issue or base.error_identification,
        learning_text=insight or reason or base.learning_text,
    )


# ------------------------------------------------------------------
# Mutation tools
# ------------------------------------------------------------------


def register_add_skill(agent: "PydanticAgent[SMDeps, Any]") -> None:
    """Register ``add_skill``."""

    @agent.tool
    def add_skill(
        ctx: RunContext[SMDeps],
        section: str,
        issue: str,
        keywords: list[str],
        insight: str | None = None,
    ) -> dict[str, Any]:
        sb = ctx.deps.skillbook
        if sb is None:
            return {"error": "skillbook unavailable"}
        normalized_keywords = _normalize_keywords(keywords)
        op_source = _derive_operation_source(
            ctx.deps.current_source,
            operation_type="ADD",
            issue=issue,
            insight=insight,
        )
        skill = sb.add_skill(
            section=section,
            issue=issue,
            keywords=normalized_keywords,
            insight=insight,
            insight_source=op_source,
        )
        ctx.deps.operations.append(
            UpdateOperation(
                type="ADD",
                section=skill.section,
                issue=issue,
                keywords=normalized_keywords,
                insight=insight,
                skill_id=skill.id,
                insight_source=op_source,
            )
        )
        return {"ok": True, "skill_id": skill.id}


def register_update_skill(agent: "PydanticAgent[SMDeps, Any]") -> None:
    """Register ``update_skill``."""

    @agent.tool
    def update_skill(
        ctx: RunContext[SMDeps],
        skill_id: str,
        issue: str,
        keywords: list[str] | None = None,
        insight: str | None = None,
    ) -> dict[str, Any]:
        sb = ctx.deps.skillbook
        if sb is None:
            return {"error": "skillbook unavailable"}
        normalized_keywords = (
            _normalize_keywords(keywords) if keywords is not None else None
        )
        op_source = _derive_operation_source(
            ctx.deps.current_source,
            operation_type="UPDATE",
            issue=issue,
            insight=insight,
        )
        skill = sb.update_skill(
            skill_id,
            issue=issue,
            keywords=normalized_keywords if normalized_keywords is not None else None,
            insight=insight,
            insight_source=op_source,
        )
        if skill is None:
            return {"error": f"skill not found: {skill_id}"}
        ctx.deps.operations.append(
            UpdateOperation(
                type="UPDATE",
                section=skill.section,
                skill_id=skill_id,
                issue=issue,
                keywords=normalized_keywords or [],
                insight=insight,
                insight_source=op_source,
            )
        )
        return {"ok": True, "skill_id": skill_id}


def register_remove_skill(agent: "PydanticAgent[SMDeps, Any]") -> None:
    """Register ``remove_skill``."""

    @agent.tool
    def remove_skill(
        ctx: RunContext[SMDeps],
        skill_id: str,
        reason: str,
    ) -> dict[str, Any]:
        sb = ctx.deps.skillbook
        if sb is None:
            return {"error": "skillbook unavailable"}
        skill = sb.get_skill(skill_id)
        if skill is None:
            return {"error": f"skill not found: {skill_id}"}
        op_source = _derive_operation_source(
            ctx.deps.current_source,
            operation_type="REMOVE",
            issue=skill.issue,
            reason=reason,
        )
        sb.remove_skill(skill_id, insight_source=op_source)
        ctx.deps.operations.append(
            UpdateOperation(
                type="REMOVE",
                section=skill.section,
                skill_id=skill_id,
                reason=reason,
                insight_source=op_source,
            )
        )
        return {"ok": True, "skill_id": skill_id}


def register_tag_skill(agent: "PydanticAgent[SMDeps, Any]") -> None:
    """Register ``tag_skill``."""

    @agent.tool
    def tag_skill(
        ctx: RunContext[SMDeps],
        skill_id: str,
        delta: Literal[1, -1, 0],
    ) -> dict[str, Any]:
        sb = ctx.deps.skillbook
        if sb is None:
            return {"error": "skillbook unavailable"}
        existing = sb.get_skill(skill_id)
        if existing is None:
            return {"error": f"skill not found: {skill_id}"}
        op_source = _derive_operation_source(
            ctx.deps.current_source,
            operation_type="TAG",
            issue=existing.issue,
            reason=f"effectiveness_delta={int(delta)}",
        )
        skill = sb.tag_skill(skill_id, delta, insight_source=op_source)
        if skill is None:
            return {"error": f"skill not found: {skill_id}"}
        ctx.deps.operations.append(
            UpdateOperation(
                type="TAG",
                section=skill.section,
                skill_id=skill_id,
                metadata={"delta": int(delta)},
                insight_source=op_source,
            )
        )
        return {
            "ok": True,
            "skill_id": skill_id,
            "helpful_count": skill.helpful_count,
            "harmful_count": skill.harmful_count,
            "neutral_count": skill.neutral_count,
        }


# ------------------------------------------------------------------
# Read-only tools
# ------------------------------------------------------------------


def register_sm_read_skill(agent: "PydanticAgent[SMDeps, Any]") -> None:
    """Register ``read_skill``."""

    @agent.tool
    def read_skill(ctx: RunContext[SMDeps], skill_id: str) -> dict[str, Any]:
        sb = ctx.deps.skillbook
        if sb is None:
            return {"error": "skillbook unavailable"}
        skill = sb.get_skill(skill_id)
        if skill is None:
            return {"error": f"skill not found: {skill_id}"}
        return {
            "id": skill.id,
            "section": skill.section,
            "keywords": list(skill.keywords),
            "issue": skill.issue,
            "insight": skill.insight,
            "active": skill.active,
            "used_count": skill.used_count,
            "helpful_count": skill.helpful_count,
            "harmful_count": skill.harmful_count,
            "neutral_count": skill.neutral_count,
            "occurrences": [source.to_dict() for source in skill.occurrences],
        }


def register_sm_search_skills(agent: "PydanticAgent[SMDeps, Any]") -> None:
    """Register ``search_skills``."""

    @agent.tool
    def search_skills(
        ctx: RunContext[SMDeps],
        query: str,
        top_k: int = 5,
        section: str | None = None,
        keywords: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        sb = ctx.deps.skillbook
        if sb is None:
            return [{"error": "skillbook unavailable"}]
        from ace.implementations.skill_rendering import retrieve_top_k

        results = retrieve_top_k(
            sb,
            query,
            top_k=top_k,
            section=section,
            keywords=keywords,
        )
        return [
            {
                "id": skill.id,
                "section": skill.section,
                "keywords": list(skill.keywords),
                "issue": skill.issue,
                "insight": skill.insight,
                "active": skill.active,
                "used_count": skill.used_count,
                "helpful_count": skill.helpful_count,
                "harmful_count": skill.harmful_count,
                "neutral_count": skill.neutral_count,
            }
            for skill in results
        ]
