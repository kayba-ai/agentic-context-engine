"""Core types for the ACE pipeline: ACESample, SkillbookView, ACEStepContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

from pipeline import StepContext

from .outputs import AgentOutput, ReflectorOutput
from .skillbook import Skill, Skillbook, UpdateBatch


# ---------------------------------------------------------------------------
# ACESample — structural protocol for step access
# ---------------------------------------------------------------------------


@runtime_checkable
class ACESample(Protocol):
    """Minimal interface that Sample satisfies.

    Steps access ``ctx.sample.question`` uniformly. This protocol makes
    the duck typing explicit and type-safe — ``Sample`` satisfies it
    structurally without inheriting from it.
    """

    @property
    def question(self) -> str: ...

    @property
    def context(self) -> str: ...

    @property
    def ground_truth(self) -> str | None: ...

    @property
    def metadata(self) -> dict: ...


# ---------------------------------------------------------------------------
# SkillbookView — read-only projection
# ---------------------------------------------------------------------------


class SkillbookView:
    """Read-only projection of a Skillbook.

    Wraps a ``Skillbook`` and exposes only read methods. Write methods
    don't exist on this class — calling them raises ``AttributeError``
    at runtime and a type error at check time.

    Safe to place on a frozen ``ACEStepContext``. Steps that need to
    write to the skillbook receive the real ``Skillbook`` via constructor
    injection.
    """

    __slots__ = ("_sb",)

    def __init__(self, skillbook: Skillbook) -> None:
        self._sb = skillbook

    # -- Read methods delegated to the underlying Skillbook --

    def as_prompt(self) -> str:
        """Return the TOON-encoded skillbook for LLM consumption."""
        return self._sb.as_prompt()

    def get_skill(self, skill_id: str) -> Skill | None:
        """Look up a skill by ID."""
        return self._sb.get_skill(skill_id)

    def skills(self, include_invalid: bool = False) -> list[Skill]:
        """Return all active skills (or all including invalid)."""
        return self._sb.skills(include_invalid=include_invalid)

    def stats(self) -> dict[str, object]:
        """Return skillbook statistics."""
        return self._sb.stats()

    def __len__(self) -> int:
        return len(self._sb.skills())

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._sb.skills())

    def __repr__(self) -> str:
        return f"SkillbookView({len(self)} skills)"


# ---------------------------------------------------------------------------
# ACEStepContext — immutable context for the ACE pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ACEStepContext(StepContext):
    """Immutable context carrying all step-to-step data for the ACE pipeline.

    The pipeline engine only knows about ``sample`` and ``metadata``; all
    ACE-specific fields live here.

    The ``skillbook`` field is a ``SkillbookView`` (read-only). Steps that
    need to write to the skillbook receive the real ``Skillbook`` via
    constructor injection.

    The ``trace`` field holds the raw execution record from any external
    system — a browser-use ``AgentHistoryList``, a LangChain result dict,
    a Claude Code transcript, or any arbitrary Python object. It has no
    enforced schema. The Reflector receives the raw trace and is
    responsible for making sense of it.
    """

    # -- Domain fields --
    skillbook: SkillbookView | None = None
    trace: object | None = None
    agent_output: AgentOutput | None = None
    reflection: ReflectorOutput | None = None
    skill_manager_output: UpdateBatch | None = None

    # -- Progress tracking --
    epoch: int = 1
    total_epochs: int = 1
    step_index: int = 0
    total_steps: int | None = None
    global_sample_index: int = 0
