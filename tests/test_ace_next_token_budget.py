"""Tests for ace_next token budget feature and active-only stats."""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar

import pytest
from pydantic import BaseModel

from ace_next.core.outputs import SkillManagerOutput, ReflectorOutput, ExtractedLearning
from ace_next.core.skillbook import Skillbook
from ace_next.implementations.skill_manager import SkillManager
from ace_next.protocols.llm import LLMClientLike

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Mock LLM that captures the prompt sent to it
# ---------------------------------------------------------------------------


class _MockLLM:
    """Minimal mock satisfying LLMClientLike; captures the prompt."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        self.last_prompt = prompt
        return ""

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        self.last_prompt = prompt
        # Return a minimal valid SkillManagerOutput
        data = {
            "reasoning": "no-op",
            "operations": [],
        }
        return response_model.model_validate(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> _MockLLM:
    return _MockLLM()


@pytest.fixture
def dummy_reflection() -> ReflectorOutput:
    return ReflectorOutput(
        reasoning="test",
        correct_approach="test",
        key_insight="test",
        extracted_learnings=[
            ExtractedLearning(learning="test learning", atomicity_score=0.9),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStatsActiveOnly:
    """Skillbook.stats() should count only active (non-invalid) skills."""

    def test_stats_active_only(self) -> None:
        sb = Skillbook()
        s1 = sb.add_skill("general", "Skill one", metadata={"helpful": 2})
        s2 = sb.add_skill("general", "Skill two", metadata={"harmful": 1})
        s3 = sb.add_skill("math", "Skill three", metadata={"neutral": 3})

        # Soft-delete one skill
        sb.remove_skill(s2.id, soft=True)

        stats = sb.stats()

        # Only 2 active skills remain
        assert stats["skills"] == 2
        # Tags should reflect only active skills
        assert stats["tags"]["helpful"] == 2
        assert stats["tags"]["harmful"] == 0  # s2 was soft-deleted
        assert stats["tags"]["neutral"] == 3


class TestTokenBudgetInPrompt:
    """When token_budget is set, the prompt stats must include token fields."""

    def test_token_budget_in_prompt(
        self, mock_llm: _MockLLM, dummy_reflection: ReflectorOutput
    ) -> None:
        sb = Skillbook()
        sb.add_skill("general", "A test skill")

        sm = SkillManager(mock_llm, token_budget=80000)
        sm.update_skills(
            reflection=dummy_reflection,
            skillbook=sb,
            question_context="test context",
            progress="1/1",
        )

        assert mock_llm.last_prompt is not None

        # Extract the Stats line from the prompt and parse it
        stats_json = _extract_stats_from_prompt(mock_llm.last_prompt)
        stats = json.loads(stats_json)

        assert "token_estimate" in stats
        assert "token_budget" in stats
        assert "over_budget" in stats
        assert stats["token_budget"] == 80000
        assert isinstance(stats["token_estimate"], int)
        assert isinstance(stats["over_budget"], bool)


class TestNoTokenBudgetInPrompt:
    """When token_budget is not set, token fields must be absent from stats."""

    def test_no_token_budget_in_prompt(
        self, mock_llm: _MockLLM, dummy_reflection: ReflectorOutput
    ) -> None:
        sb = Skillbook()
        sb.add_skill("general", "A test skill")

        sm = SkillManager(mock_llm)  # no token_budget
        sm.update_skills(
            reflection=dummy_reflection,
            skillbook=sb,
            question_context="test context",
            progress="1/1",
        )

        assert mock_llm.last_prompt is not None

        stats_json = _extract_stats_from_prompt(mock_llm.last_prompt)
        stats = json.loads(stats_json)

        assert "token_estimate" not in stats
        assert "token_budget" not in stats
        assert "over_budget" not in stats


# ---------------------------------------------------------------------------
# ASP run plan (documented, not executed)
# ---------------------------------------------------------------------------
#
# To run an ASP experiment with the v3 prompt and token budget using ace_next:
#
#   Model:   bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
#   Traces:  25 sanitised traces from haiku_sanitised_traces_subset/
#   Dedup:   0.7 threshold
#   Epochs:  1
#
#   Variant A — without token_budget:
#     sm = SkillManager(llm)
#
#   Variant B — with token_budget:
#     sm = SkillManager(llm, token_budget=80000)
#
#   Compare final skillbook sizes and stats between A and B.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_stats_from_prompt(prompt: str) -> str:
    """Extract the JSON stats string from the rendered v3 prompt.

    The v3 prompt renders stats on a line like:
        Stats: {"sections": ..., "skills": ...}
    """
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Stats:"):
            return stripped[len("Stats:"):].strip()
    raise ValueError("Could not find 'Stats:' line in prompt")
