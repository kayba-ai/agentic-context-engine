"""Tests for ace_next steps: ReflectStep, TagStep, UpdateStep, ApplyStep, learning_tail."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import (
    AgentOutput,
    ExtractedLearning,
    ReflectorOutput,
    SkillManagerOutput,
    SkillTag,
)
from ace_next.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace_next.steps import learning_tail
from ace_next.steps.apply import ApplyStep
from ace_next.steps.reflect import ReflectStep
from ace_next.steps.tag import TagStep
from ace_next.steps.update import UpdateStep


# ------------------------------------------------------------------ #
# Helpers — mock roles satisfying protocols
# ------------------------------------------------------------------ #


class MockReflector:
    """Minimal mock satisfying ReflectorLike."""

    def __init__(self, output: ReflectorOutput | None = None):
        self.output = output or ReflectorOutput(
            reasoning="test reasoning",
            correct_approach="test approach",
            key_insight="test insight",
            skill_tags=[],
        )
        self.calls: list[dict] = []

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        self.calls.append(
            {
                "question": question,
                "agent_output": agent_output,
                "ground_truth": ground_truth,
                "feedback": feedback,
                **kwargs,
            }
        )
        return self.output


class MockSkillManager:
    """Minimal mock satisfying SkillManagerLike."""

    def __init__(self, output: SkillManagerOutput | None = None):
        self.output = output or SkillManagerOutput(
            update=UpdateBatch(reasoning="test", operations=[]),
        )
        self.calls: list[dict] = []

    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        self.calls.append(
            {
                "reflection": reflection,
                "question_context": question_context,
                "progress": progress,
            }
        )
        return self.output


# ------------------------------------------------------------------ #
# ReflectStep
# ------------------------------------------------------------------ #


class TestReflectStep:
    def test_dict_trace(self):
        """Structured dict trace should extract known fields."""
        reflector = MockReflector()
        step = ReflectStep(reflector)

        trace = {
            "question": "What is 2+2?",
            "answer": "4",
            "reasoning": "simple math",
            "ground_truth": "4",
            "feedback": "Correct!",
            "skill_ids": ["math-001"],
        }
        sb = Skillbook()
        ctx = ACEStepContext(
            trace=trace,
            skillbook=SkillbookView(sb),
        )

        result = step(ctx)
        assert result.reflection is not None
        assert len(reflector.calls) == 1
        call = reflector.calls[0]
        assert call["question"] == "What is 2+2?"
        assert call["agent_output"].final_answer == "4"
        assert call["ground_truth"] == "4"
        assert call["feedback"] == "Correct!"

    def test_raw_trace(self):
        """Non-dict trace should be passed as-is via kwargs."""
        reflector = MockReflector()
        step = ReflectStep(reflector)

        raw_trace = ["step1", "step2", "step3"]
        sb = Skillbook()
        ctx = ACEStepContext(
            trace=raw_trace,
            skillbook=SkillbookView(sb),
        )

        result = step(ctx)
        assert result.reflection is not None
        assert len(reflector.calls) == 1
        call = reflector.calls[0]
        assert call["question"] == ""
        assert call["agent_output"].final_answer == ""
        assert call.get("trace") is raw_trace

    def test_provides_and_requires(self):
        step = ReflectStep(MockReflector())
        assert "trace" in step.requires
        assert "skillbook" in step.requires
        assert "reflection" in step.provides
        assert step.async_boundary is True
        assert step.max_workers == 3


# ------------------------------------------------------------------ #
# TagStep
# ------------------------------------------------------------------ #


class TestTagStep:
    def test_tag_existing_skills(self):
        sb = Skillbook()
        sb.add_skill("math", "content", skill_id="math-001")
        sb.add_skill("writing", "content", skill_id="writing-001")

        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
            skill_tags=[
                SkillTag(id="math-001", tag="helpful"),
                SkillTag(id="writing-001", tag="harmful"),
            ],
        )
        ctx = ACEStepContext(reflection=reflection)
        step = TagStep(sb)
        step(ctx)

        assert sb.get_skill("math-001").helpful == 1
        assert sb.get_skill("writing-001").harmful == 1

    def test_hallucinated_skill_id_no_crash(self):
        """Hallucinated skill IDs should be silently ignored, not crash."""
        sb = Skillbook()
        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
            skill_tags=[
                SkillTag(id="nonexistent-001", tag="helpful"),
            ],
        )
        ctx = ACEStepContext(reflection=reflection)
        step = TagStep(sb)

        result = step(ctx)
        assert result is ctx

    def test_invalid_tag_name_warns(self, caplog):
        """Invalid tag name should warn, not crash."""
        sb = Skillbook()
        sb.add_skill("math", "content", skill_id="math-001")
        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
            skill_tags=[
                SkillTag(id="math-001", tag="invalid_tag"),
            ],
        )
        ctx = ACEStepContext(reflection=reflection)
        step = TagStep(sb)

        with caplog.at_level(logging.WARNING):
            result = step(ctx)

        assert result is ctx
        assert "math-001" in caplog.text

    def test_provides_and_requires(self):
        sb = Skillbook()
        step = TagStep(sb)
        assert "reflection" in step.requires
        assert len(step.provides) == 0
        assert step.max_workers == 1


# ------------------------------------------------------------------ #
# UpdateStep
# ------------------------------------------------------------------ #


class TestUpdateStep:
    def test_generates_update_batch(self):
        sm = MockSkillManager()
        step = UpdateStep(sm)

        sb = Skillbook()
        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
        )
        trace = {"question": "What is 2+2?", "context": "math quiz"}
        ctx = ACEStepContext(
            reflection=reflection,
            skillbook=SkillbookView(sb),
            trace=trace,
            epoch=2,
            total_epochs=3,
            step_index=5,
            total_steps=10,
        )

        result = step(ctx)
        assert result.skill_manager_output is not None
        assert len(sm.calls) == 1
        call = sm.calls[0]
        assert "Epoch 2/3" in call["progress"]
        assert "sample 5/10" in call["progress"]
        assert "What is 2+2?" in call["question_context"]

    def test_non_dict_trace(self):
        """Non-dict trace should produce empty question_context."""
        sm = MockSkillManager()
        step = UpdateStep(sm)

        sb = Skillbook()
        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
        )
        ctx = ACEStepContext(
            reflection=reflection,
            skillbook=SkillbookView(sb),
            trace="raw string trace",
        )

        step(ctx)
        assert sm.calls[0]["question_context"] == ""

    def test_provides_and_requires(self):
        step = UpdateStep(MockSkillManager())
        assert "reflection" in step.requires
        assert "skillbook" in step.requires
        assert "skill_manager_output" in step.provides
        assert step.max_workers == 1


# ------------------------------------------------------------------ #
# ApplyStep
# ------------------------------------------------------------------ #


class TestApplyStep:
    def test_applies_update(self):
        sb = Skillbook()
        step = ApplyStep(sb)

        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(type="ADD", section="math", content="new skill")
            ],
        )
        ctx = ACEStepContext(skill_manager_output=batch)

        result = step(ctx)
        assert result is ctx
        assert len(sb.skills()) == 1
        assert sb.skills()[0].content == "new skill"

    def test_malformed_update_raises(self):
        """Malformed UpdateBatch should propagate the exception."""
        sb = Skillbook()
        step = ApplyStep(sb)

        ctx = ACEStepContext(skill_manager_output=None)

        with pytest.raises(Exception):
            step(ctx)

    def test_provides_and_requires(self):
        sb = Skillbook()
        step = ApplyStep(sb)
        assert "skill_manager_output" in step.requires
        assert len(step.provides) == 0
        assert step.max_workers == 1


# ------------------------------------------------------------------ #
# learning_tail helper
# ------------------------------------------------------------------ #


class TestLearningTail:
    def test_basic_tail(self):
        reflector = MockReflector()
        sm = MockSkillManager()
        sb = Skillbook()

        steps = learning_tail(reflector, sm, sb)
        assert len(steps) == 4
        assert isinstance(steps[0], ReflectStep)
        assert isinstance(steps[1], TagStep)
        assert isinstance(steps[2], UpdateStep)
        assert isinstance(steps[3], ApplyStep)

    def test_with_checkpoint(self, tmp_path):
        reflector = MockReflector()
        sm = MockSkillManager()
        sb = Skillbook()

        steps = learning_tail(
            reflector,
            sm,
            sb,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=5,
        )
        assert len(steps) == 5  # 4 + CheckpointStep

    def test_with_dedup(self):
        reflector = MockReflector()
        sm = MockSkillManager()
        sb = Skillbook()
        dedup = MagicMock()

        steps = learning_tail(
            reflector,
            sm,
            sb,
            dedup_manager=dedup,
            dedup_interval=5,
        )
        assert len(steps) == 5  # 4 + DeduplicateStep

    def test_with_both(self, tmp_path):
        reflector = MockReflector()
        sm = MockSkillManager()
        sb = Skillbook()
        dedup = MagicMock()

        steps = learning_tail(
            reflector,
            sm,
            sb,
            dedup_manager=dedup,
            dedup_interval=5,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=5,
        )
        assert len(steps) == 6  # 4 + DeduplicateStep + CheckpointStep
