"""Tests for ace steps: ReflectStep, UpdateStep, provenance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from pipeline import Pipeline

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import (
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
)
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace.steps import learning_tail
from ace.steps.reflect import ReflectStep
from ace.steps.reflection_ensemble import ReflectionEnsembleStep
from ace.steps.update import UpdateStep

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
        reflections: tuple[ReflectorOutput, ...],
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        self.calls.append(
            {
                "reflections": reflections,
                "question_context": question_context,
                "progress": progress,
            }
        )
        return self.output


class SequencedReflector(MockReflector):
    """Mock reflector that returns a distinct reflection per call."""

    def reflect(self, **kwargs: Any) -> ReflectorOutput:
        call_number = len(self.calls) + 1
        self.output = ReflectorOutput(
            reasoning=f"reasoning {call_number}",
            correct_approach="test approach",
            key_insight=f"test insight {call_number}",
        )
        return super().reflect(**kwargs)


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
        }
        sb = Skillbook()
        ctx = ACEStepContext(
            trace=trace,
            skillbook=SkillbookView(sb),
        )

        result = step(ctx)
        assert len(result.reflections) == 1
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
        assert len(result.reflections) == 1
        assert len(reflector.calls) == 1
        call = reflector.calls[0]
        assert call["question"] == ""
        assert call["agent_output"].final_answer == ""
        assert call.get("trace") is raw_trace

    def test_batch_dict_trace_is_passed_raw(self):
        """Batch dict traces should bypass structured trace extraction."""
        reflector = MockReflector()
        step = ReflectStep(reflector)

        batch_trace = {
            "tasks": [
                {"task_id": "task-0", "trace": {"question": "What is 2+2?"}},
                {"task_id": "task-1", "trace": {"question": "What is 3+3?"}},
            ]
        }
        sb = Skillbook()
        ctx = ACEStepContext(
            trace=batch_trace,
            skillbook=SkillbookView(sb),
        )

        result = step(ctx)
        assert len(result.reflections) == 1
        assert len(reflector.calls) == 1
        call = reflector.calls[0]
        assert call["question"] == ""
        assert call["agent_output"].final_answer == ""
        assert call.get("trace") is batch_trace

    def test_provides_and_requires(self):
        step = ReflectStep(MockReflector())
        assert "trace" in step.requires
        assert "skillbook" in step.requires
        assert "reflections" in step.provides
        assert step.async_boundary is True
        assert step.max_workers == 3


# ------------------------------------------------------------------ #
# ReflectionEnsembleStep
# ------------------------------------------------------------------ #


class TestReflectionEnsembleStep:
    def test_runs_same_trace_multiple_times(self):
        reflector = SequencedReflector()
        step = ReflectionEnsembleStep(reflector, ensemble_size=2, workers=2)
        trace = {
            "question": "What is 2+2?",
            "answer": "4",
            "reasoning": "simple math",
            "ground_truth": "4",
            "feedback": "Correct!",
        }
        sb = Skillbook()
        ctx = ACEStepContext(trace=trace, skillbook=SkillbookView(sb))

        result = step(ctx)

        assert len(reflector.calls) == 2
        assert len(result.reflections) == 2
        assert [r.key_insight for r in result.reflections] == [
            "test insight 1",
            "test insight 2",
        ]
        assert {call["question"] for call in reflector.calls} == {"What is 2+2?"}
        assert result.metadata["reflection_ensemble_size"] == 2
        assert result.metadata["reflection_ensemble_completed"] == 2

    def test_rejects_invalid_ensemble_size(self):
        with pytest.raises(ValueError, match="ensemble_size must be >= 1"):
            ReflectionEnsembleStep(MockReflector(), ensemble_size=0)

    def test_learning_tail_passes_ensemble_to_one_update(self):
        reflector = SequencedReflector()
        sm = MockSkillManager()
        sb = Skillbook()
        steps = learning_tail(
            reflector,
            sm,
            sb,
            reflection_ensemble_size=2,
            reflection_ensemble_workers=2,
        )
        assert isinstance(steps[0], ReflectionEnsembleStep)

        trace = {
            "question": "What is 2+2?",
            "answer": "4",
            "reasoning": "simple math",
            "ground_truth": "4",
            "feedback": "Correct!",
        }
        ctx = ACEStepContext(trace=trace, skillbook=SkillbookView(sb))
        pipe = Pipeline(steps)

        results = pipe.run([ctx])
        pipe.wait_for_background()

        assert results[0].error is None
        assert len(reflector.calls) == 2
        assert len(sm.calls) == 1
        assert [r.key_insight for r in sm.calls[0]["reflections"]] == [
            "test insight 1",
            "test insight 2",
        ]


# ------------------------------------------------------------------ #
# UpdateStep
# ------------------------------------------------------------------ #


class TestUpdateStep:
    def test_generates_update_batch(self):
        sm = MockSkillManager()
        sb = Skillbook()
        step = UpdateStep(sm, sb)

        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
        )
        trace = {"question": "What is 2+2?", "context": "math quiz"}
        ctx = ACEStepContext(
            reflections=(reflection,),
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
        sb = Skillbook()
        step = UpdateStep(sm, sb)

        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
        )
        ctx = ACEStepContext(
            reflections=(reflection,),
            skillbook=SkillbookView(sb),
            trace="raw string trace",
        )

        step(ctx)
        assert sm.calls[0]["question_context"] == ""

    def test_forwards_full_reflections_tuple(self):
        """UpdateStep forwards the entire reflections tuple to the skill manager."""
        sm = MockSkillManager()
        sb = Skillbook()
        step = UpdateStep(sm, sb)

        r1 = ReflectorOutput(reasoning="r1", correct_approach="c", key_insight="k1")
        r2 = ReflectorOutput(reasoning="r2", correct_approach="c", key_insight="k2")
        ctx = ACEStepContext(
            reflections=(r1, r2),
            skillbook=SkillbookView(sb),
        )

        step(ctx)
        assert len(sm.calls) == 1
        assert sm.calls[0]["reflections"] == (r1, r2)

    def test_provides_and_requires(self):
        sb = Skillbook()
        step = UpdateStep(MockSkillManager(), sb)
        assert "reflections" in step.requires
        assert "skillbook" in step.requires
        assert "skill_manager_output" in step.provides
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
        assert len(steps) == 2
        assert isinstance(steps[0], ReflectStep)
        assert isinstance(steps[1], UpdateStep)

    def test_step_like_reflector_is_inserted_directly(self):
        class ReflectorStep(MockReflector):
            requires = frozenset({"trace", "skillbook"})
            provides = frozenset({"reflections"})

            def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
                return ctx.replace(reflections=(self.output,))

        reflector = ReflectorStep()
        sm = MockSkillManager()
        sb = Skillbook()

        steps = learning_tail(reflector, sm, sb)

        assert steps[0] is reflector
        assert isinstance(steps[1], UpdateStep)

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
        assert len(steps) == 3  # 2 + CheckpointStep

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
        assert len(steps) == 3  # 2 + DeduplicateStep

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
        assert len(steps) == 4  # 2 + DeduplicateStep + CheckpointStep
