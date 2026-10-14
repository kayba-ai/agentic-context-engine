"""Tests for ace steps: ReflectStep, UpdateStep, provenance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import (
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
)
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace.steps import learning_tail
from ace.steps.reflect import ReflectStep
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


# ---------------------------------------------------------------------------
# DeduplicateStep
# ---------------------------------------------------------------------------


def _skillbook_with_duplicate_pair() -> tuple[Skillbook, str, str]:
    sb = Skillbook()
    keep = sb.add_skill("math", "Check units before dividing")
    dup = sb.add_skill("math", "Always check the units before a division")
    # Pre-computed embeddings so the detector makes no embedding API call.
    keep.embedding = [1.0, 0.0]
    dup.embedding = [0.99, 0.01]
    return sb, keep.id, dup.id


def _consolidation_model(operations: list[dict[str, Any]]):
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        tool = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool, args={"consolidation_operations": operations}
                )
            ]
        )

    return FunctionModel(respond)


class TestDeduplicateStep:
    def test_applies_consolidation_to_skillbook(self):
        """Regression for kayba-ai/agentic-context-engine#135."""
        from ace.deduplication import DeduplicationManager
        from ace.protocols.deduplication import DeduplicationConfig
        from ace.steps import DeduplicateStep

        sb, keep_id, dup_id = _skillbook_with_duplicate_pair()
        model = _consolidation_model(
            [
                {
                    "type": "MERGE",
                    "source_ids": [keep_id, dup_id],
                    "keep_id": keep_id,
                    "merged_content": "Check units before any division",
                    "reasoning": "same advice",
                }
            ]
        )
        manager = DeduplicationManager(DeduplicationConfig(consolidation_model=model))
        step = DeduplicateStep(manager, sb, interval=10)

        step(ACEStepContext(global_sample_index=10))

        assert [s.id for s in sb.skills()] == [keep_id]
        assert sb.get_skill(keep_id).insight == "Check units before any division"

    def test_keep_decision_stops_pair_being_reported_again(self):
        from ace.deduplication import DeduplicationManager
        from ace.protocols.deduplication import DeduplicationConfig

        sb, keep_id, dup_id = _skillbook_with_duplicate_pair()
        model = _consolidation_model(
            [{"type": "KEEP", "skill_ids": [keep_id, dup_id], "reasoning": "x"}]
        )
        manager = DeduplicationManager(DeduplicationConfig(consolidation_model=model))

        manager.consolidate(sb)

        assert len(sb.skills()) == 2
        assert manager.get_similarity_report(sb) is None

    def test_model_error_leaves_skillbook_unchanged(self):
        from pydantic_ai.models.function import FunctionModel

        from ace.deduplication import DeduplicationManager
        from ace.protocols.deduplication import DeduplicationConfig

        def fail(messages, info):
            raise RuntimeError("provider down")

        sb, keep_id, dup_id = _skillbook_with_duplicate_pair()
        manager = DeduplicationManager(
            DeduplicationConfig(consolidation_model=FunctionModel(fail))
        )

        assert manager.consolidate(sb) == []
        assert {s.id for s in sb.skills()} == {keep_id, dup_id}

    def test_without_consolidation_model_only_reports(self, caplog):
        from ace.deduplication import DeduplicationManager
        from ace.steps import DeduplicateStep

        sb, keep_id, dup_id = _skillbook_with_duplicate_pair()
        step = DeduplicateStep(DeduplicationManager(), sb, interval=10)

        with caplog.at_level(logging.INFO):
            step(ACEStepContext(global_sample_index=10))

        assert {s.id for s in sb.skills()} == {keep_id, dup_id}
        assert "Similar Skills Detected" in caplog.text

    def test_skips_samples_off_interval(self):
        from ace.steps import DeduplicateStep

        manager = MagicMock()
        step = DeduplicateStep(manager, Skillbook(), interval=10)

        step(ACEStepContext(global_sample_index=7))

        manager.consolidate.assert_not_called()
        manager.get_similarity_report.assert_not_called()
