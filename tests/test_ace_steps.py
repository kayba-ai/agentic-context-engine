"""Tests for ace steps: ReflectStep, TagStep, UpdateStep, provenance, ApplyStep."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.insight_source import TRACE_IDENTITY_METADATA_KEY, InsightSource
from ace.core.outputs import (
    AgentOutput,
    ExtractedLearning,
    ReflectorOutput,
    SkillManagerOutput,
)
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace.steps import learning_tail
from ace.steps.apply import ApplyStep
from ace.steps.attach_insight_sources import AttachInsightSourcesStep
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
            "skill_ids": ["math-001"],
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
        step = UpdateStep(sm)

        sb = Skillbook()
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
        step = UpdateStep(sm)

        sb = Skillbook()
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
        step = UpdateStep(sm)
        sb = Skillbook()

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
        step = UpdateStep(MockSkillManager())
        assert "reflections" in step.requires
        assert "skillbook" in step.requires
        assert "skill_manager_output" in step.provides
        assert step.max_workers == 1


# ------------------------------------------------------------------ #
# AttachInsightSourcesStep
# ------------------------------------------------------------------ #


class TestAttachInsightSourcesStep:
    def test_enriches_operations_with_trace_provenance(self):
        step = AttachInsightSourcesStep()
        reflection = ReflectorOutput(
            reasoning="r",
            correct_approach="c",
            key_insight="k",
            error_identification="Missed pagination signal",
            extracted_learnings=[
                ExtractedLearning(
                    learning="Check for a next-page token before stopping.",
                    evidence="The API response included next_page_token.",
                )
            ],
        )
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="ADD",
                    section="api",
                    content="Check for a next-page token before stopping.",
                    evidence="The API response included next_page_token.",
                    learning_index=0,
                )
            ],
        )
        ctx = ACEStepContext(
            trace={"question": "Why did pagination stop early?"},
            reflections=(reflection,),
            skill_manager_output=batch,
            metadata={
                TRACE_IDENTITY_METADATA_KEY: {
                    "source_system": "kayba-hosted",
                    "trace_id": "conv-123",
                    "display_name": "checkout-failure.md",
                }
            },
            epoch=2,
            step_index=4,
        )

        result = step(ctx)

        assert result.skill_manager_output is not batch
        assert batch.operations[0].insight_source is None
        source = result.skill_manager_output.operations[0].insight_source
        assert isinstance(source, InsightSource)
        assert source.trace_uid == "kayba-hosted:conv-123"
        assert source.trace_id == "conv-123"
        assert source.sample_question == "Why did pagination stop early?"
        assert source.epoch == 2
        assert source.operation_type == "ADD"

    def test_batch_context_uses_per_reflection_trace_identity(self):
        step = AttachInsightSourcesStep()
        reflections = (
            ReflectorOutput(
                reasoning="r0",
                correct_approach="c0",
                key_insight="k0",
                extracted_learnings=[
                    ExtractedLearning(
                        learning="Use the continuation token.",
                        evidence="first batch evidence",
                    )
                ],
                raw={"item_id": "task-0"},
            ),
            ReflectorOutput(
                reasoning="r1",
                correct_approach="c1",
                key_insight="k1",
                extracted_learnings=[
                    ExtractedLearning(
                        learning="Verify exact constants.",
                        evidence="second batch evidence",
                    )
                ],
                raw={"item_id": "task-1"},
            ),
        )
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="ADD",
                    section="api",
                    content="Use the continuation token.",
                    learning_index=0,
                    reflection_index=0,
                ),
                UpdateOperation(
                    type="ADD",
                    section="science",
                    content="Verify exact constants.",
                    learning_index=0,
                    reflection_index=1,
                ),
            ],
        )
        ctx = ACEStepContext(
            trace={
                "tasks": [
                    {
                        "task_id": "task-0",
                        "trace": {
                            "question": "Why did pagination stop early?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "first batch evidence",
                                }
                            ],
                        },
                    },
                    {
                        "task_id": "task-1",
                        "trace": {
                            "question": "What temperature does water boil at?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "second batch evidence",
                                }
                            ],
                        },
                    },
                ]
            },
            reflections=reflections,
            skill_manager_output=batch,
            metadata={
                TRACE_IDENTITY_METADATA_KEY: {
                    "source_system": "rr-batch",
                    "trace_id": "batch-root",
                    "display_name": "rr batch",
                }
            },
        )

        result = step(ctx)

        first_source = result.skill_manager_output.operations[0].insight_source
        second_source = result.skill_manager_output.operations[1].insight_source
        assert isinstance(first_source, InsightSource)
        assert isinstance(second_source, InsightSource)
        assert first_source.trace_uid == "rr-batch:task-0"
        assert first_source.sample_question == "Why did pagination stop early?"
        assert second_source.trace_uid == "rr-batch:task-1"
        assert second_source.sample_question == "What temperature does water boil at?"

    def test_batch_context_can_attach_multiple_sources(self):
        step = AttachInsightSourcesStep()
        reflections = (
            ReflectorOutput(
                reasoning="r0",
                correct_approach="c0",
                key_insight="k0",
                raw={"item_id": "task-0"},
            ),
            ReflectorOutput(
                reasoning="r1",
                correct_approach="c1",
                key_insight="k1",
                raw={"item_id": "task-1"},
            ),
        )
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="ADD",
                    section="general",
                    content="Generalize verification across arithmetic and factual tasks.",
                    evidence="Both Task-0 and Task-1 failed because verification was skipped.",
                    reflection_indices=[0, 1],
                )
            ],
        )
        ctx = ACEStepContext(
            trace={
                "tasks": [
                    {
                        "task_id": "task-0",
                        "trace": {
                            "question": "What is 12 * 15?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "task-0 arithmetic failure",
                                }
                            ],
                        },
                    },
                    {
                        "task_id": "task-1",
                        "trace": {
                            "question": "What is the capital of Japan?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "task-1 factual failure",
                                }
                            ],
                        },
                    },
                ]
            },
            reflections=reflections,
            skill_manager_output=batch,
            metadata={
                TRACE_IDENTITY_METADATA_KEY: {
                    "source_system": "rr-batch",
                    "trace_id": "batch-root",
                    "display_name": "rr batch",
                }
            },
        )

        result = step(ctx)

        sources = result.skill_manager_output.operations[0].insight_source
        assert isinstance(sources, list)
        assert [source.trace_uid for source in sources] == [
            "rr-batch:task-0",
            "rr-batch:task-1",
        ]
        assert [source.relation for source in sources] == ["seed", "supporting"]

    def test_single_reflection_index_matches_single_batch_item(self):
        """Without explicit reflection_indices, only the positional match is used."""
        step = AttachInsightSourcesStep()
        reflections = (
            ReflectorOutput(
                reasoning="r0",
                correct_approach="c0",
                key_insight="k0",
                raw={"item_id": "task-0"},
            ),
            ReflectorOutput(
                reasoning="r1",
                correct_approach="c1",
                key_insight="k1",
                raw={"item_id": "task-1"},
            ),
        )
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="ADD",
                    section="general",
                    content="Generalize verification across tasks.",
                    evidence=(
                        "Both Task-0 and Task-1 failed because each answer was "
                        "accepted without verification."
                    ),
                    reflection_index=0,
                )
            ],
        )
        ctx = ACEStepContext(
            trace={
                "tasks": [
                    {
                        "task_id": "task-0",
                        "trace": {
                            "question": "What is 12 * 15?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "task-0 arithmetic failure",
                                }
                            ],
                        },
                    },
                    {
                        "task_id": "task-1",
                        "trace": {
                            "question": "What is the capital of Japan?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "task-1 factual failure",
                                }
                            ],
                        },
                    },
                ]
            },
            reflections=reflections,
            skill_manager_output=batch,
            metadata={
                TRACE_IDENTITY_METADATA_KEY: {
                    "source_system": "rr-batch",
                    "trace_id": "batch-root",
                    "display_name": "rr batch",
                }
            },
        )

        result = step(ctx)

        source = result.skill_manager_output.operations[0].insight_source
        assert isinstance(source, InsightSource)
        assert source.trace_uid == "rr-batch:task-0"

    def test_none_update_is_noop(self):
        step = AttachInsightSourcesStep()
        ctx = ACEStepContext(
            trace={"question": "Q"},
            reflections=(),
            skill_manager_output=None,
        )
        assert step(ctx) is ctx

    def test_reflection_index_determines_batch_match(self):
        """reflection_index is used deterministically — no fuzzy overrides."""
        step = AttachInsightSourcesStep()
        reflections = (
            ReflectorOutput(
                reasoning="r0",
                correct_approach="c0",
                key_insight="k0",
                raw={"item_id": "task-0"},
            ),
            ReflectorOutput(
                reasoning="r1",
                correct_approach="c1",
                key_insight="k1",
                raw={"item_id": "task-1"},
            ),
        )
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="ADD",
                    section="science",
                    content="Verify factual claims against canonical sources.",
                    reflection_index=0,
                )
            ],
        )
        ctx = ACEStepContext(
            trace={
                "tasks": [
                    {
                        "task_id": "task-0",
                        "trace": {
                            "question": "What is 12 * 15?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "I computed it directly and got 170.",
                                }
                            ],
                        },
                    },
                    {
                        "task_id": "task-1",
                        "trace": {
                            "question": "What is the capital of Japan?",
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": (
                                        "Kyoto feels plausible because of its history."
                                    ),
                                }
                            ],
                        },
                    },
                ]
            },
            reflections=reflections,
            skill_manager_output=batch,
            metadata={
                TRACE_IDENTITY_METADATA_KEY: {
                    "source_system": "rr-batch",
                    "trace_id": "batch-root",
                    "display_name": "rr batch",
                }
            },
        )

        result = step(ctx)

        source = result.skill_manager_output.operations[0].insight_source
        assert isinstance(source, InsightSource)
        assert source.trace_uid == "rr-batch:task-0"
        assert source.sample_question == "What is 12 * 15?"


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

    def test_none_update_is_noop(self):
        """None skill_manager_output should be a safe no-op."""
        sb = Skillbook()
        step = ApplyStep(sb)

        ctx = ACEStepContext(skill_manager_output=None)

        result = step(ctx)
        assert result is ctx
        assert len(sb.skills()) == 0

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
        assert isinstance(steps[1], UpdateStep)
        assert isinstance(steps[2], AttachInsightSourcesStep)
        assert isinstance(steps[3], ApplyStep)

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
