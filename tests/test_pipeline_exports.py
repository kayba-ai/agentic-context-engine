"""Tests that pipeline composition classes are importable from ace.

Verifies the public API surface for pipeline-first composition.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ace.core.outputs import (
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
)
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation

# ------------------------------------------------------------------ #
# Mock roles for build_steps() tests
# ------------------------------------------------------------------ #


class MockAgent:
    def run(self, *a: Any, **kw: Any) -> AgentOutput:
        return AgentOutput(reasoning="r", final_answer="a")


class MockReflector:
    def reflect(self, *a: Any, **kw: Any) -> ReflectorOutput:
        return ReflectorOutput(
            reasoning="r",
            correct_approach="a",
            key_insight="i",
            skill_tags=[],
        )


class MockSkillManager:
    def update_skills(self, *a: Any, **kw: Any) -> SkillManagerOutput:
        return SkillManagerOutput(
            update=UpdateBatch(
                reasoning="r",
                operations=[
                    UpdateOperation(type="ADD", section="learned", content="c")
                ],
            ),
        )


# ------------------------------------------------------------------ #
# Pipeline primitives are importable from ace
# ------------------------------------------------------------------ #


class TestPipelineExports:
    def test_pipeline_class(self):
        from ace import Pipeline

        assert Pipeline is not None

    def test_branch_class(self):
        from ace import Branch

        assert Branch is not None

    def test_merge_strategy(self):
        from ace import MergeStrategy

        assert MergeStrategy is not None

    def test_step_protocol(self):
        from ace import StepProtocol

        assert StepProtocol is not None

    def test_sample_result(self):
        from ace import SampleResult

        assert SampleResult is not None


# ------------------------------------------------------------------ #
# ACE context types are importable from ace
# ------------------------------------------------------------------ #


class TestContextExports:
    def test_ace_step_context(self):
        from ace import ACEStepContext

        assert ACEStepContext is not None

    def test_skillbook_view(self):
        from ace import SkillbookView

        assert SkillbookView is not None

    def test_ace_runner(self):
        from ace import ACERunner

        assert ACERunner is not None


# ------------------------------------------------------------------ #
# All steps are importable from ace
# ------------------------------------------------------------------ #


class TestStepExports:
    @pytest.mark.parametrize(
        "name",
        [
            "AgentStep",
            "EvaluateStep",
            "ReflectStep",
            "TagStep",
            "UpdateStep",
            "AttachInsightSourcesStep",
            "ApplyStep",
            "DeduplicateStep",
            "CheckpointStep",
            "LoadTracesStep",
            "ExportSkillbookMarkdownStep",
            "ObservabilityStep",
            "PersistStep",
            "learning_tail",
        ],
    )
    def test_step_importable(self, name: str):
        import ace

        assert hasattr(ace, name), f"{name} not in ace"

    def test_all_steps_in_dunder_all(self):
        import ace

        step_names = [
            "AgentStep",
            "EvaluateStep",
            "ReflectStep",
            "TagStep",
            "UpdateStep",
            "AttachInsightSourcesStep",
            "ApplyStep",
            "DeduplicateStep",
            "CheckpointStep",
            "LoadTracesStep",
            "ExportSkillbookMarkdownStep",
            "ObservabilityStep",
            "PersistStep",
            "learning_tail",
        ]
        for name in step_names:
            assert name in ace.__all__, f"{name} not in __all__"


# ------------------------------------------------------------------ #
# build_steps() returns expected step types
# ------------------------------------------------------------------ #


class TestBuildSteps:
    def test_ace_build_steps(self):
        from ace import ACE
        from ace.steps import AgentStep, EvaluateStep, ReflectStep

        steps = ACE.build_steps(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )
        assert isinstance(steps, list)
        assert len(steps) >= 5  # Agent, Evaluate, Reflect, Tag, Update, Sources, Apply
        assert isinstance(steps[0], AgentStep)
        assert isinstance(steps[1], EvaluateStep)
        assert isinstance(steps[2], ReflectStep)

    def test_trace_analyser_build_steps(self):
        from ace import TraceAnalyser
        from ace.steps import ReflectStep

        steps = TraceAnalyser.build_steps(
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )
        assert isinstance(steps, list)
        assert len(steps) >= 5  # Reflect, Tag, Update, Sources, Apply
        assert isinstance(steps[0], ReflectStep)

    def test_ace_from_roles_delegates_to_build_steps(self):
        """from_roles() should produce the same steps as build_steps()."""
        from ace import ACE

        kwargs = dict(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )
        runner = ACE.from_roles(**kwargs)
        steps = ACE.build_steps(**kwargs)

        # Same number of steps
        assert len(runner.pipeline._steps) == len(steps)
        # Same step types
        for pipe_step, built_step in zip(runner.pipeline._steps, steps):
            assert type(pipe_step) is type(built_step)

    def test_build_steps_with_extra_steps(self):
        from ace import ACE

        class DummyStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx):
                return ctx

        steps = ACE.build_steps(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
            extra_steps=[DummyStep()],
        )
        assert isinstance(steps[-1], DummyStep)

    def test_pipeline_from_build_steps(self):
        """Pipeline constructed from build_steps() should be valid."""
        from ace import ACE, Pipeline

        steps = ACE.build_steps(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )
        pipe = Pipeline(steps)
        assert pipe is not None
        assert len(pipe._steps) == len(steps)
