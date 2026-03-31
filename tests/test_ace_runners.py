"""Tests for ace runners: ACE, TraceAnalyser, ACERunner, ACELiteLLM."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.environments import Sample, SimpleEnvironment
from ace.core.insight_source import TRACE_IDENTITY_METADATA_KEY
from ace.core.outputs import (
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
    SkillTag,
)
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace.runners.base import ACERunner

# ------------------------------------------------------------------ #
# Mock roles — satisfy protocols without any LLM dependency
# ------------------------------------------------------------------ #


class MockAgent:
    """Minimal mock satisfying AgentLike."""

    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        skillbook: Any,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        return AgentOutput(reasoning="mock reasoning", final_answer="mock answer")


class MockReflector:
    """Minimal mock satisfying ReflectorLike."""

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
        return ReflectorOutput(
            reasoning="mock reflection",
            correct_approach="mock approach",
            key_insight="mock insight",
            skill_tags=[],
        )


class MockSkillManager:
    """Minimal mock satisfying SkillManagerLike."""

    def update_skills(
        self,
        *,
        reflections: tuple[ReflectorOutput, ...],
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        return SkillManagerOutput(
            update=UpdateBatch(
                reasoning="mock update",
                operations=[
                    UpdateOperation(
                        type="ADD",
                        section="learned",
                        content="mock skill",
                    )
                ],
            ),
        )


# ------------------------------------------------------------------ #
# ACERunner base class
# ------------------------------------------------------------------ #


class TestACERunnerBase:
    def test_save_and_load(self, tmp_path):
        """save() and load() should round-trip the skillbook."""
        sb = Skillbook()
        sb.add_skill("sec", "content", skill_id="s-001")
        pipeline = MagicMock()
        runner = ACERunner(pipeline=pipeline, skillbook=sb)

        path = str(tmp_path / "sb.json")
        runner.save(path)

        # Modify skillbook
        sb.add_skill("sec", "new", skill_id="s-002")
        assert len(runner.skillbook.skills()) == 2

        # Load should replace the skillbook
        runner.load(path)
        assert len(runner.skillbook.skills()) == 1
        assert runner.skillbook.get_skill("s-001") is not None

    def test_multi_epoch_requires_sequence(self):
        """Multi-epoch with non-Sequence should raise ValueError."""
        pipeline = MagicMock()
        sb = Skillbook()
        runner = ACERunner(pipeline=pipeline, skillbook=sb)

        def gen():
            yield "item"

        with pytest.raises(ValueError, match="Sequence"):
            runner._run(gen(), epochs=2)


# ------------------------------------------------------------------ #
# load_skillbook alias correctness
# ------------------------------------------------------------------ #


class TestLoadSkillbookAlias:
    def test_langchain_alias(self):
        from ace.runners.langchain import LangChain

        assert LangChain.load_skillbook is ACERunner.load
        assert LangChain.save_skillbook is ACERunner.save

    def test_browser_use_alias(self):
        from ace.runners.browser_use import BrowserUse

        assert BrowserUse.load_skillbook is ACERunner.load
        assert BrowserUse.save_skillbook is ACERunner.save

    def test_claude_code_alias(self):
        from ace.runners.claude_code import ClaudeCode

        assert ClaudeCode.load_skillbook is ACERunner.load
        assert ClaudeCode.save_skillbook is ACERunner.save

    def test_litellm_alias(self):
        from ace.runners.litellm import ACELiteLLM

        assert ACELiteLLM.load_skillbook is ACELiteLLM.load
        assert ACELiteLLM.save_skillbook is ACELiteLLM.save

    def test_load_not_save(self):
        """Critical: load_skillbook must NOT point to save."""
        from ace.runners.langchain import LangChain

        assert LangChain.load_skillbook is not ACERunner.save
        assert LangChain.load_skillbook is not LangChain.save_skillbook


# ------------------------------------------------------------------ #
# ACE runner (full pipeline) with mocks
# ------------------------------------------------------------------ #


class TestACERunner:
    def test_from_roles_run(self):
        """ACE.from_roles().run() should complete without error with mock roles."""
        from ace.runners.ace import ACE

        env = SimpleEnvironment()

        runner = ACE.from_roles(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
            environment=env,
        )

        samples = [
            Sample(question="What is 2+2?", ground_truth="4"),
            Sample(question="Capital of France?", ground_truth="Paris"),
        ]

        results = runner.run(samples, epochs=1)
        assert len(results) == 2
        # After learning, skillbook should have skills
        assert len(runner.skillbook.skills()) > 0

    def test_multi_epoch(self):
        from ace.runners.ace import ACE

        env = SimpleEnvironment()

        runner = ACE.from_roles(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
            environment=env,
        )

        samples = [Sample(question="Q1", ground_truth="A1")]
        results = runner.run(samples, epochs=2)
        assert len(results) == 2  # 1 sample × 2 epochs

    def test_build_context_adds_trace_identity_metadata(self):
        from ace.runners.ace import ACE

        runner = ACE.from_roles(
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )
        sample = Sample(
            question="Why did pagination stop early?",
            id="conv-123",
            metadata={
                "source_system": "kayba-hosted",
                "trace_id": "conv-123",
                "display_name": "checkout-failure.md",
            },
        )

        ctx = runner._build_context(
            sample,
            epoch=1,
            total_epochs=1,
            index=1,
            total=1,
            global_sample_index=1,
        )

        identity = ctx.metadata[TRACE_IDENTITY_METADATA_KEY]
        assert identity["trace_uid"] == "kayba-hosted:conv-123"
        assert identity["display_name"] == "checkout-failure.md"


# ------------------------------------------------------------------ #
# TraceAnalyser runner
# ------------------------------------------------------------------ #


class TestTraceAnalyser:
    def test_from_roles_run(self):
        """TraceAnalyser.from_roles().run() with mock roles should work."""
        from ace.runners.trace_analyser import TraceAnalyser

        runner = TraceAnalyser.from_roles(
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )

        traces = [
            {
                "question": "What is 2+2?",
                "answer": "4",
                "reasoning": "simple",
                "ground_truth": "4",
                "feedback": "Correct!",
            },
        ]

        results = runner.run(traces)
        assert len(results) == 1
        assert len(runner.skillbook.skills()) > 0

    def test_build_context_adds_inferred_trace_identity(self):
        from ace.runners.trace_analyser import TraceAnalyser

        runner = TraceAnalyser.from_roles(
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
        )

        ctx = runner._build_context(
            {"sample_id": "trace-001", "question": "Q"},
            epoch=1,
            total_epochs=1,
            index=1,
            total=1,
            global_sample_index=1,
        )

        identity = ctx.metadata[TRACE_IDENTITY_METADATA_KEY]
        assert identity["trace_uid"] == "trace:trace-001"
        assert identity["trace_id"] == "trace-001"


# ------------------------------------------------------------------ #
# ACELiteLLM
# ------------------------------------------------------------------ #


class TestACELiteLLM:
    def _make_ace(self, **kwargs):
        from ace.runners.litellm import ACELiteLLM

        return ACELiteLLM(
            "test-model",
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
            **kwargs,
        )

    def test_ask(self):
        ace = self._make_ace()
        answer = ace.ask("What is 2+2?")
        assert answer == "mock answer"

    def test_learn_from_feedback_no_prior_ask(self):
        """learn_from_feedback with no prior ask() should return False."""
        ace = self._make_ace()
        assert ace.learn_from_feedback("good answer") is False

    def test_learn_from_feedback_after_ask(self):
        """learn_from_feedback after ask() should return True."""
        ace = self._make_ace()
        ace.ask("What is 2+2?")
        result = ace.learn_from_feedback("Correct!", ground_truth="4")
        assert result is True
        assert len(ace.skillbook.skills()) > 0

    def test_learn_from_feedback_disabled(self):
        """learn_from_feedback with learning disabled should return False."""
        ace = self._make_ace(is_learning=False)
        ace.ask("What is 2+2?")
        assert ace.learn_from_feedback("Correct!") is False

    def test_learn(self):
        ace = self._make_ace(environment=SimpleEnvironment())
        samples = [Sample(question="Q", ground_truth="A")]
        results = ace.learn(samples)
        assert len(results) == 1

    def test_learn_disabled(self):
        ace = self._make_ace(is_learning=False)
        with pytest.raises(RuntimeError, match="disabled"):
            ace.learn([Sample(question="Q", ground_truth="A")])

    def test_save_and_load(self, tmp_path):
        ace = self._make_ace()
        ace.ask("Q")
        ace.learn_from_feedback("good", ground_truth="A")

        path = str(tmp_path / "sb.json")
        ace.save(path)
        skills_before = len(ace.skillbook.skills())

        # Load into same instance
        ace.load(path)
        assert len(ace.skillbook.skills()) == skills_before

    def test_enable_disable_learning(self):
        ace = self._make_ace()
        assert ace.is_learning is True

        ace.disable_learning()
        assert ace.is_learning is False

        ace.enable_learning()
        assert ace.is_learning is True

    def test_get_strategies_empty(self):
        ace = self._make_ace()
        assert ace.get_strategies() == ""

    def test_skillbook_path_loading(self, tmp_path):
        """Constructor with skillbook_path should load from file."""
        sb = Skillbook()
        sb.add_skill("test", "content", skill_id="t-001")
        path = str(tmp_path / "sb.json")
        sb.save_to_file(path)

        from ace.runners.litellm import ACELiteLLM

        ace = ACELiteLLM(
            "test-model",
            agent=MockAgent(),
            reflector=MockReflector(),
            skill_manager=MockSkillManager(),
            skillbook_path=path,
        )
        assert ace.skillbook.get_skill("t-001") is not None
