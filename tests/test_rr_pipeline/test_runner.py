"""Tests for RRStep — PydanticAI-based Recursive Reflector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ace_next.rr.config import RecursiveConfig
from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import AgentOutput, ReflectorOutput
from ace_next.core.skillbook import Skillbook

from ace_next.rr import RRStep, RRConfig
from ace_next.rr.agent import RRDeps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    question: str = "test",
    answer: str = "a",
    reasoning: str = "r",
    ground_truth: str | None = None,
    feedback: str | None = None,
) -> ACEStepContext:
    """Build an ACEStepContext suitable for RRStep.__call__."""
    trace: dict = {
        "question": question,
        "steps": [
            {"role": "agent", "reasoning": reasoning, "answer": answer, "skill_ids": []}
        ],
    }
    if ground_truth is not None:
        trace["ground_truth"] = ground_truth
    if feedback is not None:
        trace["feedback"] = feedback
    return ACEStepContext(trace=trace, skillbook=SkillbookView(Skillbook()))


def _mock_run_result(
    *,
    reasoning: str = "mock reasoning",
    key_insight: str = "mock insight",
    correct_approach: str = "mock approach",
    extracted_learnings: list | None = None,
) -> MagicMock:
    """Create a mock PydanticAI RunResult."""
    output = ReflectorOutput(
        reasoning=reasoning,
        key_insight=key_insight,
        correct_approach=correct_approach,
        extracted_learnings=extracted_learnings or [],
    )
    result = MagicMock()
    result.output = output
    usage = MagicMock()
    usage.request_tokens = 100
    usage.response_tokens = 50
    usage.total_tokens = 150
    usage.requests = 3
    result.usage.return_value = usage
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRStep:
    """Test RRStep construction and StepProtocol."""

    def test_step_protocol_attributes(self):
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        assert "trace" in rr.requires
        assert "skillbook" in rr.requires
        assert "reflections" in rr.provides
        assert "reflection" not in rr.provides

    def test_call_produces_reflection_on_context(self):
        """RRStep.__call__ populates ctx.reflections."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        mock_result = _mock_run_result(key_insight="step test")

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            ctx = _make_ctx(
                question="What is 2+2?",
                answer="4",
                reasoning="2+2=4",
                ground_truth="4",
                feedback="Correct!",
            )
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 1
        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert result_ctx.reflections[0].key_insight == "step test"

    def test_rr_trace_metadata_populated(self):
        """Successful reflection populates rr_trace in raw."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result()

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(_make_ctx())

        output = result_ctx.reflections[0]
        assert "rr_trace" in output.raw
        assert output.raw["rr_trace"]["timed_out"] is False
        assert "usage" in output.raw

    def test_timeout_produces_output(self):
        """UsageLimitExceeded produces a timeout ReflectorOutput."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        rr = RRStep(
            "test-model",
            config=RRConfig(max_llm_calls=5, enable_subagent=False),
        )

        with patch.object(
            rr._agent, "run_sync",
            side_effect=UsageLimitExceeded("limit reached"),
        ):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        output = result_ctx.reflections[0]
        assert isinstance(output, ReflectorOutput)
        assert "usage limit" in output.reasoning.lower()
        assert output.raw.get("timeout") is True

    def test_timeout_with_ground_truth_correct(self):
        """Timeout correctly detects correct answer."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        with patch.object(
            rr._agent, "run_sync",
            side_effect=UsageLimitExceeded("limit"),
        ):
            ctx = _make_ctx(
                question="What is 2+2?",
                answer="4",
                ground_truth="4",
            )
            # reflect() via __call__ doesn't pass agent_output, so is_correct is False
            # Test directly via reflect() with agent_output
            output = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
            )

        assert isinstance(output, ReflectorOutput)
        assert "correct" in output.reasoning.lower()

    def test_error_produces_safe_output(self):
        """General exception produces a safe fallback output."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        with patch.object(
            rr._agent, "run_sync",
            side_effect=RuntimeError("unexpected error"),
        ):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        output = result_ctx.reflections[0]
        assert "failed" in output.reasoning.lower()


@pytest.mark.unit
class TestRRStepProtocol:
    """Test that RRStep satisfies structural protocols."""

    def test_satisfies_reflector_like(self):
        """RRStep satisfies ReflectorLike protocol."""
        from ace_next.protocols import ReflectorLike

        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        assert isinstance(rr, ReflectorLike)

    def test_reflect_method(self):
        """reflect() delegates to the PydanticAI agent."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result(key_insight="reflected")

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            output = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
                feedback="Correct!",
            )

        assert isinstance(output, ReflectorOutput)
        assert output.key_insight == "reflected"


@pytest.mark.unit
class TestRRBatchReflection:
    """Test batch reflection (traces with 'tasks' key)."""

    def test_batch_splits_into_per_task_outputs(self):
        """Batch with per-task results in raw produces per-task ReflectorOutputs."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        # Mock a batch result with per-task data in raw
        output = ReflectorOutput(
            reasoning="batch analysis",
            key_insight="batch insight",
            correct_approach="approach",
            raw={
                "tasks": [
                    {
                        "reasoning": "task 0 analysis",
                        "key_insight": "t0 insight",
                        "extracted_learnings": [],
                    },
                    {
                        "reasoning": "task 1 analysis",
                        "key_insight": "t1 insight",
                        "extracted_learnings": [
                            {"learning": "l1", "atomicity_score": 0.8, "evidence": "e1"}
                        ],
                    },
                ],
            },
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 200
        usage.response_tokens = 100
        usage.total_tokens = 300
        usage.requests = 5
        mock_result.usage.return_value = usage

        batch_trace = {
            "tasks": [
                {"task_id": "t0", "trace": [{"role": "user", "content": "hello"}]},
                {"task_id": "t1", "trace": [{"role": "user", "content": "world"}]},
            ]
        }
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].reasoning == "task 0 analysis"
        assert result_ctx.reflections[1].key_insight == "t1 insight"
        assert len(result_ctx.reflections[1].extracted_learnings) == 1

    def test_batch_fallback_duplicates_when_no_per_task(self):
        """When batch output lacks per-task results, duplicate the single reflection."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        output = ReflectorOutput(
            reasoning="single batch analysis",
            key_insight="single insight",
            correct_approach="approach",
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 100
        usage.response_tokens = 50
        usage.total_tokens = 150
        usage.requests = 3
        mock_result.usage.return_value = usage

        batch_trace = {
            "tasks": [
                {"task_id": "t0", "trace": []},
                {"task_id": "t1", "trace": []},
            ]
        }
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].raw["task_id"] == "t0"
        assert result_ctx.reflections[1].raw["task_id"] == "t1"


@pytest.mark.unit
class TestRROpikStep:
    """Test RROpikStep — graceful degradation and data reading."""

    def test_noop_when_opik_unavailable(self):
        """RROpikStep is a no-op when Opik is not installed."""
        from ace_next.rr.opik import RROpikStep, OPIK_AVAILABLE

        step = RROpikStep(project_name="test")
        if not OPIK_AVAILABLE:
            assert not step.enabled

    def test_noop_when_no_reflection(self):
        """RROpikStep returns ctx unchanged when reflections is empty."""
        from ace_next.rr.opik import RROpikStep

        step = RROpikStep(project_name="test")
        step.enabled = False

        ctx = ACEStepContext(skillbook=SkillbookView(Skillbook()))
        result = step(ctx)
        assert result is ctx

    def test_step_protocol_attributes(self):
        """RROpikStep has correct requires/provides."""
        from ace_next.rr.opik import RROpikStep

        step = RROpikStep(project_name="test")
        assert "reflections" in step.requires
        assert len(step.provides) == 0
