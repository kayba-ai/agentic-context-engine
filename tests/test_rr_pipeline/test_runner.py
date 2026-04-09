"""Tests for RRStep — PydanticAI-based Recursive Reflector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ace.implementations.rr.config import RecursiveConfig
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.skillbook import Skillbook

from ace.steps.rr_step import RRStep, RRConfig
from ace.implementations.rr.tools import RRDeps

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


_RUN_SYNC = "ace.core.recursive_agent.run_agent_sync"


def _mock_compaction_result(
    *,
    reasoning: str = "mock reasoning",
    key_insight: str = "mock insight",
    correct_approach: str = "mock approach",
) -> tuple[ReflectorOutput, dict]:
    """Create a mock return value for run_agent_sync."""
    output = ReflectorOutput(
        reasoning=reasoning,
        key_insight=key_insight,
        correct_approach=correct_approach,
        raw={},
    )
    metadata = {
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "requests": 3},
        "compactions": 0,
        "depth": 0,
        "iterations": 2,
        "timed_out": False,
    }
    return output, metadata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRStep:
    """Test RRStep construction and StepProtocol."""

    def test_step_protocol_attributes(self):
        rr = RRStep("test-model", config=RRConfig())
        assert "trace" in rr.requires
        assert "skillbook" in rr.requires
        assert "reflections" in rr.provides
        assert "reflection" not in rr.provides

    def test_call_produces_reflection_on_context(self):
        """RRStep.__call__ populates ctx.reflections."""
        rr = RRStep("test-model", config=RRConfig())

        output, metadata = _mock_compaction_result(key_insight="step test")

        with patch(_RUN_SYNC, return_value=(output, metadata)):
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
        rr = RRStep("test-model", config=RRConfig())
        output, metadata = _mock_compaction_result()

        with patch(_RUN_SYNC, return_value=(output, metadata)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is False
        assert "usage" in result.raw

    def test_timeout_produces_output(self):
        """Budget exhaustion produces a timeout ReflectorOutput."""
        from ace.core.recursive_agent import BudgetExhausted

        rr = RRStep("test-model", config=RRConfig())

        with patch(_RUN_SYNC, side_effect=BudgetExhausted(compaction_count=0)):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        output = result_ctx.reflections[0]
        assert isinstance(output, ReflectorOutput)
        assert "budget limit" in output.reasoning.lower()
        assert output.raw.get("timeout") is True

    def test_timeout_with_ground_truth_correct(self):
        """Timeout correctly detects correct answer."""
        from ace.core.recursive_agent import BudgetExhausted

        rr = RRStep("test-model", config=RRConfig())

        with patch(_RUN_SYNC, side_effect=BudgetExhausted(compaction_count=0)):
            output = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
            )

        assert isinstance(output, ReflectorOutput)
        assert "correct" in output.reasoning.lower()

    def test_error_produces_safe_output(self):
        """General exception produces a safe fallback output."""
        rr = RRStep("test-model", config=RRConfig())

        with patch(_RUN_SYNC, side_effect=RuntimeError("unexpected error")):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        output = result_ctx.reflections[0]
        assert "failed" in output.reasoning.lower()


@pytest.mark.unit
class TestRRStepProtocol:
    """Test that RRStep satisfies structural protocols."""

    def test_satisfies_reflector_like(self):
        """RRStep satisfies ReflectorLike protocol."""
        from ace.protocols import ReflectorLike

        rr = RRStep("test-model", config=RRConfig())
        assert isinstance(rr, ReflectorLike)

    def test_reflect_method(self):
        """reflect() delegates to the PydanticAI agent."""
        rr = RRStep("test-model", config=RRConfig())
        output, metadata = _mock_compaction_result(key_insight="reflected")

        with patch(_RUN_SYNC, return_value=(output, metadata)):
            result = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
                feedback="Correct!",
            )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "reflected"
