"""Stress tests for RR components.

Tests sandbox behavior and the PydanticAI-based RRStep entry points.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ace.rr.config import RecursiveConfig
from ace.rr.sandbox import TraceSandbox

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.skillbook import Skillbook
from ace.rr import RRConfig, RRStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    question: str = "q",
    answer: str = "4",
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
    reasoning: str = "done",
    key_insight: str = "insight",
    correct_approach: str = "approach",
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


# =========================================================================
# 1. RRStep lifecycle (PydanticAI-based)
# =========================================================================


@pytest.mark.unit
class TestLoopLifecycle:
    def test_successful_reflection(self):
        """Happy path: PydanticAI agent produces valid ReflectorOutput."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result(key_insight="insight")

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(
                _make_ctx(
                    question="What is 2+2?",
                    ground_truth="4",
                    feedback="Correct!",
                )
            )

        result = result_ctx.reflections[0]
        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "insight"

    def test_max_requests_timeout(self):
        """UsageLimitExceeded produces timeout output."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        rr = RRStep(
            "test-model",
            config=RRConfig(max_llm_calls=3, enable_subagent=False),
        )

        with patch.object(
            rr._agent,
            "run_sync",
            side_effect=UsageLimitExceeded("limit reached"),
        ):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert "usage limit" in result_ctx.reflections[0].reasoning.lower()

    def test_budget_field_in_config(self):
        """max_llm_calls config is passed to UsageLimits."""
        rr = RRStep(
            "test-model",
            config=RRConfig(max_llm_calls=42, enable_subagent=False),
        )
        assert rr.config.max_llm_calls == 42

    def test_subagent_request_budget_default(self):
        """Deep-dive sub-agents get a less fragile default request budget."""
        assert RecursiveConfig().subagent_max_requests == 15

    def test_rr_trace_metadata_on_success(self):
        """Successful reflection populates rr_trace metadata."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result()

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(_make_ctx())

        output = result_ctx.reflections[0]
        assert "rr_trace" in output.raw
        assert output.raw["rr_trace"]["timed_out"] is False
        assert isinstance(output.raw["rr_trace"]["subagent_calls"], list)

    def test_rr_trace_metadata_on_timeout(self):
        """Timeout reflection also has rr_trace metadata."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        with patch.object(
            rr._agent,
            "run_sync",
            side_effect=UsageLimitExceeded("limit"),
        ):
            result_ctx = rr(_make_ctx())

        output = result_ctx.reflections[0]
        assert "rr_trace" in output.raw
        assert output.raw["rr_trace"]["timed_out"] is True


# =========================================================================
# 2. Sandbox behavior
# =========================================================================


@pytest.mark.unit
class TestSandboxBehavior:
    def test_sandbox_variables_persist_across_iterations(self):
        """Variables set in one execution persist for the next."""
        sandbox = TraceSandbox(trace=None)
        sandbox.execute("x = 42", timeout=5.0)
        result = sandbox.execute("print(x + 1)", timeout=5.0)
        assert "43" in result.stdout

    def test_sandbox_code_modifies_injected_traces(self):
        """Mutation of injected dict is visible in later executions."""
        sandbox = TraceSandbox(trace=None)
        traces = {"question": "q", "items": [1, 2, 3]}
        sandbox.inject("traces", traces)
        sandbox.execute("traces['items'].append(4)", timeout=5.0)
        result = sandbox.execute("print(len(traces['items']))", timeout=5.0)
        assert "4" in result.stdout

    def test_sandbox_exception_produces_stderr(self):
        """Code that raises captures error in stderr."""
        sandbox = TraceSandbox(trace=None)
        result = sandbox.execute("raise RuntimeError('boom')", timeout=5.0)
        assert not result.success
        assert "RuntimeError" in result.stderr
        assert "boom" in result.stderr


# =========================================================================
# 3. Entry points (PydanticAI-based)
# =========================================================================


@pytest.mark.unit
class TestEntryPoints:
    def test_call_produces_reflection(self):
        """__call__() produces a ReflectorOutput on the context."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result(key_insight="insight")

        traces = {
            "question": "q",
            "steps": [
                {"role": "agent", "reasoning": "r", "answer": "4", "skill_ids": []}
            ],
        }
        ctx = ACEStepContext(trace=traces, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(ctx)

        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert result_ctx.reflections[0].key_insight == "insight"

    def test_reflect_method_works(self):
        """reflect() works as ReflectorLike entry point."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result(key_insight="reflected")

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            output = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
            )

        assert isinstance(output, ReflectorOutput)
        assert output.key_insight == "reflected"
