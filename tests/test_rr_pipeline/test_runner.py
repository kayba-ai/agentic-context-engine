"""Tests for RRStep — PydanticAI-based Recursive Reflector."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic_ai.settings import ModelSettings

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
        error_identification="none",
        root_cause_analysis="mock root cause",
        correct_approach=correct_approach,
        key_insight=key_insight,
        raw={},
    )
    metadata = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "requests": 3,
        },
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

        reflection, metadata = _mock_compaction_result(key_insight="step test")

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
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
        reflection, metadata = _mock_compaction_result()

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is False
        assert "usage" in result.raw

    def test_thoughts_are_exposed_in_raw(self):
        """RRStep preserves think-tool notes recorded during evidence gathering."""
        rr = RRStep("test-model", config=RRConfig())
        reflection, metadata = _mock_compaction_result()

        def _run_with_thought(*args, **kwargs):
            deps = kwargs["deps"]
            deps.thoughts.append(
                {
                    "thought": "The selected flights satisfy the requested dates.",
                    "evidence_refs": ["messages[5]", "messages[9]"],
                }
            )
            return reflection, metadata

        with patch(_RUN_SYNC, side_effect=_run_with_thought):
            result_ctx = rr(_make_ctx())

        thoughts = result_ctx.reflections[0].raw["thoughts"]
        assert thoughts == [
            {
                "thought": "The selected flights satisfy the requested dates.",
                "evidence_refs": ["messages[5]", "messages[9]"],
            }
        ]

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
        reflection, metadata = _mock_compaction_result(key_insight="reflected")

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
                feedback="Correct!",
            )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "reflected"


@pytest.mark.unit
class TestMeteredModel:
    """``MeteredModel`` fires the usage callback from the pydantic-ai model layer."""

    def test_callback_invoked_with_request_usage_and_model_name(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel
        from pydantic_ai.usage import RequestUsage

        from ace.core.metered_model import MeteredModel

        calls: list[tuple[RequestUsage, str]] = []

        def _cb(usage, model_id):
            calls.append((usage, model_id))

        inner = TestModel()
        agent = Agent(MeteredModel(inner, _cb), output_type=str)
        result = agent.run_sync("hello")

        assert result.output
        assert len(calls) >= 1
        reported_usage, model_id = calls[-1]
        assert isinstance(reported_usage, RequestUsage)
        assert reported_usage.input_tokens > 0
        assert model_id == inner.model_name

    def test_callback_exception_does_not_break_agent_run(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from ace.core.metered_model import MeteredModel

        def _cb(usage, model_id):
            raise RuntimeError("boom")

        agent = Agent(MeteredModel(TestModel(), _cb), output_type=str)
        result = agent.run_sync("hello")

        assert result.output

    def test_rrstep_accepts_prebuilt_model_instance(self):
        """Passing a pre-built ``Model`` flows through ``RRStep`` unchanged."""
        from pydantic_ai.models.test import TestModel

        test_model = TestModel()
        rr = RRStep(test_model, config=RRConfig())

        assert rr._model is test_model
        assert rr._agent.model is test_model

    def test_rrstep_wraps_model_when_usage_callback_set(self):
        """``RRStep.__init__`` routes the agent model through ``MeteredModel``."""
        from ace.core.metered_model import MeteredModel

        rr = RRStep(
            "test-model",
            config=RRConfig(usage_callback=lambda u, n: None),
        )

        assert isinstance(rr._agent.model, MeteredModel)

    def test_rrstep_does_not_wrap_when_no_callback(self):
        """Without a callback there's no wrapper overhead."""
        from ace.core.metered_model import MeteredModel

        rr = RRStep("test-model", config=RRConfig())

        assert not isinstance(rr._agent.model, MeteredModel)

    def test_rrstep_uses_prompted_reflector_output(self):
        """RR should gather evidence with tools and return structured output directly."""
        rr = RRStep("test-model", config=RRConfig())

        assert rr._agent._output_schema.mode == "prompted"
        assert rr._agent._output_schema.allows_text is True

    def test_rrstep_omits_temperature_by_default(self):
        """RR leaves temperature unset unless overridden."""
        rr = RRStep("test-model", config=RRConfig())

        assert "temperature" not in rr._agent.model_settings

    def test_rrstep_preserves_explicit_model_settings(self):
        """Callers can still override RR model settings explicitly."""
        rr = RRStep(
            "test-model",
            config=RRConfig(),
            model_settings=ModelSettings(temperature=0.7),
        )

        assert rr._agent.model_settings["temperature"] == 0.7

    def test_rrstep_specializes_execute_code_tool_description(self):
        """RR should present execute_code as an evidence tool, not a prose channel."""
        rr = RRStep("test-model", config=RRConfig())

        tool = rr._agent._function_toolset.tools["execute_code"]

        assert "evidence workbench" in tool.description
        assert "think" in tool.description
        assert "store strings/snippets" in tool.description
        assert tool.function_schema.description == tool.description
        code_schema = tool.function_schema.json_schema["properties"]["code"]
        assert "short snippet" in code_schema["description"]

    def test_small_trace_summary_includes_effort_guidance(self):
        """Small traces should discourage transcript walkthroughs."""
        rr = RRStep("test-model", config=RRConfig())

        summary = rr._build_data_summary(
            {
                "question": "q",
                "feedback": "Task PASSED",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        assert "Expected effort" in summary
        assert "2-4 focused execute_code checks" in summary
        assert "Do not produce a transcript walkthrough" in summary

    def test_prebuilt_model_and_callback_compose(self):
        """Pre-built Model + usage_callback both apply — meter wraps the instance."""
        from pydantic_ai.models.test import TestModel

        from ace.core.metered_model import MeteredModel

        inner = TestModel()
        rr = RRStep(
            inner,
            config=RRConfig(usage_callback=lambda u, n: None),
        )

        assert isinstance(rr._agent.model, MeteredModel)
        assert rr._agent.model.wrapped is inner
