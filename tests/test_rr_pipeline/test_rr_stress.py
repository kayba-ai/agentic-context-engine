"""Stress tests for RR components.

Tests sandbox behavior, code extraction, FINAL parsing, and the new
PydanticAI-based RRStep entry points.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ace_next.rr.config import RecursiveConfig
from ace_next.rr.sandbox import TraceSandbox

from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import AgentOutput, ReflectorOutput
from ace_next.core.skillbook import Skillbook
from ace_next.rr import RRConfig, RRStep

# Keep old imports for tests that still test unchanged modules
from ace_next.rr.context import RRIterationContext
from ace_next.rr.steps import (
    ExtractCodeStep,
    SandboxExecStep,
    CheckResultStep,
    _parse_final_value,
)

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
            rr._agent, "run_sync",
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
            rr._agent, "run_sync",
            side_effect=UsageLimitExceeded("limit"),
        ):
            result_ctx = rr(_make_ctx())

        output = result_ctx.reflections[0]
        assert "rr_trace" in output.raw
        assert output.raw["rr_trace"]["timed_out"] is True


# =========================================================================
# 2. Code extraction edge cases (unchanged module)
# =========================================================================


@pytest.mark.unit
class TestCodeExtractionEdgeCases:
    def test_nested_backticks_in_code(self):
        """Code containing triple backticks inside strings."""
        step = ExtractCodeStep()
        response = '```python\nx = "```hello```"\nprint(x)\n```'
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        assert result.code is not None
        assert 'x = "' in result.code

    def test_bare_code_block_no_python_tag(self):
        """Bare ``` without 'python' tag."""
        step = ExtractCodeStep()
        response = '```\nprint("bare block")\n```'
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        assert result.code is not None
        assert "bare block" in result.code

    def test_final_call_without_code_block(self):
        """FINAL() as plain text with no fences."""
        step = ExtractCodeStep()
        response = 'After analysis:\nFINAL({"reasoning": "done", "key_insight": "k", "correct_approach": "a"})'
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        assert result.code is not None
        assert "FINAL(" in result.code

    def test_empty_code_block(self):
        """Empty code block — no code extracted."""
        step = ExtractCodeStep()
        response = "```python\n```"
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        if result.code is not None:
            assert result.code.strip() == ""
        else:
            assert result.direct_response is not None


# =========================================================================
# 3. FINAL() parsing edge cases (unchanged module)
# =========================================================================


@pytest.mark.unit
class TestFinalParsingEdgeCases:
    def test_final_with_missing_fields(self):
        """FINAL with only reasoning — other fields default."""
        result = _parse_final_value({"reasoning": "only reasoning"})
        assert result.reasoning == "only reasoning"
        assert result.key_insight == ""
        assert result.correct_approach == ""
        assert result.extracted_learnings == []
        assert result.skill_tags == []

    def test_final_with_non_dict_value(self):
        """FINAL("just a string") creates ReflectorOutput."""
        result = _parse_final_value("just a string")
        assert result.reasoning == "just a string"

    def test_final_with_bad_atomicity_score(self):
        """FINAL with atomicity_score='high' should not crash."""
        value = {
            "reasoning": "r",
            "key_insight": "k",
            "correct_approach": "a",
            "extracted_learnings": [
                {"learning": "l", "atomicity_score": "high", "evidence": "e"}
            ],
            "skill_tags": [],
        }
        result = _parse_final_value(value)
        assert len(result.extracted_learnings) == 1
        assert result.extracted_learnings[0].atomicity_score == 0.0

    def test_final_after_execution_error_rejected(self):
        """FINAL when sandbox code raised should be rejected."""
        sandbox = TraceSandbox(trace=None)
        config = RecursiveConfig()
        step = CheckResultStep(sandbox, config)

        code = 'x = 1/0\nFINAL({"reasoning": "error"})'
        exec_result = sandbox.execute(code, timeout=5.0)

        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze"},),
            llm_response=f"```python\n{code}\n```",
            code=code,
            exec_result=exec_result,
            iteration=1,
        )
        result = step(ctx)
        assert not result.terminated
        assert "error" in result.feedback_messages[1]["content"].lower()


# =========================================================================
# 4. Sandbox behavior (unchanged module)
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
        config = RecursiveConfig()
        step = SandboxExecStep(sandbox, config)
        ctx = RRIterationContext(code="raise RuntimeError('boom')")
        result = step(ctx)
        assert result.exec_result is not None
        assert not result.exec_result.success
        assert "RuntimeError" in result.exec_result.stderr
        assert "boom" in result.exec_result.stderr


# =========================================================================
# 5. Entry points (PydanticAI-based)
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
