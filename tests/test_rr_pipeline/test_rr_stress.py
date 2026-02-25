"""Stress tests exercising every code path in the RRStep REPL loop.

All tests use MockLLM — no live LLM calls.
"""

import json

import pytest

from ace.llm import LLMResponse
from ace.reflector.config import RecursiveConfig
from ace.reflector.sandbox import TraceSandbox
from ace.reflector.subagent import CallBudget

from ace_next.core.outputs import AgentOutput, ReflectorOutput
from ace_next.core.skillbook import Skillbook
from ace_next.rr import RRConfig, RRStep
from ace_next.rr.context import RRIterationContext
from ace_next.rr.steps import (
    LLMCallStep,
    ExtractCodeStep,
    SandboxExecStep,
    CheckResultStep,
    _parse_final_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    """Returns queued responses; empty string when exhausted."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.call_count = 0

    def complete_messages(self, messages, **kw):
        self.call_count += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text)

    def complete(self, prompt, **kw):
        self.call_count += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text)


def _make_rr(llm, max_iterations=5, **kw):
    return RRStep(
        llm,
        config=RRConfig(max_iterations=max_iterations, enable_subagent=False, **kw),
    )


def _agent_output(answer="4"):
    return AgentOutput(reasoning="r", final_answer=answer, skill_ids=[])


FINAL_GOOD = """```python
FINAL({
    "reasoning": "done",
    "key_insight": "insight",
    "correct_approach": "approach",
    "extracted_learnings": [
        {"learning": "l1", "atomicity_score": 0.8, "evidence": "e1"}
    ],
    "skill_tags": []
})
```"""

EXPLORE = '```python\nprint(traces["question"])\n```'


# =========================================================================
# 1. Loop lifecycle
# =========================================================================


@pytest.mark.unit
class TestLoopLifecycle:
    def test_single_iteration_explore_then_final(self):
        """Happy path: explore on iter 0, FINAL on iter 1."""
        llm = MockLLM([EXPLORE, FINAL_GOOD])
        rr = _make_rr(llm)
        result = rr.reflect(
            question="What is 2+2?",
            agent_output=_agent_output(),
            skillbook=Skillbook(),
            ground_truth="4",
            feedback="Correct!",
        )
        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "insight"
        assert llm.call_count == 2

    def test_immediate_final_rejected_then_accepted(self):
        """FINAL on iteration 0 is rejected; explore then FINAL succeeds."""
        premature = '```python\nFINAL({"reasoning":"fast","key_insight":"k","correct_approach":"a"})\n```'
        llm = MockLLM([premature, EXPLORE, FINAL_GOOD])
        rr = _make_rr(llm)
        result = rr.reflect(
            question="q", agent_output=_agent_output(), skillbook=Skillbook()
        )
        assert result.key_insight == "insight"
        assert llm.call_count == 3

    def test_max_iterations_timeout(self):
        """All iterations exhaust without FINAL -> timeout output."""
        llm = MockLLM([EXPLORE] * 3)
        rr = _make_rr(llm, max_iterations=3, enable_fallback_synthesis=False)
        result = rr.reflect(
            question="q", agent_output=_agent_output(), skillbook=Skillbook()
        )
        assert isinstance(result, ReflectorOutput)
        assert "max iterations" in result.reasoning.lower()

    def test_budget_exhaustion_stops_gracefully(self):
        """Budget runs out mid-loop — LLM should NOT be called."""
        llm = MockLLM([EXPLORE] * 10)
        rr = _make_rr(llm, max_iterations=10)

        # Directly test LLMCallStep with an already-exhausted budget
        budget = CallBudget(max_calls=0)  # already exhausted
        config = RecursiveConfig()
        step = LLMCallStep(llm, config, budget)
        ctx = RRIterationContext(messages=({"role": "user", "content": "go"},))
        result = step(ctx)
        assert result.llm_response == ""
        assert llm.call_count == 0  # LLM was NOT called

    def test_direct_json_no_code_blocks(self):
        """LLM returns raw JSON without code fences."""
        raw = json.dumps(
            {
                "reasoning": "direct",
                "key_insight": "di",
                "correct_approach": "a",
                "extracted_learnings": [],
                "skill_tags": [],
            }
        )
        llm = MockLLM([raw])
        rr = _make_rr(llm)
        result = rr.reflect(
            question="q", agent_output=_agent_output(), skillbook=Skillbook()
        )
        assert result.key_insight == "di"

    def test_empty_llm_response_gets_feedback(self):
        """Empty response triggers retry feedback, then FINAL succeeds."""
        llm = MockLLM(["", EXPLORE, FINAL_GOOD])
        rr = _make_rr(llm)
        result = rr.reflect(
            question="q", agent_output=_agent_output(), skillbook=Skillbook()
        )
        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "insight"
        assert llm.call_count == 3

    def test_20_iterations_message_accumulation(self):
        """Long loop — verify messages accumulate correctly."""
        llm = MockLLM([EXPLORE] * 20)
        rr = _make_rr(llm, max_iterations=20, enable_fallback_synthesis=False)
        result = rr.reflect(
            question="q", agent_output=_agent_output(), skillbook=Skillbook()
        )
        assert "max iterations" in result.reasoning.lower()
        assert llm.call_count == 20


# =========================================================================
# 2. Code extraction edge cases
# =========================================================================


@pytest.mark.unit
class TestCodeExtractionEdgeCases:
    def test_nested_backticks_in_code(self):
        """Code containing triple backticks inside strings truncates at first closing fence.

        Regex-based extraction cannot handle nested backticks — the extracted
        code is partial.  This documents the known limitation.
        """
        step = ExtractCodeStep()
        response = '```python\nx = "```hello```"\nprint(x)\n```'
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        # Extraction stops at the first ``` after the opening fence
        assert result.code is not None
        assert 'x = "' in result.code

    def test_bare_code_block_no_python_tag(self):
        """Bare ``` without 'python' tag — should still extract if looks like Python."""
        step = ExtractCodeStep()
        response = '```\nprint("bare block")\n```'
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        assert result.code is not None
        assert "bare block" in result.code

    def test_final_call_without_code_block(self):
        """FINAL() as plain text with no fences — fallback extraction."""
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
        # Empty block extracts empty string which is falsy -> direct_response
        if result.code is not None:
            assert result.code.strip() == ""
        else:
            assert result.direct_response is not None


# =========================================================================
# 3. FINAL() parsing edge cases
# =========================================================================


@pytest.mark.unit
class TestFinalParsingEdgeCases:
    def test_final_with_missing_fields(self):
        """FINAL with only reasoning — other fields should default."""
        result = _parse_final_value({"reasoning": "only reasoning"})
        assert result.reasoning == "only reasoning"
        assert result.key_insight == ""
        assert result.correct_approach == ""
        assert result.extracted_learnings == []
        assert result.skill_tags == []

    def test_final_with_non_dict_value(self):
        """FINAL("just a string") should create ReflectorOutput with reasoning."""
        result = _parse_final_value("just a string")
        assert result.reasoning == "just a string"

    def test_final_with_bad_atomicity_score(self):
        """FINAL with atomicity_score='high' should not crash (Bug B fix)."""
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
# 4. Sandbox behavior
# =========================================================================


@pytest.mark.unit
class TestSandboxBehavior:
    def test_sandbox_variables_persist_across_iterations(self):
        """Variables set in one execution persist for the next (REPL semantics)."""
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
        """Code that raises captures the error in stderr."""
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
# 5. Dual entry points
# =========================================================================


@pytest.mark.unit
class TestDualEntryPoints:
    def test_reflect_and_call_produce_same_result(self):
        """Both reflect() and __call__() should produce a ReflectorOutput."""
        from ace_next.core.context import ACEStepContext, SkillbookView

        explore = EXPLORE
        final = FINAL_GOOD
        sb = Skillbook()

        # reflect() path
        llm1 = MockLLM([explore, final])
        rr1 = _make_rr(llm1)
        result1 = rr1.reflect(question="q", agent_output=_agent_output(), skillbook=sb)

        # __call__() path
        llm2 = MockLLM([explore, final])
        rr2 = _make_rr(llm2)
        traces = {
            "question": "q",
            "ground_truth": None,
            "feedback": None,
            "steps": [
                {"role": "agent", "reasoning": "r", "answer": "4", "skill_ids": []}
            ],
        }
        ctx = ACEStepContext(trace=traces, skillbook=SkillbookView(sb))
        result_ctx = rr2(ctx)
        result2 = result_ctx.reflection

        assert isinstance(result1, ReflectorOutput)
        assert isinstance(result2, ReflectorOutput)
        assert result1.key_insight == result2.key_insight

    def test_run_loop_direct_with_all_kwargs(self):
        """run_loop() standalone with full kwargs produces valid output."""
        llm = MockLLM([EXPLORE, FINAL_GOOD])
        rr = _make_rr(llm)

        sandbox = rr._create_sandbox(None, {"question": "q", "steps": []}, None)
        budget = CallBudget(10)
        result = rr.run_loop(
            sandbox=sandbox,
            budget=budget,
            initial_prompt="test prompt",
            timeout_args={
                "question": "q",
                "agent_output": None,
                "ground_truth": None,
                "feedback": None,
            },
        )
        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "insight"
