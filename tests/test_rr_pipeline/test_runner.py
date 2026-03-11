"""Integration tests for RRStep — the SubRunner that wires inner pipeline + REPL loop."""

import json

import pytest

from ace.llm import LLMResponse
from ace_next.rr.config import RecursiveConfig
from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import AgentOutput, ReflectorOutput
from ace_next.core.skillbook import Skillbook

from ace_next.rr import RRStep, RRConfig

# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock that returns queued responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.call_count = 0

    def set_responses(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def complete_messages(self, messages, **kwargs):
        self.call_count += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text)

    def complete(self, prompt, **kwargs):
        self.call_count += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text)


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRStep:
    """Test RRStep.__call__() — the StepProtocol entry point."""

    def test_basic_reflection(self):
        """Two iterations: explore then FINAL."""
        explore = '```python\nprint(traces["question"])\n```'
        final = """```python
FINAL({
    "reasoning": "Agent answered correctly",
    "key_insight": "Simple arithmetic handled well",
    "correct_approach": "Step-by-step calculation",
    "extracted_learnings": [
        {"learning": "Step-by-step works", "atomicity_score": 0.8, "evidence": "2+2=4"}
    ],
    "skill_tags": []
})
```"""
        llm = MockLLM([explore, final])
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))

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
        assert result_ctx.reflections[0].key_insight == "Simple arithmetic handled well"
        assert len(result_ctx.reflections[0].extracted_learnings) == 1
        assert llm.call_count == 2

    def test_timeout_produces_output(self):
        """When max iterations reached, a timeout ReflectorOutput is returned."""
        explore = '```python\nprint("still looking...")\n```'
        llm = MockLLM([explore] * 3)
        rr = RRStep(
            llm,
            config=RRConfig(
                max_iterations=3,
                enable_subagent=False,
                enable_fallback_synthesis=False,
            ),
        )

        ctx = _make_ctx(question="What is 2+2?", answer="4", ground_truth="4")
        result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 1
        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert "max iterations" in result_ctx.reflections[0].reasoning.lower()

    def test_premature_final_rejected_then_accepted(self):
        """FINAL on iteration 0 is rejected; on iteration 1 it's accepted."""
        premature_final = """```python
FINAL({"reasoning": "premature", "key_insight": "k", "correct_approach": "a"})
```"""
        explore = '```python\nprint(traces["question"])\n```'
        good_final = """```python
FINAL({
    "reasoning": "After exploring",
    "key_insight": "explored",
    "correct_approach": "ok",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
        llm = MockLLM([premature_final, explore, good_final])
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))

        result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert result_ctx.reflections[0].key_insight == "explored"
        assert llm.call_count == 3  # premature + explore + final

    def test_direct_json_response(self):
        """When LLM returns JSON without code blocks, parse directly."""
        json_response = json.dumps(
            {
                "reasoning": "direct json",
                "key_insight": "insight",
                "correct_approach": "approach",
                "extracted_learnings": [],
                "skill_tags": [],
            }
        )
        llm = MockLLM([json_response])
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))

        result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert result_ctx.reflections[0].key_insight == "insight"

    def test_step_protocol_attributes(self):
        llm = MockLLM()
        rr = RRStep(llm)
        assert "trace" in rr.requires
        assert "skillbook" in rr.requires
        assert "reflections" in rr.provides
        assert "reflection" not in rr.provides

    def test_inner_context_uses_singular_reflection(self):
        """RRIterationContext.reflection (inner) is separate from ACEStepContext.reflections (outer)."""
        from ace_next.rr.context import RRIterationContext
        from ace_next.rr.steps import CheckResultStep

        # Inner context has singular 'reflection'
        inner = RRIterationContext()
        assert hasattr(inner, "reflection")
        assert not hasattr(inner, "reflections")

        # CheckResultStep provides singular 'reflection' (inner)
        assert "reflection" in CheckResultStep.provides
        assert "reflections" not in CheckResultStep.provides

    def test_call_produces_reflection_on_context(self):
        """RRStep.__call__ populates ctx.reflections."""
        explore = '```python\nprint("exploring")\n```'
        final = """```python
FINAL({
    "reasoning": "ok",
    "key_insight": "step test",
    "correct_approach": "approach",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
        llm = MockLLM([explore, final])
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))

        ctx = ACEStepContext(
            trace={
                "question": "q",
                "ground_truth": "gt",
                "feedback": "fb",
                "steps": [
                    {"role": "agent", "reasoning": "r", "answer": "a", "skill_ids": []}
                ],
            },
            skillbook=SkillbookView(Skillbook()),
        )
        result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 1
        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert result_ctx.reflections[0].key_insight == "step test"


@pytest.mark.unit
class TestRRStepProtocol:
    """Test that RRStep satisfies structural protocols."""

    def test_satisfies_reflector_like(self):
        """RRStep satisfies ReflectorLike protocol."""
        from ace_next.protocols import ReflectorLike

        llm = MockLLM()
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))
        assert isinstance(rr, ReflectorLike)


@pytest.mark.unit
class TestRRStepTimeout:
    """Test _on_timeout kwargs-based fallback (no instance-level stashing)."""

    def test_run_loop_directly_does_not_raise_attribute_error(self):
        """run_loop() without __call__() must not raise AttributeError.

        Before the fix, _on_timeout read self._timeout_args which was only
        set internally.  Now it reads from **kwargs, so calling run_loop()
        directly produces a graceful timeout output.
        """
        explore = '```python\nprint("exploring")\n```'
        llm = MockLLM([explore] * 2)
        rr = RRStep(
            llm,
            config=RRConfig(
                max_iterations=2,
                enable_subagent=False,
                enable_fallback_synthesis=False,
            ),
        )

        # Call run_loop directly — no __call__() wrapper
        budget = __import__(
            "ace.reflector.subagent", fromlist=["CallBudget"]
        ).CallBudget(10)
        result = rr.run_loop(
            sandbox=rr._create_sandbox(
                None, {"question": "q", "steps": []}, None, budget=budget
            ),
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
        assert "max iterations" in result.reasoning.lower()

    def test_timeout_without_timeout_args_gives_safe_defaults(self):
        """If timeout_args kwarg is missing, _on_timeout still works."""
        explore = '```python\nprint("exploring")\n```'
        llm = MockLLM([explore])
        rr = RRStep(
            llm,
            config=RRConfig(
                max_iterations=1,
                enable_subagent=False,
                enable_fallback_synthesis=False,
            ),
        )

        budget = __import__(
            "ace.reflector.subagent", fromlist=["CallBudget"]
        ).CallBudget(10)
        result = rr.run_loop(
            sandbox=rr._create_sandbox(
                None, {"question": "q", "steps": []}, None, budget=budget
            ),
            budget=budget,
            initial_prompt="test prompt",
            # No timeout_args — should use safe defaults
        )
        assert isinstance(result, ReflectorOutput)
        assert "max iterations" in result.reasoning.lower()


@pytest.mark.unit
class TestRRTraceData:
    """Test that RRStep populates rr_trace on ReflectorOutput.raw."""

    def test_rr_trace_populated_on_success(self):
        """Successful reflection stores iteration log in .raw['rr_trace']."""
        explore = '```python\nprint(traces["question"])\n```'
        final = """```python
FINAL({
    "reasoning": "ok",
    "key_insight": "insight",
    "correct_approach": "approach",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
        llm = MockLLM([explore, final])
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))

        result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert "rr_trace" in result_ctx.reflections[0].raw
        rr_trace = result_ctx.reflections[0].raw["rr_trace"]
        assert rr_trace["total_iterations"] == 2
        assert rr_trace["timed_out"] is False
        assert len(rr_trace["iterations"]) == 2
        assert rr_trace["iterations"][0]["iteration"] == 0
        assert rr_trace["iterations"][1]["iteration"] == 1
        assert rr_trace["iterations"][1]["terminated"] is True
        assert isinstance(rr_trace["subagent_calls"], list)

    def test_rr_trace_populated_on_timeout(self):
        """Timeout reflection also stores iteration log."""
        explore = '```python\nprint("looking...")\n```'
        llm = MockLLM([explore] * 2)
        rr = RRStep(
            llm,
            config=RRConfig(
                max_iterations=2,
                enable_subagent=False,
                enable_fallback_synthesis=False,
            ),
        )

        result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert "rr_trace" in result_ctx.reflections[0].raw
        rr_trace = result_ctx.reflections[0].raw["rr_trace"]
        assert rr_trace["total_iterations"] == 2
        assert rr_trace["timed_out"] is True

    def test_iteration_log_has_code_and_stdout(self):
        """Each iteration entry has code and stdout fields."""
        code_response = '```python\nprint("hello")\n```'
        final = """```python
FINAL({"reasoning": "r", "key_insight": "k", "correct_approach": "a"})
```"""
        llm = MockLLM([code_response, final])
        rr = RRStep(llm, config=RRConfig(max_iterations=5, enable_subagent=False))

        result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        it0 = result_ctx.reflections[0].raw["rr_trace"]["iterations"][0]
        assert it0["code"] is not None
        assert "hello" in (it0["stdout"] or "")


@pytest.mark.unit
class TestRROpikStep:
    """Test RROpikStep — graceful degradation and data reading."""

    def test_noop_when_opik_unavailable(self):
        """RROpikStep is a no-op when Opik is not installed."""
        from ace_next.rr.opik import RROpikStep, OPIK_AVAILABLE

        step = RROpikStep(project_name="test")
        # If Opik IS installed in the test env, skip this assertion
        if not OPIK_AVAILABLE:
            assert not step.enabled

    def test_noop_when_no_reflection(self):
        """RROpikStep returns ctx unchanged when reflections is empty."""
        from ace_next.rr.opik import RROpikStep

        step = RROpikStep(project_name="test")
        # Force disabled to avoid needing real Opik client
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
