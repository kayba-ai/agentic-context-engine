"""Integration tests for RRStep — the SubRunner that wires inner pipeline + REPL loop."""

import json

import pytest

from ace.llm import LLMResponse
from ace.reflector.config import RecursiveConfig
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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRStepReflect:
    """Test RRStep.reflect() — the ReflectorLike entry point."""

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

        result = rr.reflect(
            question="What is 2+2?",
            agent_output=AgentOutput(reasoning="2+2=4", final_answer="4", skill_ids=[]),
            skillbook=Skillbook(),
            ground_truth="4",
            feedback="Correct!",
        )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "Simple arithmetic handled well"
        assert len(result.extracted_learnings) == 1
        assert llm.call_count == 2

    def test_timeout_produces_output(self):
        """When max iterations reached, a timeout ReflectorOutput is returned."""
        # All iterations return code that doesn't call FINAL
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

        result = rr.reflect(
            question="What is 2+2?",
            agent_output=AgentOutput(reasoning="2+2=4", final_answer="4", skill_ids=[]),
            skillbook=Skillbook(),
            ground_truth="4",
        )

        assert isinstance(result, ReflectorOutput)
        assert "max iterations" in result.reasoning.lower()

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

        result = rr.reflect(
            question="test",
            agent_output=AgentOutput(
                reasoning="reasoning", final_answer="answer", skill_ids=[]
            ),
            skillbook=Skillbook(),
        )

        assert result.key_insight == "explored"
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

        result = rr.reflect(
            question="test",
            agent_output=AgentOutput(reasoning="r", final_answer="a", skill_ids=[]),
            skillbook=Skillbook(),
        )

        assert result.key_insight == "insight"


@pytest.mark.unit
class TestRRStepAsStep:
    """Test RRStep.__call__() — the StepProtocol entry point."""

    def test_step_protocol_attributes(self):
        llm = MockLLM()
        rr = RRStep(llm)
        assert "trace" in rr.requires
        assert "skillbook" in rr.requires
        assert "reflection" in rr.provides

    def test_call_produces_reflection_on_context(self):
        """RRStep.__call__ populates ctx.reflection."""
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

        sb = Skillbook()
        traces = {
            "question": "q",
            "ground_truth": "gt",
            "feedback": "fb",
            "steps": [
                {"role": "agent", "reasoning": "r", "answer": "a", "skill_ids": []}
            ],
        }
        ctx = ACEStepContext(
            trace=traces,
            skillbook=SkillbookView(sb),
        )
        result_ctx = rr(ctx)

        assert result_ctx.reflection is not None
        assert isinstance(result_ctx.reflection, ReflectorOutput)
        assert result_ctx.reflection.key_insight == "step test"


@pytest.mark.unit
class TestRRStepTimeout:
    """Test _on_timeout kwargs-based fallback (no instance-level stashing)."""

    def test_run_loop_directly_does_not_raise_attribute_error(self):
        """run_loop() without reflect() must not raise AttributeError.

        Before the fix, _on_timeout read self._timeout_args which was only
        set by reflect().  Now it reads from **kwargs, so calling run_loop()
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

        # Call run_loop directly — no reflect() wrapper
        result = rr.run_loop(
            sandbox=rr._create_sandbox(None, {"question": "q", "steps": []}, None),
            budget=__import__(
                "ace.reflector.subagent", fromlist=["CallBudget"]
            ).CallBudget(10),
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

        result = rr.run_loop(
            sandbox=rr._create_sandbox(None, {"question": "q", "steps": []}, None),
            budget=__import__(
                "ace.reflector.subagent", fromlist=["CallBudget"]
            ).CallBudget(10),
            initial_prompt="test prompt",
            # No timeout_args — should use safe defaults
        )
        assert isinstance(result, ReflectorOutput)
        assert "max iterations" in result.reasoning.lower()


@pytest.mark.unit
class TestRRStepBackwardCompat:
    """Ensure RRStep satisfies ReflectorLike protocol."""

    def test_satisfies_reflector_like(self):
        from ace_next.protocols import ReflectorLike

        llm = MockLLM()
        rr = RRStep(llm)
        assert isinstance(rr, ReflectorLike)
