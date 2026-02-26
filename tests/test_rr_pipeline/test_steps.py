"""Unit tests for the four inner RR pipeline steps."""

import pytest

from ace.reflector.config import RecursiveConfig
from ace.reflector.sandbox import TraceSandbox
from ace.reflector.subagent import CallBudget
from ace.llm import LLMResponse

from ace_next.rr.context import RRIterationContext
from ace_next.rr.steps import (
    LLMCallStep,
    ExtractCodeStep,
    SandboxExecStep,
    CheckResultStep,
    _parse_final_value,
    _parse_direct_response,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class MockLLM:
    """Minimal mock matching the LLMClient interface."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.call_count = 0

    def complete_messages(self, messages, **kwargs):
        self.call_count += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text)


@pytest.fixture
def config():
    return RecursiveConfig(max_iterations=5, timeout=5.0, max_output_chars=5000)


@pytest.fixture
def sandbox():
    return TraceSandbox(trace=None)


@pytest.fixture
def budget():
    return CallBudget(max_calls=10)


# ---------------------------------------------------------------------------
# LLMCallStep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMCallStep:
    def test_calls_llm_and_sets_response(self, config, budget):
        llm = MockLLM(["```python\nprint('hi')\n```"])
        step = LLMCallStep(llm, config, budget)
        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze this"},)
        )
        result = step(ctx)
        assert result.llm_response == "```python\nprint('hi')\n```"
        assert llm.call_count == 1

    def test_empty_response(self, config, budget):
        llm = MockLLM([""])
        step = LLMCallStep(llm, config, budget)
        ctx = RRIterationContext(messages=({"role": "user", "content": "analyze"},))
        result = step(ctx)
        assert result.llm_response == ""

    def test_budget_consumed(self, config, budget):
        llm = MockLLM(["response"])
        step = LLMCallStep(llm, config, budget)
        ctx = RRIterationContext(messages=({"role": "user", "content": "go"},))
        step(ctx)
        assert budget.count == 1


# ---------------------------------------------------------------------------
# ExtractCodeStep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractCodeStep:
    def test_extracts_python_block(self):
        step = ExtractCodeStep()
        ctx = RRIterationContext(llm_response='```python\nprint("hello")\n```')
        result = step(ctx)
        assert result.code == 'print("hello")'
        assert result.direct_response is None

    def test_no_code_sets_direct_response(self):
        step = ExtractCodeStep()
        ctx = RRIterationContext(llm_response="Just some text without code")
        result = step(ctx)
        assert result.code is None
        assert result.direct_response == "Just some text without code"

    def test_final_call_extraction(self):
        step = ExtractCodeStep()
        response = 'Some text\nFINAL({"reasoning": "done"})'
        ctx = RRIterationContext(llm_response=response)
        result = step(ctx)
        assert result.code is not None
        assert "FINAL(" in result.code


# ---------------------------------------------------------------------------
# SandboxExecStep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxExecStep:
    def test_executes_code(self, sandbox, config):
        step = SandboxExecStep(sandbox, config)
        ctx = RRIterationContext(code='print("hello world")')
        result = step(ctx)
        assert result.exec_result is not None
        assert "hello world" in result.exec_result.stdout

    def test_no_code_passes_through(self, sandbox, config):
        step = SandboxExecStep(sandbox, config)
        ctx = RRIterationContext(code=None)
        result = step(ctx)
        assert result.exec_result is None

    def test_execution_error_captured(self, sandbox, config):
        step = SandboxExecStep(sandbox, config)
        ctx = RRIterationContext(code="raise ValueError('test error')")
        result = step(ctx)
        assert result.exec_result is not None
        assert not result.exec_result.success
        assert "ValueError" in result.exec_result.stderr


# ---------------------------------------------------------------------------
# CheckResultStep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckResultStep:
    def test_normal_continuation(self, sandbox, config):
        """When code runs but FINAL() not called, produce feedback messages."""
        step = CheckResultStep(sandbox, config)
        # Simulate execution with output
        exec_result = sandbox.execute('print("exploring data")', timeout=5.0)

        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze"},),
            llm_response='```python\nprint("exploring data")\n```',
            code='print("exploring data")',
            exec_result=exec_result,
            iteration=1,
        )
        result = step(ctx)
        assert not result.terminated
        assert len(result.feedback_messages) == 2
        assert result.feedback_messages[1]["role"] == "user"
        assert "exploring data" in result.feedback_messages[1]["content"]

    def test_final_accepted(self, config):
        """When FINAL() is called after iteration 0, accept it."""
        sandbox = TraceSandbox(trace=None)
        step = CheckResultStep(sandbox, config)

        code = 'FINAL({"reasoning": "done", "key_insight": "test", "correct_approach": "ok"})'
        exec_result = sandbox.execute(code, timeout=5.0)

        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze"},),
            llm_response=f"```python\n{code}\n```",
            code=code,
            exec_result=exec_result,
            iteration=1,  # > 0
        )
        result = step(ctx)
        assert result.terminated is True
        assert result.reflection is not None
        assert result.reflection.key_insight == "test"

    def test_premature_final_rejected(self, config):
        """FINAL() on iteration 0 should be rejected."""
        sandbox = TraceSandbox(trace=None)
        step = CheckResultStep(sandbox, config)

        code = 'FINAL({"reasoning": "premature"})'
        exec_result = sandbox.execute(code, timeout=5.0)

        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze"},),
            llm_response=f"```python\n{code}\n```",
            code=code,
            exec_result=exec_result,
            iteration=0,
        )
        result = step(ctx)
        assert not result.terminated
        assert "before exploring" in result.feedback_messages[1]["content"]

    def test_final_after_error_rejected(self, config):
        """FINAL() when code had errors should be rejected."""
        sandbox = TraceSandbox(trace=None)
        step = CheckResultStep(sandbox, config)

        # Code that errors but also calls FINAL somehow
        code = """
x = 1/0
FINAL({"reasoning": "error"})
"""
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

    def test_direct_response_json_parsed(self, sandbox, config):
        """When no code is found but response is valid JSON, parse it."""
        step = CheckResultStep(sandbox, config)
        json_response = (
            '{"reasoning": "direct", "key_insight": "neat", "correct_approach": "ok"}'
        )
        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze"},),
            llm_response=json_response,
            code=None,
            direct_response=json_response,
            exec_result=None,
            iteration=1,
        )
        result = step(ctx)
        assert result.terminated is True
        assert result.reflection.key_insight == "neat"

    def test_empty_response_feedback(self, sandbox, config):
        """Empty LLM response should produce retry feedback."""
        step = CheckResultStep(sandbox, config)
        ctx = RRIterationContext(
            messages=({"role": "user", "content": "analyze"},),
            llm_response="",
            code=None,
            exec_result=None,
            iteration=1,
        )
        result = step(ctx)
        assert not result.terminated
        assert "empty" in result.feedback_messages[1]["content"].lower()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParsingHelpers:
    def test_parse_final_value_dict(self):
        value = {
            "reasoning": "test reasoning",
            "key_insight": "key",
            "correct_approach": "approach",
            "extracted_learnings": [
                {"learning": "learn1", "atomicity_score": 0.8, "evidence": "ev1"}
            ],
            "skill_tags": [{"id": "sk1", "tag": "helpful"}],
        }
        result = _parse_final_value(value)
        assert result.reasoning == "test reasoning"
        assert len(result.extracted_learnings) == 1
        assert result.extracted_learnings[0].learning == "learn1"
        assert len(result.skill_tags) == 1
        assert result.skill_tags[0].tag == "helpful"

    def test_parse_final_value_string(self):
        result = _parse_final_value("just a string")
        assert result.reasoning == "just a string"

    def test_parse_direct_response_json(self):
        response = '{"reasoning": "ok", "key_insight": "i", "correct_approach": "a"}'
        result = _parse_direct_response(response)
        assert result.reasoning == "ok"

    def test_parse_direct_response_markdown_json(self):
        response = '```json\n{"reasoning": "ok", "key_insight": "i", "correct_approach": "a"}\n```'
        result = _parse_direct_response(response)
        assert result.reasoning == "ok"

    def test_parse_direct_response_invalid(self):
        with pytest.raises(Exception):
            _parse_direct_response("not json at all")
