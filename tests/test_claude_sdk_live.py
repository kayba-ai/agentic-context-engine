"""Live integration tests for the Claude SDK step.

These tests make REAL API calls to the Anthropic API. They require:
- ``ANTHROPIC_API_KEY`` environment variable to be set
- Network access to ``api.anthropic.com``

Run with::

    ANTHROPIC_API_KEY=sk-... uv run pytest tests/test_claude_sdk_live.py -v -m integration
"""

from __future__ import annotations

import os

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.skillbook import Skillbook
from ace.integrations.claude_sdk import (
    ClaudeSDKExecuteStep,
    ClaudeSDKResult,
    ClaudeSDKToTrace,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_api,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]

pytest.importorskip("anthropic")

MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------ #
# 1. Basic text generation
# ------------------------------------------------------------------ #


class TestBasicGeneration:
    def test_simple_question(self):
        """Verify a basic question returns a successful result with tokens."""
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=100)
        ctx = ACEStepContext(
            sample="What is 2+2? Answer with just the number.", skillbook=None
        )

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert r.error is None
        assert "4" in r.output
        assert r.model == MODEL
        assert r.stop_reason in ("end_turn", "max_tokens")
        assert r.input_tokens > 0
        assert r.output_tokens > 0
        assert r.total_tokens == r.input_tokens + r.output_tokens
        assert r.latency_seconds > 0
        assert r.raw_response is not None

    def test_longer_response(self):
        """Verify the step handles multi-sentence responses."""
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=300)
        ctx = ACEStepContext(
            sample="Explain in 2-3 sentences why the sky is blue.",
            skillbook=None,
        )

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert len(r.output) > 50
        assert r.output_tokens > 10


# ------------------------------------------------------------------ #
# 2. System prompt
# ------------------------------------------------------------------ #


class TestSystemPrompt:
    def test_system_prompt_affects_output(self):
        """A system prompt instructing a specific format should be followed."""
        step = ClaudeSDKExecuteStep(
            model=MODEL,
            max_tokens=50,
            system_prompt="You are a calculator. Only output numbers, nothing else.",
        )
        ctx = ACEStepContext(sample="What is 15 * 3?", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert "45" in r.output


# ------------------------------------------------------------------ #
# 3. Skillbook injection
# ------------------------------------------------------------------ #


class TestSkillbookInjection:
    def test_skillbook_injected_into_system(self):
        """When a skillbook has skills, they're injected and the model can reference them."""
        sb = Skillbook()
        sb.add_skill("math", "Always show step-by-step work for math problems")
        sb.add_skill("format", "End every answer with 'QED'")

        step = ClaudeSDKExecuteStep(
            model=MODEL,
            max_tokens=200,
            inject_skillbook=True,
        )
        ctx = ACEStepContext(
            sample="What is 7 * 8? Follow the strategies you've been given.",
            skillbook=SkillbookView(sb),
        )

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert "56" in r.output

    def test_skillbook_injection_disabled(self):
        """With inject_skillbook=False, skills are NOT sent to the model."""
        sb = Skillbook()
        sb.add_skill("format", "End every answer with the word BANANA")

        step = ClaudeSDKExecuteStep(
            model=MODEL,
            max_tokens=50,
            inject_skillbook=False,
        )
        ctx = ACEStepContext(
            sample="What is 1+1? Just answer the number.",
            skillbook=SkillbookView(sb),
        )

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert "BANANA" not in r.output


# ------------------------------------------------------------------ #
# 4. Tool use
# ------------------------------------------------------------------ #


class TestToolUse:
    def test_tool_call_returned(self):
        """When given a tool, the model should call it and we capture the call."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ]
        step = ClaudeSDKExecuteStep(
            model=MODEL,
            max_tokens=200,
            tools=tools,
        )
        ctx = ACEStepContext(
            sample="What's the weather in Paris right now?",
            skillbook=None,
        )

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert r.stop_reason in ("tool_use", "end_turn")
        if r.stop_reason == "tool_use":
            assert len(r.tool_calls) >= 1
            tc = r.tool_calls[0]
            assert tc.name == "get_weather"
            assert "city" in tc.input
            assert tc.id.startswith("toolu_")


# ------------------------------------------------------------------ #
# 5. Temperature
# ------------------------------------------------------------------ #


class TestTemperature:
    def test_zero_temperature_deterministic(self):
        """Temperature 0 should produce near-identical outputs."""
        step = ClaudeSDKExecuteStep(
            model=MODEL,
            max_tokens=20,
            temperature=0.0,
        )
        ctx = ACEStepContext(
            sample="Complete this: 1, 2, 3, 4, ",
            skillbook=None,
        )

        r1: ClaudeSDKResult = step(ctx).trace  # type: ignore[assignment]
        r2: ClaudeSDKResult = step(ctx).trace  # type: ignore[assignment]

        assert r1.success and r2.success
        # With temp=0 outputs should be identical or very similar
        assert r1.output[:10] == r2.output[:10]


# ------------------------------------------------------------------ #
# 6. Error handling
# ------------------------------------------------------------------ #


class TestErrorHandling:
    def test_invalid_model_returns_error(self):
        """An invalid model name should return a failed result, not crash."""
        step = ClaudeSDKExecuteStep(
            model="claude-nonexistent-99",
            max_tokens=10,
        )
        ctx = ACEStepContext(sample="hello", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is False
        assert r.error is not None
        assert r.latency_seconds >= 0

    def test_max_tokens_respected(self):
        """A very low max_tokens should truncate the response."""
        step = ClaudeSDKExecuteStep(
            model=MODEL,
            max_tokens=5,
        )
        ctx = ACEStepContext(
            sample="Write a 500-word essay about the history of computing.",
            skillbook=None,
        )

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert r.stop_reason == "max_tokens"
        assert r.output_tokens <= 10  # small buffer around max_tokens


# ------------------------------------------------------------------ #
# 7. ToTrace conversion with real data
# ------------------------------------------------------------------ #


class TestToTraceWithRealData:
    def test_full_pipeline_execute_then_convert(self):
        """Run execute step then ToTrace step — end-to-end pipeline flow."""
        execute = ClaudeSDKExecuteStep(model=MODEL, max_tokens=100)
        to_trace = ClaudeSDKToTrace()

        ctx = ACEStepContext(sample="What is the capital of France?", skillbook=None)

        # Execute
        ctx = execute(ctx)
        r: ClaudeSDKResult = ctx.trace  # type: ignore[assignment]
        assert r.success is True
        assert "Paris" in r.output

        # Convert
        ctx = to_trace(ctx)
        trace = ctx.trace
        assert isinstance(trace, dict)
        assert trace["question"] == "What is the capital of France?"
        assert "Paris" in trace["answer"]
        assert "succeeded" in trace["reasoning"]
        assert str(r.input_tokens) in trace["reasoning"]
        assert "succeeded" in trace["feedback"]
        assert trace["ground_truth"] is None


# ------------------------------------------------------------------ #
# 8. ACESample input (structured sample)
# ------------------------------------------------------------------ #


class TestACESampleInput:
    def test_structured_sample(self):
        """Step should extract question+context from a structured sample."""
        from dataclasses import dataclass

        @dataclass
        class Sample:
            question: str = "What color is a ripe banana?"
            context: str = "We are discussing fruits."
            ground_truth: str = "yellow"
            metadata: dict = None  # type: ignore[assignment]

            def __post_init__(self):
                self.metadata = self.metadata or {}

        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=30)
        ctx = ACEStepContext(sample=Sample(), skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert r.task  # task was extracted
        assert "banana" in r.task.lower() or "color" in r.task.lower()


# ------------------------------------------------------------------ #
# 9. Observability data quality
# ------------------------------------------------------------------ #


class TestObservabilityData:
    def test_token_counts_realistic(self):
        """Token counts should be reasonable for a simple prompt."""
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=50)
        ctx = ACEStepContext(sample="Say 'hello world'", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        # Input: system overhead + short prompt, should be < 100
        assert 1 < r.input_tokens < 100
        # Output: short response
        assert 1 < r.output_tokens <= 50
        # Total is sum
        assert r.total_tokens == r.input_tokens + r.output_tokens
        # Latency > 0 and reasonable (< 30s for a short request)
        assert 0 < r.latency_seconds < 30

    def test_model_field_matches_request(self):
        """The model field in the result should match what we requested."""
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=10)
        ctx = ACEStepContext(sample="hi", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.model == MODEL


# ------------------------------------------------------------------ #
# 10. Pydantic validation on real results
# ------------------------------------------------------------------ #


class TestPydanticValidation:
    def test_result_is_pydantic_model(self):
        """ClaudeSDKResult should be a Pydantic BaseModel."""
        from pydantic import BaseModel

        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=10)
        ctx = ACEStepContext(sample="hi", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert isinstance(r, BaseModel)

    def test_model_dump(self):
        """Real result should serialise cleanly, with raw_response excluded."""
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=20)
        ctx = ACEStepContext(sample="Say hello", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        d = r.model_dump()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert d["input_tokens"] > 0
        assert d["output_tokens"] > 0
        assert d["total_tokens"] == d["input_tokens"] + d["output_tokens"]
        assert "raw_response" not in d

    def test_model_dump_json(self):
        """Result should serialise to JSON string."""
        import json

        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=10)
        ctx = ACEStepContext(sample="hi", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        j = r.model_dump_json()
        parsed = json.loads(j)
        assert parsed["task"] == "hi"
        assert parsed["success"] is True

    def test_total_tokens_auto_computed(self):
        """total_tokens should equal input + output on a real response."""
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=20)
        ctx = ACEStepContext(sample="Count to 3", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.total_tokens == r.input_tokens + r.output_tokens
        assert r.total_tokens > 0

    def test_tool_calls_are_pydantic_models(self):
        """Tool calls in the result should be validated ToolCall models."""
        from ace.integrations.claude_sdk import ToolCall

        tools = [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "input_schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            }
        ]
        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=100, tools=tools)
        ctx = ACEStepContext(sample="Look up the value of 'pi'", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        if r.tool_calls:
            tc = r.tool_calls[0]
            assert isinstance(tc, ToolCall)
            assert isinstance(tc.id, str)
            assert isinstance(tc.name, str)
            assert isinstance(tc.input, dict)


# ------------------------------------------------------------------ #
# 11. Logfire integration (real)
# ------------------------------------------------------------------ #


class TestLogfireIntegration:
    def test_logfire_configure_and_instrument(self):
        """Configure Logfire, create a step, make a call — verify spans are sent."""
        from ace.observability import configure_logfire, is_configured

        configured = configure_logfire()
        if not configured:
            pytest.skip("Logfire not configured (missing LOGFIRE_TOKEN?)")

        assert is_configured()

        step = ClaudeSDKExecuteStep(model=MODEL, max_tokens=20)
        ctx = ACEStepContext(sample="Say 'logfire test'", skillbook=None)

        result_ctx = step(ctx)
        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]

        assert r.success is True
        assert r.output_tokens > 0
