"""Tests for the Claude SDK integration step."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.skillbook import Skillbook
from ace.integrations.claude_sdk import (
    ClaudeSDKExecuteStep,
    ClaudeSDKResult,
    ClaudeSDKToTrace,
    ToolCall,
)

# ------------------------------------------------------------------ #
# Helpers — mock Anthropic API objects
# ------------------------------------------------------------------ #


def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_tool_block(
    tool_id: str = "toolu_01", name: str = "calculator", inp: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=inp or {})


def _make_usage(input_tokens: int = 100, output_tokens: int = 50) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def _make_response(
    content: list | None = None,
    model: str = "claude-sonnet-4-20250514",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> SimpleNamespace:
    if content is None:
        content = [_make_text_block("Hello, world!")]
    return SimpleNamespace(
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=_make_usage(input_tokens, output_tokens),
    )


def _make_mock_client(response: SimpleNamespace | None = None) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response or _make_response()
    return client


@dataclass
class FakeSample:
    question: str = "What is 2+2?"
    context: str = "math quiz"
    ground_truth: str = "4"
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# ------------------------------------------------------------------ #
# ClaudeSDKResult
# ------------------------------------------------------------------ #


class TestClaudeSDKResult:
    def test_defaults(self):
        r = ClaudeSDKResult(task="test", success=True)
        assert r.task == "test"
        assert r.success is True
        assert r.output == ""
        assert r.error is None
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.total_tokens == 0
        assert r.latency_seconds == 0.0
        assert r.tool_calls == []
        assert r.cited_skill_ids == []
        assert r.raw_response is None

    def test_full(self):
        tc = ToolCall(id="toolu_01", name="calc", input={"expr": "1+1"})
        r = ClaudeSDKResult(
            task="hello",
            success=True,
            output="world",
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_seconds=1.5,
            tool_calls=[tc],
            cited_skill_ids=["math-001"],
        )
        assert r.total_tokens == 150
        assert r.tool_calls[0].name == "calc"

    def test_auto_compute_total_tokens(self):
        """total_tokens is auto-computed from input + output when left at 0."""
        r = ClaudeSDKResult(
            task="test", success=True, input_tokens=100, output_tokens=50
        )
        assert r.total_tokens == 150

    def test_explicit_total_not_overwritten(self):
        """An explicitly set total_tokens is preserved."""
        r = ClaudeSDKResult(
            task="test",
            success=True,
            input_tokens=100,
            output_tokens=50,
            total_tokens=999,
        )
        assert r.total_tokens == 999

    def test_negative_tokens_rejected(self):
        """Negative token counts should be rejected by validation."""
        with pytest.raises(Exception):
            ClaudeSDKResult(task="test", success=True, input_tokens=-1)

    def test_negative_latency_rejected(self):
        """Negative latency should be rejected by validation."""
        with pytest.raises(Exception):
            ClaudeSDKResult(task="test", success=True, latency_seconds=-0.1)

    def test_serialization(self):
        """Result should serialise to dict/JSON (raw_response excluded)."""
        r = ClaudeSDKResult(
            task="test",
            success=True,
            input_tokens=10,
            output_tokens=5,
            raw_response=object(),
        )
        d = r.model_dump()
        assert d["task"] == "test"
        assert "raw_response" not in d  # excluded

    def test_tool_call_validation(self):
        """ToolCall should validate required fields."""
        tc = ToolCall(id="toolu_01", name="calc")
        assert tc.input == {}

        with pytest.raises(Exception):
            ToolCall(name="calc")  # missing id


# ------------------------------------------------------------------ #
# ClaudeSDKExecuteStep — contracts
# ------------------------------------------------------------------ #


class TestClaudeSDKExecuteStepContracts:
    def test_requires_and_provides(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(client=client)
        assert "sample" in step.requires
        assert "skillbook" in step.requires
        assert "trace" in step.provides

    def test_not_available_raises_without_client(self):
        with patch("ace.integrations.claude_sdk.ANTHROPIC_SDK_AVAILABLE", False):
            with pytest.raises(ImportError, match="anthropic SDK not installed"):
                ClaudeSDKExecuteStep()

    def test_injected_client_skips_availability_check(self):
        with patch("ace.integrations.claude_sdk.ANTHROPIC_SDK_AVAILABLE", False):
            step = ClaudeSDKExecuteStep(client=_make_mock_client())
            assert "sample" in step.requires

    def test_invalid_max_tokens_rejected(self):
        with pytest.raises(ValidationError):
            ClaudeSDKExecuteStep(client=_make_mock_client(), max_tokens=0)

    def test_invalid_temperature_rejected(self):
        with pytest.raises(ValidationError):
            ClaudeSDKExecuteStep(client=_make_mock_client(), temperature=1.5)


# ------------------------------------------------------------------ #
# ClaudeSDKExecuteStep — execution
# ------------------------------------------------------------------ #


class TestClaudeSDKExecuteStepExecution:
    def test_basic_call(self):
        response = _make_response(
            content=[_make_text_block("The answer is 4")],
            input_tokens=80,
            output_tokens=20,
        )
        client = _make_mock_client(response)
        step = ClaudeSDKExecuteStep(client=client, model="claude-sonnet-4-20250514")

        ctx = ACEStepContext(sample="What is 2+2?", skillbook=None)
        result_ctx = step(ctx)

        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]
        assert r.success is True
        assert r.output == "The answer is 4"
        assert r.input_tokens == 80
        assert r.output_tokens == 20
        assert r.total_tokens == 100
        assert r.latency_seconds >= 0
        assert r.model == "claude-sonnet-4-20250514"
        assert r.stop_reason == "end_turn"

        # Verify API was called correctly
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_with_system_prompt(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(
            client=client,
            system_prompt="You are a math tutor.",
        )

        ctx = ACEStepContext(sample="What is 2+2?", skillbook=None)
        step(ctx)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a math tutor."

    def test_skillbook_injection(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(client=client, inject_skillbook=True)

        sb = Skillbook()
        sb.add_skill("math", "Always show your work")
        ctx = ACEStepContext(sample="What is 2+2?", skillbook=SkillbookView(sb))
        step(ctx)

        call_kwargs = client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert "Strategic Knowledge" in call_kwargs["system"]

    def test_skillbook_injection_with_system_prompt(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(
            client=client,
            system_prompt="You are a tutor.",
            inject_skillbook=True,
        )

        sb = Skillbook()
        sb.add_skill("math", "Show work")
        ctx = ACEStepContext(sample="test", skillbook=SkillbookView(sb))
        step(ctx)

        call_kwargs = client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert "You are a tutor." in system
        assert "Strategic Knowledge" in system

    def test_skillbook_injection_disabled(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(client=client, inject_skillbook=False)

        sb = Skillbook()
        sb.add_skill("math", "Show work")
        ctx = ACEStepContext(sample="test", skillbook=SkillbookView(sb))
        step(ctx)

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_empty_skillbook_no_system(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(client=client, inject_skillbook=True)

        sb = Skillbook()
        ctx = ACEStepContext(sample="test", skillbook=SkillbookView(sb))
        step(ctx)

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_with_tools(self):
        tools = [
            {
                "name": "calculator",
                "description": "A calculator",
                "input_schema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                },
            }
        ]
        response = _make_response(
            content=[
                _make_tool_block("toolu_01", "calculator", {"expr": "2+2"}),
                _make_text_block("The result is 4"),
            ]
        )
        client = _make_mock_client(response)
        step = ClaudeSDKExecuteStep(client=client, tools=tools)

        ctx = ACEStepContext(sample="Calculate 2+2", skillbook=None)
        result_ctx = step(ctx)

        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]
        assert r.success is True
        assert r.output == "The result is 4"
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "calculator"
        assert r.tool_calls[0].input == {"expr": "2+2"}

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools

    def test_api_error_handled(self):
        client = _make_mock_client()
        client.messages.create.side_effect = RuntimeError("API down")
        step = ClaudeSDKExecuteStep(client=client)

        ctx = ACEStepContext(sample="test", skillbook=None)
        result_ctx = step(ctx)

        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]
        assert r.success is False
        assert "API down" in r.error
        assert r.latency_seconds >= 0

    def test_temperature_and_max_tokens(self):
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(client=client, temperature=0.7, max_tokens=1024)

        ctx = ACEStepContext(sample="test", skillbook=None)
        step(ctx)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1024


# ------------------------------------------------------------------ #
# ClaudeSDKExecuteStep — task extraction
# ------------------------------------------------------------------ #


class TestClaudeSDKTaskExtraction:
    def test_string_sample(self):
        assert ClaudeSDKExecuteStep._extract_task("hello") == "hello"

    def test_sample_with_question(self):
        sample = FakeSample(question="What is 2+2?", context="")
        result = ClaudeSDKExecuteStep._extract_task(sample)
        assert result == "What is 2+2?"

    def test_sample_with_context(self):
        sample = FakeSample(question="What is 2+2?", context="math quiz")
        result = ClaudeSDKExecuteStep._extract_task(sample)
        assert "What is 2+2?" in result
        assert "Context: math quiz" in result

    def test_arbitrary_object(self):
        result = ClaudeSDKExecuteStep._extract_task(42)
        assert result == "42"


# ------------------------------------------------------------------ #
# ClaudeSDKExecuteStep — skill ID extraction
# ------------------------------------------------------------------ #


class TestClaudeSDKSkillExtraction:
    def test_extracts_skill_ids(self):
        response = _make_response(
            content=[
                _make_text_block(
                    "Following [math-00001], the answer is 4. "
                    "Also [general-00042] applies."
                )
            ]
        )
        client = _make_mock_client(response)
        step = ClaudeSDKExecuteStep(client=client)

        ctx = ACEStepContext(sample="test", skillbook=None)
        result_ctx = step(ctx)

        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]
        assert "math-00001" in r.cited_skill_ids
        assert "general-00042" in r.cited_skill_ids

    def test_no_skill_ids(self):
        response = _make_response(content=[_make_text_block("No citations here")])
        client = _make_mock_client(response)
        step = ClaudeSDKExecuteStep(client=client)

        ctx = ACEStepContext(sample="test", skillbook=None)
        result_ctx = step(ctx)

        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]
        assert r.cited_skill_ids == []


# ------------------------------------------------------------------ #
# ClaudeSDKExecuteStep — observability logging
# ------------------------------------------------------------------ #


class TestClaudeSDKObservability:
    def test_auto_instruments_anthropic_when_logfire_configured(self):
        mock_logfire = MagicMock()
        mock_ctx = MagicMock()
        mock_logfire.instrument_anthropic.return_value = mock_ctx

        with (
            patch("ace.observability.is_configured", return_value=True),
            patch.dict("sys.modules", {"logfire": mock_logfire}),
        ):
            step = ClaudeSDKExecuteStep(client=_make_mock_client())

        mock_logfire.instrument_anthropic.assert_called_once_with(step._client)
        mock_ctx.__enter__.assert_not_called()

    def test_logs_metrics(self, caplog):
        response = _make_response(input_tokens=200, output_tokens=100)
        client = _make_mock_client(response)
        step = ClaudeSDKExecuteStep(client=client)

        with caplog.at_level(logging.INFO, logger="ace.integrations.claude_sdk"):
            ctx = ACEStepContext(sample="test", skillbook=None)
            step(ctx)

        assert "ClaudeSDK:" in caplog.text
        assert "tokens=" in caplog.text

    def test_logs_error(self, caplog):
        client = _make_mock_client()
        client.messages.create.side_effect = RuntimeError("boom")
        step = ClaudeSDKExecuteStep(client=client)

        with caplog.at_level(logging.ERROR, logger="ace.integrations.claude_sdk"):
            ctx = ACEStepContext(sample="test", skillbook=None)
            step(ctx)

        assert "failed" in caplog.text.lower()

    def test_logfire_span_on_success(self):
        """When Logfire is configured, __call__ opens a span with attributes."""
        mock_span = MagicMock()
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)

        response = _make_response(input_tokens=50, output_tokens=25)
        client = _make_mock_client(response)
        step = ClaudeSDKExecuteStep(client=client)

        with (
            patch(
                "ace.integrations.claude_sdk._get_logfire", return_value=mock_logfire
            ),
        ):
            ctx = ACEStepContext(sample="What is 2+2?", skillbook=None)
            result_ctx = step(ctx)

        # Span was opened
        mock_logfire.span.assert_called_once()
        call_kwargs = mock_logfire.span.call_args
        assert call_kwargs[0][0] == "ClaudeSDKExecuteStep"
        assert call_kwargs[1]["model"] == "claude-sonnet-4-20250514"

        # Attributes were set on the span
        attr_calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["success"] is True
        assert attr_calls["input_tokens"] == 50
        assert attr_calls["output_tokens"] == 25
        assert attr_calls["total_tokens"] == 75
        assert "error" not in attr_calls

        # logfire.info was called with metrics
        mock_logfire.info.assert_called_once()
        info_kwargs = mock_logfire.info.call_args[1]
        assert info_kwargs["input_tokens"] == 50
        assert info_kwargs["output_tokens"] == 25

    def test_logfire_span_on_failure(self):
        """On API error, span captures error attribute and logfire.error is called."""
        mock_span = MagicMock()
        mock_logfire = MagicMock()
        mock_logfire.span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_logfire.span.return_value.__exit__ = MagicMock(return_value=False)

        client = _make_mock_client()
        client.messages.create.side_effect = RuntimeError("rate limited")
        step = ClaudeSDKExecuteStep(client=client)

        with (
            patch(
                "ace.integrations.claude_sdk._get_logfire", return_value=mock_logfire
            ),
        ):
            ctx = ACEStepContext(sample="test", skillbook=None)
            step(ctx)

        # Span captured the error attribute
        attr_calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
        assert attr_calls["success"] is False
        assert "rate limited" in attr_calls["error"]

        # logfire.error was called
        mock_logfire.error.assert_called_once()
        error_kwargs = mock_logfire.error.call_args[1]
        assert "rate limited" in error_kwargs["error"]

    def test_no_logfire_noop(self):
        """When Logfire is not configured, execution proceeds without spans."""
        client = _make_mock_client()
        step = ClaudeSDKExecuteStep(client=client)

        with patch("ace.integrations.claude_sdk._get_logfire", return_value=None):
            ctx = ACEStepContext(sample="test", skillbook=None)
            result_ctx = step(ctx)

        r: ClaudeSDKResult = result_ctx.trace  # type: ignore[assignment]
        assert r.success is True


# ------------------------------------------------------------------ #
# ClaudeSDKToTrace
# ------------------------------------------------------------------ #


class TestClaudeSDKToTrace:
    def test_requires_and_provides(self):
        step = ClaudeSDKToTrace()
        assert "trace" in step.requires
        assert "trace" in step.provides

    def test_success_trace(self):
        r = ClaudeSDKResult(
            task="What is 2+2?",
            success=True,
            output="4",
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_seconds=1.2,
            cited_skill_ids=["math-001"],
        )
        ctx = ACEStepContext(trace=r)
        result_ctx = ClaudeSDKToTrace()(ctx)

        trace = result_ctx.trace
        assert trace["question"] == "What is 2+2?"
        assert trace["answer"] == "4"
        assert trace["skill_ids"] == ["math-001"]
        assert "succeeded" in trace["reasoning"]
        assert "claude-sonnet-4-20250514" in trace["reasoning"]
        assert "100" in trace["reasoning"]  # input tokens
        assert "50" in trace["reasoning"]  # output tokens
        assert "1.2" in trace["reasoning"]  # latency
        assert "succeeded" in trace["feedback"]
        assert trace["ground_truth"] is None

    def test_failure_trace(self):
        r = ClaudeSDKResult(
            task="fail",
            success=False,
            error="API timeout",
            model="claude-sonnet-4-20250514",
            latency_seconds=30.0,
        )
        ctx = ACEStepContext(trace=r)
        result_ctx = ClaudeSDKToTrace()(ctx)

        trace = result_ctx.trace
        assert trace["question"] == "fail"
        assert trace["answer"] == ""
        assert "failed" in trace["reasoning"]
        assert "API timeout" in trace["reasoning"]
        assert "failed" in trace["feedback"]
        assert "API timeout" in trace["feedback"]

    def test_tool_calls_in_reasoning(self):
        r = ClaudeSDKResult(
            task="calc",
            success=True,
            output="4",
            model="claude-sonnet-4-20250514",
            tool_calls=[
                ToolCall(id="toolu_01", name="calculator", input={"expr": "2+2"}),
                ToolCall(id="toolu_02", name="formatter"),
            ],
        )
        ctx = ACEStepContext(trace=r)
        result_ctx = ClaudeSDKToTrace()(ctx)

        trace = result_ctx.trace
        assert "Tool calls (2)" in trace["reasoning"]
        assert "calculator" in trace["reasoning"]
        assert "formatter" in trace["reasoning"]
