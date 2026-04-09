"""Stress tests for RR components.

Tests sandbox behavior and the PydanticAI-based RRStep entry points.
"""

import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from ace.implementations.rr.config import RecursiveConfig
from ace.core.sandbox import TraceSandbox, create_readonly_sandbox

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.skillbook import Skillbook
from ace.steps.rr_step import RRConfig, RRStep
from ace.implementations.rr.tools import RRDeps

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


_RUN_SYNC = "ace.core.recursive_agent.run_agent_sync"


def _mock_compaction_result(
    *,
    reasoning: str = "done",
    key_insight: str = "insight",
    correct_approach: str = "approach",
    timed_out: bool = False,
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
        "timed_out": timed_out,
    }

    return output, metadata


# =========================================================================
# 1. RRStep lifecycle (PydanticAI-based)
# =========================================================================


@pytest.mark.unit
class TestLoopLifecycle:
    def test_successful_reflection(self):
        """Happy path: PydanticAI agent produces valid ReflectorOutput."""
        rr = RRStep("test-model", config=RRConfig())
        output, metadata = _mock_compaction_result(key_insight="insight")

        with patch(_RUN_SYNC, return_value=(output, metadata)):
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
        """Budget exhaustion produces timeout output."""
        rr = RRStep(
            "test-model",
            config=RRConfig(max_requests=3),
        )

        output, metadata = _mock_compaction_result(
            reasoning="Analysis reached budget limit.",
            timed_out=True,
        )

        with patch(_RUN_SYNC, return_value=(output, metadata)):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert "budget limit" in result_ctx.reflections[0].reasoning.lower()

    def test_budget_field_in_config(self):
        """max_tokens and max_requests config fields exist."""
        rr = RRStep(
            "test-model",
            config=RRConfig(max_tokens=100_000, max_requests=42),
        )
        assert rr.config.max_tokens == 100_000
        assert rr.config.max_requests == 42

    def test_config_build_usage_limits(self):
        """build_usage_limits() produces correct UsageLimits."""
        cfg = RecursiveConfig(max_tokens=500_000, max_requests=50, context_window=128_000)
        limits = cfg.build_usage_limits()
        assert limits.total_tokens_limit == 500_000
        assert limits.request_limit == 50

    def test_config_build_usage_limits_with_remaining(self):
        """build_usage_limits() uses remaining_tokens when provided."""
        cfg = RecursiveConfig(max_tokens=500_000, max_requests=50)
        limits = cfg.build_usage_limits(remaining_tokens=100_000)
        assert limits.total_tokens_limit == 100_000
        assert limits.request_limit == 50


    def test_rr_trace_metadata_on_success(self):
        """Successful reflection populates rr_trace metadata."""
        rr = RRStep("test-model", config=RRConfig())
        output, metadata = _mock_compaction_result()

        with patch(_RUN_SYNC, return_value=(output, metadata)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is False
        assert isinstance(result.raw["rr_trace"]["subagent_calls"], list)

    def test_rr_trace_metadata_on_timeout(self):
        """Budget exhaustion produces rr_trace with timed_out=True."""
        from ace.core.recursive_agent import BudgetExhausted

        rr = RRStep("test-model", config=RRConfig())

        with patch(_RUN_SYNC, side_effect=BudgetExhausted(compaction_count=1)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is True


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

    def test_registered_helpers_persist_and_run(self):
        """Registered helpers should persist across execute_code calls."""
        sandbox = TraceSandbox(trace=None)
        sandbox.inject("traces", {"values": [1, 2, 3]})
        result = sandbox.execute(
            """
register_helper(
    "sum_values",
    "def sum_values():\\n    return sum(traces['values'])\\n",
    "Return the sum of traces['values']",
)
print(run_helper("sum_values"))
            """.strip(),
            timeout=5.0,
        )

        assert result.success
        assert "6" in result.stdout
        assert sandbox.namespace["list_helpers"]()[0]["name"] == "sum_values"

    def test_registered_helpers_are_rehydrated_in_snapshots(self):
        """Sub-agent snapshots should recreate helpers against the child namespace."""
        parent = TraceSandbox(trace=None)
        parent.inject("traces", {"values": [1, 2, 3]})
        parent.execute(
            """
register_helper(
    "sum_values",
    "def sum_values():\\n    return sum(traces['values'])\\n",
    "Return the sum of traces['values']",
)
            """.strip(),
            timeout=5.0,
        )

        child = create_readonly_sandbox(parent)
        child.namespace["traces"]["values"].append(4)
        result = child.execute('print(run_helper("sum_values"))', timeout=5.0)

        assert result.success
        assert "10" in result.stdout
        assert parent.namespace["traces"]["values"] == [1, 2, 3]

    def test_batch_accessors_handle_nested_task_payloads(self):
        """Batch helpers should expose a stable view over nested task items."""
        sandbox = TraceSandbox(trace=None)
        batch_items = [
            {
                "task_id": "task_0",
                "question": "Where is my order?",
                "feedback": "Task FAILED (reward=0.0)",
                "trace": {
                    "messages": [
                        {"role": "assistant", "content": "Hi!"},
                        {"role": "user", "content": "Where is my order?"},
                    ]
                },
            }
        ]
        sandbox.inject("batch_items", batch_items)
        sandbox.inject("item_ids", ["task_0"])

        result = sandbox.execute(
            """
item = get_batch_item(0)
print(get_item_id(0))
print(get_item_question(item))
print(get_item_feedback(item))
messages = get_item_messages(item)
print(len(messages))
print(get_message_text(messages[1]))
print(preview_item(0)["payload_type"])
            """.strip(),
            timeout=5.0,
        )

        assert result.success
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "task_0"
        assert lines[1] == "Where is my order?"
        assert lines[2] == "Task FAILED (reward=0.0)"
        assert lines[3] == "2"
        assert lines[4] == "Where is my order?"
        assert lines[5] == "dict"


# =========================================================================
# 3. Entry points (PydanticAI-based)
# =========================================================================


@pytest.mark.unit
class TestEntryPoints:
    def test_call_produces_reflection(self):
        """__call__() produces a ReflectorOutput on the context."""
        rr = RRStep("test-model", config=RRConfig())
        output, metadata = _mock_compaction_result(key_insight="insight")

        traces = {
            "question": "q",
            "steps": [
                {"role": "agent", "reasoning": "r", "answer": "4", "skill_ids": []}
            ],
        }
        ctx = ACEStepContext(trace=traces, skillbook=SkillbookView(Skillbook()))

        with patch(_RUN_SYNC, return_value=(output, metadata)):
            result_ctx = rr(ctx)

        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert result_ctx.reflections[0].key_insight == "insight"

    def test_reflect_method_works(self):
        """reflect() works as ReflectorLike entry point."""
        rr = RRStep("test-model", config=RRConfig())
        output, metadata = _mock_compaction_result(key_insight="reflected")

        with patch(_RUN_SYNC, return_value=(output, metadata)):
            result = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
            )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "reflected"


# =========================================================================
# 4. New architecture tests
# =========================================================================


@pytest.mark.unit
class TestRecurseToolRegistration:
    def test_recurse_tool_registered_at_non_leaf_depth(self):
        """Agent at depth 0 with max_depth=2 should have recurse tool."""
        from ace.core.recursive_agent import AgenticConfig, RecursiveAgent

        ra = RecursiveAgent(
            "test-model", output_type=ReflectorOutput, system_prompt="test",
            config=AgenticConfig(max_depth=2),
        )
        agent = ra._create_agent(depth=0)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "recurse" in tool_names

    def test_recurse_tool_not_registered_at_max_depth(self):
        """Agent at max_depth should NOT have recurse tool."""
        from ace.core.recursive_agent import AgenticConfig, RecursiveAgent

        ra = RecursiveAgent(
            "test-model", output_type=ReflectorOutput, system_prompt="test",
            config=AgenticConfig(max_depth=2),
        )
        agent = ra._create_agent(depth=2)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "recurse" not in tool_names

    def test_recurse_tool_not_registered_at_depth_zero_max_zero(self):
        """Agent at depth=0, max_depth=0 should NOT have recurse tool."""
        from ace.core.recursive_agent import AgenticConfig, RecursiveAgent

        ra = RecursiveAgent(
            "test-model", output_type=ReflectorOutput, system_prompt="test",
            config=AgenticConfig(max_depth=0),
        )
        agent = ra._create_agent(depth=0)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "recurse" not in tool_names


@pytest.mark.unit
class TestMicrocompaction:
    def test_microcompact_clears_old_tool_results(self):
        """_microcompact should clear old tool results, keeping recent ones."""
        rr = RRStep("test-model", config=RRConfig())

        # Build a message list with 5 tool results
        messages = []
        for i in range(5):
            messages.append(ModelRequest(parts=[
                ToolReturnPart(
                    tool_name="execute_code",
                    content=f"result {i}",
                    tool_call_id=f"call_{i}",
                ),
            ]))

        from ace.core.recursive_agent import microcompact
        compacted = microcompact(messages, keep_recent=2, tool_names=("execute_code",))

        # Should NOT be the same object (changes were made)
        assert compacted is not messages

        # First 3 should be cleared, last 2 kept
        for i in range(3):
            assert "[cleared" in compacted[i].parts[0].content
        assert compacted[3].parts[0].content == "result 3"
        assert compacted[4].parts[0].content == "result 4"

    def test_microcompact_returns_same_when_nothing_to_clear(self):
        """_microcompact returns identity when keep_recent >= total tool results."""
        rr = RRStep("test-model", config=RRConfig())

        messages = [
            ModelRequest(parts=[
                ToolReturnPart(tool_name="execute_code", content="result 0", tool_call_id="call_0"),
            ]),
        ]

        from ace.core.recursive_agent import microcompact
        result = microcompact(messages, keep_recent=3, tool_names=("execute_code",))
        assert result is messages  # identity = no change

    def test_microcompact_ignores_non_tool_messages(self):
        """_microcompact should not touch model response messages."""
        rr = RRStep("test-model", config=RRConfig())

        messages = [
            ModelResponse(parts=[TextPart(content="thinking...")]),
            ModelRequest(parts=[
                ToolReturnPart(tool_name="execute_code", content="old result", tool_call_id="call_0"),
            ]),
            ModelResponse(parts=[TextPart(content="more thinking...")]),
            ModelRequest(parts=[
                ToolReturnPart(tool_name="execute_code", content="new result", tool_call_id="call_1"),
            ]),
        ]

        from ace.core.recursive_agent import microcompact
        compacted = microcompact(messages, keep_recent=1, tool_names=("execute_code",))
        assert compacted is not messages
        # First tool result cleared, second kept
        assert "[cleared" in compacted[1].parts[0].content
        assert compacted[3].parts[0].content == "new result"
        # Model responses untouched
        assert compacted[0].parts[0].content == "thinking..."
        assert compacted[2].parts[0].content == "more thinking..."


@pytest.mark.unit
class TestBudgetExhausted:
    def test_is_budget_exhausted_tokens(self):
        """is_budget_exhausted detects total token limit."""
        from ace.core.recursive_agent import is_budget_exhausted
        limits = UsageLimits(total_tokens_limit=1000, request_limit=50)
        usage = MagicMock()
        usage.total_tokens = 1000
        usage.requests = 5
        assert is_budget_exhausted(limits, usage) is True

    def test_is_budget_exhausted_requests(self):
        """is_budget_exhausted detects request limit."""
        from ace.core.recursive_agent import is_budget_exhausted
        limits = UsageLimits(total_tokens_limit=100_000, request_limit=10)
        usage = MagicMock()
        usage.total_tokens = 500
        usage.requests = 10
        assert is_budget_exhausted(limits, usage) is True

    def test_is_budget_not_exhausted(self):
        """is_budget_exhausted returns False when under budget."""
        from ace.core.recursive_agent import is_budget_exhausted
        limits = UsageLimits(total_tokens_limit=100_000, request_limit=50)
        usage = MagicMock()
        usage.total_tokens = 500
        usage.requests = 5
        assert is_budget_exhausted(limits, usage) is False
