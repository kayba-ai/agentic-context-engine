"""Tests for RRStep — PydanticAI-based Recursive Reflector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ace.rr.config import RecursiveConfig
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.skillbook import Skillbook

from ace.rr import RRStep, RRConfig
from ace.rr.agent import RRDeps

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


def _mock_run_result(
    *,
    reasoning: str = "mock reasoning",
    key_insight: str = "mock insight",
    correct_approach: str = "mock approach",
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRRStep:
    """Test RRStep construction and StepProtocol."""

    def test_step_protocol_attributes(self):
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        assert "trace" in rr.requires
        assert "skillbook" in rr.requires
        assert "reflections" in rr.provides
        assert "reflection" not in rr.provides

    def test_call_produces_reflection_on_context(self):
        """RRStep.__call__ populates ctx.reflections."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        mock_result = _mock_run_result(key_insight="step test")

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
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
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result()

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            result_ctx = rr(_make_ctx())

        output = result_ctx.reflections[0]
        assert "rr_trace" in output.raw
        assert output.raw["rr_trace"]["timed_out"] is False
        assert "usage" in output.raw
        assert output.raw["max_llm_calls_scope"] == "main_agent_session"

    def test_timeout_produces_output(self):
        """UsageLimitExceeded produces a timeout ReflectorOutput."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        rr = RRStep(
            "test-model",
            config=RRConfig(max_llm_calls=5, enable_subagent=False),
        )

        with patch.object(
            rr._agent,
            "run_sync",
            side_effect=UsageLimitExceeded("limit reached"),
        ):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        output = result_ctx.reflections[0]
        assert isinstance(output, ReflectorOutput)
        assert "usage limit" in output.reasoning.lower()
        assert output.raw.get("timeout") is True
        assert output.raw["max_llm_calls_scope"] == "main_agent_session"

    def test_timeout_with_ground_truth_correct(self):
        """Timeout correctly detects correct answer."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        with patch.object(
            rr._agent,
            "run_sync",
            side_effect=UsageLimitExceeded("limit"),
        ):
            ctx = _make_ctx(
                question="What is 2+2?",
                answer="4",
                ground_truth="4",
            )
            # reflect() via __call__ doesn't pass agent_output, so is_correct is False
            # Test directly via reflect() with agent_output
            output = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
            )

        assert isinstance(output, ReflectorOutput)
        assert "correct" in output.reasoning.lower()

    def test_error_produces_safe_output(self):
        """General exception produces a safe fallback output."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))

        with patch.object(
            rr._agent,
            "run_sync",
            side_effect=RuntimeError("unexpected error"),
        ):
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

        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        assert isinstance(rr, ReflectorLike)

    def test_reflect_method(self):
        """reflect() delegates to the PydanticAI agent."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = _mock_run_result(key_insight="reflected")

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            output = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
                feedback="Correct!",
            )

        assert isinstance(output, ReflectorOutput)
        assert output.key_insight == "reflected"


@pytest.mark.unit
class TestRRBatchReflection:
    """Test generic batch reflection paths."""

    def _make_batch_rr(self):
        """Create an RRStep with the orchestrator agent eagerly initialized."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        rr._ensure_batch_agents()
        return rr

    def test_batch_splits_into_per_task_outputs(self):
        """Batch with per-item results in raw produces per-item ReflectorOutputs."""
        rr = self._make_batch_rr()

        # Mock a batch result with per-item data in raw
        output = ReflectorOutput(
            reasoning="batch analysis",
            key_insight="batch insight",
            correct_approach="approach",
            raw={
                "items": [
                    {
                        "reasoning": "task 0 analysis",
                        "key_insight": "t0 insight",
                        "extracted_learnings": [],
                    },
                    {
                        "reasoning": "task 1 analysis",
                        "key_insight": "t1 insight",
                        "extracted_learnings": [
                            {"learning": "l1", "atomicity_score": 0.8, "evidence": "e1"}
                        ],
                    },
                ],
            },
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 200
        usage.response_tokens = 100
        usage.total_tokens = 300
        usage.requests = 5
        mock_result.usage.return_value = usage

        batch_trace = {
            "tasks": [
                {"item_id": "t0", "trace": [{"role": "user", "content": "hello"}]},
                {"item_id": "t1", "trace": [{"role": "user", "content": "world"}]},
            ]
        }
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._orchestrator_agent, "run_sync", return_value=mock_result):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].reasoning == "task 0 analysis"
        assert result_ctx.reflections[1].key_insight == "t1 insight"
        assert len(result_ctx.reflections[1].extracted_learnings) == 1
        assert result_ctx.reflections[0].raw["item_id"] == "t0"

    def test_batch_missing_per_item_results_fails_loudly(self):
        """Direct batch output without per-item results should raise."""
        rr = self._make_batch_rr()

        output = ReflectorOutput(
            reasoning="single batch analysis",
            key_insight="single insight",
            correct_approach="approach",
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 100
        usage.response_tokens = 50
        usage.total_tokens = 150
        usage.requests = 3
        mock_result.usage.return_value = usage

        batch_trace = {
            "tasks": [
                {"task_id": "t0", "trace": []},
                {"task_id": "t1", "trace": []},
            ]
        }
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._orchestrator_agent, "run_sync", return_value=mock_result):
            with pytest.raises(RuntimeError, match="raw\\['items'\\]"):
                rr(ctx)

    def test_raw_list_batch_is_supported_without_preprocessing(self):
        """A raw list of trace items should route through generic batch mode."""
        rr = self._make_batch_rr()

        output = ReflectorOutput(
            reasoning="raw list analysis",
            key_insight="raw list insight",
            correct_approach="approach",
            raw={
                "items": [
                    {
                        "reasoning": "item 0",
                        "key_insight": "i0",
                        "extracted_learnings": [],
                    },
                    {
                        "reasoning": "item 1",
                        "key_insight": "i1",
                        "extracted_learnings": [],
                    },
                ]
            },
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 200
        usage.response_tokens = 100
        usage.total_tokens = 300
        usage.requests = 5
        mock_result.usage.return_value = usage

        batch_trace = [
            {"item_id": "i0", "messages": [{"role": "user", "content": "hello"}]},
            {"item_id": "i1", "messages": [{"role": "user", "content": "world"}]},
        ]
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._orchestrator_agent, "run_sync", return_value=mock_result):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].raw["item_id"] == "i0"
        assert result_ctx.reflections[1].raw["item_id"] == "i1"

    def test_combined_steps_batch_routes_through_generic_batch_mode(self):
        """Legacy combined-step batches should batch without inner normalization."""
        rr = self._make_batch_rr()

        output = ReflectorOutput(
            reasoning="normalized batch analysis",
            key_insight="normalized insight",
            correct_approach="approach",
            raw={
                "items": [
                    {
                        "reasoning": "task 0 analysis",
                        "key_insight": "t0 insight",
                        "extracted_learnings": [],
                    },
                    {
                        "reasoning": "task 1 analysis",
                        "key_insight": "t1 insight",
                        "extracted_learnings": [],
                    },
                ],
            },
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 200
        usage.response_tokens = 100
        usage.total_tokens = 300
        usage.requests = 5
        mock_result.usage.return_value = usage

        combined_trace = {
            "question": "Analyze 2 agent execution traces",
            "steps": [
                {
                    "role": "conversation",
                    "id": "task_0",
                    "content": {
                        "question": "Where is my order?",
                        "feedback": "reward=1.0",
                        "steps": [{"role": "user", "content": "order status"}],
                    },
                },
                {
                    "role": "conversation",
                    "id": "task_1",
                    "content": {
                        "question": "I need a refund",
                        "feedback": "reward=0.0",
                        "steps": [{"role": "user", "content": "refund help"}],
                    },
                },
            ],
        }
        ctx = ACEStepContext(trace=combined_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._orchestrator_agent, "run_sync", return_value=mock_result):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].raw["item_id"] == "task_0"
        assert result_ctx.reflections[1].raw["item_id"] == "task_1"

    def test_batch_sandbox_injects_generic_helper_variables(self):
        """Batch sandbox should expose generic helper data and helper registry tools."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        batch_trace = {
            "tasks": [
                {
                    "task_id": "t0",
                    "question": "Where is my order?",
                    "feedback": "reward=1.0",
                    "trace": [{"role": "user", "content": "order status"}],
                },
                {
                    "task_id": "t1",
                    "question": "I need a refund",
                    "feedback": "reward=0.0",
                    "trace": [{"role": "user", "content": "refund help"}],
                },
            ]
        }

        sandbox = rr._create_sandbox(
            trace_obj=None,
            traces=batch_trace,
            skillbook=SkillbookView(Skillbook()),
        )

        assert sandbox.namespace["batch_items"] == batch_trace["tasks"]
        assert sandbox.namespace["item_ids"] == ["t0", "t1"]
        assert sandbox.namespace["item_id_to_index"] == {"t0": 0, "t1": 1}
        assert "survey_items" in sandbox.namespace
        assert sandbox.namespace["survey_items"][0].startswith("Inspect batch_items[0]")
        assert sandbox.namespace["item_preview_by_id"]["t0"]["question_preview"]
        assert callable(sandbox.namespace["register_helper"])
        assert callable(sandbox.namespace["run_helper"])

    def test_batch_prompt_mentions_helper_registration_and_survey_items(self):
        """Batch prompt should surface helper registration and generic survey items."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        batch_trace = {
            "tasks": [
                {
                    "task_id": "t0",
                    "question": "Where is my order?",
                    "feedback": "reward=1.0",
                    "trace": [{"role": "user", "content": "order status"}],
                },
                {
                    "task_id": "t1",
                    "question": "I need a refund",
                    "feedback": "reward=0.0",
                    "trace": [{"role": "user", "content": "refund help"}],
                },
                {
                    "task_id": "t2",
                    "question": "Can I exchange this?",
                    "feedback": "reward=1.0",
                    "trace": [{"role": "user", "content": "exchange help"}],
                },
            ]
        }

        prompt = rr._build_initial_prompt(
            traces=batch_trace,
            skillbook=SkillbookView(Skillbook()),
            trace_obj=None,
        )

        assert "register_helper" in prompt
        assert "helper_registry" in prompt
        assert "survey_items" in prompt
        assert 'raw["items"]' in prompt
        assert "Inspect batch_items[0] (item_id='t0')" in prompt

    def test_batch_prompt_uses_nested_trace_messages_for_previews(self):
        """Wrapped batch items should surface nested trace messages in previews."""
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        batch_trace = {
            "tasks": [
                {
                    "task_id": "task_0",
                    "question": "Please cancel my reservation.",
                    "feedback": "Task PASSED (reward=1.0)",
                    "trace": {
                        "question": "Please cancel my reservation.",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "Hi! How can I help you today?",
                            },
                            {
                                "role": "user",
                                "content": "Please cancel my reservation.",
                            },
                        ],
                        "reasoning": "Ask for identifiers and confirm refund policy.",
                        "answer": "I can help with that.",
                    },
                }
            ]
        }

        sandbox = rr._create_sandbox(
            trace_obj=None,
            traces=batch_trace,
            skillbook=SkillbookView(Skillbook()),
        )
        preview = sandbox.namespace["item_preview_by_id"]["task_0"]

        assert preview["message_count"] == 2
        assert "Hi! How can I help you today?" in preview["first_message_preview"]

        prompt = rr._build_initial_prompt(
            traces=batch_trace,
            skillbook=SkillbookView(Skillbook()),
            trace_obj=None,
        )

        assert "| `task_0` | 2 messages |" in prompt
        assert "task_0: PASS, 2 messages" in prompt
