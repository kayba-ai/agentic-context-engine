"""Tests for RRStep — PydanticAI-based Recursive Reflector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ace.steps.rr.config import RecursiveConfig
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.skillbook import Skillbook

from ace.steps.rr import RRStep, RRConfig
from ace.steps.rr.tools import RRDeps

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


def _mock_compaction_result(
    *,
    reasoning: str = "mock reasoning",
    key_insight: str = "mock insight",
    correct_approach: str = "mock approach",
) -> tuple[ReflectorOutput, RRDeps]:
    """Create a mock return value for _run_with_compaction."""
    output = ReflectorOutput(
        reasoning=reasoning,
        key_insight=key_insight,
        correct_approach=correct_approach,
        raw={
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "requests": 3},
            "rr_trace": {"total_iterations": 2, "subagent_calls": [], "timed_out": False, "compactions": 0, "depth": 0},
        },
    )
    deps = MagicMock(spec=RRDeps)
    deps.iteration = 2
    return output, deps


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

        output, deps = _mock_compaction_result(key_insight="step test")

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
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
        output, deps = _mock_compaction_result()

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is False
        assert "usage" in result.raw

    def test_timeout_produces_output(self):
        """Budget exhaustion produces a timeout ReflectorOutput."""
        rr = RRStep(
            "test-model",
            config=RRConfig(),
        )

        timeout_output = ReflectorOutput(
            reasoning="Analysis reached budget limit.",
            error_identification="budget_exhausted",
            key_insight="Session reached budget limit before completing",
            correct_approach="Consider increasing budget or simplifying the analysis",
            raw={"timeout": True, "rr_trace": {"total_iterations": 0, "subagent_calls": [], "timed_out": True, "compactions": 0, "depth": 0}},
        )
        deps = MagicMock(spec=RRDeps)
        deps.iteration = 0


        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(timeout_output, deps)):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        output = result_ctx.reflections[0]
        assert isinstance(output, ReflectorOutput)
        assert "budget limit" in output.reasoning.lower()
        assert output.raw.get("timeout") is True

    def test_timeout_with_ground_truth_correct(self):
        """Timeout correctly detects correct answer."""
        rr = RRStep("test-model", config=RRConfig())

        timeout_output = ReflectorOutput(
            reasoning="budget limit",
            key_insight="",
            correct_approach="",
            raw={"timeout": True, "rr_trace": {"total_iterations": 0, "subagent_calls": [], "timed_out": True, "compactions": 0, "depth": 0}},
        )
        deps = MagicMock(spec=RRDeps)
        deps.iteration = 0


        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(timeout_output, deps)):
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

        error_output = ReflectorOutput(
            reasoning="Recursive analysis failed: unexpected error",
            correct_approach="",
            key_insight="",
            raw={"error": "unexpected error"},
        )
        deps = MagicMock(spec=RRDeps)

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(error_output, deps)):
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
        output, deps = _mock_compaction_result(key_insight="reflected")

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
            result = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
                feedback="Correct!",
            )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "reflected"


@pytest.mark.unit
class TestRRBatchReflection:
    """Test generic batch reflection paths."""

    def test_batch_splits_into_per_task_outputs(self):
        """Batch with per-item results in raw produces per-item ReflectorOutputs."""
        rr = RRStep("test-model", config=RRConfig())

        output = ReflectorOutput(
            reasoning="batch analysis",
            key_insight="batch insight",
            correct_approach="approach",
            raw={
                "items": [
                    {
                        "reasoning": "task 0 analysis",
                        "key_insight": "t0 insight",
                    },
                    {
                        "reasoning": "task 1 analysis",
                        "key_insight": "t1 insight",
                    },
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "requests": 5},
                "rr_trace": {"total_iterations": 3, "subagent_calls": [], "timed_out": False, "compactions": 0, "depth": 0},
            },
        )
        deps = MagicMock(spec=RRDeps)
        deps.iteration = 3


        batch_trace = {
            "tasks": [
                {"item_id": "t0", "trace": [{"role": "user", "content": "hello"}]},
                {"item_id": "t1", "trace": [{"role": "user", "content": "world"}]},
            ]
        }
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].reasoning == "task 0 analysis"
        assert result_ctx.reflections[1].key_insight == "t1 insight"
        assert result_ctx.reflections[0].raw["item_id"] == "t0"

    def test_batch_missing_per_item_results_fails_loudly(self):
        """Direct batch output without per-item results should raise."""
        rr = RRStep("test-model", config=RRConfig())

        output = ReflectorOutput(
            reasoning="single batch analysis",
            key_insight="single insight",
            correct_approach="approach",
        )
        deps = MagicMock(spec=RRDeps)

        batch_trace = {
            "tasks": [
                {"task_id": "t0", "trace": []},
                {"task_id": "t1", "trace": []},
            ]
        }
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
            # No raw["items"] → single reflection, not split
            result_ctx = rr(ctx)
            assert len(result_ctx.reflections) == 1

    def test_raw_list_batch_is_supported_without_preprocessing(self):
        """A raw list of trace items should route through batch splitting."""
        rr = RRStep("test-model", config=RRConfig())

        output = ReflectorOutput(
            reasoning="raw list analysis",
            key_insight="raw list insight",
            correct_approach="approach",
            raw={
                "items": [
                    {
                        "reasoning": "item 0",
                        "key_insight": "i0",
                    },
                    {
                        "reasoning": "item 1",
                        "key_insight": "i1",
                    },
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "requests": 5},
                "rr_trace": {"total_iterations": 3, "subagent_calls": [], "timed_out": False, "compactions": 0, "depth": 0},
            },
        )
        deps = MagicMock(spec=RRDeps)

        batch_trace = [
            {"item_id": "i0", "messages": [{"role": "user", "content": "hello"}]},
            {"item_id": "i1", "messages": [{"role": "user", "content": "world"}]},
        ]
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].raw["item_id"] == "i0"
        assert result_ctx.reflections[1].raw["item_id"] == "i1"

    def test_combined_steps_batch_routes_through_generic_batch_mode(self):
        """Legacy combined-step batches should batch without inner normalization."""
        rr = RRStep("test-model", config=RRConfig())

        output = ReflectorOutput(
            reasoning="normalized batch analysis",
            key_insight="normalized insight",
            correct_approach="approach",
            raw={
                "items": [
                    {
                        "reasoning": "task 0 analysis",
                        "key_insight": "t0 insight",
                    },
                    {
                        "reasoning": "task 1 analysis",
                        "key_insight": "t1 insight",
                    },
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "requests": 5},
                "rr_trace": {"total_iterations": 3, "subagent_calls": [], "timed_out": False, "compactions": 0, "depth": 0},
            },
        )
        deps = MagicMock(spec=RRDeps)

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

        with patch.object(rr, "_run_with_compaction", new_callable=AsyncMock, return_value=(output, deps)):
            result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        assert result_ctx.reflections[0].raw["item_id"] == "task_0"
        assert result_ctx.reflections[1].raw["item_id"] == "task_1"

    def test_batch_sandbox_injects_generic_helper_variables(self):
        """Batch sandbox should expose generic helper data and helper registry tools."""
        rr = RRStep("test-model", config=RRConfig())
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
        rr = RRStep("test-model", config=RRConfig())
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
        rr = RRStep("test-model", config=RRConfig())
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
