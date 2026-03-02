"""Tests for ace_next/steps/opik.py — OpikStep pipeline step."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import (
    AgentOutput,
    ExtractedLearning,
    ReflectorOutput,
)
from ace_next.core.skillbook import Skillbook, UpdateBatch, UpdateOperation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skillbook(n: int = 3) -> SkillbookView:
    sb = Skillbook()
    for i in range(n):
        sb.add_skill(section="general", content=f"strategy {i}")
    return SkillbookView(sb)


def _base_ctx(**overrides) -> ACEStepContext:
    defaults = dict(
        sample=None,
        metadata={},
        skillbook=_make_skillbook(),
        epoch=2,
        total_epochs=5,
        step_index=4,
        global_sample_index=7,
    )
    defaults.update(overrides)
    return ACEStepContext(**defaults)


def _reflection(**raw_extra) -> ReflectorOutput:
    return ReflectorOutput(
        reasoning="looks good",
        correct_approach="do X",
        key_insight="insight A",
        extracted_learnings=[
            ExtractedLearning(learning="l1"),
            ExtractedLearning(learning="l2"),
        ],
        raw=raw_extra,
    )


def _claude_rr_reflection() -> ReflectorOutput:
    return _reflection(
        claude_rr=True,
        cost_info={
            "total_cost_usd": 0.042,
            "num_turns": 3,
            "duration_ms": 12500,
        },
    )


# ---------------------------------------------------------------------------
# _build_input
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildInput(unittest.TestCase):
    def _step(self):
        with patch("ace_next.steps.opik.OPIK_AVAILABLE", False):
            from ace_next.steps.opik import OpikStep

            return OpikStep()

    def test_dict_trace_with_question_and_context(self):
        step = self._step()
        ctx = _base_ctx(trace={"question": "What is 2+2?", "context": "math"})
        result = step._build_input(ctx)
        self.assertEqual(result["question"], "What is 2+2?")
        self.assertEqual(result["context"], "math")

    def test_dict_trace_without_context(self):
        step = self._step()
        ctx = _base_ctx(trace={"question": "hi"})
        result = step._build_input(ctx)
        self.assertEqual(result["question"], "hi")
        self.assertNotIn("context", result)

    def test_sample_with_question_attr(self):
        step = self._step()
        sample = MagicMock()
        sample.question = "sample Q"
        sample.context = "sample ctx"
        ctx = _base_ctx(sample=sample, trace=None)
        result = step._build_input(ctx)
        self.assertEqual(result["question"], "sample Q")
        self.assertEqual(result["context"], "sample ctx")

    def test_string_sample(self):
        step = self._step()
        ctx = _base_ctx(sample="do this task", trace=None)
        result = step._build_input(ctx)
        self.assertEqual(result["task"], "do this task")

    def test_empty_when_no_trace_no_sample(self):
        step = self._step()
        ctx = _base_ctx(trace=None, sample=None)
        result = step._build_input(ctx)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _build_output
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildOutput(unittest.TestCase):
    def _step(self):
        with patch("ace_next.steps.opik.OPIK_AVAILABLE", False):
            from ace_next.steps.opik import OpikStep

            return OpikStep()

    def test_agent_output(self):
        step = self._step()
        ao = AgentOutput(
            reasoning="because",
            final_answer="42",
            skill_ids=["s1", "s2"],
        )
        ctx = _base_ctx(agent_output=ao)
        result = step._build_output(ctx)
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["reasoning"], "because")
        self.assertEqual(result["skill_ids_cited"], ["s1", "s2"])

    def test_dict_trace_fallback(self):
        step = self._step()
        ctx = _base_ctx(trace={"answer": "fallback answer"}, agent_output=None)
        result = step._build_output(ctx)
        self.assertEqual(result["answer"], "fallback answer")

    def test_empty_when_no_output(self):
        step = self._step()
        ctx = _base_ctx(agent_output=None, trace=None)
        result = step._build_output(ctx)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _build_metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildMetadata(unittest.TestCase):
    def _step(self):
        with patch("ace_next.steps.opik.OPIK_AVAILABLE", False):
            from ace_next.steps.opik import OpikStep

            return OpikStep()

    def test_baseline_metadata(self):
        step = self._step()
        ctx = _base_ctx()
        meta = step._build_metadata(ctx)
        self.assertEqual(meta["epoch"], 2)
        self.assertEqual(meta["total_epochs"], 5)
        self.assertEqual(meta["step_index"], 4)
        self.assertEqual(meta["global_sample_index"], 7)
        self.assertEqual(meta["skill_count"], 3)
        self.assertNotIn("key_insight", meta)

    def test_with_reflection(self):
        step = self._step()
        ctx = _base_ctx(reflection=_reflection())
        meta = step._build_metadata(ctx)
        self.assertEqual(meta["key_insight"], "insight A")
        self.assertEqual(meta["learnings_count"], 2)
        self.assertNotIn("reflector_type", meta)

    def test_with_claude_rr_reflection(self):
        step = self._step()
        ctx = _base_ctx(reflection=_claude_rr_reflection())
        meta = step._build_metadata(ctx)
        self.assertEqual(meta["key_insight"], "insight A")
        self.assertEqual(meta["learnings_count"], 2)
        self.assertEqual(meta["reflector_type"], "claude_rr")
        self.assertAlmostEqual(meta["rr_cost_usd"], 0.042)
        self.assertEqual(meta["rr_turns"], 3)
        self.assertEqual(meta["rr_duration_ms"], 12500)

    def test_claude_rr_reflection_without_cost_info(self):
        step = self._step()
        ctx = _base_ctx(reflection=_reflection(claude_rr=True))
        meta = step._build_metadata(ctx)
        self.assertEqual(meta["reflector_type"], "claude_rr")
        self.assertNotIn("rr_cost_usd", meta)

    def test_with_skill_manager_output(self):
        step = self._step()
        ops = [
            UpdateOperation(type="ADD", section="general", skill_id="s1", content="c1"),
            UpdateOperation(type="ADD", section="general", skill_id="s2", content="c2"),
            UpdateOperation(
                type="UPDATE", section="general", skill_id="s3", content="c3"
            ),
        ]
        batch = UpdateBatch(reasoning="update", operations=ops)
        ctx = _base_ctx(skill_manager_output=batch)
        meta = step._build_metadata(ctx)
        self.assertEqual(meta["operations_count"], 3)
        self.assertEqual(meta["operation_types"], {"ADD": 2, "UPDATE": 1})


# ---------------------------------------------------------------------------
# _build_feedback_scores
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildFeedbackScores(unittest.TestCase):
    def _step(self):
        with patch("ace_next.steps.opik.OPIK_AVAILABLE", False):
            from ace_next.steps.opik import OpikStep

            return OpikStep()

    def test_correct_feedback(self):
        step = self._step()
        ctx = _base_ctx(trace={"feedback": "Correct! Well done."})
        scores = step._build_feedback_scores(ctx)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["name"], "accuracy")
        self.assertEqual(scores[0]["value"], 1.0)

    def test_incorrect_feedback(self):
        step = self._step()
        ctx = _base_ctx(trace={"feedback": "Wrong answer."})
        scores = step._build_feedback_scores(ctx)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["value"], 0.0)

    def test_no_feedback(self):
        step = self._step()
        ctx = _base_ctx(trace={"question": "q"})
        scores = step._build_feedback_scores(ctx)
        self.assertEqual(scores, [])

    def test_no_trace(self):
        step = self._step()
        ctx = _base_ctx(trace=None)
        scores = step._build_feedback_scores(ctx)
        self.assertEqual(scores, [])


# ---------------------------------------------------------------------------
# __call__ behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpikStepCall(unittest.TestCase):
    def test_noop_when_disabled(self):
        with patch("ace_next.steps.opik.OPIK_AVAILABLE", False):
            from ace_next.steps.opik import OpikStep

            step = OpikStep()
        self.assertFalse(step.enabled)
        ctx = _base_ctx()
        result = step(ctx)
        self.assertIs(result, ctx)

    def test_logs_trace_when_enabled(self):
        with (
            patch("ace_next.steps.opik.OPIK_AVAILABLE", True),
            patch("ace_next.steps.opik._opik_disabled", return_value=False),
            patch("ace_next.steps.opik._opik") as mock_opik_mod,
        ):
            mock_client = MagicMock()
            mock_opik_mod.Opik.return_value = mock_client
            mock_trace = MagicMock()
            mock_client.trace.return_value = mock_trace

            from ace_next.steps.opik import OpikStep

            step = OpikStep(project_name="test-project", tags=["test"])

        self.assertTrue(step.enabled)

        ctx = _base_ctx(
            trace={"question": "q", "feedback": "Correct"},
            agent_output=AgentOutput(reasoning="r", final_answer="a", skill_ids=["s1"]),
            reflection=_claude_rr_reflection(),
        )

        result = step(ctx)
        self.assertIs(result, ctx)

        mock_client.trace.assert_called_once()
        call_kwargs = mock_client.trace.call_args[1]
        self.assertEqual(call_kwargs["name"], "ace_pipeline")
        self.assertEqual(call_kwargs["input"]["question"], "q")
        self.assertEqual(call_kwargs["output"]["answer"], "a")
        self.assertEqual(call_kwargs["metadata"]["reflector_type"], "claude_rr")
        self.assertAlmostEqual(call_kwargs["metadata"]["rr_cost_usd"], 0.042)
        self.assertIn("epoch-2", call_kwargs["tags"])
        mock_trace.end.assert_called_once()

    def test_exception_in_log_trace_is_swallowed(self):
        with (
            patch("ace_next.steps.opik.OPIK_AVAILABLE", True),
            patch("ace_next.steps.opik._opik_disabled", return_value=False),
            patch("ace_next.steps.opik._opik") as mock_opik_mod,
        ):
            mock_client = MagicMock()
            mock_opik_mod.Opik.return_value = mock_client
            mock_client.trace.side_effect = RuntimeError("boom")

            from ace_next.steps.opik import OpikStep

            step = OpikStep()

        ctx = _base_ctx()
        result = step(ctx)
        self.assertIs(result, ctx)


if __name__ == "__main__":
    unittest.main()
