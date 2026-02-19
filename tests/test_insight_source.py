"""Unit tests for InsightSource and TraceReference dataclasses."""

import unittest

import pytest

from ace.insight_source import (
    InsightSource,
    TraceReference,
    build_insight_source,
)


@pytest.mark.unit
class TestTraceReference(unittest.TestCase):
    """Test TraceReference serialization."""

    def test_structured_roundtrip(self):
        """Test roundtrip with structured step indices."""
        ref = TraceReference(
            step_indices=[0, 2, 5],
            action_types=["tool_call:search", "tool_call:submit"],
        )
        d = ref.to_dict()
        restored = TraceReference.from_dict(d)

        self.assertEqual(restored.step_indices, [0, 2, 5])
        self.assertEqual(
            restored.action_types, ["tool_call:search", "tool_call:submit"]
        )
        self.assertIsNone(restored.text_excerpt)
        self.assertIsNone(restored.excerpt_location)

    def test_text_fallback_roundtrip(self):
        """Test roundtrip with text excerpt fallback."""
        ref = TraceReference(
            text_excerpt="The agent failed to check the boundary condition",
            excerpt_location="reasoning",
        )
        d = ref.to_dict()
        restored = TraceReference.from_dict(d)

        self.assertEqual(
            restored.text_excerpt, "The agent failed to check the boundary condition"
        )
        self.assertEqual(restored.excerpt_location, "reasoning")
        self.assertIsNone(restored.step_indices)

    def test_compact_serialization_omits_none(self):
        """Test that to_dict omits None fields."""
        ref = TraceReference(step_indices=[1, 2])
        d = ref.to_dict()

        self.assertIn("step_indices", d)
        self.assertNotIn("action_types", d)
        self.assertNotIn("text_excerpt", d)
        self.assertNotIn("excerpt_location", d)

    def test_empty_reference(self):
        """Test empty TraceReference roundtrip."""
        ref = TraceReference()
        d = ref.to_dict()
        self.assertEqual(d, {})
        restored = TraceReference.from_dict(d)
        self.assertIsNone(restored.step_indices)


@pytest.mark.unit
class TestInsightSource(unittest.TestCase):
    """Test InsightSource serialization."""

    def test_full_roundtrip(self):
        """Test roundtrip with all fields populated."""
        source = InsightSource(
            sample_question="What is 2+2?",
            epoch=1,
            step=3,
            trace_refs=[
                TraceReference(step_indices=[0, 1]),
                TraceReference(text_excerpt="hint", excerpt_location="feedback"),
            ],
            learning_text="Always double-check arithmetic",
        )
        d = source.to_dict()
        restored = InsightSource.from_dict(d)

        self.assertEqual(restored.sample_question, "What is 2+2?")
        self.assertEqual(restored.epoch, 1)
        self.assertEqual(restored.step, 3)
        self.assertEqual(len(restored.trace_refs), 2)
        self.assertEqual(restored.trace_refs[0].step_indices, [0, 1])
        self.assertEqual(restored.trace_refs[1].text_excerpt, "hint")
        self.assertEqual(restored.learning_text, "Always double-check arithmetic")

    def test_minimal_roundtrip(self):
        """Test roundtrip with only required fields."""
        source = InsightSource(
            sample_question="Test",
            epoch=0,
            step=0,
        )
        d = source.to_dict()
        restored = InsightSource.from_dict(d)

        self.assertEqual(restored.sample_question, "Test")
        self.assertIsNone(restored.learning_text)
        self.assertIsNone(restored.sample_id)
        self.assertEqual(restored.trace_refs, [])

    def test_error_identification_roundtrip(self):
        """Test roundtrip with error_identification populated."""
        source = InsightSource(
            sample_question="What is 2+2?",
            epoch=1,
            step=3,
            error_identification="The agent multiplied instead of adding",
        )
        d = source.to_dict()
        self.assertIn("error_identification", d)
        self.assertEqual(
            d["error_identification"], "The agent multiplied instead of adding"
        )

        restored = InsightSource.from_dict(d)
        self.assertEqual(
            restored.error_identification,
            "The agent multiplied instead of adding",
        )

    def test_error_identification_none_by_default(self):
        """Test that error_identification is None when not provided."""
        source = InsightSource(
            sample_question="Q",
            epoch=0,
            step=0,
        )
        self.assertIsNone(source.error_identification)
        d = source.to_dict()
        self.assertNotIn("error_identification", d)

        restored = InsightSource.from_dict(d)
        self.assertIsNone(restored.error_identification)

    def test_compact_serialization_omits_none(self):
        """Test that to_dict omits None/empty optional fields."""
        source = InsightSource(
            sample_question="Q",
            epoch=1,
            step=1,
        )
        d = source.to_dict()

        self.assertNotIn("trace_refs", d)
        self.assertNotIn("learning_text", d)
        self.assertNotIn("error_identification", d)
        self.assertNotIn("sample_id", d)
        # Required fields are always present
        self.assertIn("sample_question", d)
        self.assertIn("epoch", d)
        self.assertIn("step", d)


@pytest.mark.unit
class TestBuildInsightSource(unittest.TestCase):
    """Test build_insight_source attaches metadata to operations."""

    def _make_agent_output(self, reasoning="Step 1: think", trace_context=None):
        class FakeAgentOutput:
            pass

        ao = FakeAgentOutput()
        ao.reasoning = reasoning
        ao.trace_context = trace_context
        return ao

    def _make_reflection(self, error_id="", key_insight="insight", learnings=None):
        class FakeLearning:
            def __init__(self, text):
                self.learning = text

        class FakeReflection:
            pass

        r = FakeReflection()
        r.error_identification = error_id
        r.key_insight = key_insight
        r.extracted_learnings = [FakeLearning(l) for l in (learnings or [])]
        return r

    def _make_operation(
        self, op_type="ADD", content="test content", learning_index=None
    ):
        class FakeOp:
            pass

        op = FakeOp()
        op.type = op_type
        op.content = content
        op.insight_source = None
        op.learning_index = learning_index
        return op

    def test_attaches_to_add_operation(self):
        ops = [self._make_operation("ADD", "verify calculations", learning_index=0)]
        build_insight_source(
            sample_question="What is 2+2?",
            epoch=1,
            step=5,
            error_identification="calculation error",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(
                error_id="calculation error",
                learnings=["verify calculations carefully"],
            ),
            operations=ops,
        )
        self.assertIsNotNone(ops[0].insight_source)
        self.assertEqual(ops[0].insight_source["epoch"], 1)
        self.assertEqual(ops[0].insight_source["step"], 5)

    def test_skips_tag_operation(self):
        ops = [self._make_operation("TAG")]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(),
            operations=ops,
        )
        self.assertIsNone(ops[0].insight_source)

    def test_text_excerpt_fallback(self):
        ops = [self._make_operation("ADD", "new strategy")]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(reasoning="Long reasoning text"),
            reflection=self._make_reflection(),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertIsNotNone(src)
        refs = src.get("trace_refs", [])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["excerpt_location"], "reasoning")
        self.assertIn("Long reasoning", refs[0]["text_excerpt"])

    def test_question_capped(self):
        long_q = "x" * 500
        ops = [self._make_operation("ADD", "content")]
        build_insight_source(
            sample_question=long_q,
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(),
            operations=ops,
        )
        self.assertEqual(len(ops[0].insight_source["sample_question"]), 200)

    def test_error_identification_stored(self):
        """Test that error_identification text is stored on the insight source."""
        ops = [self._make_operation("ADD", "new strategy")]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="The agent failed to carry the remainder",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertEqual(
            src["error_identification"],
            "The agent failed to carry the remainder",
        )

    def test_error_identification_capped(self):
        """Test that long error_identification is capped at _MAX_EXCERPT_LEN (200)."""
        long_error = "x" * 2000
        ops = [self._make_operation("ADD", "content")]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification=long_error,
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertEqual(len(src["error_identification"]), 200)

    def test_empty_error_identification_stored_as_none(self):
        """Test that empty error_identification is stored as None (omitted)."""
        ops = [self._make_operation("ADD", "content")]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertNotIn("error_identification", src)

    def test_learning_index_resolves_to_learning_text(self):
        """Test that learning_index resolves to the correct learning_text."""
        ops = [self._make_operation("ADD", "strategy A", learning_index=1)]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(
                learnings=["first learning", "second learning"],
            ),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertEqual(src["learning_text"], "second learning")

    def test_learning_index_none_gives_none_learning_text(self):
        """Test that missing learning_index results in learning_text=None."""
        ops = [self._make_operation("ADD", "strategy")]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(
                learnings=["a learning"],
            ),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertNotIn("learning_text", src)

    def test_learning_index_out_of_range_gives_none(self):
        """Test that out-of-range learning_index results in learning_text=None."""
        ops = [self._make_operation("ADD", "strategy", learning_index=99)]
        build_insight_source(
            sample_question="Q",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(
                learnings=["only one learning"],
            ),
            operations=ops,
        )
        src = ops[0].insight_source
        self.assertNotIn("learning_text", src)

    def test_sample_id_roundtrip(self):
        """Test that sample_id survives serialization roundtrip."""
        source = InsightSource(
            sample_question="-",
            epoch=1,
            step=1,
            sample_id="trace_001.md",
        )
        d = source.to_dict()
        self.assertEqual(d["sample_id"], "trace_001.md")

        restored = InsightSource.from_dict(d)
        self.assertEqual(restored.sample_id, "trace_001.md")

    def test_sample_id_none_omitted(self):
        """Test that sample_id is omitted from dict when None."""
        source = InsightSource(
            sample_question="Q",
            epoch=0,
            step=0,
        )
        d = source.to_dict()
        self.assertNotIn("sample_id", d)

    def test_build_insight_source_with_sample_id(self):
        """Test that sample_id flows through build_insight_source."""
        ops = [self._make_operation("ADD", "strategy")]
        build_insight_source(
            sample_question="-",
            epoch=1,
            step=1,
            error_identification="",
            agent_output=self._make_agent_output(),
            reflection=self._make_reflection(),
            operations=ops,
            sample_id="session_42.toon",
        )
        src = ops[0].insight_source
        self.assertEqual(src["sample_id"], "session_42.toon")


if __name__ == "__main__":
    unittest.main()
