"""Tests for RRIterationContext."""

import pytest

from ace_next.rr.context import RRIterationContext


@pytest.mark.unit
class TestRRIterationContext:
    """Test RRIterationContext creation and immutability."""

    def test_defaults(self):
        ctx = RRIterationContext()
        assert ctx.messages == ()
        assert ctx.iteration == 0
        assert ctx.llm_response is None
        assert ctx.code is None
        assert ctx.direct_response is None
        assert ctx.exec_result is None
        assert ctx.terminated is False
        assert ctx.reflection is None
        assert ctx.feedback_messages == ()

    def test_frozen(self):
        ctx = RRIterationContext()
        with pytest.raises(AttributeError):
            ctx.iteration = 5  # type: ignore[misc]

    def test_replace(self):
        ctx = RRIterationContext(messages=({"role": "user", "content": "hi"},))
        ctx2 = ctx.replace(iteration=3, llm_response="hello")
        assert ctx2.iteration == 3
        assert ctx2.llm_response == "hello"
        # Original unchanged
        assert ctx.iteration == 0
        assert ctx.llm_response is None
        # Preserved field
        assert ctx2.messages == ({"role": "user", "content": "hi"},)

    def test_replace_terminated(self):
        ctx = RRIterationContext()
        ctx2 = ctx.replace(terminated=True, reflection={"key": "value"})
        assert ctx2.terminated is True
        assert ctx2.reflection == {"key": "value"}

    def test_inherits_step_context_fields(self):
        """RRIterationContext inherits sample and metadata from StepContext."""
        ctx = RRIterationContext(sample="test_sample")
        assert ctx.sample == "test_sample"
