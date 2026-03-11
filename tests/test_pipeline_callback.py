"""Tests for Pipeline.run() on_sample_done callback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pipeline import Pipeline, SampleResult, StepContext


class PassthroughStep:
    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class FailingStep:
    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        raise RuntimeError("boom")


class TestOnSampleDone:
    def test_callback_called_per_sample(self):
        pipe = Pipeline([PassthroughStep()])
        contexts = [StepContext(sample=i) for i in range(5)]
        cb = MagicMock()

        pipe.run(contexts, on_sample_done=cb)

        assert cb.call_count == 5
        for call in cb.call_args_list:
            result = call[0][0]
            assert isinstance(result, SampleResult)
            assert result.error is None

    def test_callback_called_on_error(self):
        pipe = Pipeline([FailingStep()])
        contexts = [StepContext(sample="x")]
        cb = MagicMock()

        pipe.run(contexts, on_sample_done=cb)

        assert cb.call_count == 1
        result = cb.call_args[0][0]
        assert isinstance(result, SampleResult)
        assert isinstance(result.error, RuntimeError)
        assert result.failed_at == "FailingStep"

    def test_none_callback_is_noop(self):
        pipe = Pipeline([PassthroughStep()])
        contexts = [StepContext(sample=1)]

        # Should not raise
        results = pipe.run(contexts, on_sample_done=None)
        assert len(results) == 1
        assert results[0].error is None

    def test_callback_with_multiple_workers(self):
        pipe = Pipeline([PassthroughStep()])
        contexts = [StepContext(sample=i) for i in range(10)]
        cb = MagicMock()

        pipe.run(contexts, workers=4, on_sample_done=cb)

        assert cb.call_count == 10
