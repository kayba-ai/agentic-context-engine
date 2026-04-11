"""Tests for PipelineHook, CancellationToken, and cancel_token_var."""

from __future__ import annotations

import threading
import time

import pytest

from pipeline import (
    CancellationToken,
    Pipeline,
    PipelineCancelled,
    PipelineHook,
    SampleResult,
    StepContext,
    cancel_token_var,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class PassthroughStep:
    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class SlowStep:
    """Step that sleeps briefly — useful for cancel-during-run tests."""

    requires = frozenset()
    provides = frozenset()

    def __init__(self, delay: float = 0.1):
        self._delay = delay

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(self._delay)
        return ctx


class FailingStep:
    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        raise RuntimeError("boom")


class RecordingHook:
    """Collects (event, step_name) tuples for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def before_step(self, step_name: str, ctx: StepContext) -> None:
        self.events.append(("before", step_name))

    def after_step(self, step_name: str, ctx: StepContext) -> None:
        self.events.append(("after", step_name))


class BrokenHook:
    """Hook that raises on every call."""

    def before_step(self, step_name: str, ctx: StepContext) -> None:
        raise ValueError("hook broken before")

    def after_step(self, step_name: str, ctx: StepContext) -> None:
        raise ValueError("hook broken after")


# ==================================================================
# PipelineHook tests
# ==================================================================


class TestPipelineHooks:
    def test_hooks_fire_before_and_after_each_step(self):
        hook = RecordingHook()
        pipe = Pipeline([PassthroughStep(), PassthroughStep()], hooks=[hook])
        pipe.run([StepContext(sample=1)])

        assert hook.events == [
            ("before", "PassthroughStep"),
            ("after", "PassthroughStep"),
            ("before", "PassthroughStep"),
            ("after", "PassthroughStep"),
        ]

    def test_hooks_fire_per_sample(self):
        hook = RecordingHook()
        pipe = Pipeline([PassthroughStep()], hooks=[hook])
        pipe.run([StepContext(sample=1), StepContext(sample=2)])

        assert len(hook.events) == 4  # 2 samples × (before + after)

    def test_multiple_hooks(self):
        hook1 = RecordingHook()
        hook2 = RecordingHook()
        pipe = Pipeline([PassthroughStep()], hooks=[hook1, hook2])
        pipe.run([StepContext(sample=1)])

        assert hook1.events == [
            ("before", "PassthroughStep"),
            ("after", "PassthroughStep"),
        ]
        assert hook2.events == [
            ("before", "PassthroughStep"),
            ("after", "PassthroughStep"),
        ]

    def test_broken_hook_does_not_kill_pipeline(self):
        broken = BrokenHook()
        recorder = RecordingHook()
        pipe = Pipeline([PassthroughStep()], hooks=[broken, recorder])
        results = pipe.run([StepContext(sample=1)])

        # Pipeline still succeeds
        assert len(results) == 1
        assert results[0].error is None
        assert results[0].output is not None
        # Second hook still fired
        assert len(recorder.events) == 2

    def test_hooks_receive_correct_step_name(self):
        hook = RecordingHook()
        pipe = Pipeline([SlowStep(delay=0)], hooks=[hook])
        pipe.run([StepContext(sample=1)])

        assert hook.events[0] == ("before", "SlowStep")
        assert hook.events[1] == ("after", "SlowStep")

    def test_after_hook_not_called_on_step_error(self):
        hook = RecordingHook()
        pipe = Pipeline([FailingStep()], hooks=[hook])
        results = pipe.run([StepContext(sample=1)])

        # before fires, step raises, after does NOT fire for that step
        assert hook.events == [("before", "FailingStep")]
        assert isinstance(results[0].error, RuntimeError)

    def test_no_hooks_is_backward_compatible(self):
        pipe = Pipeline([PassthroughStep()])
        results = pipe.run([StepContext(sample=1)])

        assert len(results) == 1
        assert results[0].error is None

    def test_hook_satisfies_protocol(self):
        hook = RecordingHook()
        assert isinstance(hook, PipelineHook)


# ==================================================================
# CancellationToken tests
# ==================================================================


class TestCancellationToken:
    def test_not_cancelled_initially(self):
        token = CancellationToken()
        assert not token.is_cancelled

    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled

    def test_cancel_is_idempotent(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled

    def test_cancel_is_thread_safe(self):
        token = CancellationToken()

        def cancel_from_thread():
            time.sleep(0.01)
            token.cancel()

        t = threading.Thread(target=cancel_from_thread)
        t.start()
        t.join()
        assert token.is_cancelled


# ==================================================================
# Pipeline cancellation tests
# ==================================================================


class TestPipelineCancellation:
    def test_pre_cancelled_token_cancels_immediately(self):
        token = CancellationToken()
        token.cancel()

        pipe = Pipeline([PassthroughStep(), PassthroughStep()])
        results = pipe.run([StepContext(sample=1)], cancel_token=token)

        assert len(results) == 1
        assert isinstance(results[0].error, PipelineCancelled)
        assert results[0].failed_at == "PassthroughStep"
        assert results[0].output is None

    def test_cancel_between_steps(self):
        """Cancel after the first step; second step should not run."""
        call_count = 0

        class CountingStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                nonlocal call_count
                call_count += 1
                return ctx

        token = CancellationToken()

        class CancelAfterFirstHook:
            def before_step(self, step_name, ctx):
                pass

            def after_step(self, step_name, ctx):
                # Cancel after the first step completes
                token.cancel()

        pipe = Pipeline(
            [CountingStep(), CountingStep()],
            hooks=[CancelAfterFirstHook()],
        )
        results = pipe.run([StepContext(sample=1)], cancel_token=token)

        assert call_count == 1  # Only first step ran
        assert isinstance(results[0].error, PipelineCancelled)

    def test_cancel_stops_remaining_samples(self):
        token = CancellationToken()
        samples_started = []

        class TrackingStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                samples_started.append(ctx.sample)
                if ctx.sample == 0:
                    token.cancel()
                return ctx

        pipe = Pipeline([TrackingStep()])
        results = pipe.run(
            [StepContext(sample=i) for i in range(5)],
            cancel_token=token,
        )

        # First sample ran (triggered cancel), rest should be cancelled
        assert 0 in samples_started
        cancelled = [r for r in results if isinstance(r.error, PipelineCancelled)]
        assert len(cancelled) >= 1  # At least some were cancelled

    def test_no_token_runs_normally(self):
        pipe = Pipeline([PassthroughStep()])
        results = pipe.run([StepContext(sample=1)], cancel_token=None)

        assert len(results) == 1
        assert results[0].error is None

    def test_on_sample_done_fires_on_cancellation(self):
        token = CancellationToken()
        token.cancel()
        cb_results = []

        pipe = Pipeline([PassthroughStep()])
        pipe.run(
            [StepContext(sample=1)],
            cancel_token=token,
            on_sample_done=lambda r: cb_results.append(r),
        )

        assert len(cb_results) == 1
        assert isinstance(cb_results[0].error, PipelineCancelled)

    def test_cancelled_result_has_correct_shape(self):
        token = CancellationToken()
        token.cancel()

        pipe = Pipeline([PassthroughStep()])
        results = pipe.run([StepContext(sample="x")], cancel_token=token)

        r = results[0]
        assert r.sample == "x"
        assert r.output is None
        assert isinstance(r.error, PipelineCancelled)
        assert r.failed_at is not None


# ==================================================================
# cancel_token_var contextvar tests
# ==================================================================


class TestCancelTokenVar:
    def test_contextvar_is_none_by_default(self):
        assert cancel_token_var.get(None) is None

    def test_contextvar_set_during_pipeline_run(self):
        """Steps can read the cancel_token_var set by the pipeline."""
        observed_tokens = []

        class TokenReadingStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                observed_tokens.append(cancel_token_var.get(None))
                return ctx

        token = CancellationToken()
        pipe = Pipeline([TokenReadingStep()])
        pipe.run([StepContext(sample=1)], cancel_token=token)

        assert len(observed_tokens) == 1
        assert observed_tokens[0] is token

    def test_contextvar_is_none_without_cancel_token(self):
        """When no cancel_token is passed, the contextvar is None inside steps."""
        observed_tokens = []

        class TokenReadingStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                observed_tokens.append(cancel_token_var.get(None))
                return ctx

        pipe = Pipeline([TokenReadingStep()])
        pipe.run([StepContext(sample=1)])

        assert len(observed_tokens) == 1
        assert observed_tokens[0] is None

    def test_contextvar_reset_after_run(self):
        """The contextvar is reset after run() completes."""
        token = CancellationToken()
        pipe = Pipeline([PassthroughStep()])
        pipe.run([StepContext(sample=1)], cancel_token=token)

        # After run, the contextvar should be back to default
        assert cancel_token_var.get(None) is None

    def test_contextvar_visible_across_multiple_steps(self):
        """All steps in the same pipeline run see the same token."""
        observed_tokens = []

        class TokenReadingStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                observed_tokens.append(cancel_token_var.get(None))
                return ctx

        token = CancellationToken()
        pipe = Pipeline([TokenReadingStep(), TokenReadingStep(), TokenReadingStep()])
        pipe.run([StepContext(sample=1)], cancel_token=token)

        assert len(observed_tokens) == 3
        assert all(t is token for t in observed_tokens)

    def test_contextvar_per_sample(self):
        """Each sample in the same run sees the same token."""
        observed_tokens = []

        class TokenReadingStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                observed_tokens.append(cancel_token_var.get(None))
                return ctx

        token = CancellationToken()
        pipe = Pipeline([TokenReadingStep()])
        pipe.run(
            [StepContext(sample=i) for i in range(3)],
            cancel_token=token,
        )

        assert len(observed_tokens) == 3
        assert all(t is token for t in observed_tokens)
