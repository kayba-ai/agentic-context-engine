"""Tests for the generic SubRunner base class."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import pytest

from pipeline import Pipeline, StepContext, StepProtocol
from ace_next.core import SubRunner


# ---------------------------------------------------------------------------
# Minimal concrete SubRunner — a counter that loops until reaching a target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterContext(StepContext):
    """Simple context that tracks a counter value."""

    counter: int = 0
    target: int = 5
    done: bool = False


class IncrementStep:
    """Adds 1 to counter and checks if target is reached."""

    requires = frozenset()
    provides = frozenset({"counter", "done"})

    def __call__(self, ctx: CounterContext) -> CounterContext:  # type: ignore[override]
        new_counter = ctx.counter + 1
        return ctx.replace(counter=new_counter, done=new_counter >= ctx.target)


class CounterRunner(SubRunner):
    """Counts up to a target using an inner pipeline with IncrementStep."""

    requires = frozenset({"target"})
    provides = frozenset({"counter"})

    def __init__(self, target: int = 5, max_iterations: int = 20) -> None:
        super().__init__(max_iterations=max_iterations)
        self.target = target

    def _build_inner_pipeline(self, **kwargs):
        return Pipeline([IncrementStep()])

    def _build_initial_context(self, **kwargs):
        return CounterContext(counter=0, target=self.target)

    def _is_done(self, ctx):
        return getattr(ctx, "done", False)

    def _extract_result(self, ctx):
        return getattr(ctx, "counter", None)

    def _accumulate(self, ctx):
        return ctx  # counter is already incremented by the step

    def __call__(self, ctx: StepContext) -> StepContext:
        result = self.run_loop()
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "counter": result})
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubRunnerBasics:
    """Test the SubRunner loop with a minimal concrete implementation."""

    def test_reaches_target(self):
        runner = CounterRunner(target=5, max_iterations=20)
        result = runner.run_loop()
        assert result == 5

    def test_reaches_target_exactly(self):
        runner = CounterRunner(target=1, max_iterations=10)
        result = runner.run_loop()
        assert result == 1

    def test_zero_iterations_raises(self):
        """max_iterations=0 means the loop body never runs → timeout."""
        runner = CounterRunner(target=1, max_iterations=0)
        with pytest.raises(RuntimeError, match="max_iterations"):
            runner.run_loop()


class TestSubRunnerTimeout:
    """Test max_iterations timeout behavior."""

    def test_timeout_raises_by_default(self):
        runner = CounterRunner(target=100, max_iterations=3)
        with pytest.raises(RuntimeError, match="max_iterations.*3"):
            runner.run_loop()

    def test_custom_on_timeout(self):
        """Subclass can override _on_timeout to return a fallback."""

        class GracefulRunner(CounterRunner):
            def _on_timeout(self, last_ctx, iteration, **kwargs):
                return -1  # sentinel

        runner = GracefulRunner(target=100, max_iterations=3)
        result = runner.run_loop()
        assert result == -1

    def test_on_timeout_receives_kwargs_from_run_loop(self):
        """kwargs passed to run_loop() are forwarded to _on_timeout()."""
        captured: dict = {}

        class KwargsCapture(CounterRunner):
            def _on_timeout(self, last_ctx, iteration, **kwargs):
                captured.update(kwargs)
                return -1

        runner = KwargsCapture(target=100, max_iterations=2)
        result = runner.run_loop(foo="bar", baz=42)
        assert result == -1
        assert captured == {"foo": "bar", "baz": 42}


class TestSubRunnerProtocol:
    """Test that SubRunner subclasses satisfy StepProtocol."""

    def test_satisfies_step_protocol(self):
        runner = CounterRunner(target=5)
        assert isinstance(runner, StepProtocol)

    def test_has_requires_and_provides(self):
        runner = CounterRunner(target=5)
        assert "target" in runner.requires
        assert "counter" in runner.provides

    def test_call_updates_context(self):
        runner = CounterRunner(target=3, max_iterations=10)
        ctx = StepContext(sample="test")
        result_ctx = runner(ctx)
        assert result_ctx.metadata["counter"] == 3


class TestSubRunnerInnerPipeline:
    """Test that the inner pipeline is called per iteration."""

    def test_inner_pipeline_called_per_iteration(self):
        """Each iteration should invoke the inner pipeline exactly once."""
        iteration_log: list[int] = []

        class TrackingStep:
            """Records each iteration it sees."""

            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: CounterContext) -> CounterContext:  # type: ignore[override]
                iteration_log.append(ctx.counter)
                return ctx

        class TrackedRunner(CounterRunner):
            def _build_inner_pipeline(self, **kwargs):
                return Pipeline([IncrementStep(), TrackingStep()])

        runner = TrackedRunner(target=3, max_iterations=10)
        result = runner.run_loop()
        assert result == 3
        assert len(iteration_log) == 3  # 3 iterations to reach target=3

    def test_pipeline_built_once_per_run_loop(self):
        """_build_inner_pipeline is called once per run_loop invocation."""
        build_count = 0

        class TrackedRunner(CounterRunner):
            def _build_inner_pipeline(self, **kwargs):
                nonlocal build_count
                build_count += 1
                return super()._build_inner_pipeline(**kwargs)

        runner = TrackedRunner(target=3, max_iterations=10)
        runner.run_loop()
        assert build_count == 1

        # Second call builds a new pipeline
        runner.run_loop()
        assert build_count == 2


class TestSubRunnerAccumulate:
    """Test the accumulate step between iterations."""

    def test_accumulate_transforms_context(self):
        """Verify that _accumulate is called between iterations."""

        class DoubleCounterRunner(CounterRunner):
            """Doubles the counter each iteration via accumulate."""

            def _accumulate(self, ctx):
                # Double the counter before next iteration
                return ctx.replace(counter=ctx.counter * 2)

        runner = DoubleCounterRunner(target=10, max_iterations=20)
        # Iteration 0: counter 0 → IncrementStep → 1, not done, accumulate → 2
        # Iteration 1: counter 2 → IncrementStep → 3, not done, accumulate → 6
        # Iteration 2: counter 6 → IncrementStep → 7, not done, accumulate → 14
        # Iteration 3: counter 14 → IncrementStep → 15, done (>=10)
        result = runner.run_loop()
        assert result == 15


class TestSubRunnerNesting:
    """Test that a SubRunner can be used as a step in a Pipeline."""

    def test_nested_in_pipeline(self):
        runner = CounterRunner(target=3, max_iterations=10)
        pipe = Pipeline([runner])
        ctx = StepContext(sample="test")
        result = pipe(ctx)
        assert result.metadata["counter"] == 3
