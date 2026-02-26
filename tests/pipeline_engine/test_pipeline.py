"""Unit tests for Pipeline — construction, validation, and execution."""

from __future__ import annotations

import asyncio
import time
import warnings
from types import MappingProxyType

import pytest

from pipeline import (
    Branch,
    MergeStrategy,
    Pipeline,
    PipelineConfigError,
    PipelineOrderError,
    SampleResult,
    StepContext,
    StepProtocol,
)
from tests.pipeline_engine.conftest import (
    Boom,
    BoundaryStep,
    Noop,
    Recorder,
    SetA,
    SetB,
    SetC,
    SlowBoundaryStep,
    Slow,
)

# ---------------------------------------------------------------------------
# Construction & contract inference
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineConstruction:
    def test_empty_pipeline_has_empty_contracts(self):
        p = Pipeline()
        assert p.requires == frozenset()
        assert p.provides == frozenset()

    def test_list_constructor_accepted(self):
        p = Pipeline([SetA(), SetB()])
        assert "a" in p.provides
        assert "b" in p.provides

    def test_then_returns_self(self):
        p = Pipeline()
        result = p.then(Noop())
        assert result is p

    def test_then_updates_provides(self):
        p = Pipeline().then(SetA())
        assert "a" in p.provides

    def test_then_chain_updates_provides_cumulatively(self):
        p = Pipeline().then(SetA()).then(SetB()).then(SetC())
        assert {"a", "b", "c"} <= p.provides

    def test_external_requires_inferred(self):
        """Fields needed by the first step that no prior step provides."""
        p = Pipeline().then(SetB())  # SetB.requires = {"a"}, nothing provides "a"
        assert "a" in p.requires

    def test_internally_satisfied_requires_not_in_external(self):
        """When A→B, 'a' is provided internally so not in pipeline.requires."""
        p = Pipeline().then(SetA()).then(SetB())
        assert "a" not in p.requires

    def test_branch_method_appends_branch(self):
        # Use two independent branches (no cross-dependency)
        p = Pipeline().then(SetA())
        p2 = p.branch(Pipeline().then(Noop()), Pipeline().then(Noop()))
        # branch() returns self; a Branch is the last step
        assert isinstance(p2._steps[-1], Branch)

    def test_pipeline_satisfies_step_protocol(self):
        p = Pipeline().then(SetA())
        assert isinstance(p, StepProtocol)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineValidation:
    def test_order_error_when_b_before_a(self):
        with pytest.raises(PipelineOrderError, match="a"):
            Pipeline().then(SetB()).then(SetA())

    def test_order_error_at_construction_with_list(self):
        with pytest.raises(PipelineOrderError):
            Pipeline([SetB(), SetA()])

    def test_no_order_error_for_external_input(self):
        """SetB requires 'a', but 'a' is not provided by any step → external input.
        This is valid — the caller is expected to put 'a' in the initial context."""
        p = Pipeline().then(SetB())  # should NOT raise
        assert "a" in p.requires

    def test_config_error_duplicate_async_boundary(self):
        class B1:
            requires = frozenset()
            provides = frozenset({"p"})
            async_boundary = True

            def __call__(self, ctx):
                return ctx

        class B2:
            requires = frozenset()
            provides = frozenset({"q"})
            async_boundary = True

            def __call__(self, ctx):
                return ctx

        with pytest.raises(PipelineConfigError, match="duplicate"):
            Pipeline().then(B1()).then(B2())

    def test_config_error_boundary_inside_branch(self):
        with pytest.raises(PipelineConfigError, match="Branch"):
            Pipeline().branch(Pipeline().then(BoundaryStep()))

    def test_warning_for_boundary_on_nested_pipeline(self):
        """async_boundary on a Pipeline-as-step must emit a warning (not error)."""
        inner = Pipeline().then(SetA())
        inner.async_boundary = True  # manually set

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Pipeline().then(inner)

        assert any(
            "async_boundary" in str(w.message) for w in caught
        ), "Expected a warning about async_boundary on nested pipeline"

    def test_validation_runs_on_each_then_call(self):
        # SetA provides "a"; SetB provides "b"; SetC requires "b".
        # Adding SetC before SetB (so "b" is internally provided but out of order)
        # must raise PipelineOrderError.
        p = Pipeline().then(SetA()).then(SetB())  # "a" → "b" in order
        with pytest.raises(PipelineOrderError):
            # Now add a step that requires "a" again, but "a" was already consumed;
            # add SetB *again* before SetC would work, but adding SetC before SetB
            # in a fresh pipeline is the right test:
            Pipeline().then(SetA()).then(SetC()).then(
                SetB()
            )  # "c" needs "b", "b" comes after


# ---------------------------------------------------------------------------
# __call__ (nested step mode)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineCall:
    def test_call_runs_all_steps(self):
        p = Pipeline().then(SetA()).then(SetB())
        ctx = p(StepContext(sample="s"))
        assert ctx.metadata["a"] == 1
        assert ctx.metadata["b"] == 2

    def test_call_ignores_async_boundary(self):
        """When used as nested step, async_boundary should not split execution."""
        p = Pipeline().then(SetA()).then(BoundaryStep()).then(SetB())

        # BoundaryStep requires nothing, provides "bg_result"
        # SetC would need "b", so we use SetA (provides "a"), then BoundaryStep,
        # then Noop — all should run.
        class AfterBoundary:
            requires = frozenset()
            provides = frozenset({"after"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "after": True})
                )

        p2 = Pipeline().then(SetA()).then(BoundaryStep()).then(AfterBoundary())
        ctx = p2(StepContext(sample="s"))
        assert ctx.metadata.get("a") == 1
        assert ctx.metadata.get("bg_result") is True
        assert ctx.metadata.get("after") is True

    def test_empty_pipeline_call_passthrough(self):
        ctx = StepContext(sample="original")
        out = Pipeline()(ctx)
        assert out == ctx


# ---------------------------------------------------------------------------
# run() — basic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineRun:
    def test_single_step_single_sample(self):
        results = Pipeline().then(SetA()).run([StepContext(sample="s")])
        assert len(results) == 1
        assert results[0].output.metadata["a"] == 1
        assert results[0].error is None

    def test_multi_step_chain(self):
        results = (
            Pipeline()
            .then(SetA())
            .then(SetB())
            .then(SetC())
            .run([StepContext(sample="s")])
        )
        out = results[0].output
        assert out.metadata["a"] == 1
        assert out.metadata["b"] == 2
        assert out.metadata["c"] == 4

    def test_multiple_samples(self):
        results = (
            Pipeline()
            .then(SetA())
            .run([StepContext(sample=s) for s in ("s1", "s2", "s3")])
        )
        assert len(results) == 3
        assert all(r.output.metadata["a"] == 1 for r in results)

    def test_sample_value_in_result(self):
        results = Pipeline().then(Noop()).run([StepContext(sample="hello")])
        assert results[0].sample == "hello"

    def test_empty_pipeline_passes_context_through(self):
        results = Pipeline().run([StepContext(sample="s")])
        assert results[0].output is not None
        assert results[0].output.sample == "s"

    def test_run_returns_sample_result_list(self):
        results = (
            Pipeline()
            .then(Noop())
            .run([StepContext(sample="a"), StepContext(sample="b")])
        )
        assert all(isinstance(r, SampleResult) for r in results)

    @pytest.mark.slow
    def test_workers_N_runs_faster_than_sequential(self):
        delay = 0.05
        contexts = [StepContext(sample=i) for i in range(4)]
        pipe = Pipeline().then(Slow(delay))

        t0 = time.monotonic()
        pipe.run(contexts, workers=1)
        seq_time = time.monotonic() - t0

        t0 = time.monotonic()
        pipe.run(contexts, workers=4)
        par_time = time.monotonic() - t0

        # Parallel should be at least 2× faster
        assert (
            par_time < seq_time / 2
        ), f"workers=4 ({par_time:.2f}s) not faster than workers=1 ({seq_time:.2f}s)"


# ---------------------------------------------------------------------------
# run() — error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineRunErrors:
    def test_step_failure_sets_error(self):
        results = Pipeline().then(Boom()).run([StepContext(sample="s")])
        assert results[0].error is not None
        assert isinstance(results[0].error, RuntimeError)

    def test_step_failure_sets_failed_at_name(self):
        results = Pipeline().then(Boom()).run([StepContext(sample="s")])
        assert results[0].failed_at == "Boom"

    def test_step_failure_output_is_none(self):
        results = Pipeline().then(Boom()).run([StepContext(sample="s")])
        assert results[0].output is None

    def test_other_samples_continue_after_one_failure(self):
        """A failing sample must not prevent other samples from being processed."""

        class FailFirst:
            requires = frozenset()
            provides = frozenset()
            call_count = 0

            def __call__(self, ctx):
                FailFirst.call_count += 1
                if ctx.sample == "bad":
                    raise RuntimeError("bad sample")
                return ctx

        FailFirst.call_count = 0
        results = (
            Pipeline()
            .then(FailFirst())
            .run([StepContext(sample=s) for s in ("ok1", "bad", "ok2")])
        )
        assert len(results) == 3
        errors = [r for r in results if r.error is not None]
        successes = [r for r in results if r.error is None]
        assert len(errors) == 1
        assert len(successes) == 2
        assert errors[0].sample == "bad"

    def test_failed_at_is_correct_step_name(self):
        class FirstStep:
            requires = frozenset()
            provides = frozenset({"p"})

            def __call__(self, ctx):
                return ctx.replace(metadata={**ctx.metadata})

        class FailingStep:
            requires = frozenset({"p"})
            provides = frozenset()

            def __call__(self, ctx):
                raise ValueError("fail")

        results = (
            Pipeline()
            .then(FirstStep())
            .then(FailingStep())
            .run([StepContext(sample="s")])
        )
        assert results[0].failed_at == "FailingStep"


# ---------------------------------------------------------------------------
# run() — async_boundary + background
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineAsyncBoundary:
    def test_foreground_returns_before_background_completes(self):
        """run() must return while the background tail is still executing.

        We verify this by using a slow background step (0.3 s sleep) and a
        threading.Event: if run() blocks until the background finishes, the
        Event will already be set when we check it — which would fail the test.
        """
        import threading as _threading
        from types import MappingProxyType as _MappingProxyType

        bg_completed = _threading.Event()

        class SlowBg:
            requires = frozenset()
            provides = frozenset({"bg"})
            async_boundary = True
            max_workers = 1

            def __call__(self, ctx):
                time.sleep(0.3)
                bg_completed.set()
                return ctx.replace(
                    metadata=_MappingProxyType({**ctx.metadata, "bg": True})
                )

        pipe = Pipeline().then(SlowBg())
        results = pipe.run([StepContext(sample="s")])

        # run() returned — background must NOT have finished yet
        assert (
            not bg_completed.is_set()
        ), "run() blocked until background completed; it should return immediately"
        assert len(results) == 1

        pipe.wait_for_background(timeout=5.0)
        assert bg_completed.is_set()

    def test_wait_for_background_completes_output(self):
        pipe = Pipeline().then(SetA()).then(BoundaryStep())
        results = pipe.run([StepContext(sample="s")])
        pipe.wait_for_background(timeout=5.0)
        assert results[0].output is not None
        assert results[0].output.metadata.get("bg_result") is True
        assert results[0].error is None

    def test_background_failure_captured_in_result(self):
        class BGBoom:
            requires = frozenset()
            provides = frozenset({"bg"})
            async_boundary = True
            max_workers = 1

            def __call__(self, ctx):
                raise RuntimeError("bg_boom")

        pipe = Pipeline().then(BGBoom())
        results = pipe.run([StepContext(sample="s")])
        pipe.wait_for_background(timeout=5.0)
        assert results[0].error is not None
        assert results[0].failed_at == "BGBoom"
        assert results[0].output is None

    def test_wait_for_background_no_threads_is_noop(self):
        """wait_for_background() on a pipeline with no async_boundary must not raise."""
        pipe = Pipeline().then(SetA())  # no boundary → no background threads
        pipe.run([StepContext(sample="s")])
        pipe.wait_for_background(timeout=1.0)  # must be a silent no-op

    def test_wait_for_background_timeout_raises(self):
        pipe = Pipeline().then(SlowBoundaryStep())
        pipe.run([StepContext(sample="s")])
        with pytest.raises(TimeoutError):
            pipe.wait_for_background(timeout=0.05)

    def test_multiple_samples_all_get_background_result(self):
        pipe = Pipeline().then(BoundaryStep())
        results = pipe.run([StepContext(sample=s) for s in ("a", "b", "c")])
        pipe.wait_for_background(timeout=5.0)
        assert all(r.output is not None for r in results)
        assert all(r.output.metadata.get("bg_result") is True for r in results)

    @pytest.mark.slow
    def test_background_max_workers_1_serializes_execution(self):
        """With max_workers=1, background steps cannot interleave."""
        from tests.pipeline_engine.conftest import SerialStep

        SerialStep._log.clear()

        class TriggerBoundary:
            requires = frozenset()
            provides = frozenset({"trigger"})
            async_boundary = True
            max_workers = 3  # multiple samples can start background at once

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "trigger": True})
                )

        # SerialStep has max_workers=1; two samples must not interleave
        pipe = Pipeline().then(TriggerBoundary()).then(SerialStep())
        results = pipe.run(
            [StepContext(sample="x"), StepContext(sample="y")], workers=2
        )
        pipe.wait_for_background(timeout=5.0)

        log = SerialStep._log
        # For correct serialization: start-X must be immediately followed by end-X
        for i in range(0, len(log), 2):
            assert log[i].startswith("start"), f"log[{i}] = {log[i]}"
            sample = log[i].split("-")[1]
            assert log[i + 1] == f"end-{sample}", f"Interleaved: {log}"


# ---------------------------------------------------------------------------
# run_async()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineRunAsync:
    def test_run_async_same_results_as_run(self):
        pipe = Pipeline().then(SetA()).then(SetB())
        contexts = [StepContext(sample="s1"), StepContext(sample="s2")]
        sync_results = pipe.run(contexts)
        async_results = asyncio.run(pipe.run_async(contexts))
        assert len(async_results) == 2
        for s, a in zip(sync_results, async_results):
            assert s.output == a.output

    def test_run_async_handles_step_failure(self):
        pipe = Pipeline().then(Boom())
        results = asyncio.run(pipe.run_async([StepContext(sample="s")]))
        assert results[0].error is not None

    def test_run_async_workers_respected(self):
        """Multiple samples run concurrently with workers>1."""
        delay = 0.05
        pipe = Pipeline().then(Slow(delay))
        contexts = [StepContext(sample=i) for i in range(4)]
        t0 = time.monotonic()
        asyncio.run(pipe.run_async(contexts, workers=4))
        elapsed = time.monotonic() - t0
        assert elapsed < delay * 2, f"Expected concurrency, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineNesting:
    def test_inner_pipeline_used_as_step(self):
        inner = Pipeline().then(SetA()).then(SetB())
        outer = Pipeline().then(inner).then(SetC())
        results = outer.run([StepContext(sample="s")])
        out = results[0].output
        assert out.metadata["a"] == 1
        assert out.metadata["b"] == 2
        assert out.metadata["c"] == 4

    def test_nested_pipeline_contracts_inferred(self):
        inner = Pipeline().then(SetA()).then(SetB())
        assert "a" in inner.provides
        assert "b" in inner.provides
        # Inner pipeline doesn't need anything external
        assert inner.requires == frozenset()

    def test_nested_pipeline_satisfies_step_protocol(self):
        inner = Pipeline().then(SetA())
        assert isinstance(inner, StepProtocol)


# ---------------------------------------------------------------------------
# Contract validation with subclass fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPipelineContractWithSubclass:
    """Validate that requires/provides work correctly with subclass named fields."""

    def test_order_error_when_subclass_field_required_before_provided(self):
        """Step B requires a subclass field that Step A provides, but B comes first."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass(frozen=True)
        class SubCtx(StepContext):
            agent_output: Any = None

        class WriteOutput:
            requires = frozenset()
            provides = frozenset({"agent_output"})

            def __call__(self, ctx):
                return ctx.replace(agent_output="answer")

        class ReadOutput:
            requires = frozenset({"agent_output"})
            provides = frozenset({"result"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType(
                        {**ctx.metadata, "result": ctx.agent_output}
                    )
                )

        # Correct order works
        p = Pipeline().then(WriteOutput()).then(ReadOutput())
        assert "agent_output" not in p.requires  # internally satisfied

        # Wrong order raises
        with pytest.raises(PipelineOrderError, match="agent_output"):
            Pipeline().then(ReadOutput()).then(WriteOutput())

    def test_subclass_field_as_external_input(self):
        """A step requires a subclass field not provided by any step → external input."""

        class NeedsOutput:
            requires = frozenset({"agent_output"})
            provides = frozenset({"score"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "score": 1.0})
                )

        p = Pipeline().then(NeedsOutput())
        assert "agent_output" in p.requires  # external — caller must provide

    def test_subclass_context_flows_through_pipeline_run(self):
        """Pipeline.run() with subclass contexts preserves subclass type."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass(frozen=True)
        class RunCtx(StepContext):
            answer: Any = None

        class SetAnswer:
            requires = frozenset()
            provides = frozenset({"answer"})

            def __call__(self, ctx):
                return ctx.replace(answer=f"solved_{ctx.sample}")

        results = Pipeline().then(SetAnswer()).run([RunCtx(sample="q1")])
        out = results[0].output
        assert isinstance(out, RunCtx)
        assert out.answer == "solved_q1"

    def test_subclass_context_with_call_mode(self):
        """Pipeline.__call__ with subclass context preserves subclass type."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass(frozen=True)
        class CallCtx(StepContext):
            tag: str = ""

        class SetTag:
            requires = frozenset()
            provides = frozenset({"tag"})

            def __call__(self, ctx):
                return ctx.replace(tag="tagged")

        out = Pipeline().then(SetTag())(CallCtx(sample="s"))
        assert isinstance(out, CallCtx)
        assert out.tag == "tagged"
