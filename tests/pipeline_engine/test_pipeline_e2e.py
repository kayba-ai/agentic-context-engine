"""End-to-end tests for the pipeline engine.

These tests exercise realistic multi-step scenarios — no ACE imports.
Dummy steps simulate the Agent → Evaluate → Reflect → Update pattern and
integration-style pipelines so the full plumbing is exercised.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import MappingProxyType

import pytest

from pipeline import (
    Branch,
    MergeStrategy,
    Pipeline,
    SampleResult,
    StepContext,
)


# ---------------------------------------------------------------------------
# Domain-agnostic dummy steps (ACE-shaped but no ACE imports)
# ---------------------------------------------------------------------------

class AgentStep:
    """Reads ctx.sample (a named field, not metadata), writes agent_output."""
    requires = frozenset()
    provides = frozenset({"agent_output"})
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(agent_output=f"answer_for_{ctx.sample}")


class EvaluateStep:
    """Reads agent_output + environment from metadata, writes environment_result."""
    requires = frozenset({"agent_output"})
    provides = frozenset({"environment_result"})
    def __call__(self, ctx: StepContext) -> StepContext:
        correct = ctx.agent_output == ctx.metadata.get("expected")
        return ctx.replace(
            environment_result={"correct": correct, "feedback": "ok" if correct else "wrong"}
        )


class ReflectStep:
    """Background step: reads agent_output + environment_result, writes reflection."""
    requires = frozenset({"agent_output", "environment_result"})
    provides = frozenset({"reflection"})
    async_boundary = True
    max_workers = 3

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.01)   # simulate LLM latency
        return ctx.replace(
            reflection={"insight": "reflected", "correct": ctx.environment_result["correct"]}
        )


class UpdateStep:
    """Background step: reads reflection, writes skill_manager_output. Serialized."""
    requires = frozenset({"reflection"})
    provides = frozenset({"skill_manager_output"})
    max_workers = 1
    _updates: list = []
    _lock = threading.Lock()

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.005)
        with self._lock:
            UpdateStep._updates.append(ctx.reflection)
        return ctx.replace(skill_manager_output={"updated": True})


class LogStep:
    """Side-effect step that records sample name (for Branch tests)."""
    requires = frozenset()
    provides = frozenset()
    def __init__(self):
        self.log: list[str] = []
        self._lock = threading.Lock()
    def __call__(self, ctx: StepContext) -> StepContext:
        with self._lock:
            self.log.append(ctx.sample)
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "logged": True}))


class MetricStep:
    """Writes a metric to metadata (for Branch tests alongside ReflectStep)."""
    requires = frozenset()
    provides = frozenset()
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "metric": len(str(ctx.sample))})
        )


# ---------------------------------------------------------------------------
# E2E: full 4-step pipeline (no async_boundary first)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullPipelineChain:
    def _pipe(self) -> Pipeline:
        return (
            Pipeline()
            .then(AgentStep())
            .then(EvaluateStep())
        )

    def test_single_sample_correct_answer(self):
        sample_ctx_kwargs = {"metadata": {"expected": "answer_for_q1"}}
        ctx = StepContext(sample="q1", **sample_ctx_kwargs)
        # Run via __call__ (nested mode)
        out = self._pipe()(ctx)
        assert out.agent_output == "answer_for_q1"
        assert out.environment_result["correct"] is True

    def test_multiple_samples_run(self):
        # Samples without expected = all "wrong"
        results = self._pipe().run(["q1", "q2", "q3"])
        assert len(results) == 3
        assert all(r.error is None for r in results)
        assert all(r.output.agent_output.startswith("answer_for_") for r in results)

    def test_data_flows_correctly_through_chain(self):
        results = self._pipe().run(["hello"])
        out = results[0].output
        assert out.sample == "hello"
        assert out.agent_output == "answer_for_hello"
        assert out.environment_result is not None
        assert "correct" in out.environment_result


# ---------------------------------------------------------------------------
# E2E: async_boundary — fire-and-forget pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestAsyncBoundaryPipeline:
    def _pipe(self) -> Pipeline:
        UpdateStep._updates.clear()
        return (
            Pipeline()
            .then(AgentStep())
            .then(EvaluateStep())
            .then(ReflectStep())   # async_boundary = True
            .then(UpdateStep())
        )

    def test_run_returns_before_background_finishes(self):
        pipe = self._pipe()
        t0 = time.monotonic()
        results = pipe.run(["q1"], workers=1)
        foreground_time = time.monotonic() - t0
        # Should return quickly (foreground only: agent + evaluate)
        # Background runs Reflect (0.01s) + Update (0.005s) asynchronously
        assert len(results) == 1
        pipe.wait_for_background(timeout=5.0)

    def test_all_steps_complete_after_wait(self):
        pipe = self._pipe()
        results = pipe.run(["q1", "q2"])
        pipe.wait_for_background(timeout=5.0)
        assert all(r.output is not None for r in results)
        assert all(r.output.skill_manager_output == {"updated": True} for r in results)

    def test_update_serialized_across_samples(self):
        """UpdateStep.max_workers=1 — updates must not interleave."""
        UpdateStep._updates.clear()
        pipe = self._pipe()
        samples = [f"s{i}" for i in range(5)]
        pipe.run(samples, workers=3)
        pipe.wait_for_background(timeout=10.0)
        # All 5 samples must have triggered an update
        assert len(UpdateStep._updates) == 5

    def test_failed_foreground_not_sent_to_background(self):
        class FailEval:
            requires = frozenset({"agent_output"})
            provides = frozenset({"environment_result"})
            def __call__(self, ctx): raise RuntimeError("eval failed")

        UpdateStep._updates.clear()
        pipe = (
            Pipeline()
            .then(AgentStep())
            .then(FailEval())
            .then(ReflectStep())
            .then(UpdateStep())
        )
        results = pipe.run(["q1"])
        pipe.wait_for_background(timeout=2.0)
        # Error in foreground → no background submission
        assert results[0].error is not None
        assert results[0].failed_at == "FailEval"
        assert len(UpdateStep._updates) == 0


# ---------------------------------------------------------------------------
# E2E: Branch inside a pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBranchInPipeline:
    def test_branch_parallel_reflect_and_log(self):
        log_step = LogStep()
        pipe = (
            Pipeline()
            .then(AgentStep())
            .then(EvaluateStep())
            .branch(
                Pipeline().then(MetricStep()),
                Pipeline().then(log_step),
                merge=MergeStrategy.RAISE_ON_CONFLICT,
            )
        )
        results = pipe.run(["hello", "world"])
        assert len(results) == 2
        assert all(r.error is None for r in results)
        # Both branches ran
        assert all(r.output.metadata.get("metric") is not None for r in results)
        assert sorted(log_step.log) == ["hello", "world"]

    def test_step_after_branch_receives_merged_context(self):
        class Summarize:
            requires = frozenset()
            provides = frozenset({"summary"})
            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "summary": "done"})
                )

        pipe = (
            Pipeline()
            .then(AgentStep())
            .branch(
                Pipeline().then(MetricStep()),
                Pipeline().then(LogStep()),
            )
            .then(Summarize())
        )
        results = pipe.run(["test"])
        assert results[0].output.metadata.get("summary") == "done"

    def test_branch_failure_captured_in_sample_result(self):
        class BranchBoom:
            requires = frozenset()
            provides = frozenset()
            def __call__(self, ctx): raise RuntimeError("branch_fail")

        pipe = (
            Pipeline()
            .then(AgentStep())
            .branch(
                Pipeline().then(MetricStep()),
                Pipeline().then(BranchBoom()),
            )
        )
        results = pipe.run(["s"])
        assert results[0].error is not None
        assert results[0].failed_at == "Branch"


# ---------------------------------------------------------------------------
# E2E: nested pipeline reuse
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNestedPipelineReuse:
    def test_inner_pipeline_reused_in_two_outer_pipelines(self):
        inner = Pipeline().then(AgentStep()).then(EvaluateStep())

        outer_a = Pipeline().then(inner)
        outer_b = Pipeline().then(inner).then(MetricStep())

        r_a = outer_a.run(["q1"])
        r_b = outer_b.run(["q1"])

        assert r_a[0].output.agent_output == "answer_for_q1"
        assert r_b[0].output.metadata.get("metric") is not None

    def test_deeply_nested_pipelines(self):
        level1 = Pipeline().then(AgentStep())
        level2 = Pipeline().then(level1).then(EvaluateStep())
        level3 = Pipeline().then(level2).then(MetricStep())

        results = level3.run(["deep"])
        out = results[0].output
        assert out.agent_output == "answer_for_deep"
        assert out.environment_result is not None
        assert out.metadata.get("metric") is not None


# ---------------------------------------------------------------------------
# E2E: multiple run() calls on same instance
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMultipleRunCalls:
    def test_two_run_calls_accumulate_background_work(self):
        UpdateStep._updates.clear()
        pipe = (
            Pipeline()
            .then(AgentStep())
            .then(EvaluateStep())
            .then(ReflectStep())
            .then(UpdateStep())
        )
        pipe.run(["a", "b"])
        pipe.run(["c", "d"])
        pipe.wait_for_background(timeout=10.0)
        assert len(UpdateStep._updates) == 4

    def test_pipeline_state_not_contaminated_between_runs(self):
        pipe = Pipeline().then(AgentStep()).then(EvaluateStep())
        r1 = pipe.run(["sample_x"])
        r2 = pipe.run(["sample_y"])
        assert r1[0].output.sample == "sample_x"
        assert r2[0].output.sample == "sample_y"


# ---------------------------------------------------------------------------
# E2E: async steps inside pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAsyncStepsInPipeline:
    def test_async_step_runs_via_run(self):
        class AsyncAgent:
            requires = frozenset()
            provides = frozenset({"agent_output"})
            async def __call__(self, ctx: StepContext) -> StepContext:
                await asyncio.sleep(0)
                return ctx.replace(agent_output="async_answer")

        results = Pipeline().then(AsyncAgent()).run(["s"])
        assert results[0].output.agent_output == "async_answer"

    def test_mixed_sync_async_steps(self):
        class AsyncAgent:
            requires = frozenset()
            provides = frozenset({"agent_output"})
            async def __call__(self, ctx: StepContext) -> StepContext:
                await asyncio.sleep(0)
                return ctx.replace(agent_output="async")

        class SyncEval:
            requires = frozenset({"agent_output"})
            provides = frozenset({"environment_result"})
            def __call__(self, ctx: StepContext) -> StepContext:
                return ctx.replace(environment_result={"score": 1.0})

        results = (
            Pipeline()
            .then(AsyncAgent())
            .then(SyncEval())
            .run(["s"])
        )
        assert results[0].output.agent_output == "async"
        assert results[0].output.environment_result["score"] == 1.0

    def test_run_async_entry_point(self):
        results = asyncio.run(
            Pipeline().then(AgentStep()).then(EvaluateStep()).run_async(["q1", "q2"])
        )
        assert len(results) == 2
        assert all(r.error is None for r in results)


# ---------------------------------------------------------------------------
# E2E: background executor shared across pipeline instances
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestSharedBackgroundExecutor:
    def test_same_step_class_uses_same_executor(self):
        """Two Pipeline instances sharing the same step class share the executor."""
        class SharedBg:
            requires = frozenset()
            provides = frozenset({"done"})
            async_boundary = True
            max_workers = 1   # single-threaded shared pool
            call_count = 0
            _lock = threading.Lock()

            def __call__(self, ctx):
                with SharedBg._lock:
                    SharedBg.call_count += 1
                time.sleep(0.01)
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "done": True})
                )

        SharedBg.call_count = 0
        # Reset class executor so the test is independent
        if hasattr(SharedBg, "_executor") and SharedBg._executor is not None:
            SharedBg._executor.shutdown(wait=False)
            SharedBg._executor = None

        pipe_a = Pipeline().then(SharedBg())
        pipe_b = Pipeline().then(SharedBg())

        results_a = pipe_a.run(["a1", "a2"])
        results_b = pipe_b.run(["b1", "b2"])

        pipe_a.wait_for_background(timeout=5.0)
        pipe_b.wait_for_background(timeout=5.0)

        assert SharedBg.call_count == 4
        assert all(r.output.metadata.get("done") for r in results_a + results_b)
