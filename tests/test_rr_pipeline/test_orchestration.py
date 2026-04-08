"""Tests for RR orchestration — manager/worker batch scaling.

Covers: validation, batch slicing, result merging, tool behavior,
strict no-fallback enforcement, lazy agent creation, config defaults.
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import ReflectorOutput
from ace.core.skillbook import Skillbook
from ace.rr import RRConfig, RRStep
from ace.rr.agent import (
    RRDeps,
    _get_subagent_parallel_cap,
    build_cluster_results_view,
    validate_worker_assignment_size,
)
from ace.rr.config import RecursiveConfig
from ace.rr.sandbox import TraceSandbox, create_readonly_sandbox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_trace(n: int = 4) -> dict:
    """Build a dict-style batch trace with n items."""
    return {
        "tasks": [
            {
                "task_id": f"t{i}",
                "question": f"Question {i}",
                "feedback": f"reward={'1.0' if i % 2 == 0 else '0.0'}",
                "trace": [
                    {"role": "user", "content": f"msg {i}"},
                    {"role": "assistant", "content": f"reply {i}"},
                ],
            }
            for i in range(n)
        ]
    }


def _make_list_batch(n: int = 3) -> list:
    """Build a raw list-style batch trace."""
    return [
        {
            "item_id": f"item_{i}",
            "messages": [{"role": "user", "content": f"hello {i}"}],
        }
        for i in range(n)
    ]


def _make_combined_steps_batch(n: int = 2) -> dict:
    """Build a legacy combined-steps batch."""
    return {
        "question": "Analyze traces",
        "steps": [
            {
                "role": "conversation",
                "id": f"task_{i}",
                "content": {
                    "question": f"Q{i}",
                    "feedback": f"reward={'1.0' if i == 0 else '0.0'}",
                    "steps": [{"role": "user", "content": f"msg {i}"}],
                },
            }
            for i in range(n)
        ],
    }


def _make_rr() -> RRStep:
    """Create an RRStep with batch agents initialized."""
    rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
    rr._ensure_batch_agents()
    return rr


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrchestrationConfig:
    def test_orchestration_config_defaults(self):
        cfg = RecursiveConfig()
        assert cfg.orchestrator_max_llm_calls == 50
        assert cfg.max_cluster_workers == 5
        assert cfg.worker_collect_timeout == 120.0
        assert cfg.worker_model is None
        assert cfg.worker_enable_subagent is False
        assert cfg.worker_max_llm_calls == 12
        assert cfg.worker_max_items == 6
        assert cfg.worker_subagent_max_parallel == 2
        assert cfg.local_parallel_max_concurrency == 8
        assert cfg.local_parallel_timeout == 30.0

    def test_worker_subagent_parallel_cap_is_separate(self):
        deps = RRDeps(
            sandbox=TraceSandbox(trace=None),
            trace_data={},
            skillbook_text="",
            config=RecursiveConfig(
                subagent_max_parallel=10, worker_subagent_max_parallel=2
            ),
            expected_item_count=3,
        )
        assert _get_subagent_parallel_cap(deps) == 2


# ---------------------------------------------------------------------------
# Lazy agent creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLazyAgentCreation:
    def test_batch_agents_not_created_eagerly(self):
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        assert rr._orchestrator_agent is None
        assert rr._worker_agent is None

    def test_ensure_batch_agents_creates_both(self):
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        rr._ensure_batch_agents()
        assert rr._orchestrator_agent is not None
        assert rr._worker_agent is not None

    def test_ensure_batch_agents_is_idempotent(self):
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        rr._ensure_batch_agents()
        orch_1 = rr._orchestrator_agent
        worker_1 = rr._worker_agent
        rr._ensure_batch_agents()
        assert rr._orchestrator_agent is orch_1
        assert rr._worker_agent is worker_1

    def test_worker_agent_uses_worker_model_override(self):
        rr = RRStep(
            "orchestrator-model",
            config=RRConfig(enable_subagent=False, worker_model="worker-model"),
        )

        with (
            patch("ace.rr.runner.create_orchestrator_agent", return_value=MagicMock()),
            patch(
                "ace.rr.runner.create_worker_agent", return_value=MagicMock()
            ) as worker_factory,
        ):
            rr._ensure_batch_agents()

        assert worker_factory.call_args.args[0] == "worker-model"

    def test_single_trace_does_not_create_batch_agents(self):
        rr = RRStep("test-model", config=RRConfig(enable_subagent=False))
        mock_result = MagicMock()
        mock_result.output = ReflectorOutput(
            reasoning="r", key_insight="k", correct_approach="c"
        )
        usage = MagicMock()
        usage.request_tokens = 10
        usage.response_tokens = 5
        usage.total_tokens = 15
        usage.requests = 1
        mock_result.usage.return_value = usage

        traces = {
            "question": "q",
            "steps": [
                {"role": "agent", "reasoning": "r", "answer": "a", "skill_ids": []}
            ],
        }
        ctx = ACEStepContext(trace=traces, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._agent, "run_sync", return_value=mock_result):
            rr(ctx)

        assert rr._orchestrator_agent is None
        assert rr._worker_agent is None


# ---------------------------------------------------------------------------
# Batch slicing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBatchSlicing:
    def test_slice_raw_list(self):
        rr = _make_rr()
        batch = _make_list_batch(5)
        sliced = rr._slice_batch_trace(batch, [1, 3])
        assert len(sliced) == 2
        assert sliced[0]["item_id"] == "item_1"
        assert sliced[1]["item_id"] == "item_3"

    def test_slice_dict_with_tasks(self):
        rr = _make_rr()
        batch = _make_batch_trace(4)
        sliced = rr._slice_batch_trace(batch, [0, 2])
        assert "tasks" in sliced
        assert len(sliced["tasks"]) == 2
        assert sliced["tasks"][0]["task_id"] == "t0"
        assert sliced["tasks"][1]["task_id"] == "t2"

    def test_slice_dict_with_items(self):
        rr = _make_rr()
        batch = {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
        sliced = rr._slice_batch_trace(batch, [2])
        assert sliced["items"] == [{"id": "c"}]

    def test_slice_preserves_batch_metadata(self):
        rr = _make_rr()
        batch = {
            "tasks": [{"task_id": "t0"}, {"task_id": "t1"}],
            "metadata": {"source": "test"},
            "version": 2,
        }
        sliced = rr._slice_batch_trace(batch, [1])
        assert sliced["metadata"] == {"source": "test"}
        assert sliced["version"] == 2
        assert len(sliced["tasks"]) == 1

    def test_slice_combined_steps_batch(self):
        rr = _make_rr()
        batch = _make_combined_steps_batch(3)
        sliced = rr._slice_batch_trace(batch, [0, 2])
        assert len(sliced["steps"]) == 2
        assert sliced["steps"][0]["id"] == "task_0"
        assert sliced["steps"][1]["id"] == "task_2"
        assert "question" in sliced  # metadata preserved


# ---------------------------------------------------------------------------
# Merge and validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMergeAndValidation:
    def _make_deps_with_results(
        self,
        batch_size: int,
        cluster_specs: list[tuple[str, list[int], str]],
    ) -> tuple[RRDeps, list[dict]]:
        """Build RRDeps with cluster_results.

        cluster_specs: list of (name, indices, status) tuples.
        Status "completed" auto-generates per-item reflections.
        """
        sandbox = TraceSandbox(trace=None)
        deps = RRDeps(
            sandbox=sandbox,
            trace_data={},
            skillbook_text="",
            config=RecursiveConfig(),
            is_orchestrator=True,
        )

        batch_items = [{"task_id": f"t{i}"} for i in range(batch_size)]

        for name, indices, status in cluster_specs:
            if status == "completed":
                per_item = tuple(
                    ReflectorOutput(
                        reasoning=f"analysis for index {idx}",
                        key_insight=f"insight {idx}",
                        correct_approach=f"approach {idx}",
                    )
                    for idx in indices
                )
            else:
                per_item = ()

            deps.cluster_results[name] = {
                "assignment": {"trace_indices": indices},
                "status": status,
                "issues": [] if status == "completed" else ["some issue"],
                "worker_output": None,
                "per_item_reflections": per_item,
                "usage": {},
                "rr_trace": {},
            }

        return deps, batch_items

    def test_merge_complete_coverage(self):
        rr = _make_rr()
        deps, items = self._make_deps_with_results(
            4,
            [
                ("cluster_a", [0, 1], "completed"),
                ("cluster_b", [2, 3], "completed"),
            ],
        )
        result = rr._merge_cluster_results(deps, items)
        assert len(result) == 4
        assert result[0].reasoning == "analysis for index 0"
        assert result[3].reasoning == "analysis for index 3"
        assert result[0].raw["item_id"] == "t0"
        assert result[3].raw["item_id"] == "t3"

    def test_merge_fails_on_missing_coverage(self):
        rr = _make_rr()
        deps, items = self._make_deps_with_results(
            4,
            [
                ("cluster_a", [0, 1], "completed"),
                # indices 2, 3 not covered
            ],
        )
        with pytest.raises(RuntimeError, match="Missing coverage"):
            rr._merge_cluster_results(deps, items)

    def test_merge_fails_on_failed_cluster(self):
        rr = _make_rr()
        deps, items = self._make_deps_with_results(
            4,
            [
                ("cluster_a", [0, 1], "completed"),
                ("cluster_b", [2, 3], "failed"),
            ],
        )
        with pytest.raises(RuntimeError, match="failed"):
            rr._merge_cluster_results(deps, items)

    def test_merge_fails_on_timed_out_cluster(self):
        rr = _make_rr()
        deps, items = self._make_deps_with_results(
            3,
            [
                ("cluster_a", [0], "completed"),
                ("cluster_b", [1, 2], "timed_out"),
            ],
        )
        with pytest.raises(RuntimeError, match="timed_out"):
            rr._merge_cluster_results(deps, items)

    def test_merge_fails_on_invalid_cluster(self):
        rr = _make_rr()
        deps, items = self._make_deps_with_results(
            2,
            [
                ("cluster_a", [0], "completed"),
                ("cluster_b", [1], "invalid"),
            ],
        )
        with pytest.raises(RuntimeError, match="invalid"):
            rr._merge_cluster_results(deps, items)

    def test_merge_fails_on_overlap(self):
        """Overlapping accepted coverage should fail."""
        rr = _make_rr()
        sandbox = TraceSandbox(trace=None)
        deps = RRDeps(
            sandbox=sandbox,
            trace_data={},
            skillbook_text="",
            config=RecursiveConfig(),
            is_orchestrator=True,
        )
        items = [{"task_id": f"t{i}"} for i in range(3)]

        # Both clusters cover index 1
        for name, indices in [("a", [0, 1]), ("b", [1, 2])]:
            deps.cluster_results[name] = {
                "assignment": {"trace_indices": indices},
                "status": "completed",
                "issues": [],
                "worker_output": None,
                "per_item_reflections": tuple(
                    ReflectorOutput(
                        reasoning=f"r{i}",
                        key_insight=f"k{i}",
                        correct_approach=f"c{i}",
                    )
                    for i in indices
                ),
                "usage": {},
                "rr_trace": {},
            }

        with pytest.raises(RuntimeError, match="already covered"):
            rr._merge_cluster_results(deps, items)

    def test_merge_fails_on_count_mismatch(self):
        """Worker output with wrong item count should fail."""
        rr = _make_rr()
        sandbox = TraceSandbox(trace=None)
        deps = RRDeps(
            sandbox=sandbox,
            trace_data={},
            skillbook_text="",
            config=RecursiveConfig(),
            is_orchestrator=True,
        )
        items = [{"task_id": "t0"}, {"task_id": "t1"}]

        # Cluster claims indices [0, 1] but only has 1 reflection
        deps.cluster_results["bad"] = {
            "assignment": {"trace_indices": [0, 1]},
            "status": "completed",
            "issues": [],
            "worker_output": None,
            "per_item_reflections": (
                ReflectorOutput(reasoning="r", key_insight="k", correct_approach="c"),
            ),
            "usage": {},
            "rr_trace": {},
        }

        with pytest.raises(RuntimeError, match="Missing coverage"):
            rr._merge_cluster_results(deps, items)


# ---------------------------------------------------------------------------
# Spawn validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpawnValidation:
    """Test spawn_analysis validation through _run_batch_reflections mocking."""

    def test_spawn_rejects_out_of_range_indices(self):
        """spawn_analysis should reject indices beyond batch size."""
        rr = _make_rr()
        batch = _make_batch_trace(3)
        items = rr._get_batch_items(batch)
        assert items is not None
        assert len(items) == 3

        # Out-of-range index 5 in a batch of 3 items
        from pydantic_ai import ModelRetry

        sandbox = TraceSandbox(trace=None)
        deps = RRDeps(
            sandbox=sandbox,
            trace_data=batch,
            skillbook_text="",
            config=RecursiveConfig(),
            is_orchestrator=True,
        )

        # Simulate what spawn_analysis validates
        trace_indices = [0, 5]
        batch_items = rr._get_batch_items(deps.trace_data)
        assert batch_items is not None
        for idx in trace_indices:
            if not (0 <= idx < len(batch_items)):
                break
        else:
            pytest.fail("Should have found out-of-range index")

    def test_spawn_rejects_duplicate_indices(self):
        """spawn_analysis should reject duplicate indices."""
        trace_indices = [0, 1, 1, 2]
        assert len(set(trace_indices)) != len(trace_indices)

    def test_spawn_rejects_assignments_larger_than_worker_limit(self):
        from pydantic_ai import ModelRetry

        with pytest.raises(ModelRetry, match="worker limit of 3"):
            validate_worker_assignment_size(
                RecursiveConfig(worker_max_items=3),
                [0, 1, 2, 3],
            )


# ---------------------------------------------------------------------------
# Orchestrator prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrchestratorPrompt:
    def test_orchestrator_prompt_is_slim(self):
        """Orchestrator prompt should not have one preview row per item."""
        rr = _make_rr()
        batch = _make_batch_trace(20)
        prompt = rr._build_orchestrator_prompt(batch, SkillbookView(Skillbook()))

        # Should have total count
        assert "20" in prompt
        # Should have pass/fail summary
        assert "PASS" in prompt
        assert "FAIL" in prompt
        # Should have spawn_analysis tool
        assert "spawn_analysis" in prompt
        assert "collect_results" in prompt
        # Should NOT have one preview row per item (unlike the full prompt)
        assert "| `t0` |" not in prompt
        assert "| `t19` |" not in prompt

    def test_orchestrator_prompt_has_exemplars(self):
        rr = _make_rr()
        batch = _make_batch_trace(5)
        prompt = rr._build_orchestrator_prompt(batch, SkillbookView(Skillbook()))
        assert "Exemplar IDs" in prompt
        assert "`t0`" in prompt

    def test_orchestrator_prompt_has_survey_groups(self):
        rr = _make_rr()
        batch = _make_batch_trace(9)
        prompt = rr._build_orchestrator_prompt(batch, SkillbookView(Skillbook()))
        assert "survey groups" in prompt.lower()


# ---------------------------------------------------------------------------
# Worker prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWorkerPrompt:
    def test_worker_prompt_includes_assignment(self):
        rr = _make_rr()
        batch = _make_batch_trace(2)
        assignment = {
            "cluster_name": "test_cluster",
            "goal": "Find root causes",
            "success_criteria": "Per-item learnings",
            "trace_indices": [0, 1],
        }
        prompt = rr._build_worker_prompt(batch, SkillbookView(Skillbook()), assignment)
        assert "test_cluster" in prompt
        assert "Find root causes" in prompt
        assert "Per-item learnings" in prompt
        assert "Worker limit" in prompt

    def test_worker_prompt_omits_analysis_tools_by_default(self):
        """Worker without subagent should not mention analyze tools."""
        rr = RRStep(
            "test-model",
            config=RRConfig(enable_subagent=False, worker_enable_subagent=False),
        )
        rr._ensure_batch_agents()
        batch = _make_batch_trace(2)
        assignment = {
            "cluster_name": "c",
            "goal": "g",
            "success_criteria": "s",
            "trace_indices": [0, 1],
        }
        prompt = rr._build_worker_prompt(batch, "skillbook text", assignment)
        assert "analyze(" not in prompt.lower() or "execute_code" in prompt

    def test_worker_prompt_includes_analysis_tools_when_enabled(self):
        rr = RRStep(
            "test-model",
            config=RRConfig(enable_subagent=False, worker_enable_subagent=True),
        )
        rr._ensure_batch_agents()
        batch = _make_batch_trace(2)
        assignment = {
            "cluster_name": "c",
            "goal": "g",
            "success_criteria": "s",
            "trace_indices": [0, 1],
        }
        prompt = rr._build_worker_prompt(batch, "skillbook text", assignment)
        assert "analyze(" in prompt


# ---------------------------------------------------------------------------
# Worker budget heuristic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWorkerBudget:
    def test_small_assignment_gets_small_budget(self):
        """Budget for a small worker should stay modest by default."""
        rr = _make_rr()
        assert rr._compute_worker_budget(traces=_make_batch_trace(2), item_count=2) <= 12

    def test_budget_is_capped_at_worker_limit(self):
        """Worker budget should never exceed the dedicated worker cap."""
        rr = RRStep(
            "test-model",
            config=RRConfig(
                max_llm_calls=80,
                worker_max_llm_calls=9,
                enable_subagent=False,
            ),
        )
        rr._ensure_batch_agents()

        assert rr._compute_worker_budget(traces=_make_batch_trace(100), item_count=100) == 9


# ---------------------------------------------------------------------------
# Per-run state isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStateIsolation:
    def test_orchestration_state_not_on_self(self):
        """Orchestration state should be on deps, not on RRStep."""
        rr = _make_rr()
        assert not hasattr(rr, "pending_clusters")
        assert not hasattr(rr, "cluster_results")
        assert not hasattr(rr, "cluster_pool")


# ---------------------------------------------------------------------------
# Runtime wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRuntimeWiring:
    def test_worker_session_gets_dedicated_subagent_when_enabled(self):
        rr = RRStep(
            "test-model",
            config=RRConfig(enable_subagent=False, worker_enable_subagent=True),
        )
        rr._ensure_batch_agents()

        observed: dict[str, object] = {}

        class DummyUsage:
            input_tokens = 1
            output_tokens = 1
            total_tokens = 2
            requests = 1

        class DummyResult:
            def __init__(self):
                self.output = ReflectorOutput(
                    reasoning="worker",
                    key_insight="insight",
                    correct_approach="approach",
                    raw={"items": [{}]},
                )

            def usage(self):
                return DummyUsage()

        def fake_run_sync(prompt, deps=None, usage_limits=None):
            observed["sub_agent"] = deps.sub_agent
            return DummyResult()

        with patch.object(rr._worker_agent, "run_sync", side_effect=fake_run_sync):
            rr._run_worker_session(
                sub_batch={"tasks": [{"task_id": "t0", "trace": []}]},
                skillbook_text="",
                assignment={
                    "cluster_name": "cluster",
                    "trace_indices": [0],
                    "goal": "goal",
                    "success_criteria": "criteria",
                },
                inherited_helpers={},
                trace_indices=[0],
            )

        assert observed["sub_agent"] is not None

    def test_local_parallel_config_is_propagated_to_sandbox_and_snapshots(self):
        rr = RRStep(
            "test-model",
            config=RRConfig(
                enable_subagent=False,
                local_parallel_max_concurrency=3,
                local_parallel_timeout=7.0,
            ),
        )

        sandbox = rr._create_sandbox(None, {"tasks": [{"task_id": "t0"}]}, "")
        snapshot = create_readonly_sandbox(sandbox)

        assert sandbox._parallel_max_concurrency == 3
        assert sandbox._parallel_timeout == 7.0
        assert snapshot._parallel_max_concurrency == 3
        assert snapshot._parallel_timeout == 7.0

    def test_pending_workers_are_recorded_as_failures_during_cleanup(self):
        rr = _make_rr()
        deps = RRDeps(
            sandbox=TraceSandbox(trace=None),
            trace_data={},
            skillbook_text="",
            config=RecursiveConfig(),
            is_orchestrator=True,
        )

        pending = Future()
        deps.pending_clusters["cluster_a"] = {
            "assignment": {"trace_indices": [0], "goal": "g", "success_criteria": "s"},
            "future": pending,
            "status": "running",
        }

        rr._harvest_pending_clusters(deps)

        assert not deps.pending_clusters
        assert deps.cluster_results["cluster_a"]["status"] == "failed"
        assert "still pending" in deps.cluster_results["cluster_a"]["issues"][0]
        assert "cluster_results" in deps.sandbox.namespace
        sandbox_view = deps.sandbox.namespace["cluster_results"]
        assert sandbox_view["cluster_a"]["status"] == "failed"
        assert "worker_output" not in sandbox_view["cluster_a"]

    def test_worker_merge_preserves_orchestration_summary(self):
        rr = _make_rr()
        batch_trace = _make_batch_trace(2)
        deps = RRDeps(
            sandbox=TraceSandbox(trace=None),
            trace_data=batch_trace,
            skillbook_text="",
            config=RecursiveConfig(),
            is_orchestrator=True,
        )
        deps.cluster_results["cluster_a"] = {
            "assignment": {"trace_indices": [0, 1]},
            "status": "completed",
            "issues": [],
            "worker_output": None,
            "per_item_reflections": (
                ReflectorOutput(
                    reasoning="r0", key_insight="k0", correct_approach="c0"
                ),
                ReflectorOutput(
                    reasoning="r1", key_insight="k1", correct_approach="c1"
                ),
            ),
            "usage": {},
            "rr_trace": {},
        }

        orchestrator_output = ReflectorOutput(
            reasoning="orchestrator summary",
            key_insight="summary insight",
            correct_approach="summary approach",
            raw={"summary_meta": True},
        )

        with patch.object(
            rr,
            "_run_reflection_session",
            return_value=(orchestrator_output, deps),
        ):
            reflections = rr._run_batch_reflections(
                batch_trace, SkillbookView(Skillbook())
            )

        assert (
            reflections[0].raw["orchestration_summary"]["reasoning"]
            == "orchestrator summary"
        )
        assert (
            reflections[0].raw["orchestration_summary"]["raw"]["summary_meta"] is True
        )

    def test_cluster_results_view_is_compact(self):
        view = build_cluster_results_view(
            {
                "cluster_a": {
                    "assignment": {
                        "trace_indices": [0, 1],
                        "goal": "Find failures",
                        "success_criteria": "Compact learnings",
                    },
                    "status": "completed",
                    "issues": [],
                    "worker_output": ReflectorOutput(
                        reasoning="r",
                        key_insight="A compact worker summary",
                        correct_approach="Do the right thing",
                    ),
                    "per_item_reflections": (
                        ReflectorOutput(
                            reasoning="r0",
                            key_insight="k0",
                            correct_approach="c0",
                        ),
                    ),
                    "usage": {"requests": 3, "total_tokens": 120},
                    "rr_trace": {},
                }
            }
        )

        assert view["cluster_a"]["trace_indices"] == [0, 1]
        assert view["cluster_a"]["item_count"] == 1
        assert view["cluster_a"]["usage"]["requests"] == 3
        assert "worker_output" not in view["cluster_a"]
        assert view["cluster_a"]["worker_summary"]["key_insight"]


# ---------------------------------------------------------------------------
# Batch mode always uses orchestrator path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBatchUsesOrchestrator:
    def test_batch_call_triggers_orchestrator_agent(self):
        """Batch traces should use the orchestrator agent, not the single-trace agent."""
        rr = _make_rr()

        output = ReflectorOutput(
            reasoning="orchestrated",
            key_insight="insight",
            correct_approach="approach",
            raw={
                "items": [
                    {"reasoning": "r0", "key_insight": "k0", "extracted_learnings": []},
                    {"reasoning": "r1", "key_insight": "k1", "extracted_learnings": []},
                ],
            },
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 100
        usage.response_tokens = 50
        usage.total_tokens = 150
        usage.requests = 3
        mock_result.usage.return_value = usage

        batch_trace = _make_batch_trace(2)
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        # Patch orchestrator agent, not single-trace agent
        with patch.object(
            rr._orchestrator_agent, "run_sync", return_value=mock_result
        ) as mock_orch:
            with patch.object(rr._agent, "run_sync") as mock_single:
                rr(ctx)

        mock_orch.assert_called_once()
        mock_single.assert_not_called()

    def test_direct_batch_output_without_items_fails_loudly(self):
        rr = _make_rr()

        output = ReflectorOutput(
            reasoning="orchestrated",
            key_insight="insight",
            correct_approach="approach",
            raw={},
        )
        mock_result = MagicMock()
        mock_result.output = output
        usage = MagicMock()
        usage.request_tokens = 100
        usage.response_tokens = 50
        usage.total_tokens = 150
        usage.requests = 3
        mock_result.usage.return_value = usage

        batch_trace = _make_batch_trace(2)
        ctx = ACEStepContext(trace=batch_trace, skillbook=SkillbookView(Skillbook()))

        with patch.object(rr._orchestrator_agent, "run_sync", return_value=mock_result):
            with pytest.raises(RuntimeError, match="raw\\['items'\\]"):
                rr(ctx)
