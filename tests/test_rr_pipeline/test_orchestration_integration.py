"""Integration tests for RR orchestration with real LLMs.

These tests verify that the orchestration pipeline works end-to-end:
- Orchestrator prompt is understood by the LLM
- Direct analysis path produces valid per-item output
- Worker delegation path works (spawn_analysis + collect_results)
- Result collection, validation, and merge work correctly

Run with: uv run pytest tests/test_rr_pipeline/test_orchestration_integration.py -v -s
"""

from __future__ import annotations

import os

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import ReflectorOutput
from ace.core.skillbook import Skillbook
from ace.rr import RRConfig, RRStep

# Skip if no Bedrock key
BEDROCK_KEY = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
HAS_LLM = bool(BEDROCK_KEY)

MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _make_small_batch(n: int = 3) -> dict:
    """Build a small batch trace for integration testing."""
    return {
        "tasks": [
            {
                "task_id": f"task_{i}",
                "question": f"Customer question {i}: {'Where is my order?' if i == 0 else 'I need a refund' if i == 1 else 'Can I change my address?'}",
                "feedback": f"reward={'1.0' if i % 2 == 0 else '0.0'}",
                "trace": [
                    {
                        "role": "system",
                        "content": "You are a helpful customer service agent.",
                    },
                    {
                        "role": "user",
                        "content": f"Customer question {i}",
                    },
                    {
                        "role": "assistant",
                        "content": f"Agent response {i}: {'I found your order' if i == 0 else 'Processing refund' if i == 1 else 'Address updated'}",
                    },
                ],
            }
            for i in range(n)
        ]
    }


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM, reason="No LLM API key available")
class TestOrchestrationDirectAnalysis:
    """Test the direct analysis path (orchestrator analyzes without spawning workers)."""

    def test_small_batch_direct_analysis(self):
        """Small batch: orchestrator should analyze directly and produce per-item output."""
        rr = RRStep(
            MODEL,
            config=RRConfig(
                orchestrator_max_llm_calls=30,
                max_llm_calls=20,
                enable_subagent=False,
                enable_fallback_synthesis=False,
            ),
        )

        batch = _make_small_batch(2)
        ctx = ACEStepContext(trace=batch, skillbook=SkillbookView(Skillbook()))

        result_ctx = rr(ctx)

        # Should produce per-item reflections
        assert result_ctx.reflections is not None
        assert len(result_ctx.reflections) == 2

        for i, ref in enumerate(result_ctx.reflections):
            assert isinstance(ref, ReflectorOutput)
            assert ref.reasoning, f"Item {i} has empty reasoning"
            # Should have item_id
            assert "item_id" in ref.raw, f"Item {i} missing item_id in raw"

        print("\n=== Direct Analysis Results ===")
        for i, ref in enumerate(result_ctx.reflections):
            print(f"\n--- Item {i} (id={ref.raw.get('item_id')}) ---")
            print(f"  Reasoning: {ref.reasoning[:200]}...")
            print(f"  Key insight: {ref.key_insight[:200]}")
            print(f"  Learnings: {len(ref.extracted_learnings)}")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM, reason="No LLM API key available")
class TestOrchestrationWorkerDelegation:
    """Test the worker delegation path (orchestrator spawns workers)."""

    def test_medium_batch_with_workers(self):
        """Medium batch: orchestrator should spawn workers and merge results.

        Uses sequential workers (max_cluster_workers=1) to avoid rate limits
        and a moderate batch size to encourage delegation.
        """
        rr = RRStep(
            MODEL,
            config=RRConfig(
                orchestrator_max_llm_calls=40,
                max_llm_calls=25,
                max_cluster_workers=1,  # Sequential to avoid rate limits
                worker_collect_timeout=300.0,
                enable_subagent=False,
                worker_enable_subagent=False,
                enable_fallback_synthesis=False,
            ),
        )

        # 4 items — enough to encourage delegation
        batch = _make_small_batch(4)
        ctx = ACEStepContext(trace=batch, skillbook=SkillbookView(Skillbook()))

        result_ctx = rr(ctx)

        assert result_ctx.reflections is not None
        assert len(result_ctx.reflections) == 4

        for i, ref in enumerate(result_ctx.reflections):
            assert isinstance(ref, ReflectorOutput)
            assert ref.reasoning, f"Item {i} has empty reasoning"

        print("\n=== Worker Delegation Results ===")
        for i, ref in enumerate(result_ctx.reflections):
            print(f"\n--- Item {i} (id={ref.raw.get('item_id')}) ---")
            print(f"  Reasoning: {ref.reasoning[:200]}...")
            print(f"  Key insight: {ref.key_insight[:200]}")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LLM, reason="No LLM API key available")
class TestOrchestrationEndToEnd:
    """End-to-end orchestration tests."""

    def test_orchestrator_produces_rr_trace_metadata(self):
        """Orchestrated batch should produce rr_trace metadata on each item."""
        rr = RRStep(
            MODEL,
            config=RRConfig(
                orchestrator_max_llm_calls=25,
                max_llm_calls=15,
                enable_subagent=False,
            ),
        )

        batch = _make_small_batch(2)
        ctx = ACEStepContext(trace=batch, skillbook=SkillbookView(Skillbook()))

        result_ctx = rr(ctx)

        assert len(result_ctx.reflections) == 2
        for ref in result_ctx.reflections:
            assert "rr_trace" in ref.raw
            rr_trace = ref.raw["rr_trace"]
            assert "total_iterations" in rr_trace or "timed_out" in rr_trace

    def test_batch_agents_created_lazily(self):
        """Batch agents should only be created when batch mode is triggered."""
        rr = RRStep(
            MODEL,
            config=RRConfig(
                orchestrator_max_llm_calls=15,
                max_llm_calls=10,
                enable_subagent=False,
            ),
        )

        # Before batch call
        assert rr._orchestrator_agent is None
        assert rr._worker_agent is None

        batch = _make_small_batch(2)
        ctx = ACEStepContext(trace=batch, skillbook=SkillbookView(Skillbook()))

        result_ctx = rr(ctx)

        # After batch call
        assert rr._orchestrator_agent is not None
        assert rr._worker_agent is not None
        assert len(result_ctx.reflections) == 2
