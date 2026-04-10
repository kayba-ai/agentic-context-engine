"""Regression tests for the ACEBenchAgent snapshot persistence bug.

Before the fix, ``_get_agent_class()`` created a new class object on
every call. The tau2 registry therefore captured class #1 at register
time, while ``run_task()`` asked ``_get_agent_class()`` for a fresh
class object each time it needed a snapshot — so ``get_snapshot()``
always read empty class-level state and persisted ``agent_config`` as
all-null on every trace.

The fix caches the first materialized class at module level. These
tests lock in the identity invariant and also exercise the
``_serialize_usage`` helper added to preserve per-message token counts.
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_agent_cache():
    """Reset the module-level cache before and after each test.

    ``_get_agent_class()`` populates a module-level singleton the first
    time it is called; tests need a clean slate so results don't leak
    between test methods.
    """
    from ace_eval.e2e import agent as agent_module

    agent_module._CACHED_AGENT_CLASS = None
    yield
    agent_module._CACHED_AGENT_CLASS = None


class TestAgentClassCache:
    def test_same_class_on_repeat_calls(self):
        """Two calls must return the same class object."""
        from ace_eval.e2e.agent import _get_agent_class

        cls_a = _get_agent_class()
        cls_b = _get_agent_class()
        assert cls_a is cls_b

    def test_snapshot_readable_across_calls(self):
        """Snapshot state set on the class object survives re-fetches."""
        from ace_eval.e2e.agent import _get_agent_class

        cls_a = _get_agent_class()
        cls_a._last_system_prompt = "ACE system prompt"
        cls_a._last_model = "bedrock/my-model"
        cls_a._last_model_args = {"temperature": 0.0}

        cls_b = _get_agent_class()
        snap = cls_b.get_snapshot()

        assert snap["system_prompt"] == "ACE system prompt"
        assert snap["model"] == "bedrock/my-model"
        assert snap["model_args"] == {"temperature": 0.0}

    def test_register_and_retrieve_share_class(self):
        """Simulate the tau2 register -> run -> snapshot lifecycle."""
        from ace_eval.e2e.agent import _get_agent_class

        registered_cls = _get_agent_class()
        registered_cls._last_tools = [{"name": "search_flights"}]
        registered_cls._last_runtime_interventions = {"self_verification": True}

        # This mirrors what tau_bench.py does when it reads the snapshot
        lookup_cls = _get_agent_class()
        snapshot = lookup_cls.get_snapshot()

        assert snapshot["tools"] == [{"name": "search_flights"}]
        assert snapshot["runtime_interventions"] == {"self_verification": True}


class TestSerializeUsage:
    def test_dict_input_passthrough(self):
        from ace_eval.e2e.benchmarks.tau_bench import _serialize_usage

        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "extra_field": "kept",
        }
        result = _serialize_usage(usage)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 25
        assert result["total_tokens"] == 125
        assert result["extra_field"] == "kept"

    def test_dict_drops_none_values(self):
        from ace_eval.e2e.benchmarks.tau_bench import _serialize_usage

        usage = {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": 10}
        result = _serialize_usage(usage)
        assert "completion_tokens" not in result
        assert result["prompt_tokens"] == 10
        assert result["total_tokens"] == 10

    def test_pydantic_model_dump(self):
        from ace_eval.e2e.benchmarks.tau_bench import _serialize_usage

        class FakeUsage:
            def model_dump(self, exclude_none: bool = True):
                return {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}

        result = _serialize_usage(FakeUsage())
        assert result == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}

    def test_attribute_fallback(self):
        from ace_eval.e2e.benchmarks.tau_bench import _serialize_usage

        class PlainUsage:
            prompt_tokens = 15
            completion_tokens = 3
            total_tokens = 18
            other = "ignored"

        result = _serialize_usage(PlainUsage())
        assert result == {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}

    def test_missing_fields_return_empty(self):
        from ace_eval.e2e.benchmarks.tau_bench import _serialize_usage

        class EmptyUsage:
            pass

        result = _serialize_usage(EmptyUsage())
        assert result == {}
