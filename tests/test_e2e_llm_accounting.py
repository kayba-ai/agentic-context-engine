"""Tests for trace-based LLM accounting fallback in ace_eval.e2e.llm_accounting."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from ace_eval.e2e.llm_accounting import (
    TraceUsageTotals,
    totals_from_messages,
    totals_from_outcome_map,
    totals_from_outcomes,
    totals_from_trace,
    totals_from_trial_map,
)
from ace_eval.e2e.types import TaskOutcome, TraceOutcome


pytestmark = pytest.mark.unit


def _make_outcome(messages: list[dict], task_index: int = 0) -> TaskOutcome:
    return TaskOutcome(
        task_index=task_index,
        reward=1.0,
        trace=MappingProxyType({"messages": messages}),
        outcome=TraceOutcome.PASS,
        wall_clock_seconds=0.0,
    )


class TestTotalsFromMessages:
    def test_empty(self):
        assert totals_from_messages([]) == TraceUsageTotals()

    def test_skips_non_assistant(self):
        messages = [
            {"role": "user", "content": "hi", "usage": {"prompt_tokens": 5}},
            {"role": "tool", "content": "x", "usage": {"prompt_tokens": 9}},
        ]
        totals = totals_from_messages(messages)
        assert totals == TraceUsageTotals()

    def test_sums_assistant_usage(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "ok",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                },
            },
            {
                "role": "assistant",
                "content": "more",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    # no total_tokens -> derived
                },
            },
        ]
        totals = totals_from_messages(messages)
        assert totals.input_tokens == 150
        assert totals.output_tokens == 35
        assert totals.total_tokens == 185  # 125 + (50 + 10)
        assert totals.llm_calls == 2
        assert totals.sources == 1

    def test_ignores_assistant_without_usage(self):
        messages = [{"role": "assistant", "content": "no usage reported"}]
        totals = totals_from_messages(messages)
        assert totals.llm_calls == 0
        assert totals.sources == 0


class TestTotalsFromTrace:
    def test_none_trace(self):
        assert totals_from_trace(None) == TraceUsageTotals()

    def test_trace_without_messages(self):
        assert totals_from_trace({"question": "x"}) == TraceUsageTotals()

    def test_trace_with_messages(self):
        trace = {
            "messages": [
                {
                    "role": "assistant",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            ]
        }
        totals = totals_from_trace(trace)
        assert totals.input_tokens == 10
        assert totals.output_tokens == 5
        assert totals.total_tokens == 15
        assert totals.llm_calls == 1
        assert totals.sources == 1


class TestTotalsFromOutcomes:
    def test_empty_mapping(self):
        assert totals_from_outcome_map({}) == TraceUsageTotals()
        assert totals_from_outcomes([]) == TraceUsageTotals()

    def test_sums_across_outcomes(self):
        outcome_a = _make_outcome(
            [
                {
                    "role": "assistant",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
            ],
            task_index=1,
        )
        outcome_b = _make_outcome(
            [
                {
                    "role": "assistant",
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 4,
                        "total_tokens": 24,
                    },
                },
                {
                    "role": "assistant",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                },
            ],
            task_index=2,
        )
        totals = totals_from_outcome_map({1: outcome_a, 2: outcome_b})
        assert totals.input_tokens == 33
        assert totals.output_tokens == 7
        assert totals.total_tokens == 40
        assert totals.llm_calls == 3
        assert totals.sources == 2  # both outcomes contributed

    def test_trial_map(self):
        outcome_a = _make_outcome(
            [
                {
                    "role": "assistant",
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            ],
            task_index=0,
        )
        outcome_b = _make_outcome(
            [
                {
                    "role": "assistant",
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    },
                }
            ],
            task_index=0,
        )
        totals = totals_from_trial_map({0: [outcome_a, outcome_b]})
        assert totals.llm_calls == 2
        assert totals.input_tokens == 12
        assert totals.output_tokens == 3
