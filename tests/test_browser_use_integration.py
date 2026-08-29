"""Tests for the browser-use integration steps."""

from __future__ import annotations

from typing import Any

import pytest

from ace.core.context import ACEStepContext
from ace.integrations import browser_use as browser_use_integration


class FakeHistory:
    """Small AgentHistoryList stand-in for semantic outcome tests."""

    def __init__(self, semantic_success: bool | None) -> None:
        self.semantic_success = semantic_success
        self.history: list[Any] = []

    def is_successful(self) -> bool | None:
        return self.semantic_success

    def final_result(self) -> str:
        return "partial browser output"

    def number_of_steps(self) -> int:
        return 2

    def total_duration_seconds(self) -> float:
        return 1.25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("semantic_success", "expected_success"),
    [(True, True), (False, False), (None, False)],
)
async def test_execute_step_uses_browser_task_outcome(
    monkeypatch: pytest.MonkeyPatch,
    semantic_success: bool | None,
    expected_success: bool,
) -> None:
    """A returned history is not proof that the browser task succeeded."""

    history = FakeHistory(semantic_success)

    class FakeAgent:
        def __init__(self, **_: Any) -> None:
            pass

        async def run(self) -> FakeHistory:
            return history

    monkeypatch.setattr(browser_use_integration, "BROWSER_USE_AVAILABLE", True)
    monkeypatch.setattr(browser_use_integration, "Agent", FakeAgent)

    step = browser_use_integration.BrowserExecuteStep(browser_llm=object())
    result_context = await step(ACEStepContext(sample="Find the top HN post"))

    assert isinstance(result_context.trace, browser_use_integration.BrowserResult)
    assert result_context.trace.success is expected_success
