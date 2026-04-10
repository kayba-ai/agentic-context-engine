"""Tests for the SkillManager diagnostics pipeline steps."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import SkillManagerOutput
from ace.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace.steps.apply import ApplyStep
from ace.steps.update import UpdateStep

from ace_eval.e2e.diagnostics import (
    DiagnosticUpdateStep,
    SKILL_MANAGER_RAW_METADATA_KEY,
    SkillManagerDiagnosticsWriterStep,
    install_diagnostic_steps,
)

pytestmark = pytest.mark.unit


class FakeSkillManager:
    """SkillManager stub that returns a canned SkillManagerOutput."""

    def __init__(self, *, update: UpdateBatch, raw: dict | None = None) -> None:
        self._update = update
        self._raw = dict(raw) if raw else {}
        self.calls: list[dict] = []

    def update_skills(
        self,
        *,
        reflections,
        skillbook,
        question_context: str,
        progress: str,
        **kwargs,
    ) -> SkillManagerOutput:
        self.calls.append(
            {
                "reflections": reflections,
                "skillbook": skillbook,
                "question_context": question_context,
                "progress": progress,
            }
        )
        output = SkillManagerOutput(update=self._update)
        output.raw = dict(self._raw)
        return output


def _make_ctx(*, trace: dict | None = None) -> ACEStepContext:
    skillbook = Skillbook()
    return ACEStepContext(
        skillbook=SkillbookView(skillbook),
        trace=trace or {"question": "a question", "context": "", "task_id": "task_42"},
        epoch=1,
        total_epochs=1,
        step_index=0,
        total_steps=1,
        global_sample_index=0,
    )


class TestDiagnosticUpdateStep:
    def test_emits_skill_manager_output(self):
        update = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(type="ADD", section="s", content="a" * 100)
            ],
        )
        fm = FakeSkillManager(
            update=update,
            raw={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "best_of_n": {
                    "n_candidates": 3,
                    "n_generated": 3,
                    "scores": [0.7, 0.8, 0.6],
                    "selected_score": 0.8,
                },
            },
        )
        step = DiagnosticUpdateStep(fm)
        ctx = _make_ctx()

        new_ctx = step(ctx)

        assert new_ctx.skill_manager_output is update
        assert len(fm.calls) == 1

    def test_captures_raw_in_metadata(self):
        update = UpdateBatch(reasoning="r", operations=[])
        raw_payload = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
            "best_of_n": {
                "n_candidates": 3,
                "n_generated": 3,
                "scores": [0.4, 0.9, 0.5],
                "selected_score": 0.9,
            },
        }
        fm = FakeSkillManager(update=update, raw=raw_payload)
        step = DiagnosticUpdateStep(fm)

        new_ctx = step(_make_ctx())

        captured = new_ctx.metadata[SKILL_MANAGER_RAW_METADATA_KEY]
        assert captured["usage"]["prompt_tokens"] == 100
        assert captured["best_of_n"]["selected_score"] == 0.9
        # wall_clock_seconds should be added by the step itself
        assert "wall_clock_seconds" in captured

    def test_same_requires_provides_as_update_step(self):
        assert DiagnosticUpdateStep.requires == UpdateStep.requires
        assert DiagnosticUpdateStep.provides == UpdateStep.provides


class TestSkillManagerDiagnosticsWriterStep:
    def test_writes_jsonl_entry(self, tmp_path: Path):
        artifact = tmp_path / "sub" / "diagnostics.jsonl"
        writer = SkillManagerDiagnosticsWriterStep(artifact)

        update = UpdateBatch(
            reasoning="r",
            operations=[
                UpdateOperation(type="ADD", section="s", content="c1"),
                UpdateOperation(type="UPDATE", section="s", content="c2", skill_id="sk1"),
            ],
        )
        raw = {
            "usage": {"prompt_tokens": 55, "completion_tokens": 7, "total_tokens": 62},
            "best_of_n": {
                "n_candidates": 3,
                "n_generated": 3,
                "scores": [0.2, 0.5, 0.8],
                "selected_score": 0.8,
            },
            "wall_clock_seconds": 1.5,
        }
        ctx = ACEStepContext(
            skillbook=SkillbookView(Skillbook()),
            trace={"task_id": "airline_demo"},
            skill_manager_output=update,
            metadata=MappingProxyType({SKILL_MANAGER_RAW_METADATA_KEY: raw}),
            epoch=2,
            total_epochs=3,
            step_index=5,
            total_steps=10,
            global_sample_index=42,
        )

        writer(ctx)

        assert artifact.exists()
        lines = artifact.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["epoch"] == 2
        assert entry["step_index"] == 5
        assert entry["total_steps"] == 10
        assert entry["global_sample_index"] == 42
        assert entry["task_id"] == "airline_demo"
        assert entry["n_operations"] == 2
        assert entry["operation_types"] == ["ADD", "UPDATE"]
        assert entry["used_best_of_n"] is True
        assert entry["n_candidates"] == 3
        assert entry["prompt_tokens"] == 55
        assert entry["completion_tokens"] == 7
        assert entry["total_tokens"] == 62
        assert entry["wall_clock_seconds"] == 1.5
        assert entry["best_of_n"]["selected_score"] == 0.8

    def test_absent_metadata_still_writes_baseline_entry(self, tmp_path: Path):
        artifact = tmp_path / "diagnostics.jsonl"
        writer = SkillManagerDiagnosticsWriterStep(artifact)

        ctx = ACEStepContext(
            skillbook=SkillbookView(Skillbook()),
            trace={"task_id": "demo"},
            skill_manager_output=UpdateBatch(reasoning="r", operations=[]),
            metadata=MappingProxyType({}),
        )
        writer(ctx)

        entry = json.loads(artifact.read_text().strip())
        assert entry["used_best_of_n"] is False
        assert entry["n_candidates"] == 1
        assert entry["n_operations"] == 0


class TestInstallDiagnosticSteps:
    def test_replaces_update_step_and_appends_writer(self, tmp_path: Path):
        update = UpdateBatch(reasoning="r", operations=[])
        fm = FakeSkillManager(update=update)
        skillbook = Skillbook()
        tail = [
            UpdateStep(fm),
            ApplyStep(skillbook),
        ]

        result = install_diagnostic_steps(
            tail, skill_manager=fm, artifact_path=tmp_path / "d.jsonl"
        )

        assert isinstance(result[0], DiagnosticUpdateStep)
        assert isinstance(result[1], ApplyStep)
        assert isinstance(result[2], SkillManagerDiagnosticsWriterStep)

    def test_raises_when_no_update_step(self, tmp_path: Path):
        update = UpdateBatch(reasoning="r", operations=[])
        fm = FakeSkillManager(update=update)
        with pytest.raises(ValueError, match="UpdateStep"):
            install_diagnostic_steps(
                [ApplyStep(Skillbook())],
                skill_manager=fm,
                artifact_path=tmp_path / "d.jsonl",
            )
