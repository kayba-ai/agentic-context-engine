"""Tests for OpenClaw integration — OpenClawToTraceStep and end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import (
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
)
from ace_next.core.skillbook import Skillbook, UpdateBatch, UpdateOperation
from ace_next.integrations.openclaw import OpenClawToTraceStep
from ace_next.steps import learning_tail
from ace_next.steps.load_traces import LoadTracesStep

from pipeline import Pipeline


# ------------------------------------------------------------------ #
# Helpers — mock roles
# ------------------------------------------------------------------ #


class MockReflector:
    """Minimal mock satisfying ReflectorLike."""

    def __init__(self, output: ReflectorOutput | None = None):
        self.output = output or ReflectorOutput(
            reasoning="test reasoning",
            correct_approach="test approach",
            key_insight="test insight",
            skill_tags=[],
        )
        self.calls: list[dict] = []

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        self.calls.append(
            {
                "question": question,
                "agent_output": agent_output,
                "ground_truth": ground_truth,
                "feedback": feedback,
                **kwargs,
            }
        )
        return self.output


class MockSkillManager:
    """Minimal mock satisfying SkillManagerLike."""

    def __init__(self, output: SkillManagerOutput | None = None):
        self.output = output or SkillManagerOutput(
            update=UpdateBatch(reasoning="test", operations=[]),
        )
        self.calls: list[dict] = []

    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        self.calls.append(
            {
                "reflection": reflection,
                "question_context": question_context,
                "progress": progress,
            }
        )
        return self.output


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a minimal OpenClaw session JSONL file."""
    events = [
        {
            "type": "session",
            "id": "s1",
            "timestamp": "2026-01-01T00:00:00Z",
            "version": 1,
            "cwd": "/app",
        },
        {
            "type": "message",
            "id": "m1",
            "parentId": "s1",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, help me debug this."}],
            },
        },
        {
            "type": "message",
            "id": "m2",
            "parentId": "m1",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze the issue..."},
                    {"type": "text", "text": "I'll help you debug this."},
                    {
                        "type": "toolCall",
                        "id": "tc1",
                        "name": "Read",
                        "arguments": {"file_path": "/app/main.py"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "id": "m3",
            "parentId": "m2",
            "timestamp": "2026-01-01T00:00:03Z",
            "message": {
                "role": "toolResult",
                "content": [
                    {"type": "text", "text": "def main():\n    print('hello')"}
                ],
            },
        },
    ]
    path = tmp_path / "test-session.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


# ------------------------------------------------------------------ #
# OpenClawToTraceStep tests
# ------------------------------------------------------------------ #


class TestOpenClawToTraceStep:
    def test_requires_provides(self):
        step = OpenClawToTraceStep()
        assert step.requires == frozenset({"trace"})
        assert step.provides == frozenset({"trace"})

    def test_converts_to_trace_dict(self):
        """Step should convert raw events into a structured trace dict."""
        raw_events = [
            {"type": "session", "id": "s1", "cwd": "/app"},
            {
                "type": "message",
                "id": "m1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            },
            {
                "type": "message",
                "id": "m2",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there"}],
                },
            },
        ]
        ctx = ACEStepContext(trace=raw_events)
        result = OpenClawToTraceStep()(ctx)

        trace = result.trace
        assert isinstance(trace, dict)
        assert trace["question"] == "User: Hello"
        assert trace["answer"] == "Hi there"
        assert trace["skill_ids"] == []
        assert trace["ground_truth"] is None
        assert "reasoning" in trace
        assert "feedback" in trace

    def test_none_trace(self):
        """Step should handle None trace gracefully."""
        ctx = ACEStepContext(trace=None)
        result = OpenClawToTraceStep()(ctx)
        assert result.trace is None

    def test_empty_list_trace(self):
        """Step should handle empty list trace gracefully."""
        ctx = ACEStepContext(trace=[])
        result = OpenClawToTraceStep()(ctx)
        assert result.trace == []


# ------------------------------------------------------------------ #
# End-to-end: LoadTracesStep → OpenClawToTraceStep → learning_tail
# ------------------------------------------------------------------ #


class TestOpenClawEndToEnd:
    def test_load_and_convert(self, sample_jsonl: Path):
        """LoadTracesStep → OpenClawToTraceStep should produce trace data."""
        load_step = LoadTracesStep()
        convert_step = OpenClawToTraceStep()

        ctx = ACEStepContext(sample=str(sample_jsonl))
        ctx = load_step(ctx)
        assert isinstance(ctx.trace, list)
        assert len(ctx.trace) == 4

        ctx = convert_step(ctx)
        # Converted to structured trace dict
        assert isinstance(ctx.trace, dict)
        assert "question" in ctx.trace
        assert "reasoning" in ctx.trace
        assert "answer" in ctx.trace
        assert ctx.trace["skill_ids"] == []
        assert ctx.trace["ground_truth"] is None

    def test_full_pipeline_with_mocks(self, sample_jsonl: Path):
        """Full pipeline: load → convert → reflect → tag → update → apply."""
        reflector = MockReflector()
        skill_manager = MockSkillManager()
        skillbook = Skillbook()

        load_step = LoadTracesStep()
        convert_step = OpenClawToTraceStep()

        steps = [
            load_step,
            convert_step,
            *learning_tail(reflector, skill_manager, skillbook),
        ]

        pipeline = Pipeline(steps)

        ctx = ACEStepContext(
            sample=str(sample_jsonl),
            skillbook=SkillbookView(skillbook),
        )

        result = pipeline.run([ctx])
        pipeline.wait_for_background()

        assert len(result) == 1
        assert len(reflector.calls) == 1
        assert len(skill_manager.calls) == 1

    def test_pipeline_with_add_operation(self, sample_jsonl: Path):
        """Pipeline with a SkillManager that adds a skill."""
        add_op = UpdateOperation(
            type="ADD",
            section="debugging",
            content="Use structured logging for better debug traces",
            skill_id=None,
            metadata={"helpful": 1, "harmful": 0, "neutral": 0},
        )
        sm_output = SkillManagerOutput(
            update=UpdateBatch(reasoning="Found useful pattern", operations=[add_op]),
        )
        reflector = MockReflector()
        skill_manager = MockSkillManager(output=sm_output)
        skillbook = Skillbook()

        steps = [
            LoadTracesStep(),
            OpenClawToTraceStep(),
            *learning_tail(reflector, skill_manager, skillbook),
        ]

        pipeline = Pipeline(steps)
        ctx = ACEStepContext(
            sample=str(sample_jsonl),
            skillbook=SkillbookView(skillbook),
        )

        pipeline.run([ctx])
        pipeline.wait_for_background()

        # Skillbook should now have one skill
        assert len(skillbook.skills()) == 1
        skill = skillbook.skills()[0]
        assert skill.section == "debugging"
        assert "structured logging" in skill.content

    def test_empty_session_skipped(self, tmp_path: Path):
        """Empty JSONL should produce empty trace."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        load_step = LoadTracesStep()
        convert_step = OpenClawToTraceStep()

        ctx = ACEStepContext(sample=str(path))
        ctx = load_step(ctx)
        assert ctx.trace == []

        ctx = convert_step(ctx)
        assert ctx.trace == []
