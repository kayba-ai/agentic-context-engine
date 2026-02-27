"""Tests for LoadTracesStep — generic JSONL file loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ace_next.core.context import ACEStepContext
from ace_next.steps.load_traces import LoadTracesStep


@pytest.fixture
def load_step():
    return LoadTracesStep()


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL file with valid events."""
    path = tmp_path / "session.jsonl"
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-01-01T00:00:00Z"},
        {
            "type": "message",
            "id": "m1",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        },
        {
            "type": "message",
            "id": "m2",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi there"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


class TestLoadTracesStep:
    def test_requires_provides(self, load_step: LoadTracesStep):
        assert load_step.requires == frozenset({"sample"})
        assert load_step.provides == frozenset({"trace"})

    def test_valid_jsonl(self, load_step: LoadTracesStep, sample_jsonl: Path):
        ctx = ACEStepContext(sample=str(sample_jsonl))
        result = load_step(ctx)

        assert isinstance(result.trace, list)
        assert len(result.trace) == 3
        assert result.trace[0]["type"] == "session"
        assert result.trace[1]["type"] == "message"
        assert result.trace[2]["type"] == "message"

    def test_empty_file(self, load_step: LoadTracesStep, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        ctx = ACEStepContext(sample=str(path))
        result = load_step(ctx)

        assert result.trace == []

    def test_missing_file(self, load_step: LoadTracesStep, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"

        ctx = ACEStepContext(sample=str(path))
        result = load_step(ctx)

        assert result.trace == []

    def test_skips_unparseable_lines(
        self, load_step: LoadTracesStep, tmp_path: Path
    ):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            '{"type": "session", "id": "s1"}\n'
            "this is not json\n"
            '{"type": "message", "id": "m1"}\n'
            "\n"  # blank line
            "also not json\n"
        )

        ctx = ACEStepContext(sample=str(path))
        result = load_step(ctx)

        assert len(result.trace) == 2
        assert result.trace[0]["type"] == "session"
        assert result.trace[1]["type"] == "message"

    def test_preserves_full_event_data(
        self, load_step: LoadTracesStep, tmp_path: Path
    ):
        """Verify no truncation of event fields."""
        long_text = "x" * 10000
        event = {
            "type": "message",
            "id": "m1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": long_text},
                    {
                        "type": "toolCall",
                        "id": "tc1",
                        "name": "Read",
                        "arguments": {"file_path": "/a/b/c", "data": long_text},
                    },
                ],
            },
        }
        path = tmp_path / "full.jsonl"
        path.write_text(json.dumps(event) + "\n")

        ctx = ACEStepContext(sample=str(path))
        result = load_step(ctx)

        assert len(result.trace) == 1
        msg = result.trace[0]["message"]
        assert msg["content"][0]["thinking"] == long_text
        assert msg["content"][1]["arguments"]["data"] == long_text

    def test_returns_new_context(self, load_step: LoadTracesStep, sample_jsonl: Path):
        """Step should return a new context, not mutate the original."""
        ctx = ACEStepContext(sample=str(sample_jsonl))
        result = load_step(ctx)

        assert result is not ctx
        assert ctx.trace is None  # original unchanged
        assert result.trace is not None

    def test_with_real_sample_fixture(self, load_step: LoadTracesStep):
        """Test with the actual sample JSONL from the examples directory."""
        sample = Path(__file__).resolve().parents[1] / "examples" / "openclaw"
        jsonl_files = list(sample.glob("*.jsonl"))
        if not jsonl_files:
            pytest.skip("No sample JSONL files in examples/openclaw/")

        ctx = ACEStepContext(sample=str(jsonl_files[0]))
        result = load_step(ctx)

        assert isinstance(result.trace, list)
        assert len(result.trace) > 0
        # Should contain message events (sample file may have corrupted first line)
        types = {e.get("type") for e in result.trace}
        assert "message" in types
