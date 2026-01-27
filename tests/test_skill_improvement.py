"""Tests for the skill improvement integration.

Following project conventions (pytest + unittest.TestCase patterns).
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Type, TypeVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from ace.integrations.claude_code_skill_improvement import (
    ParsedTranscript,
    PublishOutput,
    SkillImprover,
    SkillImprovementResult,
    TranscriptParser,
    TranscriptTooLargeError,
    Turn,
    ToolCall,
    render_full_chat,
    get_transcript_summary,
)
from ace.integrations.claude_code_skill_improvement.improver import (
    _compute_diff,
    _check_diff_threshold,
    _normalize_for_comparison,
)

T = TypeVar("T", bound=BaseModel)


class MockLLMClient:
    """Mock LLM client for testing skill improvement."""

    def __init__(self):
        self._responses = []
        self._call_history = []

    def set_response(self, response: str) -> None:
        """Queue a response."""
        self._responses.append(response)

    def set_responses(self, responses: list) -> None:
        """Queue multiple responses."""
        self._responses.extend(responses)

    def complete(self, prompt: str, **kwargs: Any):
        """Return queued response."""
        self._call_history.append({"prompt": prompt, "kwargs": kwargs})
        if not self._responses:
            raise RuntimeError("No queued responses")
        response = self._responses.pop(0)
        from ace.llm import LLMResponse
        return LLMResponse(text=response)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """Mock structured output."""
        self._call_history.append({
            "prompt": prompt,
            "response_model": response_model,
            "kwargs": kwargs,
        })
        if not self._responses:
            raise RuntimeError("No queued responses")
        response = self._responses.pop(0)
        data = json.loads(response)
        return response_model.model_validate(data)

    @property
    def call_history(self) -> list:
        return self._call_history

    def reset(self) -> None:
        self._responses = []
        self._call_history = []


# Sample transcript data for testing
SAMPLE_TRANSCRIPT_JSONL = """{"type":"user","sessionId":"test-session","cwd":"/test/project","message":{"content":[{"type":"text","text":"Fix the bug in main.py"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"I'll fix the bug."},{"type":"tool_use","id":"tool1","name":"Read","input":{"file_path":"/test/project/main.py"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tool1","content":"def main():\\n    pass"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"I see the issue."},{"type":"tool_use","id":"tool2","name":"Edit","input":{"file_path":"/test/project/main.py","old_string":"pass","new_string":"print('Hello')"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tool2","content":"File edited successfully"}]}}
"""

SAMPLE_SKILL_CONTENT = """---
name: my-skill
description: Test skill
---

# My Skill

## Strategies

- Always read files before editing
- Use descriptive commit messages
"""


class TestTranscriptParser(unittest.TestCase):
    """Tests for TranscriptParser."""

    def setUp(self):
        self.parser = TranscriptParser()

    def test_parse_valid_transcript(self):
        """Test parsing a valid JSONL transcript."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(SAMPLE_TRANSCRIPT_JSONL)
            f.flush()

            transcript = self.parser.parse(f.name)

            assert transcript.session_id == "test-session"
            assert transcript.cwd == "/test/project"
            assert transcript.total_tool_calls == 2
            assert transcript.failed_tool_calls == 0
            assert len(transcript.turns) >= 1

    def test_parse_missing_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse("/nonexistent/path.jsonl")

    def test_parse_empty_transcript(self):
        """Test parsing an empty transcript."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()

            transcript = self.parser.parse(f.name)

            assert transcript.total_tool_calls == 0
            assert len(transcript.turns) == 0

    def test_transcript_too_large(self):
        """Test that TranscriptTooLargeError is raised for oversized files."""
        # Create parser with very low limit
        parser = TranscriptParser(max_size_bytes=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(SAMPLE_TRANSCRIPT_JSONL)
            f.flush()

            with pytest.raises(TranscriptTooLargeError):
                parser.parse(f.name)


class TestRenderFullChat(unittest.TestCase):
    """Tests for render_full_chat function."""

    def test_render_basic_transcript(self):
        """Test rendering a basic transcript."""
        transcript = ParsedTranscript(
            session_id="test-123",
            cwd="/test/dir",
            turns=[
                Turn(
                    user_prompt="Hello",
                    assistant_text="Hi there!",
                    tool_calls=[],
                )
            ],
            total_tool_calls=0,
            successful_tool_calls=0,
            failed_tool_calls=0,
        )

        result = render_full_chat(transcript)

        assert "test-123" in result
        assert "/test/dir" in result
        assert "Hello" in result
        assert "Hi there!" in result

    def test_render_with_tool_calls(self):
        """Test rendering transcript with tool calls."""
        transcript = ParsedTranscript(
            session_id="test-456",
            cwd="/project",
            turns=[
                Turn(
                    user_prompt="Read the file",
                    assistant_text="Reading...",
                    tool_calls=[
                        ToolCall(
                            name="Read",
                            input={"file_path": "/project/main.py"},
                            tool_use_id="tool1",
                            result="content",
                            is_error=False,
                        )
                    ],
                )
            ],
            total_tool_calls=1,
            successful_tool_calls=1,
            failed_tool_calls=0,
        )

        result = render_full_chat(transcript)

        assert "Read" in result
        assert "/project/main.py" in result
        assert "OK" in result

    def test_render_with_error(self):
        """Test rendering transcript with tool error."""
        transcript = ParsedTranscript(
            session_id="test-789",
            cwd="/project",
            turns=[
                Turn(
                    user_prompt="Run command",
                    assistant_text="Running...",
                    tool_calls=[
                        ToolCall(
                            name="Bash",
                            input={"command": "invalid_cmd"},
                            tool_use_id="tool1",
                            result="command not found",
                            is_error=True,
                        )
                    ],
                )
            ],
            total_tool_calls=1,
            successful_tool_calls=0,
            failed_tool_calls=1,
        )

        result = render_full_chat(transcript)

        assert "ERROR" in result
        assert "command not found" in result

    def test_render_determinism(self):
        """Test that rendering is deterministic."""
        transcript = ParsedTranscript(
            session_id="test-det",
            cwd="/test",
            turns=[
                Turn(
                    user_prompt="Test prompt",
                    assistant_text="Test response",
                    tool_calls=[
                        ToolCall(
                            name="Read",
                            input={"file_path": "/test/file.py"},
                            tool_use_id="t1",
                        )
                    ],
                )
            ],
            total_tool_calls=1,
            successful_tool_calls=1,
            failed_tool_calls=0,
        )

        result1 = render_full_chat(transcript)
        result2 = render_full_chat(transcript)

        assert result1 == result2


class TestGetTranscriptSummary(unittest.TestCase):
    """Tests for get_transcript_summary function."""

    def test_summary_basic(self):
        """Test getting summary of a basic transcript."""
        transcript = ParsedTranscript(
            session_id="test-sum",
            cwd="/project",
            turns=[
                Turn(
                    user_prompt="Do something",
                    assistant_text="Done",
                    tool_calls=[
                        ToolCall(name="Read", input={}, tool_use_id="t1"),
                        ToolCall(name="Edit", input={}, tool_use_id="t2"),
                    ],
                )
            ],
            total_tool_calls=2,
            successful_tool_calls=2,
            failed_tool_calls=0,
        )

        summary = get_transcript_summary(transcript)

        assert summary["session_id"] == "test-sum"
        assert summary["cwd"] == "/project"
        assert summary["turns"] == 1
        assert summary["total_tool_calls"] == 2
        assert summary["success_rate"] == 100.0
        assert "Read" in summary["tools_used"]
        assert "Edit" in summary["tools_used"]


class TestDiffComputation(unittest.TestCase):
    """Tests for diff computation utilities."""

    def test_compute_diff_no_changes(self):
        """Test diff computation with no changes."""
        content = "line 1\nline 2\n"
        diff = _compute_diff(content, content)
        assert diff == ""

    def test_compute_diff_with_changes(self):
        """Test diff computation with changes."""
        original = "line 1\nline 2\n"
        updated = "line 1\nline 2 modified\n"
        diff = _compute_diff(original, updated)
        assert "-line 2" in diff
        assert "+line 2 modified" in diff

    def test_check_diff_threshold_acceptable(self):
        """Test diff threshold check with acceptable changes."""
        original = "line 1\nline 2\nline 3\nline 4\nline 5\n"
        updated = "line 1\nline 2 mod\nline 3\nline 4\nline 5\n"
        assert _check_diff_threshold(original, updated, 50.0)

    def test_check_diff_threshold_exceeded(self):
        """Test diff threshold check with excessive changes."""
        original = "line 1\nline 2\n"
        updated = "completely\ndifferent\ncontent\n"
        # This should exceed the threshold with small original
        result = _check_diff_threshold(original, updated, 10.0)
        assert not result

    def test_normalize_for_comparison(self):
        """Test text normalization for duplicate checking."""
        text1 = "  Always use  descriptive names!  "
        text2 = "always use descriptive names"
        assert _normalize_for_comparison(text1) == _normalize_for_comparison(text2)


class TestPublishOutput(unittest.TestCase):
    """Tests for PublishOutput model."""

    def test_publish_output_validation(self):
        """Test PublishOutput model validation."""
        data = {
            "updated_skill_text": "# Updated content",
            "accepted": [
                {"change": "Added strategy", "reason": "Useful", "evidence": "Line 10"}
            ],
            "rejected": [
                {"learning": "Session-specific", "reason": "Too specific"}
            ],
        }
        output = PublishOutput.model_validate(data)

        assert output.updated_skill_text == "# Updated content"
        assert len(output.accepted) == 1
        assert len(output.rejected) == 1

    def test_publish_output_defaults(self):
        """Test PublishOutput default values."""
        data = {"updated_skill_text": "content"}
        output = PublishOutput.model_validate(data)

        assert output.accepted == []
        assert output.rejected == []


class TestSkillImprovementResult(unittest.TestCase):
    """Tests for SkillImprovementResult model."""

    def test_result_no_changes(self):
        """Test result with no changes."""
        result = SkillImprovementResult(
            original_content="content",
            updated_content="content",
            diff="",
            has_changes=False,
        )
        assert not result.has_changes
        assert result.diff == ""

    def test_result_with_changes(self):
        """Test result with changes."""
        result = SkillImprovementResult(
            original_content="old",
            updated_content="new",
            diff="-old\n+new",
            has_changes=True,
            backup_path="/backup/file.md",
        )
        assert result.has_changes
        assert "-old" in result.diff
        assert result.backup_path == "/backup/file.md"


class TestSkillImproverIntegration(unittest.TestCase):
    """Integration tests for SkillImprover with mocked LLM."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMClient()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_files(self):
        """Create test skill and transcript files."""
        skill_path = Path(self.temp_dir) / "SKILL.md"
        skill_path.write_text(SAMPLE_SKILL_CONTENT)

        transcript_path = Path(self.temp_dir) / "session.jsonl"
        transcript_path.write_text(SAMPLE_TRANSCRIPT_JSONL)

        return str(skill_path), str(transcript_path)

    def test_improve_dry_run(self):
        """Test improvement in dry-run mode (no file writes)."""
        skill_path, transcript_path = self._create_test_files()

        # Set up mock responses for Reflector, SkillManager, Publisher
        reflector_response = json.dumps({
            "reasoning": "The session shows good practices",
            "error_identification": "",
            "root_cause_analysis": "",
            "correct_approach": "Reading before editing",
            "key_insight": "Always read files first",
            "extracted_learnings": [
                {"learning": "Read files before editing", "atomicity_score": 0.9, "evidence": "Step 1"}
            ],
            "skill_tags": [],
        })

        skill_manager_response = json.dumps({
            "update": {
                "reasoning": "Adding new insight from session",
                "operations": [
                    {"type": "ADD", "section": "general", "content": "Read files before editing"}
                ]
            },
        })

        publisher_response = json.dumps({
            "updated_skill_text": SAMPLE_SKILL_CONTENT + "\n- Read files before editing them",
            "accepted": [
                {"change": "Added read-first strategy", "reason": "Useful pattern", "evidence": "Step 1-2"}
            ],
            "rejected": [],
        })

        self.mock_llm.set_responses([
            reflector_response,
            skill_manager_response,
            publisher_response,
        ])

        # Create improver with mock LLM
        with patch.object(
            SkillImprover, '__init__',
            lambda self, **kwargs: None
        ):
            improver = SkillImprover.__new__(SkillImprover)
            improver.llm = self.mock_llm
            improver.max_lines_added = 20
            improver.max_diff_percent = 30.0
            improver.transcript_parser = TranscriptParser()

            # Set up required attributes
            from ace.roles import Reflector, SkillManager, AgentOutput
            from ace.skillbook import Skillbook

            improver.reflector = Reflector(self.mock_llm)
            improver.skill_manager = SkillManager(self.mock_llm)
            improver._AgentOutput = AgentOutput
            improver._Skillbook = Skillbook

            result = improver.improve(
                skill_path=skill_path,
                transcript_path=transcript_path,
                dry_run=True,
            )

        # Verify dry run doesn't modify original file
        original_content = Path(skill_path).read_text()
        assert original_content == SAMPLE_SKILL_CONTENT

        # Verify result has changes
        assert result.has_changes
        assert result.diff
        assert result.backup_path is None  # No backup in dry-run

    def test_improve_no_changes(self):
        """Test when no changes are needed."""
        skill_path, transcript_path = self._create_test_files()

        # Set up mock responses that result in no changes
        reflector_response = json.dumps({
            "reasoning": "Session already follows best practices",
            "error_identification": "",
            "root_cause_analysis": "",
            "correct_approach": "Current approach is good",
            "key_insight": "No new insights",
            "extracted_learnings": [],
            "skill_tags": [],
        })

        skill_manager_response = json.dumps({
            "update": {"reasoning": "No changes needed", "operations": []},
        })

        publisher_response = json.dumps({
            "updated_skill_text": SAMPLE_SKILL_CONTENT,
            "accepted": [],
            "rejected": [],
        })

        self.mock_llm.set_responses([
            reflector_response,
            skill_manager_response,
            publisher_response,
        ])

        with patch.object(
            SkillImprover, '__init__',
            lambda self, **kwargs: None
        ):
            improver = SkillImprover.__new__(SkillImprover)
            improver.llm = self.mock_llm
            improver.max_lines_added = 20
            improver.max_diff_percent = 30.0
            improver.transcript_parser = TranscriptParser()

            from ace.roles import Reflector, SkillManager, AgentOutput
            from ace.skillbook import Skillbook

            improver.reflector = Reflector(self.mock_llm)
            improver.skill_manager = SkillManager(self.mock_llm)
            improver._AgentOutput = AgentOutput
            improver._Skillbook = Skillbook

            result = improver.improve(
                skill_path=skill_path,
                transcript_path=transcript_path,
                dry_run=True,
            )

        assert not result.has_changes
        assert result.diff == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
