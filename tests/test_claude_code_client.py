"""Tests for ace/llm_providers/claude_code_client.py."""

import json
import subprocess
import unittest
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestFindClaudeCli(unittest.TestCase):
    """Tests for _find_claude_cli()."""

    @patch(
        "ace.llm_providers.claude_code_client.shutil.which",
        return_value="/usr/bin/claude",
    )
    def test_found_via_which(self, mock_which):
        from ace.llm_providers.claude_code_client import _find_claude_cli

        result = _find_claude_cli()
        self.assertEqual(result, "/usr/bin/claude")

    @patch("ace.llm_providers.claude_code_client.shutil.which", return_value=None)
    @patch("ace.llm_providers.claude_code_client.os.name", "posix")
    def test_not_found_returns_none(self, mock_which):
        from ace.llm_providers.claude_code_client import _find_claude_cli

        result = _find_claude_cli()
        self.assertIsNone(result)


@pytest.mark.unit
class TestIsClaudeCodeCliAvailable(unittest.TestCase):

    def test_returns_bool(self):
        from ace.llm_providers.claude_code_client import is_claude_code_cli_available

        result = is_claude_code_cli_available()
        self.assertIsInstance(result, bool)


@pytest.mark.unit
class TestClaudeCodeLLMClientInit(unittest.TestCase):

    @patch("ace.llm_providers.claude_code_client.CLAUDE_CODE_CLI_AVAILABLE", False)
    def test_raises_when_cli_unavailable(self):
        from ace.llm_providers.claude_code_client import ClaudeCodeLLMClient

        with self.assertRaises(RuntimeError) as ctx:
            ClaudeCodeLLMClient()
        self.assertIn("Claude Code CLI not found", str(ctx.exception))


@pytest.mark.unit
class TestExtractJson(unittest.TestCase):
    """Tests for ClaudeCodeLLMClient._extract_json()."""

    def _get_client_instance(self):
        """Create a client instance without CLI check."""
        from ace.llm_providers.claude_code_client import ClaudeCodeLLMClient

        obj = ClaudeCodeLLMClient.__new__(ClaudeCodeLLMClient)
        return obj

    def test_json_in_markdown_code_block(self):
        client = self._get_client_instance()
        text = '```json\n{"key": "value"}\n```'
        result = client._extract_json(text)
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")

    def test_bare_json_object(self):
        client = self._get_client_instance()
        text = '{"name": "test", "count": 42}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        self.assertEqual(parsed["name"], "test")
        self.assertEqual(parsed["count"], 42)

    def test_nested_brackets_with_escapes(self):
        client = self._get_client_instance()
        text = '{"outer": {"inner": "val\\"ue"}}'
        result = client._extract_json(text)
        parsed = json.loads(result)
        self.assertIn("outer", parsed)

    def test_no_json_returns_original(self):
        client = self._get_client_instance()
        text = "This is just plain text with no JSON"
        result = client._extract_json(text)
        self.assertEqual(result, text.strip())

    def test_json_array(self):
        client = self._get_client_instance()
        text = "[1, 2, 3]"
        result = client._extract_json(text)
        parsed = json.loads(result)
        self.assertEqual(parsed, [1, 2, 3])

    def test_json_with_trailing_text(self):
        client = self._get_client_instance()
        text = '{"key": "value"} and some extra text'
        result = client._extract_json(text)
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")


@pytest.mark.unit
class TestClaudeCodeComplete(unittest.TestCase):
    """Tests for complete() with mocked subprocess."""

    def _make_client(self):
        from ace.llm_providers.claude_code_client import (
            ClaudeCodeLLMClient,
            ClaudeCodeLLMConfig,
        )

        obj = ClaudeCodeLLMClient.__new__(ClaudeCodeLLMClient)
        obj.config = ClaudeCodeLLMConfig()
        obj.model = obj.config.model
        return obj

    @patch("ace.llm_providers.claude_code_client._CLAUDE_CLI_PATH", "/usr/bin/claude")
    @patch("ace.llm_providers.claude_code_client.subprocess.run")
    def test_complete_success(self, mock_run):
        client = self._make_client()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello world",
            stderr="",
        )
        result = client.complete("Say hello")
        self.assertEqual(result.text, "Hello world")
        self.assertFalse(result.raw.get("error", False))

    @patch("ace.llm_providers.claude_code_client._CLAUDE_CLI_PATH", "/usr/bin/claude")
    @patch("ace.llm_providers.claude_code_client.subprocess.run")
    def test_complete_failure(self, mock_run):
        client = self._make_client()
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Something went wrong",
        )
        result = client.complete("Bad prompt")
        self.assertIn("Error", result.text)
        self.assertTrue(result.raw["error"])

    @patch("ace.llm_providers.claude_code_client._CLAUDE_CLI_PATH", "/usr/bin/claude")
    @patch("ace.llm_providers.claude_code_client.subprocess.run")
    def test_complete_with_system_message(self, mock_run):
        client = self._make_client()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="response",
            stderr="",
        )
        client.complete("user prompt", system="system instructions")
        call_kwargs = mock_run.call_args
        prompt_input = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input")
        self.assertIn("system instructions", prompt_input)
        self.assertIn("user prompt", prompt_input)

    @patch("ace.llm_providers.claude_code_client._CLAUDE_CLI_PATH", "/usr/bin/claude")
    @patch("ace.llm_providers.claude_code_client.subprocess.run")
    def test_complete_timeout(self, mock_run):
        client = self._make_client()
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)
        result = client.complete("slow prompt")
        self.assertIn("timed out", result.text)
        self.assertTrue(result.raw["error"])

    @patch("ace.llm_providers.claude_code_client._CLAUDE_CLI_PATH", None)
    def test_complete_cli_not_found(self):
        client = self._make_client()
        result = client.complete("test")
        self.assertIn("not found", result.text)
        self.assertTrue(result.raw["error"])


@pytest.mark.unit
class TestCompleteStructured(unittest.TestCase):
    """Tests for complete_structured() with mocked subprocess."""

    def _make_client(self):
        from ace.llm_providers.claude_code_client import (
            ClaudeCodeLLMClient,
            ClaudeCodeLLMConfig,
        )

        obj = ClaudeCodeLLMClient.__new__(ClaudeCodeLLMClient)
        obj.config = ClaudeCodeLLMConfig()
        obj.model = obj.config.model
        return obj

    @patch("ace.llm_providers.claude_code_client._CLAUDE_CLI_PATH", "/usr/bin/claude")
    @patch("ace.llm_providers.claude_code_client.subprocess.run")
    def test_structured_retry_on_validation_error(self, mock_run):
        """Test that complete_structured retries on parse failure."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str
            value: int

        client = self._make_client()

        # First call returns invalid JSON, second returns valid
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="not json at all", stderr=""),
            MagicMock(
                returncode=0,
                stdout='{"name": "test", "value": 42}',
                stderr="",
            ),
        ]

        result = client.complete_structured("Give me data", MyModel)
        self.assertEqual(result.name, "test")
        self.assertEqual(result.value, 42)
        self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
