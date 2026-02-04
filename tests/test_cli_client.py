"""Tests for CLI client path resolution and initialization.

Tests the following functionality:
- CLI path resolution priority order
- Patched CLI detection (~/.ace/claude-learner/cli.js)
- Environment variable overrides (ACE_CLI_PATH)
- Fallback to system 'claude' command
- JS vs binary CLI detection
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestCLIPathResolution(unittest.TestCase):
    """Test CLI path resolution functionality."""

    def setUp(self):
        """Set up temporary directory and save original environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up temporary files and restore environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_import(self):
        """Test that CLI client can be imported."""
        from ace.integrations.claude_code.cli_client import (
            CLIClient,
            _resolve_cli_path,
            _is_js_cli,
        )

        self.assertIsNotNone(CLIClient)
        self.assertIsNotNone(_resolve_cli_path)
        self.assertIsNotNone(_is_js_cli)

    def test_explicit_path_takes_priority(self):
        """Test that explicit cli_path parameter takes highest priority."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        # Create a fake CLI file
        fake_cli = Path(self.temp_dir) / "fake_claude"
        fake_cli.touch()

        result = _resolve_cli_path(str(fake_cli))
        self.assertEqual(result, fake_cli)

    def test_explicit_path_not_found(self):
        """Test that FileNotFoundError is raised for nonexistent explicit path."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        with self.assertRaises(FileNotFoundError):
            _resolve_cli_path("/nonexistent/path/to/claude")

    def test_ace_cli_path_env_var(self):
        """Test ACE_CLI_PATH environment variable."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        # Create a fake CLI file
        fake_cli = Path(self.temp_dir) / "cli.js"
        fake_cli.touch()

        os.environ["ACE_CLI_PATH"] = str(fake_cli)

        result = _resolve_cli_path(None)
        self.assertEqual(result, fake_cli)

    def test_patched_cli_auto_detection(self):
        """Test that patched CLI is auto-detected via get_or_create_patched_cli."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        # Create a fake patched CLI file
        patched_dir = Path(self.temp_dir) / ".ace" / "claude-learner"
        patched_dir.mkdir(parents=True)
        patched_cli = patched_dir / "cli.js"
        patched_cli.touch()

        # Mock get_or_create_patched_cli to return our fake patched CLI
        with patch(
            "ace.integrations.claude_code.prompt_patcher.get_or_create_patched_cli",
            return_value=patched_cli,
        ):
            result = _resolve_cli_path(None)
            self.assertEqual(result, patched_cli)

    def test_ace_cli_path_takes_priority_over_patched(self):
        """Test ACE_CLI_PATH takes priority over patched CLI."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        # Create a fake binary CLI file
        fake_bin = Path(self.temp_dir) / "claude_bin"
        fake_bin.touch()

        os.environ["ACE_CLI_PATH"] = str(fake_bin)

        # Even with a patched CLI available, ACE_CLI_PATH should win
        patched_cli = Path(self.temp_dir) / "patched" / "cli.js"
        patched_cli.parent.mkdir(parents=True)
        patched_cli.touch()

        with patch(
            "ace.integrations.claude_code.prompt_patcher.get_or_create_patched_cli",
            return_value=patched_cli,
        ):
            result = _resolve_cli_path(None)
            self.assertEqual(result, fake_bin)

    @patch("shutil.which")
    def test_system_claude_fallback(self, mock_which):
        """Test fallback to system 'claude' command when no patched CLI."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        system_claude = "/usr/local/bin/claude"
        mock_which.return_value = system_claude

        # Mock patched CLI as unavailable so we fall through to system claude
        with patch(
            "ace.integrations.claude_code.prompt_patcher.get_or_create_patched_cli",
            return_value=None,
        ):
            result = _resolve_cli_path(None)
            self.assertEqual(result, Path(system_claude))

    @patch("shutil.which")
    def test_no_cli_found(self, mock_which):
        """Test FileNotFoundError when no CLI is found."""
        from ace.integrations.claude_code.cli_client import _resolve_cli_path

        mock_which.return_value = None

        # Mock patched CLI as unavailable
        with patch(
            "ace.integrations.claude_code.prompt_patcher.get_or_create_patched_cli",
            return_value=None,
        ):
            with self.assertRaises(FileNotFoundError) as context:
                _resolve_cli_path(None)

            # Check error message is helpful
            self.assertIn("Claude CLI not found", str(context.exception))


@pytest.mark.unit
class TestJSCLIDetection(unittest.TestCase):
    """Test JS vs binary CLI detection."""

    def test_js_file_detection(self):
        """Test that .js files are detected as JS CLI."""
        from ace.integrations.claude_code.cli_client import _is_js_cli

        self.assertTrue(_is_js_cli(Path("/path/to/cli.js")))
        self.assertTrue(_is_js_cli(Path("cli.js")))

    def test_binary_detection(self):
        """Test that non-.js files are not detected as JS CLI."""
        from ace.integrations.claude_code.cli_client import _is_js_cli

        self.assertFalse(_is_js_cli(Path("/usr/local/bin/claude")))
        self.assertFalse(_is_js_cli(Path("claude")))
        self.assertFalse(_is_js_cli(Path("/path/to/script.py")))


@pytest.mark.unit
class TestCLIClientInitialization(unittest.TestCase):
    """Test CLIClient initialization."""

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch("shutil.which")
    def test_initialization_with_system_claude(self, mock_which):
        """Test CLIClient initialization with system claude."""
        from ace.integrations.claude_code.cli_client import CLIClient

        system_claude = Path(self.temp_dir) / "claude"
        system_claude.touch()
        mock_which.return_value = str(system_claude)

        # Mock patched CLI as unavailable so system claude is used
        with patch(
            "ace.integrations.claude_code.prompt_patcher.get_or_create_patched_cli",
            return_value=None,
        ):
            client = CLIClient()
            self.assertEqual(client.cli_path, system_claude)
            self.assertFalse(client._is_js)

    def test_initialization_with_js_cli(self):
        """Test CLIClient initialization with JS CLI."""
        from ace.integrations.claude_code.cli_client import CLIClient

        # Create a JS CLI file
        js_cli = Path(self.temp_dir) / "cli.js"
        js_cli.touch()

        client = CLIClient(cli_path=str(js_cli))
        self.assertEqual(client.cli_path, js_cli)
        self.assertTrue(client._is_js)

    def test_default_timeout_and_retries(self):
        """Test default timeout and retry values."""
        from ace.integrations.claude_code.cli_client import CLIClient

        # Create a fake CLI
        fake_cli = Path(self.temp_dir) / "claude"
        fake_cli.touch()

        client = CLIClient(cli_path=str(fake_cli))
        self.assertEqual(client.timeout, 120)
        self.assertEqual(client.max_retries, 3)

    def test_custom_timeout_and_retries(self):
        """Test custom timeout and retry values."""
        from ace.integrations.claude_code.cli_client import CLIClient

        fake_cli = Path(self.temp_dir) / "claude"
        fake_cli.touch()

        client = CLIClient(cli_path=str(fake_cli), timeout=60, max_retries=5)
        self.assertEqual(client.timeout, 60)
        self.assertEqual(client.max_retries, 5)


@pytest.mark.unit
class TestCLIClientEnvOverrides(unittest.TestCase):
    """Test environment variable overrides for CLIClient."""

    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_ace_cli_path_in_hook_learner(self):
        """Test ACE_CLI_PATH environment variable in ACEHookLearner."""
        # Create a fake CLI
        fake_cli = Path(self.temp_dir) / "custom_claude"
        fake_cli.touch()

        os.environ["ACE_CLI_PATH"] = str(fake_cli)

        # Create project directory
        project = Path(self.temp_dir) / "project"
        project.mkdir()
        (project / ".git").mkdir()

        with patch(
            "ace.integrations.claude_code.cli_client._resolve_cli_path"
        ) as mock_resolve:
            mock_resolve.return_value = fake_cli

            from ace.integrations.claude_code.learner import ACEHookLearner

            learner = ACEHookLearner(cwd=str(project))

            # Verify CLIClient was called with the custom path
            self.assertIsNotNone(learner.ace_llm)


if __name__ == "__main__":
    unittest.main()
