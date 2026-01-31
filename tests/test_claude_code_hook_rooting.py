"""Tests for Claude Code hook project root detection.

Tests the following functionality:
- find_project_root() with various marker combinations
- .ace-root marker taking priority over .git (monorepo scenario)
- ACE_PROJECT_DIR environment override
- Home directory fallback when no project root found
- Transcript cwd extraction
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestFindProjectRoot(unittest.TestCase):
    """Test project root detection functionality."""

    def setUp(self):
        """Set up temporary directory structure for testing."""
        # Resolve symlinks (macOS /var -> /private/var)
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up temporary files and restore environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_import(self):
        """Test that module can be imported."""
        from ace.integrations.claude_code.learner import (
            find_project_root,
            DEFAULT_MARKERS,
        )

        self.assertIsNotNone(find_project_root)
        self.assertIn(".ace-root", DEFAULT_MARKERS)
        # .claude should NOT be a marker (would match ~/.claude)
        self.assertNotIn(".claude", DEFAULT_MARKERS)

    def test_find_project_root_with_git(self):
        """Test finding project root via .git directory."""
        from ace.integrations.claude_code.learner import find_project_root

        # Create nested directory with .git at root
        project_root = Path(self.temp_dir) / "project"
        nested = project_root / "src" / "nested"
        nested.mkdir(parents=True)
        (project_root / ".git").mkdir()

        result = find_project_root(nested)
        self.assertEqual(result, project_root)

    def test_ace_root_marker_takes_priority(self):
        """Test that .ace-root file takes priority over .git."""
        from ace.integrations.claude_code.learner import find_project_root

        # Create structure: monorepo/.ace-root + monorepo/packages/foo/.git
        monorepo = Path(self.temp_dir) / "monorepo"
        monorepo.mkdir(parents=True)
        (monorepo / ".ace-root").touch()  # File, not directory

        package_foo = monorepo / "packages" / "foo"
        package_foo.mkdir(parents=True)
        (package_foo / ".git").mkdir()

        nested = package_foo / "src"
        nested.mkdir(parents=True)

        # Starting from nested, should find monorepo (via .ace-root) not package_foo (via .git)
        result = find_project_root(nested)
        self.assertEqual(result, monorepo)

    def test_no_markers_returns_none(self):
        """Test that no markers returns None."""
        from ace.integrations.claude_code.learner import find_project_root

        # Create directory with no markers
        no_markers = Path(self.temp_dir) / "no_markers"
        no_markers.mkdir(parents=True)

        result = find_project_root(no_markers)
        self.assertIsNone(result)

    def test_ace_project_dir_override(self):
        """Test ACE_PROJECT_DIR environment variable override."""
        from ace.integrations.claude_code.learner import find_project_root

        # Create two directories
        override_dir = Path(self.temp_dir) / "override"
        override_dir.mkdir()

        other_dir = Path(self.temp_dir) / "other"
        (other_dir / ".git").mkdir(parents=True)

        # Set environment variable
        os.environ["ACE_PROJECT_DIR"] = str(override_dir)

        # Should return override dir even when starting from other_dir
        result = find_project_root(other_dir)
        self.assertEqual(result, override_dir)

    def test_ace_project_dir_invalid_path(self):
        """Test ACE_PROJECT_DIR with invalid path falls back to markers."""
        from ace.integrations.claude_code.learner import find_project_root

        # Set invalid environment variable
        os.environ["ACE_PROJECT_DIR"] = "/nonexistent/path"

        # Create directory with marker
        project = Path(self.temp_dir) / "project"
        project.mkdir()
        (project / ".git").mkdir()

        # Should fall back to marker-based detection
        result = find_project_root(project)
        self.assertEqual(result, project)

    def test_pyproject_toml_marker(self):
        """Test finding root via pyproject.toml."""
        from ace.integrations.claude_code.learner import find_project_root

        project = Path(self.temp_dir) / "python_project"
        nested = project / "src" / "module"
        nested.mkdir(parents=True)
        (project / "pyproject.toml").touch()

        result = find_project_root(nested)
        self.assertEqual(result, project)

    def test_package_json_marker(self):
        """Test finding root via package.json."""
        from ace.integrations.claude_code.learner import find_project_root

        project = Path(self.temp_dir) / "node_project"
        nested = project / "src" / "components"
        nested.mkdir(parents=True)
        (project / "package.json").touch()

        result = find_project_root(nested)
        self.assertEqual(result, project)


@pytest.mark.unit
class TestACELearnerInit(unittest.TestCase):
    """Test ACELearner initialization with various project configurations."""

    def setUp(self):
        """Set up temporary directory structure for testing."""
        # Resolve symlinks (macOS /var -> /private/var)
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up temporary files and restore environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_ace_learner_detects_project_root(self):
        """Test ACELearner correctly detects project root from .git."""
        project = Path(self.temp_dir) / "project"
        project.mkdir()
        (project / ".git").mkdir()

        with patch("ace.integrations.claude_code.learner.CLIClient") as mock_cli:
            mock_cli.return_value = MagicMock()

            from ace.integrations.claude_code.learner import ACELearner

            learner = ACELearner(cwd=str(project))
            self.assertEqual(learner.project_root, project)
            self.assertEqual(learner.ace_dir, project / ".ace")

    def test_ace_learner_with_explicit_project_root(self):
        """Test ACELearner with explicit project_root parameter."""
        project = Path(self.temp_dir) / "project"
        project.mkdir()

        with patch("ace.integrations.claude_code.learner.CLIClient") as mock_cli:
            mock_cli.return_value = MagicMock()

            from ace.integrations.claude_code.learner import ACELearner

            learner = ACELearner(cwd=str(self.temp_dir), project_root=project)
            self.assertEqual(learner.project_root, project)


@pytest.mark.unit
class TestTranscriptFirstBehavior(unittest.TestCase):
    """Test transcript-first parsing in learn_from_hook_input."""

    def setUp(self):
        """Set up temporary directory structure and fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up temporary files and restore environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def _create_minimal_transcript(self, cwd: str) -> Path:
        """Create a minimal valid transcript for testing."""
        transcript_path = Path(self.temp_dir) / "transcript.jsonl"
        entries = [
            {"type": "user", "cwd": cwd, "message": {"content": []}},
            {"type": "assistant", "message": {"content": []}},
        ]
        with transcript_path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return transcript_path

    def test_transcript_cwd_extraction(self):
        """Test that cwd is correctly extracted from transcript."""
        from ace.integrations.claude_code.learner import _extract_cwd_from_transcript

        # Create project directory
        project = Path(self.temp_dir) / "project"
        project.mkdir()
        (project / ".git").mkdir()

        # Create transcript with specific cwd
        transcript_cwd = str(project / "src")
        transcript_path = self._create_minimal_transcript(transcript_cwd)

        extracted_cwd = _extract_cwd_from_transcript(transcript_path)

        self.assertEqual(extracted_cwd, transcript_cwd)

    def test_learn_from_nonexistent_transcript(self):
        """Test that learning from nonexistent transcript fails gracefully."""
        from ace.integrations.claude_code.learner import ACELearner

        # Create project directory
        project = Path(self.temp_dir) / "project"
        project.mkdir()
        (project / ".git").mkdir()

        with patch("ace.integrations.claude_code.learner.CLIClient") as mock_cli:
            mock_cli.return_value = MagicMock()

            learner = ACELearner(cwd=str(project))

            # Test nonexistent transcript
            result = learner.learn_from_transcript(
                Path("/nonexistent/transcript.jsonl")
            )
            self.assertFalse(result)


@pytest.mark.unit
class TestHomeFallback(unittest.TestCase):
    """Test home directory fallback when no project root found."""

    def setUp(self):
        """Set up temporary directory structure."""
        # Resolve symlinks (macOS /var -> /private/var)
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ace_learner_home_fallback(self):
        """Test that ACELearner falls back to home directory when no project found."""
        # Create directory with no project markers
        no_project = Path(self.temp_dir) / "no_project"
        no_project.mkdir()

        # Mock CLIClient to avoid actual CLI dependency
        with patch("ace.integrations.claude_code.learner.CLIClient") as mock_cli:
            mock_cli.return_value = MagicMock()

            from ace.integrations.claude_code.learner import ACELearner

            learner = ACELearner(cwd=str(no_project))

            # Should fallback to home directory
            self.assertEqual(learner.project_root, Path.home())
            self.assertEqual(learner.ace_dir, Path.home() / ".ace")


if __name__ == "__main__":
    unittest.main()
