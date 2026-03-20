"""Tests for path safety utilities — directory traversal prevention."""

import os
import tempfile
from pathlib import Path

import pytest

from ace.path_safety import safe_resolve, safe_resolve_within


@pytest.mark.unit
class TestSafeResolve:
    """Tests for safe_resolve()."""

    def test_absolute_path_passes(self, tmp_path):
        result = safe_resolve(tmp_path / "file.json")
        assert result == (tmp_path / "file.json").resolve()

    def test_relative_path_resolves(self):
        result = safe_resolve("file.json")
        expected = Path(os.path.realpath("file.json"))
        assert result == expected

    def test_traversal_collapses_to_valid_path(self, tmp_path):
        """A path with .. that resolves inside a valid location should work."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        # tmp_path/a/b/../file.json -> tmp_path/a/file.json (valid)
        result = safe_resolve(nested / ".." / "file.json")
        assert result == (tmp_path / "a" / "file.json")
        assert ".." not in result.parts

    def test_string_input(self, tmp_path):
        result = safe_resolve(str(tmp_path / "file.json"))
        assert isinstance(result, Path)

    def test_returns_path_object(self, tmp_path):
        result = safe_resolve(tmp_path / "file.json")
        assert isinstance(result, Path)


@pytest.mark.unit
class TestSafeResolveWithin:
    """Tests for safe_resolve_within()."""

    def test_path_inside_base_passes(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        target = base / "checkpoint_10.json"
        result = safe_resolve_within(target, base)
        assert result == target.resolve()

    def test_nested_path_inside_base_passes(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        target = base / "sub" / "checkpoint_10.json"
        result = safe_resolve_within(target, base)
        assert ".." not in result.parts

    def test_path_escaping_base_raises(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        # Trying to escape up one level from the checkpoint dir
        target = base / ".." / "evil.json"
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_within(target, base)

    def test_sibling_directory_raises(self, tmp_path):
        base = tmp_path / "checkpoints"
        sibling = tmp_path / "other"
        base.mkdir()
        sibling.mkdir()
        target = sibling / "file.json"
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_within(target, base)

    def test_completely_unrelated_path_raises(self, tmp_path):
        base = tmp_path / "checkpoints"
        base.mkdir()
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_within("/etc/passwd", base)

    def test_double_traversal_raises(self, tmp_path):
        base = tmp_path / "a" / "b" / "checkpoints"
        base.mkdir(parents=True)
        target = base / ".." / ".." / "evil.json"
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_resolve_within(target, base)


@pytest.mark.unit
class TestSkillbookSaveTraversal:
    """Integration tests: verify that Skillbook.save_to_file rejects traversal."""

    def test_save_to_file_normal_path(self, tmp_path):
        from ace import Skillbook

        sb = Skillbook()
        sb.add_skill(section="test", content="test content")
        path = str(tmp_path / "output.json")
        sb.save_to_file(path)
        assert (tmp_path / "output.json").exists()

    def test_save_to_file_traversal_is_normalised(self, tmp_path):
        """A traversal that resolves to a valid absolute path is normalised."""
        from ace import Skillbook

        sb = Skillbook()
        sb.add_skill(section="test", content="test content")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        # This resolves to tmp_path/a/output.json (valid, no escape)
        path = str(nested / ".." / "output.json")
        sb.save_to_file(path)
        assert (tmp_path / "a" / "output.json").exists()


@pytest.mark.unit
class TestAceNextSkillbookSaveTraversal:
    """Integration tests: verify ace_next Skillbook.save_to_file rejects traversal."""

    def test_save_to_file_normal_path(self, tmp_path):
        from ace_next.core.skillbook import Skillbook

        sb = Skillbook()
        sb.add_skill(section="test", content="test content")
        path = str(tmp_path / "output.json")
        sb.save_to_file(path)
        assert (tmp_path / "output.json").exists()

    def test_save_to_file_traversal_is_normalised(self, tmp_path):
        """A traversal that resolves to a valid absolute path is normalised."""
        from ace_next.core.skillbook import Skillbook

        sb = Skillbook()
        sb.add_skill(section="test", content="test content")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        path = str(nested / ".." / "output.json")
        sb.save_to_file(path)
        assert (tmp_path / "a" / "output.json").exists()


@pytest.mark.unit
class TestCheckpointStepTraversal:
    """Verify CheckpointStep rejects traversal in the directory parameter."""

    def test_checkpoint_dir_traversal_rejected(self, tmp_path):
        from ace_next.core.skillbook import Skillbook
        from ace_next.steps.checkpoint import CheckpointStep

        sb = Skillbook()
        safe_dir = tmp_path / "ckpts"
        safe_dir.mkdir()

        # This should resolve normally
        step = CheckpointStep(str(safe_dir), sb, interval=1)
        assert step.directory == safe_dir.resolve()

    def test_checkpoint_dir_normalises_traversal(self, tmp_path):
        from ace_next.core.skillbook import Skillbook
        from ace_next.steps.checkpoint import CheckpointStep

        sb = Skillbook()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        # ../.. resolves to tmp_path, which is a valid dir
        step = CheckpointStep(str(nested / ".." / ".."), sb, interval=1)
        assert step.directory == tmp_path.resolve()
