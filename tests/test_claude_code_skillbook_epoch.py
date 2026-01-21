"""Tests for preventing stale skillbook writes after `ace-learn clear`.

The daemon can run learning jobs concurrently. Without coordination, an in-flight
job that loaded the old skillbook could overwrite a freshly-cleared skillbook.
"""

import tempfile
import unittest
from pathlib import Path

import pytest


class DummyLLM:
    """Minimal LLM stub to avoid Instructor wrapping in role constructors."""

    def complete_structured(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("DummyLLM should not be called in these tests")


@pytest.mark.unit
class TestSkillbookEpoch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_persist_update_skips_when_epoch_changed(self):
        from ace.integrations.claude_code.hook import ACEHookLearner, _bump_skillbook_epoch
        from ace.skillbook import Skillbook
        from ace.updates import UpdateBatch, UpdateOperation

        skill_dir = self.temp_dir / "skills"
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Seed an epoch and a non-empty skillbook.
        epoch_path = skill_dir / ".ace-skillbook-epoch"
        epoch_path.write_text("epoch-1", encoding="utf-8")

        sb = Skillbook()
        sb.add_skill(section="general", content="keep me")
        sb.save_to_file(str(skill_dir / "skillbook.json"))

        learner = ACEHookLearner(
            cwd=str(self.temp_dir),
            skill_dir=skill_dir,
            ace_llm=DummyLLM(),
        )

        # Simulate `ace-learn clear` happening while the learner is in-flight.
        _bump_skillbook_epoch(skill_dir)
        Skillbook().save_to_file(str(skill_dir / "skillbook.json"))

        update = UpdateBatch(
            reasoning="add a skill",
            operations=[UpdateOperation(type="ADD", section="general", content="new")],
        )

        saved = learner._persist_skillbook_update(update)
        self.assertFalse(saved)

        loaded = Skillbook.load_from_file(str(skill_dir / "skillbook.json"))
        self.assertEqual(len(loaded.skills()), 0)

    def test_persist_update_applies_when_epoch_matches(self):
        from ace.integrations.claude_code.hook import ACEHookLearner
        from ace.skillbook import Skillbook
        from ace.updates import UpdateBatch, UpdateOperation

        skill_dir = self.temp_dir / "skills"
        skill_dir.mkdir(parents=True, exist_ok=True)

        (skill_dir / ".ace-skillbook-epoch").write_text("epoch-1", encoding="utf-8")

        sb = Skillbook()
        sb.add_skill(section="general", content="keep me")
        sb.save_to_file(str(skill_dir / "skillbook.json"))

        learner = ACEHookLearner(
            cwd=str(self.temp_dir),
            skill_dir=skill_dir,
            ace_llm=DummyLLM(),
        )

        update = UpdateBatch(
            reasoning="add a skill",
            operations=[UpdateOperation(type="ADD", section="general", content="new")],
        )

        saved = learner._persist_skillbook_update(update)
        self.assertTrue(saved)

        loaded = Skillbook.load_from_file(str(skill_dir / "skillbook.json"))
        self.assertEqual(len(loaded.skills()), 2)

