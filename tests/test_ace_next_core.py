"""Tests for ace_next core: Skillbook, SkillbookView, ACEStepContext."""

from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import AgentOutput, ReflectorOutput
from ace_next.core.skillbook import (
    Skill,
    Skillbook,
    UpdateBatch,
    UpdateOperation,
)


# ------------------------------------------------------------------ #
# Skillbook CRUD
# ------------------------------------------------------------------ #


class TestSkillbookCRUD:
    def test_add_and_get_skill(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "Use division for fractions")
        assert skill.section == "math"
        assert skill.content == "Use division for fractions"
        assert sb.get_skill(skill.id) is skill

    def test_add_skill_custom_id(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "content", skill_id="custom-001")
        assert skill.id == "custom-001"
        assert sb.get_skill("custom-001") is skill

    def test_update_skill(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "old content")
        updated = sb.update_skill(skill.id, content="new content")
        assert updated is not None
        assert updated.content == "new content"

    def test_update_nonexistent_skill(self):
        sb = Skillbook()
        assert sb.update_skill("missing-id", content="x") is None

    def test_tag_skill(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "content")
        sb.tag_skill(skill.id, "helpful")
        sb.tag_skill(skill.id, "helpful")
        sb.tag_skill(skill.id, "harmful")
        assert skill.helpful == 2
        assert skill.harmful == 1
        assert skill.neutral == 0

    def test_tag_invalid_tag(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "content")
        with pytest.raises(ValueError, match="Unsupported tag"):
            sb.tag_skill(skill.id, "invalid_tag")

    def test_tag_nonexistent_skill(self):
        sb = Skillbook()
        assert sb.tag_skill("missing-id", "helpful") is None

    def test_remove_skill_hard(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "content")
        sb.remove_skill(skill.id)
        assert sb.get_skill(skill.id) is None
        assert len(sb.skills()) == 0

    def test_remove_skill_soft(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "content")
        sb.remove_skill(skill.id, soft=True)
        assert sb.get_skill(skill.id) is not None
        assert skill.status == "invalid"
        assert len(sb.skills()) == 0  # active only
        assert len(sb.skills(include_invalid=True)) == 1

    def test_remove_nonexistent_skill(self):
        sb = Skillbook()
        sb.remove_skill("missing-id")  # should not raise

    def test_skills_list(self):
        sb = Skillbook()
        sb.add_skill("math", "a")
        sb.add_skill("math", "b")
        sb.add_skill("writing", "c")
        assert len(sb.skills()) == 3

    def test_generate_id_increments(self):
        sb = Skillbook()
        s1 = sb.add_skill("math", "a")
        s2 = sb.add_skill("math", "b")
        assert s1.id != s2.id
        assert "math" in s1.id
        assert "math" in s2.id


# ------------------------------------------------------------------ #
# Skillbook serialization
# ------------------------------------------------------------------ #


class TestSkillbookSerialization:
    def test_round_trip(self):
        sb = Skillbook()
        sb.add_skill("math", "content A", skill_id="math-001")
        sb.add_skill("writing", "content B", skill_id="writing-001")
        sb.tag_skill("math-001", "helpful")

        data = sb.to_dict()
        restored = Skillbook.from_dict(data)

        assert len(restored.skills()) == 2
        assert restored.get_skill("math-001").helpful == 1
        assert restored.get_skill("writing-001").content == "content B"

    def test_json_round_trip(self):
        sb = Skillbook()
        sb.add_skill("sec", "content", skill_id="sec-001")
        json_str = sb.dumps()
        restored = Skillbook.loads(json_str)
        assert restored.get_skill("sec-001").content == "content"

    def test_file_round_trip(self, tmp_path):
        sb = Skillbook()
        sb.add_skill("sec", "content", skill_id="sec-001")
        path = str(tmp_path / "sb.json")
        sb.save_to_file(path)
        restored = Skillbook.load_from_file(path)
        assert restored.get_skill("sec-001").content == "content"

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            Skillbook.load_from_file("/nonexistent/path.json")

    def test_loads_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            Skillbook.loads("not json")

    def test_from_dict_malformed_sections(self):
        """Malformed sections (string instead of list) should not crash.

        The code iterates the value — a string is iterable so each char
        becomes an entry.  This is a known quirk (review item #7).
        """
        payload = {
            "skills": {},
            "sections": {"bad": "not-a-list"},
            "next_id": 0,
        }
        sb = Skillbook.from_dict(payload)
        # String is iterable, so list("not-a-list") = ['n','o','t',...]
        assert isinstance(sb._sections.get("bad"), list)

    def test_from_dict_missing_fields(self):
        """Missing optional fields should use defaults."""
        payload = {
            "skills": {
                "s1": {
                    "id": "s1",
                    "section": "a",
                    "content": "x",
                    "helpful": 0,
                    "harmful": 0,
                    "neutral": 0,
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                }
            },
            "sections": {"a": ["s1"]},
        }
        sb = Skillbook.from_dict(payload)
        skill = sb.get_skill("s1")
        assert skill is not None
        assert skill.embedding is None
        assert skill.status == "active"
        assert skill.sources == []


# ------------------------------------------------------------------ #
# Skillbook update operations
# ------------------------------------------------------------------ #


class TestSkillbookUpdates:
    def test_apply_add(self):
        sb = Skillbook()
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(type="ADD", section="math", content="new skill")
            ],
        )
        sb.apply_update(batch)
        assert len(sb.skills()) == 1
        assert sb.skills()[0].content == "new skill"

    def test_apply_update(self):
        sb = Skillbook()
        skill = sb.add_skill("math", "old", skill_id="math-001")
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="UPDATE",
                    section="math",
                    content="new",
                    skill_id="math-001",
                )
            ],
        )
        sb.apply_update(batch)
        assert skill.content == "new"

    def test_apply_tag(self):
        sb = Skillbook()
        sb.add_skill("math", "content", skill_id="math-001")
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(
                    type="TAG",
                    section="math",
                    skill_id="math-001",
                    metadata={"helpful": 1},
                )
            ],
        )
        sb.apply_update(batch)
        assert sb.get_skill("math-001").helpful == 1

    def test_apply_remove(self):
        sb = Skillbook()
        sb.add_skill("math", "content", skill_id="math-001")
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(type="REMOVE", section="math", skill_id="math-001")
            ],
        )
        sb.apply_update(batch)
        assert sb.get_skill("math-001") is None

    def test_apply_update_missing_skill_id(self):
        """UPDATE/TAG/REMOVE without skill_id should be skipped silently."""
        sb = Skillbook()
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(type="UPDATE", section="math", content="x"),
                UpdateOperation(type="TAG", section="math", metadata={"helpful": 1}),
                UpdateOperation(type="REMOVE", section="math"),
            ],
        )
        sb.apply_update(batch)  # should not raise
        assert len(sb.skills()) == 0


# ------------------------------------------------------------------ #
# Skillbook thread safety
# ------------------------------------------------------------------ #


class TestSkillbookThreadSafety:
    def test_concurrent_add_and_tag(self):
        """Concurrent add_skill and tag_skill should not corrupt state."""
        sb = Skillbook()
        errors = []
        n_add = 50
        n_tag = 50

        def adder():
            try:
                for i in range(n_add):
                    sb.add_skill("concurrent", f"skill-{i}")
            except Exception as e:
                errors.append(e)

        def tagger():
            try:
                for _ in range(n_tag):
                    skills = sb.skills()
                    if skills:
                        try:
                            sb.tag_skill(skills[0].id, "helpful")
                        except (ValueError, KeyError):
                            pass  # skill may have been removed
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=tagger),
            threading.Thread(target=adder),
            threading.Thread(target=tagger),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        # All skills should be present (2 adders × 50 each)
        assert len(sb.skills()) == n_add * 2

    def test_lock_is_reentrant(self):
        """apply_update calls add_skill internally — lock must be reentrant."""
        sb = Skillbook()
        batch = UpdateBatch(
            reasoning="test",
            operations=[
                UpdateOperation(type="ADD", section="sec", content="a"),
                UpdateOperation(type="ADD", section="sec", content="b"),
            ],
        )
        sb.apply_update(batch)
        assert len(sb.skills()) == 2


# ------------------------------------------------------------------ #
# SkillbookView
# ------------------------------------------------------------------ #


class TestSkillbookView:
    def test_read_methods(self):
        sb = Skillbook()
        sb.add_skill("math", "content", skill_id="m-001")
        view = SkillbookView(sb)

        assert len(view) == 1
        assert view.get_skill("m-001").content == "content"
        assert len(view.skills()) == 1
        assert "skills" in view.stats()

    def test_no_write_methods(self):
        sb = Skillbook()
        view = SkillbookView(sb)

        assert not hasattr(view, "add_skill")
        assert not hasattr(view, "update_skill")
        assert not hasattr(view, "tag_skill")
        assert not hasattr(view, "remove_skill")
        assert not hasattr(view, "apply_update")

    def test_iteration(self):
        sb = Skillbook()
        sb.add_skill("a", "x")
        sb.add_skill("b", "y")
        view = SkillbookView(sb)

        skills = list(view)
        assert len(skills) == 2

    def test_repr(self):
        sb = Skillbook()
        sb.add_skill("a", "x")
        view = SkillbookView(sb)
        assert "1 skills" in repr(view)


# ------------------------------------------------------------------ #
# ACEStepContext
# ------------------------------------------------------------------ #


class TestACEStepContext:
    def test_frozen(self):
        ctx = ACEStepContext(sample="test")
        with pytest.raises(FrozenInstanceError):
            ctx.sample = "other"

    def test_replace(self):
        ctx = ACEStepContext(sample="test", epoch=1)
        ctx2 = ctx.replace(epoch=2)
        assert ctx.epoch == 1
        assert ctx2.epoch == 2

    def test_defaults(self):
        ctx = ACEStepContext()
        assert ctx.sample is None
        assert ctx.skillbook is None
        assert ctx.trace is None
        assert ctx.agent_output is None
        assert ctx.reflection is None
        assert ctx.skill_manager_output is None
        assert ctx.epoch == 1
        assert ctx.total_epochs == 1
        assert ctx.step_index == 0

    def test_replace_with_skillbook_view(self):
        sb = Skillbook()
        view = SkillbookView(sb)
        ctx = ACEStepContext(skillbook=view)
        assert ctx.skillbook is view

    def test_replace_with_outputs(self):
        agent_out = AgentOutput(reasoning="r", final_answer="a")
        ctx = ACEStepContext()
        ctx2 = ctx.replace(agent_output=agent_out)
        assert ctx2.agent_output is agent_out
        assert ctx.agent_output is None  # original unchanged


# ------------------------------------------------------------------ #
# UpdateOperation / UpdateBatch parsing
# ------------------------------------------------------------------ #


class TestUpdateOperationParsing:
    def test_from_json_add(self):
        op = UpdateOperation.from_json(
            {"type": "ADD", "section": "math", "content": "skill content"}
        )
        assert op.type == "ADD"
        assert op.section == "math"
        assert op.content == "skill content"

    def test_from_json_tag_filters_metadata(self):
        op = UpdateOperation.from_json(
            {
                "type": "TAG",
                "section": "math",
                "skill_id": "m-001",
                "metadata": {"helpful": 1, "invalid_key": 5},
            }
        )
        assert "helpful" in op.metadata
        assert "invalid_key" not in op.metadata

    def test_from_json_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid operation type"):
            UpdateOperation.from_json({"type": "INVALID", "section": "x"})

    def test_batch_from_json(self):
        batch = UpdateBatch.from_json(
            {
                "reasoning": "test reasoning",
                "operations": [
                    {"type": "ADD", "section": "a", "content": "x"},
                    {"type": "ADD", "section": "b", "content": "y"},
                ],
            }
        )
        assert batch.reasoning == "test reasoning"
        assert len(batch.operations) == 2

    def test_batch_round_trip(self):
        batch = UpdateBatch(
            reasoning="r",
            operations=[UpdateOperation(type="ADD", section="s", content="c")],
        )
        data = batch.to_json()
        restored = UpdateBatch.from_json(data)
        assert restored.reasoning == "r"
        assert len(restored.operations) == 1
        assert restored.operations[0].type == "ADD"
