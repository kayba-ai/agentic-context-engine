"""Tests for Skillbook functionality including persistence."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

from ace import Skillbook, UpdateBatch, UpdateOperation


@pytest.mark.unit
class TestSkillbook(unittest.TestCase):
    """Test Skillbook class functionality."""

    def setUp(self):
        """Set up test skillbook with sample data."""
        self.skillbook = Skillbook()

        # Add test skills
        self.skill1 = self.skillbook.add_skill(
            section="general",
            content="Always be clear",
            metadata={"helpful": 5, "harmful": 0},
        )

        self.skill2 = self.skillbook.add_skill(
            section="math",
            content="Show your work",
            metadata={"helpful": 3, "harmful": 1},
        )

    def test_add_skill(self):
        """Test adding skills to skillbook."""
        skill = self.skillbook.add_skill(section="test", content="Test content")

        self.assertIsNotNone(skill)
        self.assertEqual(skill.section, "test")
        self.assertEqual(skill.content, "Test content")
        self.assertEqual(len(self.skillbook.skills()), 3)

    def test_update_skill(self):
        """Test updating existing skill."""
        updated = self.skillbook.update_skill(
            self.skill1.id, content="Updated content", metadata={"helpful": 10}
        )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.content, "Updated content")
        self.assertEqual(updated.helpful, 10)

    def test_tag_skill(self):
        """Test tagging skills."""
        self.skillbook.tag_skill(self.skill1.id, "helpful", 2)
        skill = self.skillbook.get_skill(self.skill1.id)

        self.assertEqual(skill.helpful, 7)  # 5 + 2

    def test_skill_to_llm_dict_excludes_timestamps(self):
        """Test that to_llm_dict filters out created_at and updated_at."""
        from ace.skillbook import Skill

        skill = Skill(
            id="test-001",
            section="test_section",
            content="Test strategy content",
            helpful=5,
            harmful=1,
            neutral=2,
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
        )

        llm_dict = skill.to_llm_dict()

        # Should include LLM-relevant fields
        self.assertEqual(llm_dict["id"], "test-001")
        self.assertEqual(llm_dict["section"], "test_section")
        self.assertEqual(llm_dict["content"], "Test strategy content")
        self.assertEqual(llm_dict["helpful"], 5)
        self.assertEqual(llm_dict["harmful"], 1)
        self.assertEqual(llm_dict["neutral"], 2)

        # Should exclude timestamps
        self.assertNotIn("created_at", llm_dict)
        self.assertNotIn("updated_at", llm_dict)

        # Should have exactly 6 fields
        self.assertEqual(len(llm_dict), 6)

    def test_remove_skill(self):
        """Test removing skills."""
        self.skillbook.remove_skill(self.skill1.id)

        self.assertIsNone(self.skillbook.get_skill(self.skill1.id))
        self.assertEqual(len(self.skillbook.skills()), 1)

    def test_apply_update(self):
        """Test applying update operations."""
        update = UpdateBatch(
            reasoning="Test update operations",
            operations=[
                UpdateOperation(type="ADD", section="new", content="New strategy"),
                UpdateOperation(
                    type="UPDATE",
                    section="",  # Section is required but not used for UPDATE
                    skill_id=self.skill1.id,
                    content="Modified content",
                ),
                UpdateOperation(
                    type="TAG",
                    section="",  # Section is required but not used for TAG
                    skill_id=self.skill2.id,
                    metadata={"harmful": 2},
                ),
            ],
        )

        self.skillbook.apply_update(update)

        # Check ADD operation
        self.assertEqual(len(self.skillbook.skills()), 3)

        # Check UPDATE operation
        skill1 = self.skillbook.get_skill(self.skill1.id)
        self.assertEqual(skill1.content, "Modified content")

        # Check TAG operation
        skill2 = self.skillbook.get_skill(self.skill2.id)
        self.assertEqual(skill2.harmful, 3)  # 1 + 2

    def test_dumps_loads(self):
        """Test JSON serialization and deserialization."""
        # Serialize
        json_str = self.skillbook.dumps()
        self.assertIsInstance(json_str, str)

        # Verify it's valid JSON
        data = json.loads(json_str)
        self.assertIn("skills", data)
        self.assertIn("sections", data)

        # Deserialize
        loaded = Skillbook.loads(json_str)

        # Verify content matches
        self.assertEqual(len(loaded.skills()), len(self.skillbook.skills()))

        for original, loaded_skill in zip(self.skillbook.skills(), loaded.skills()):
            self.assertEqual(original.id, loaded_skill.id)
            self.assertEqual(original.content, loaded_skill.content)
            self.assertEqual(original.helpful, loaded_skill.helpful)

    def test_save_to_file(self):
        """Test saving skillbook to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Save skillbook
            self.skillbook.save_to_file(temp_path)

            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))

            # Verify content is valid JSON
            with open(temp_path, "r") as f:
                data = json.load(f)

            self.assertIn("skills", data)
            self.assertIn("sections", data)
            self.assertEqual(len(data["skills"]), 2)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_file(self):
        """Test loading skillbook from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Save original
            self.skillbook.save_to_file(temp_path)

            # Load from file
            loaded = Skillbook.load_from_file(temp_path)

            # Verify content matches
            self.assertEqual(len(loaded.skills()), 2)

            loaded_skills = {b.id: b for b in loaded.skills()}
            original_skills = {b.id: b for b in self.skillbook.skills()}

            for sid, original in original_skills.items():
                loaded_skill = loaded_skills[sid]
                self.assertEqual(loaded_skill.content, original.content)
                self.assertEqual(loaded_skill.section, original.section)
                self.assertEqual(loaded_skill.helpful, original.helpful)
                self.assertEqual(loaded_skill.harmful, original.harmful)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_creates_parent_dirs(self):
        """Test that save_to_file creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "skillbook.json")

            # Parent dirs don't exist yet
            self.assertFalse(os.path.exists(os.path.dirname(nested_path)))

            # Save should create them
            self.skillbook.save_to_file(nested_path)

            # Verify file was created
            self.assertTrue(os.path.exists(nested_path))

            # Verify we can load it back
            loaded = Skillbook.load_from_file(nested_path)
            self.assertEqual(len(loaded.skills()), 2)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            Skillbook.load_from_file("nonexistent_file.json")

        self.assertIn("not found", str(context.exception))

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                Skillbook.load_from_file(temp_path)
        finally:
            os.remove(temp_path)

    def test_as_prompt(self):
        """Test skillbook prompt generation in TOON format."""
        prompt = self.skillbook.as_prompt()

        # Check for TOON format with tab-delimited header
        self.assertIn("skills[2", prompt)  # Array length
        self.assertIn("{id", prompt)  # Field declarations
        self.assertIn("general", prompt)
        self.assertIn("math", prompt)
        self.assertIn("Always be clear", prompt)
        self.assertIn("Show your work", prompt)

        # Check tab delimiters are used
        self.assertIn("\t", prompt)

        # Check helpful/harmful values are present as tab-separated numbers
        lines = prompt.split("\n")
        data_lines = [line for line in lines if line and not line.startswith("skills")]
        self.assertEqual(len(data_lines), 2)  # Two skill rows

        # Check first skill: general-00001\tgeneral\tAlways be clear\t5\t0\t0
        self.assertIn("general-00001", data_lines[0])
        self.assertIn("Always be clear", data_lines[0])
        parts = data_lines[0].split("\t")
        self.assertEqual(parts[3], "5")  # helpful
        self.assertEqual(parts[4], "0")  # harmful

        # Check second skill: math-00002\tmath\tShow your work\t3\t1\t0
        self.assertIn("math-00002", data_lines[1])
        self.assertIn("Show your work", data_lines[1])
        parts = data_lines[1].split("\t")
        self.assertEqual(parts[3], "3")  # helpful
        self.assertEqual(parts[4], "1")  # harmful

    def test_stats(self):
        """Test skillbook statistics."""
        stats = self.skillbook.stats()

        self.assertEqual(stats["sections"], 2)
        self.assertEqual(stats["skills"], 2)
        self.assertEqual(stats["tags"]["helpful"], 8)  # 5 + 3
        self.assertEqual(stats["tags"]["harmful"], 1)
        self.assertEqual(stats["tags"]["neutral"], 0)

    def test_empty_skillbook_serialization(self):
        """Test that empty skillbook can be saved and loaded."""
        empty = Skillbook()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Save empty skillbook
            empty.save_to_file(temp_path)

            # Load it back
            loaded = Skillbook.load_from_file(temp_path)

            # Verify it's empty
            self.assertEqual(len(loaded.skills()), 0)
            self.assertEqual(loaded.stats()["skills"], 0)
            self.assertEqual(loaded.stats()["sections"], 0)

        finally:
            os.remove(temp_path)

    def test_as_prompt_returns_valid_toon(self):
        """Test that as_prompt() returns valid TOON format."""
        from toon import decode

        # Get TOON output
        toon_output = self.skillbook.as_prompt()

        # Should be valid TOON - decode it
        decoded = decode(toon_output)

        # Verify structure
        self.assertIn("skills", decoded)
        self.assertEqual(len(decoded["skills"]), 2)

        # Verify skill IDs are preserved
        skill_ids = {b["id"] for b in decoded["skills"]}
        self.assertIn("general-00001", skill_ids)
        self.assertIn("math-00002", skill_ids)

    def test_as_prompt_empty_skillbook(self):
        """Test that empty skillbook is handled gracefully."""
        empty = Skillbook()

        # Should not crash
        result = empty.as_prompt()

        # Should return valid TOON for empty skills array
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_markdown_debug_method(self):
        """Test that _as_markdown_debug() provides human-readable format."""
        markdown = self.skillbook._as_markdown_debug()

        # Should be markdown format
        self.assertIn("##", markdown)  # Section headers
        self.assertIn("- [", markdown)  # Skill points
        self.assertIn("general", markdown)  # Section name
        self.assertIn("Always be clear", markdown)  # Content
        self.assertIn("helpful=5", markdown)  # Counters

    def test_dumps_exclude_embeddings(self):
        """Test that exclude_embeddings=True sets embeddings to None."""
        # Create skill with embedding
        skill = self.skillbook.add_skill(section="test", content="Test")
        skill.embedding = [0.1, 0.2, 0.3]  # Simulate embedding

        # Without exclude_embeddings - embedding preserved
        json_str = self.skillbook.dumps()
        data = json.loads(json_str)
        self.assertEqual(data["skills"][skill.id]["embedding"], [0.1, 0.2, 0.3])

        # With exclude_embeddings=True - embedding set to None
        json_str_excluded = self.skillbook.dumps(exclude_embeddings=True)
        data_excluded = json.loads(json_str_excluded)
        self.assertIsNone(data_excluded["skills"][skill.id]["embedding"])

    def test_save_to_file_exclude_embeddings(self):
        """Test save_to_file with exclude_embeddings option."""
        skill = self.skillbook.add_skill(section="test", content="Test")
        skill.embedding = [0.1, 0.2, 0.3]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Save with embeddings excluded
            self.skillbook.save_to_file(temp_path, exclude_embeddings=True)

            with open(temp_path) as f:
                data = json.load(f)

            self.assertIsNone(data["skills"][skill.id]["embedding"])
        finally:
            os.remove(temp_path)

    def test_to_dict_exclude_embeddings(self):
        """Test to_dict with exclude_embeddings option."""
        skill = self.skillbook.add_skill(section="test", content="Test")
        skill.embedding = [0.1, 0.2, 0.3]

        # Default: embedding preserved
        data = self.skillbook.to_dict()
        self.assertEqual(data["skills"][skill.id]["embedding"], [0.1, 0.2, 0.3])

        # exclude_embeddings=True: embedding set to None
        data_excluded = self.skillbook.to_dict(exclude_embeddings=True)
        self.assertIsNone(data_excluded["skills"][skill.id]["embedding"])

    def test_exclude_embeddings_does_not_modify_original(self):
        """Test that exclude_embeddings doesn't modify the in-memory skillbook."""
        skill = self.skillbook.add_skill(section="test", content="Test")
        skill.embedding = [0.1, 0.2, 0.3]

        # Serialize with exclusion
        self.skillbook.dumps(exclude_embeddings=True)

        # Original skill should still have embedding
        self.assertEqual(skill.embedding, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
