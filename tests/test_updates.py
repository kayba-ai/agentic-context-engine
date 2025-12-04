"""Unit tests for UpdateOperation and UpdateBatch classes."""

import unittest

import pytest

from ace.updates import UpdateOperation, UpdateBatch


@pytest.mark.unit
class TestUpdateOperation(unittest.TestCase):
    """Test UpdateOperation class."""

    def test_add_operation(self):
        """Test creating ADD operation."""
        op = UpdateOperation(type="ADD", section="math", content="Show your work")
        self.assertEqual(op.type, "ADD")
        self.assertEqual(op.section, "math")
        self.assertEqual(op.content, "Show your work")
        self.assertIsNone(op.skill_id)

    def test_tag_operation(self):
        """Test creating TAG operation."""
        op = UpdateOperation(
            type="TAG",
            section="general",
            skill_id="skill_123",
            metadata={"helpful": 1},
        )
        self.assertEqual(op.type, "TAG")
        self.assertEqual(op.skill_id, "skill_123")
        self.assertEqual(op.metadata["helpful"], 1)

    def test_update_operation(self):
        """Test creating UPDATE operation."""
        op = UpdateOperation(
            type="UPDATE",
            section="general",
            skill_id="skill_123",
            content="Updated content",
        )
        self.assertEqual(op.type, "UPDATE")
        self.assertEqual(op.skill_id, "skill_123")
        self.assertEqual(op.content, "Updated content")

    def test_remove_operation(self):
        """Test creating REMOVE operation."""
        op = UpdateOperation(type="REMOVE", section="general", skill_id="skill_123")
        self.assertEqual(op.type, "REMOVE")
        self.assertEqual(op.skill_id, "skill_123")

    def test_from_json_add(self):
        """Test parsing ADD operation from JSON."""
        payload = {"type": "ADD", "section": "math", "content": "Test content"}
        op = UpdateOperation.from_json(payload)
        self.assertEqual(op.type, "ADD")
        self.assertEqual(op.section, "math")
        self.assertEqual(op.content, "Test content")

    def test_from_json_tag(self):
        """Test parsing TAG operation from JSON."""
        payload = {
            "type": "TAG",
            "section": "general",
            "skill_id": "skill_123",
            "metadata": {"helpful": 1, "harmful": 0},
        }
        op = UpdateOperation.from_json(payload)
        self.assertEqual(op.type, "TAG")
        self.assertEqual(op.skill_id, "skill_123")
        self.assertEqual(op.metadata["helpful"], 1)
        self.assertEqual(op.metadata["harmful"], 0)

    def test_from_json_tag_filters_invalid_metadata(self):
        """Test TAG operation filters invalid metadata keys."""
        payload = {
            "type": "TAG",
            "section": "general",
            "skill_id": "skill_123",
            "metadata": {
                "helpful": 1,
                "invalid_key": 5,  # Should be filtered out
                "harmful": 2,
            },
        }
        op = UpdateOperation.from_json(payload)
        self.assertIn("helpful", op.metadata)
        self.assertIn("harmful", op.metadata)
        self.assertNotIn("invalid_key", op.metadata)

    def test_from_json_invalid_type(self):
        """Test parsing invalid operation type raises ValueError."""
        payload = {"type": "INVALID", "section": "general"}
        with self.assertRaises(ValueError) as context:
            UpdateOperation.from_json(payload)
        self.assertIn("Invalid operation type", str(context.exception))

    def test_from_json_case_insensitive_type(self):
        """Test operation type is case-insensitive."""
        payload = {"type": "add", "section": "general", "content": "Test"}  # lowercase
        op = UpdateOperation.from_json(payload)
        self.assertEqual(op.type, "ADD")  # Should be uppercased

    def test_to_json_add(self):
        """Test serializing ADD operation to JSON."""
        op = UpdateOperation(type="ADD", section="math", content="Test content")
        json_data = op.to_json()
        self.assertEqual(json_data["type"], "ADD")
        self.assertEqual(json_data["section"], "math")
        self.assertEqual(json_data["content"], "Test content")
        self.assertNotIn("skill_id", json_data)  # Should not include None fields

    def test_to_json_tag(self):
        """Test serializing TAG operation to JSON."""
        op = UpdateOperation(
            type="TAG",
            section="general",
            skill_id="skill_123",
            metadata={"helpful": 1},
        )
        json_data = op.to_json()
        self.assertEqual(json_data["type"], "TAG")
        self.assertEqual(json_data["skill_id"], "skill_123")
        self.assertEqual(json_data["metadata"], {"helpful": 1})

    def test_to_json_remove(self):
        """Test serializing REMOVE operation to JSON."""
        op = UpdateOperation(type="REMOVE", section="general", skill_id="skill_123")
        json_data = op.to_json()
        self.assertEqual(json_data["type"], "REMOVE")
        self.assertEqual(json_data["skill_id"], "skill_123")
        self.assertNotIn("content", json_data)

    def test_roundtrip_json(self):
        """Test operation survives JSON serialization round-trip."""
        original = UpdateOperation(
            type="UPDATE",
            section="general",
            skill_id="skill_123",
            content="Updated",
            metadata={"helpful": 2},
        )
        json_data = original.to_json()
        restored = UpdateOperation.from_json(json_data)

        self.assertEqual(original.type, restored.type)
        self.assertEqual(original.section, restored.section)
        self.assertEqual(original.content, restored.content)
        self.assertEqual(original.skill_id, restored.skill_id)
        self.assertEqual(original.metadata, restored.metadata)


@pytest.mark.unit
class TestUpdateBatch(unittest.TestCase):
    """Test UpdateBatch class."""

    def test_create_empty_batch(self):
        """Test creating empty batch."""
        batch = UpdateBatch(reasoning="No changes needed")
        self.assertEqual(batch.reasoning, "No changes needed")
        self.assertEqual(len(batch.operations), 0)

    def test_create_batch_with_operations(self):
        """Test creating batch with operations."""
        ops = [
            UpdateOperation(type="ADD", section="math", content="New strategy"),
            UpdateOperation(
                type="TAG", section="general", skill_id="b1", metadata={"helpful": 1}
            ),
        ]
        batch = UpdateBatch(reasoning="Multiple updates", operations=ops)
        self.assertEqual(len(batch.operations), 2)
        self.assertEqual(batch.operations[0].type, "ADD")
        self.assertEqual(batch.operations[1].type, "TAG")

    def test_from_json_empty(self):
        """Test parsing empty batch from JSON."""
        payload = {"reasoning": "No changes", "operations": []}
        batch = UpdateBatch.from_json(payload)
        self.assertEqual(batch.reasoning, "No changes")
        self.assertEqual(len(batch.operations), 0)

    def test_from_json_with_operations(self):
        """Test parsing batch with operations from JSON."""
        payload = {
            "reasoning": "Add new strategies",
            "operations": [
                {"type": "ADD", "section": "math", "content": "Strategy 1"},
                {"type": "ADD", "section": "general", "content": "Strategy 2"},
            ],
        }
        batch = UpdateBatch.from_json(payload)
        self.assertEqual(batch.reasoning, "Add new strategies")
        self.assertEqual(len(batch.operations), 2)
        self.assertEqual(batch.operations[0].content, "Strategy 1")
        self.assertEqual(batch.operations[1].content, "Strategy 2")

    def test_from_json_missing_reasoning(self):
        """Test parsing batch with missing reasoning."""
        payload = {"operations": []}
        batch = UpdateBatch.from_json(payload)
        self.assertEqual(batch.reasoning, "")  # Should default to empty string

    def test_from_json_invalid_operations(self):
        """Test parsing batch with invalid operations (should skip)."""
        payload = {
            "reasoning": "Test",
            "operations": [
                {"type": "ADD", "section": "math", "content": "Valid"},
                "invalid_operation",  # Not a dict, should be skipped
                {"type": "ADD", "section": "general", "content": "Also valid"},
            ],
        }
        batch = UpdateBatch.from_json(payload)
        self.assertEqual(len(batch.operations), 2)  # Only valid ones
        self.assertEqual(batch.operations[0].content, "Valid")
        self.assertEqual(batch.operations[1].content, "Also valid")

    def test_to_json_empty(self):
        """Test serializing empty batch to JSON."""
        batch = UpdateBatch(reasoning="No changes")
        json_data = batch.to_json()
        self.assertEqual(json_data["reasoning"], "No changes")
        self.assertEqual(json_data["operations"], [])

    def test_to_json_with_operations(self):
        """Test serializing batch with operations to JSON."""
        ops = [
            UpdateOperation(type="ADD", section="math", content="Strategy"),
            UpdateOperation(
                type="TAG", section="general", skill_id="b1", metadata={"helpful": 1}
            ),
        ]
        batch = UpdateBatch(reasoning="Updates", operations=ops)
        json_data = batch.to_json()

        self.assertEqual(json_data["reasoning"], "Updates")
        self.assertEqual(len(json_data["operations"]), 2)
        self.assertEqual(json_data["operations"][0]["type"], "ADD")
        self.assertEqual(json_data["operations"][1]["type"], "TAG")

    def test_roundtrip_json(self):
        """Test batch survives JSON serialization round-trip."""
        original_ops = [
            UpdateOperation(type="ADD", section="math", content="New"),
            UpdateOperation(type="REMOVE", section="old", skill_id="b1"),
        ]
        original = UpdateBatch(reasoning="Test reasoning", operations=original_ops)

        json_data = original.to_json()
        restored = UpdateBatch.from_json(json_data)

        self.assertEqual(original.reasoning, restored.reasoning)
        self.assertEqual(len(original.operations), len(restored.operations))
        for orig_op, rest_op in zip(original.operations, restored.operations):
            self.assertEqual(orig_op.type, rest_op.type)
            self.assertEqual(orig_op.section, rest_op.section)
            self.assertEqual(orig_op.content, rest_op.content)
            self.assertEqual(orig_op.skill_id, rest_op.skill_id)


if __name__ == "__main__":
    unittest.main()
