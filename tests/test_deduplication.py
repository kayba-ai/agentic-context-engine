"""Tests for bullet deduplication feature."""

import unittest
from typing import List, Optional

from ace.deduplication import (
    DeduplicationConfig,
    DeduplicationManager,
)
from ace.deduplication.detector import SimilarityDetector
from ace.deduplication.operations import (
    DeleteOp,
    KeepOp,
    MergeOp,
    UpdateOp,
    apply_consolidation_operations,
)
from ace.deduplication.prompts import (
    format_pair_for_logging,
    generate_similarity_report,
)
from ace.playbook import Bullet, Playbook


class TestDeduplicationConfig(unittest.TestCase):
    """Tests for DeduplicationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeduplicationConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.embedding_model, "text-embedding-3-small")
        self.assertEqual(config.embedding_provider, "litellm")
        self.assertEqual(config.similarity_threshold, 0.85)
        self.assertEqual(config.min_pairs_to_report, 1)
        self.assertTrue(config.within_section_only)
        self.assertEqual(config.local_model_name, "all-MiniLM-L6-v2")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeduplicationConfig(
            enabled=False,
            similarity_threshold=0.90,
            embedding_provider="sentence_transformers",
            within_section_only=False,
        )
        self.assertFalse(config.enabled)
        self.assertEqual(config.similarity_threshold, 0.90)
        self.assertEqual(config.embedding_provider, "sentence_transformers")
        self.assertFalse(config.within_section_only)


class TestSimilarityDetector(unittest.TestCase):
    """Tests for SimilarityDetector."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        detector = SimilarityDetector()
        vec = [1.0, 0.0, 0.0]
        similarity = detector.cosine_similarity(vec, vec)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        detector = SimilarityDetector()
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        similarity = detector.cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        detector = SimilarityDetector()
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        similarity = detector.cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(similarity, -1.0, places=5)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector returns 0."""
        detector = SimilarityDetector()
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 0.0, 0.0]
        similarity = detector.cosine_similarity(vec_a, vec_b)
        self.assertEqual(similarity, 0.0)

    def test_detect_similar_pairs_empty_playbook(self):
        """Test detection on empty playbook returns empty list."""
        detector = SimilarityDetector()
        playbook = Playbook()
        pairs = detector.detect_similar_pairs(playbook)
        self.assertEqual(len(pairs), 0)

    def test_detect_similar_pairs_with_embeddings(self):
        """Test detection finds similar pairs when embeddings are set."""
        detector = SimilarityDetector(DeduplicationConfig(similarity_threshold=0.8))
        playbook = Playbook()

        # Add bullets with manually set embeddings (section, content)
        bullet_a = playbook.add_bullet("general", "Use caching for performance")
        bullet_b = playbook.add_bullet("general", "Use caching to improve speed")
        bullet_c = playbook.add_bullet("general", "Log all errors to file")

        # Set embeddings - a and b are similar, c is different
        bullet_a.embedding = [0.9, 0.1, 0.0]
        bullet_b.embedding = [0.85, 0.15, 0.05]
        bullet_c.embedding = [0.1, 0.1, 0.9]

        pairs = detector.detect_similar_pairs(playbook)

        # Should find one similar pair (a, b)
        self.assertEqual(len(pairs), 1)
        pair_ids = {pairs[0][0].id, pairs[0][1].id}
        self.assertEqual(pair_ids, {bullet_a.id, bullet_b.id})

    def test_detect_respects_keep_decisions(self):
        """Test that pairs with KEEP decisions are skipped."""
        detector = SimilarityDetector(DeduplicationConfig(similarity_threshold=0.5))
        playbook = Playbook()

        bullet_a = playbook.add_bullet("general", "Strategy A")
        bullet_b = playbook.add_bullet("general", "Strategy B")

        # Set similar embeddings
        bullet_a.embedding = [1.0, 0.0, 0.0]
        bullet_b.embedding = [0.9, 0.1, 0.0]

        # Before KEEP decision, should find pair
        pairs_before = detector.detect_similar_pairs(playbook)
        self.assertEqual(len(pairs_before), 1)

        # Add KEEP decision
        from ace.playbook import SimilarityDecision

        decision = SimilarityDecision(
            decision="KEEP",
            reasoning="They serve different purposes",
            decided_at="2024-01-01T00:00:00Z",
            similarity_at_decision=0.95,
        )
        playbook.set_similarity_decision(bullet_a.id, bullet_b.id, decision)

        # After KEEP decision, should skip pair
        pairs_after = detector.detect_similar_pairs(playbook)
        self.assertEqual(len(pairs_after), 0)


class TestConsolidationOperations(unittest.TestCase):
    """Tests for consolidation operations."""

    def test_merge_operation_combines_counters(self):
        """Test MergeOp combines helpful/harmful counters."""
        playbook = Playbook()
        bullet_a = playbook.add_bullet("general", "Strategy A")
        bullet_b = playbook.add_bullet("general", "Strategy B")

        # Set some counters
        bullet_a.helpful = 5
        bullet_a.harmful = 1
        bullet_b.helpful = 3
        bullet_b.harmful = 2

        op = MergeOp(
            source_ids=[bullet_a.id, bullet_b.id],
            keep_id=bullet_a.id,
            merged_content="Combined strategy",
            reasoning="Same strategy",
        )

        apply_consolidation_operations([op], playbook)

        # Check merged bullet
        merged = playbook.get_bullet(bullet_a.id)
        self.assertEqual(merged.content, "Combined strategy")
        self.assertEqual(merged.helpful, 8)  # 5 + 3
        self.assertEqual(merged.harmful, 3)  # 1 + 2
        self.assertIsNone(merged.embedding)  # Invalidated

        # Check source bullet is soft-deleted
        deleted = playbook.get_bullet(bullet_b.id)
        self.assertEqual(deleted.status, "invalid")

    def test_delete_operation_soft_deletes(self):
        """Test DeleteOp performs soft delete."""
        playbook = Playbook()
        bullet = playbook.add_bullet("general", "To be deleted")

        op = DeleteOp(bullet_id=bullet.id, reasoning="Redundant")

        apply_consolidation_operations([op], playbook)

        # Bullet should be soft-deleted (inactive)
        deleted = playbook.get_bullet(bullet.id)
        self.assertEqual(deleted.status, "invalid")

        # Should not appear in active bullets
        active_bullets = playbook.bullets(include_invalid=False)
        self.assertEqual(len(active_bullets), 0)

    def test_keep_operation_stores_decision(self):
        """Test KeepOp stores similarity decision."""
        playbook = Playbook()
        bullet_a = playbook.add_bullet("general", "Strategy A")
        bullet_b = playbook.add_bullet("general", "Strategy B")

        op = KeepOp(
            bullet_ids=[bullet_a.id, bullet_b.id],
            differentiation="Different contexts",
            reasoning="Both needed",
        )

        apply_consolidation_operations([op], playbook)

        # Check decision is stored
        self.assertTrue(playbook.has_keep_decision(bullet_a.id, bullet_b.id))

    def test_update_operation_changes_content(self):
        """Test UpdateOp changes bullet content."""
        playbook = Playbook()
        bullet = playbook.add_bullet("general", "Original content")
        bullet.embedding = [1.0, 0.0, 0.0]

        op = UpdateOp(
            bullet_id=bullet.id,
            new_content="Updated content with [Batch] tag",
            reasoning="Clarify context",
        )

        apply_consolidation_operations([op], playbook)

        # Check content updated
        updated = playbook.get_bullet(bullet.id)
        self.assertEqual(updated.content, "Updated content with [Batch] tag")
        self.assertIsNone(updated.embedding)  # Invalidated


class TestPromptGeneration(unittest.TestCase):
    """Tests for prompt generation utilities."""

    def test_generate_similarity_report_empty(self):
        """Test empty pairs returns empty string."""
        report = generate_similarity_report([])
        self.assertEqual(report, "")

    def test_generate_similarity_report_includes_pairs(self):
        """Test report includes pair information."""
        bullet_a = Bullet(id="general-00001", content="Strategy A", section="general")
        bullet_b = Bullet(id="general-00002", content="Strategy B", section="general")
        bullet_a.helpful = 5
        bullet_b.harmful = 2

        pairs = [(bullet_a, bullet_b, 0.92)]
        report = generate_similarity_report(pairs)

        # Check key elements are present
        self.assertIn("Similar Bullets Detected", report)
        self.assertIn("general-00001", report)
        self.assertIn("general-00002", report)
        self.assertIn("92%", report)  # Similarity percentage
        self.assertIn("MERGE", report)  # Operation type in format
        self.assertIn("consolidation_operations", report)

    def test_format_pair_for_logging(self):
        """Test logging format includes key info."""
        bullet_a = Bullet(
            id="general-00001",
            content="A very long strategy description that should be truncated",
            section="general",
        )
        bullet_b = Bullet(
            id="general-00002", content="Another strategy", section="general"
        )

        log_str = format_pair_for_logging(bullet_a, bullet_b, 0.88)

        self.assertIn("general-00001", log_str)
        self.assertIn("general-00002", log_str)
        self.assertIn("88%", log_str)
        self.assertIn("...", log_str)  # Truncation indicator


class TestDeduplicationManager(unittest.TestCase):
    """Tests for DeduplicationManager."""

    def test_disabled_config_returns_none(self):
        """Test disabled config returns None for similarity report."""
        config = DeduplicationConfig(enabled=False)
        manager = DeduplicationManager(config)
        playbook = Playbook()

        report = manager.get_similarity_report(playbook)
        self.assertIsNone(report)

    def test_parse_consolidation_operations_merge(self):
        """Test parsing MERGE operation from response."""
        manager = DeduplicationManager()
        response = {
            "consolidation_operations": [
                {
                    "type": "MERGE",
                    "source_ids": ["a", "b"],
                    "keep_id": "a",
                    "merged_content": "Combined",
                    "reasoning": "Same thing",
                }
            ]
        }

        ops = manager.parse_consolidation_operations(response)

        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], MergeOp)
        self.assertEqual(ops[0].source_ids, ["a", "b"])
        self.assertEqual(ops[0].keep_id, "a")

    def test_parse_consolidation_operations_mixed(self):
        """Test parsing multiple operation types."""
        manager = DeduplicationManager()
        response = {
            "consolidation_operations": [
                {"type": "DELETE", "bullet_id": "x", "reasoning": "Redundant"},
                {"type": "KEEP", "bullet_ids": ["y", "z"], "reasoning": "Different"},
                {
                    "type": "UPDATE",
                    "bullet_id": "w",
                    "new_content": "New",
                    "reasoning": "Clarify",
                },
            ]
        }

        ops = manager.parse_consolidation_operations(response)

        self.assertEqual(len(ops), 3)
        self.assertIsInstance(ops[0], DeleteOp)
        self.assertIsInstance(ops[1], KeepOp)
        self.assertIsInstance(ops[2], UpdateOp)

    def test_parse_consolidation_operations_invalid_type(self):
        """Test parsing ignores unknown operation types."""
        manager = DeduplicationManager()
        response = {
            "consolidation_operations": [
                {"type": "UNKNOWN", "data": "something"},
                {"type": "DELETE", "bullet_id": "x", "reasoning": "Valid"},
            ]
        }

        ops = manager.parse_consolidation_operations(response)

        # Should only parse the valid DELETE
        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], DeleteOp)

    def test_parse_consolidation_operations_not_list(self):
        """Test parsing handles non-list gracefully."""
        manager = DeduplicationManager()
        response = {"consolidation_operations": "not a list"}

        ops = manager.parse_consolidation_operations(response)
        self.assertEqual(len(ops), 0)


class TestPlaybookDeduplicationIntegration(unittest.TestCase):
    """Integration tests for playbook deduplication features."""

    def test_similarity_decision_serialization(self):
        """Test similarity decisions are serialized/deserialized correctly."""
        playbook = Playbook()
        bullet_a = playbook.add_bullet("general", "Strategy A")
        bullet_b = playbook.add_bullet("general", "Strategy B")

        from ace.playbook import SimilarityDecision

        decision = SimilarityDecision(
            decision="KEEP",
            reasoning="Different purposes",
            decided_at="2024-01-01T00:00:00Z",
            similarity_at_decision=0.90,
        )
        playbook.set_similarity_decision(bullet_a.id, bullet_b.id, decision)

        # Serialize and deserialize
        data = playbook.to_dict()
        restored = Playbook.from_dict(data)

        # Check decision preserved
        self.assertTrue(restored.has_keep_decision(bullet_a.id, bullet_b.id))
        retrieved = restored.get_similarity_decision(bullet_a.id, bullet_b.id)
        self.assertEqual(retrieved.reasoning, "Different purposes")

    def test_bullet_embedding_field(self):
        """Test bullet embedding field is preserved."""
        playbook = Playbook()
        bullet = playbook.add_bullet("general", "Strategy")
        bullet.embedding = [0.1, 0.2, 0.3]

        data = playbook.to_dict()
        restored = Playbook.from_dict(data)

        restored_bullet = restored.get_bullet(bullet.id)
        self.assertEqual(restored_bullet.embedding, [0.1, 0.2, 0.3])

    def test_soft_delete_preserves_bullet(self):
        """Test soft delete keeps bullet but marks as invalid."""
        playbook = Playbook()
        bullet = playbook.add_bullet("general", "Strategy")
        bullet_id = bullet.id

        playbook.remove_bullet(bullet_id, soft=True)

        # Bullet still exists
        self.assertIsNotNone(playbook.get_bullet(bullet_id))

        # But is marked invalid
        self.assertEqual(playbook.get_bullet(bullet_id).status, "invalid")

        # And not in active bullets
        active = playbook.bullets(include_invalid=False)
        self.assertEqual(len(active), 0)


if __name__ == "__main__":
    unittest.main()
