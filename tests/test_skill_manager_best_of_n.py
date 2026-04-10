"""Tests for SkillManager best-of-N candidate selection (L3 intervention)."""

import pytest

from ace.core.skillbook import UpdateBatch, UpdateOperation
from ace.implementations.skill_manager import _score_update_batch


# ---------------------------------------------------------------------------
# _score_update_batch tests
# ---------------------------------------------------------------------------


def _make_batch(ops: list[tuple[str, str, str]]) -> UpdateBatch:
    """Helper: (type, section, content) triples → UpdateBatch."""
    return UpdateBatch(
        reasoning="test",
        operations=[
            UpdateOperation(type=t, section=s, content=c) for t, s, c in ops
        ],
    )


class TestScoreUpdateBatch:
    """Unit tests for the scoring function."""

    def test_empty_batch_scores_zero(self):
        batch = UpdateBatch(reasoning="nothing", operations=[])
        assert _score_update_batch(batch, ()) == 0.0

    def test_remove_only_batch_scores_zero(self):
        batch = UpdateBatch(
            reasoning="cleanup",
            operations=[
                UpdateOperation(type="REMOVE", section="s", skill_id="sk_1")
            ],
        )
        assert _score_update_batch(batch, ()) == 0.0

    def test_specific_content_scores_higher(self):
        short = _make_batch([("ADD", "s", "Be careful.")])
        long = _make_batch([
            ("ADD", "s", "When the customer requests a booking change within 24 hours of departure, always verify the fare class before confirming the change.")
        ])
        assert _score_update_batch(long, ()) > _score_update_batch(short, ())

    def test_focused_batch_scores_higher(self):
        focused = _make_batch([("ADD", "s", "x" * 200)])
        bloated = _make_batch([
            ("ADD", "s", "x" * 200),
            ("ADD", "s", "y" * 200),
            ("ADD", "s", "z" * 200),
            ("ADD", "s", "w" * 200),
            ("ADD", "s", "v" * 200),
            ("ADD", "s", "u" * 200),
        ])
        assert _score_update_batch(focused, ()) > _score_update_batch(bloated, ())

    def test_non_redundant_scores_higher(self):
        existing = ("Always verify fare class before booking changes.",)
        novel = _make_batch([
            ("ADD", "s", "Check seat availability before suggesting upgrades.")
        ])
        redundant = _make_batch([
            ("ADD", "s", "Always verify fare class before booking changes.")
        ])
        assert _score_update_batch(novel, existing) > _score_update_batch(
            redundant, existing
        )

    def test_score_in_unit_range(self):
        batch = _make_batch([
            ("ADD", "s", "a" * 300),
            ("UPDATE", "s", "b" * 100),
        ])
        score = _score_update_batch(batch, ("existing skill content",))
        assert 0.0 <= score <= 1.0

    def test_no_existing_skills_gives_full_non_redundancy(self):
        batch = _make_batch([("ADD", "s", "new skill content")])
        score = _score_update_batch(batch, ())
        # Non-redundancy component should be 1.0 (no overlap)
        # Focus component should be 1.0 (1 op <= 3)
        # Only specificity varies
        assert score > 0.3  # at least focus + non_redundancy contribution


@pytest.mark.unit
class TestSkillManagerNCandidatesInit:
    """Test that n_candidates parameter is accepted and stored."""

    def test_default_n_candidates_is_one(self):
        from unittest.mock import patch

        with patch("ace.implementations.skill_manager.resolve_model"):
            from ace.implementations.skill_manager import SkillManager

            sm = SkillManager("test-model")
            assert sm._n_candidates == 1

    def test_custom_n_candidates(self):
        from unittest.mock import patch

        with patch("ace.implementations.skill_manager.resolve_model"):
            from ace.implementations.skill_manager import SkillManager

            sm = SkillManager("test-model", n_candidates=3)
            assert sm._n_candidates == 3

    def test_n_candidates_clamped_to_one(self):
        from unittest.mock import patch

        with patch("ace.implementations.skill_manager.resolve_model"):
            from ace.implementations.skill_manager import SkillManager

            sm = SkillManager("test-model", n_candidates=0)
            assert sm._n_candidates == 1
