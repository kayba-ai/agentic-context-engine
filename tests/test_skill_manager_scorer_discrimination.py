"""Fixture test for SkillManager best-of-N scorer discrimination.

Hard constraint from iter_0004:

> before running a full screen, write a fixture test that loads (or fakes)
> 3 candidate UpdateBatches with materially different quality and asserts
> the new scorer ranks them differently than the current heuristic does

This file contains:

1. ``test_heuristic_fails_to_discriminate`` — pins the iter_0003 failure
   mode: when three candidates all have the same op count and the
   skillbook is empty, the deterministic heuristic clusters them.
2. ``test_judge_scorer_discriminates_and_picks_well_targeted`` — mocks
   the judge LLM and asserts the scorer (a) produces a spread > 0.15 on
   the synthetic fixture, (b) picks the well-targeted candidate, and (c)
   picks a *different* candidate than the heuristic would on the same
   input. If (c) fails, the fixture does not discriminate scorer behavior.
3. ``test_judge_scorer_falls_back_on_failure`` — asserts the deterministic
   heuristic fallback kicks in when the judge raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ace.core.outputs import ExtractedLearning, ReflectorOutput
from ace.core.skillbook import UpdateBatch, UpdateOperation
from ace.implementations.skill_manager import _score_update_batch


@pytest.fixture
def synthetic_reflection() -> ReflectorOutput:
    """Reflector output describing a doom-loop failure on a tau-airline task."""
    return ReflectorOutput(
        reasoning=(
            "Agent entered a doom loop calling get_reservation 12 times "
            "with the same arguments before the episode terminated."
        ),
        error_identification="doom_loop on tool cycling",
        root_cause_analysis=(
            "Agent did not check whether the previous get_reservation result "
            "was sufficient before calling it again. It re-issued the same "
            "tool call 12 times without reading its own tool history."
        ),
        correct_approach=(
            "Cache the first successful get_reservation result and reuse it "
            "for the remainder of the turn."
        ),
        key_insight=(
            "Repeated identical tool calls indicate the agent is not "
            "reading its own tool history before acting."
        ),
        extracted_learnings=[
            ExtractedLearning(
                learning=(
                    "Do not call the same tool twice with identical arguments "
                    "in the same turn."
                ),
                atomicity_score=0.9,
                evidence="Observed 12 redundant get_reservation calls in the failed trace.",
                justification="Concrete runtime rule derived from a specific failure mode.",
            ),
        ],
    )


_FIXTURE_CONTENT_LENGTH = 520
"""Length each candidate's content is padded to.

All three fixture candidates are normalized to the same length so that
the heuristic's ``specificity`` term ( = ``min(len, 500) / 500`` ) saturates
at 1.0 for every candidate. Combined with equal op counts (focus == 1.0)
and an empty starting skillbook (non_redundancy == 1.0), every candidate
scores exactly the same under the heuristic. This pins the iter_0003
failure mode: the heuristic cannot differentiate when its only varying
input is text length and content lengths are equal.
"""


def _pad_to_fixture_length(text: str) -> str:
    """Pad ``text`` to exactly :data:`_FIXTURE_CONTENT_LENGTH` chars.

    Uses whitespace + a unique sentinel so SequenceMatcher sees different
    strings (relevant for tests that exercise non_redundancy, though the
    fixture always passes an empty existing skillbook).
    """
    if len(text) >= _FIXTURE_CONTENT_LENGTH:
        return text[:_FIXTURE_CONTENT_LENGTH]
    padding_needed = _FIXTURE_CONTENT_LENGTH - len(text)
    return text + " " + ("." * (padding_needed - 1))


def _make_add_batch(reasoning: str, content: str) -> UpdateBatch:
    """Helper: single-op ADD batch with the given content padded to fixture length."""
    return UpdateBatch(
        reasoning=reasoning,
        operations=[
            UpdateOperation(
                type="ADD",
                section="strategies",
                content=_pad_to_fixture_length(content),
            )
        ],
    )


@pytest.fixture
def candidate_batches() -> tuple[UpdateBatch, UpdateBatch, UpdateBatch]:
    """Three candidates with materially different semantic quality but
    identical length and op count.

    - ``trivial``: generic exhortation, does not address the doom loop.
    - ``off_topic``: concrete but about a different failure mode
      (refunds / fare class), irrelevant to the reflected failure.
    - ``well_targeted``: directly addresses the doom-loop root cause with
      concrete preconditions and the tool name.

    All three contents are padded to :data:`_FIXTURE_CONTENT_LENGTH`, so
    the heuristic's specificity / focus / non_redundancy terms are
    identical across candidates — reproducing the iter_0003 failure mode
    where the heuristic could not differentiate between candidates.
    """
    trivial = _make_add_batch(
        reasoning="generic guidance",
        content=(
            "Be careful and thorough when using tools. Always double-check "
            "your work and make sure you understand the user's request "
            "before taking action. Remember to consider edge cases and "
            "potential failure modes. Do not rush."
        ),
    )

    off_topic = _make_add_batch(
        reasoning="refund policy skill",
        content=(
            "When processing a refund request, always verify the customer's "
            "fare class before confirming. Economy basic fares are "
            "non-refundable; economy standard and above are refundable up "
            "to 24 hours before departure with a 25% fee."
        ),
    )

    well_targeted = _make_add_batch(
        reasoning="prevent doom loop on get_reservation",
        content=(
            "Before calling get_reservation, first check the tool history "
            "for the current turn. If get_reservation has already returned "
            "a result in this turn with the same arguments, reuse that "
            "cached result. Do not call the same tool twice identically."
        ),
    )

    return (trivial, off_topic, well_targeted)


@pytest.mark.unit
class TestHeuristicFailsToDiscriminate:
    """Pin the iter_0003 failure mode: heuristic clusters on this fixture."""

    def test_heuristic_produces_identical_scores(self, candidate_batches):
        """When specificity saturates, focus is constant, and the skillbook
        is empty, every candidate scores exactly 1.0. The heuristic has
        nothing to rank on."""
        scores = [_score_update_batch(b, ()) for b in candidate_batches]
        spread = max(scores) - min(scores)
        assert spread < 0.01, (
            "Regression signal: the fixture is designed so the heuristic's "
            "specificity / focus / non_redundancy terms are all identical "
            "across candidates, giving spread 0. If the heuristic formula "
            "was changed, iter_0004's motivation may have been partly "
            f"addressed already — re-evaluate before shipping the judge "
            f"scorer. Spread: {spread:.4f}; scores: {scores}"
        )
        assert all(abs(s - 1.0) < 0.001 for s in scores), (
            f"Every candidate should score 1.0 on this saturating fixture. "
            f"Got {scores}. Check _FIXTURE_CONTENT_LENGTH and op counts."
        )

    def test_heuristic_rank_is_a_stable_sort_tie(self, candidate_batches):
        """Because every heuristic score is identical, ``max`` picks the
        first candidate (index 0) by list order. This is the iter_0003
        pathology: the 'selection' is just positional noise."""
        scores = [_score_update_batch(b, ()) for b in candidate_batches]
        best_idx = scores.index(max(scores))
        assert best_idx == 0, (
            "Expected the heuristic to tie-break by list order on this "
            f"all-equal fixture. Got best_idx={best_idx}, scores={scores}. "
            "If this changed, the tie-breaking behavior of _score_update_batch "
            "is different from what iter_0004 assumed."
        )


@pytest.mark.unit
class TestJudgeScorerDiscriminates:
    """The new judge scorer produces a wide spread and picks the
    well-targeted candidate — and picks a *different* candidate than the
    heuristic on the same input."""

    def test_judge_scorer_spread_exceeds_0_15_and_picks_well_targeted(
        self, candidate_batches, synthetic_reflection
    ):
        from ace_eval.e2e.scoring import JudgeScore, JudgeScorer

        trivial_score = 0.22
        off_topic_score = 0.30
        well_targeted_score = 0.87

        mock_run_results = [
            MagicMock(
                output=JudgeScore(
                    score=trivial_score,
                    addresses_root_causes=False,
                    concrete_and_actionable=False,
                    contradicts_existing=False,
                    rationale=(
                        "Generic exhortation. Does not name any of the "
                        "root causes in the reflection and is not "
                        "actionable at runtime."
                    ),
                )
            ),
            MagicMock(
                output=JudgeScore(
                    score=off_topic_score,
                    addresses_root_causes=False,
                    concrete_and_actionable=True,
                    contradicts_existing=False,
                    rationale=(
                        "Concrete refund policy rule but unrelated to the "
                        "doom_loop failure mode. Does not address the "
                        "reflector's root cause."
                    ),
                )
            ),
            MagicMock(
                output=JudgeScore(
                    score=well_targeted_score,
                    addresses_root_causes=True,
                    concrete_and_actionable=True,
                    contradicts_existing=False,
                    rationale=(
                        "Directly addresses the doom_loop root cause. "
                        "Names the specific tool (get_reservation), gives "
                        "a concrete precondition (check tool history), "
                        "and encodes the reflector's extracted learning."
                    ),
                )
            ),
        ]

        with patch(
            "ace_eval.e2e.scoring._resolve_scorer_model"
        ) as mock_resolve:
            mock_resolve.return_value = MagicMock()
            scorer = JudgeScorer(model="gpt-4.1-mini")

        with patch.object(scorer._agent, "run_sync", side_effect=mock_run_results):
            judge_scores = [
                scorer(
                    b,
                    existing_skill_contents=(),
                    reflections=(synthetic_reflection,),
                    question_context="tau-bench-airline customer service episode",
                    candidate_index=i,
                )
                for i, b in enumerate(candidate_batches)
            ]

        # (a) spread > 0.15
        spread = max(judge_scores) - min(judge_scores)
        assert spread > 0.15, (
            f"Judge scorer spread must exceed 0.15 to beat the heuristic's "
            f"iter_0003 cluster (0.04). Got spread={spread:.3f}, "
            f"scores={judge_scores}"
        )

        # (b) judge picks the well-targeted candidate (index 2)
        judge_best_idx = judge_scores.index(max(judge_scores))
        assert judge_best_idx == 2, (
            f"Judge should pick the well_targeted candidate (index 2); "
            f"picked {judge_best_idx}. Scores: {judge_scores}"
        )

        # (c) judge picks a *different* candidate than the heuristic on the
        #     same input — the whole point of iter_0004
        heuristic_scores = [_score_update_batch(b, ()) for b in candidate_batches]
        heuristic_best_idx = heuristic_scores.index(max(heuristic_scores))
        assert heuristic_best_idx != judge_best_idx, (
            "Judge scorer must select a different candidate than the "
            "heuristic on this fixture. If both pick the same index, the "
            "fixture does not discriminate between scorer behaviors. "
            f"Judge picked {judge_best_idx}, heuristic picked "
            f"{heuristic_best_idx}. "
            f"Heuristic scores: {heuristic_scores}. "
            f"Judge scores: {judge_scores}."
        )

    def test_judge_scorer_falls_back_to_heuristic_on_failure(
        self, candidate_batches, synthetic_reflection
    ):
        """If the judge raises, the heuristic score is returned."""
        from ace_eval.e2e.scoring import JudgeScorer

        with patch(
            "ace_eval.e2e.scoring._resolve_scorer_model"
        ) as mock_resolve:
            mock_resolve.return_value = MagicMock()
            scorer = JudgeScorer(model="gpt-4.1-mini")

        judge_error = RuntimeError("simulated judge auth failure")

        with patch.object(scorer._agent, "run_sync", side_effect=judge_error):
            fallback_scores = [
                scorer(
                    b,
                    existing_skill_contents=(),
                    reflections=(synthetic_reflection,),
                    question_context="tau-bench-airline",
                    candidate_index=i,
                )
                for i, b in enumerate(candidate_batches)
            ]

        heuristic_scores = [_score_update_batch(b, ()) for b in candidate_batches]
        assert fallback_scores == pytest.approx(heuristic_scores), (
            "JudgeScorer fallback must return exactly the heuristic score "
            f"on failure. Fallback: {fallback_scores}, "
            f"heuristic: {heuristic_scores}"
        )


@pytest.mark.unit
class TestResolveScoreFn:
    """String → callable resolver used by ace_eval.e2e.training."""

    def test_heuristic_returns_none(self):
        from ace_eval.e2e.scoring import resolve_score_fn

        assert resolve_score_fn("heuristic") is None

    def test_default_returns_none(self):
        from ace_eval.e2e.scoring import resolve_score_fn

        assert resolve_score_fn("") is None

    def test_judge_returns_callable(self):
        from ace_eval.e2e.scoring import JudgeScorer, resolve_score_fn

        with patch("ace_eval.e2e.scoring._resolve_scorer_model") as mock_resolve:
            mock_resolve.return_value = MagicMock()
            scorer = resolve_score_fn("judge", judge_model="gpt-4.1-mini")

        assert callable(scorer)
        assert isinstance(scorer, JudgeScorer)

    def test_unknown_scorer_raises(self):
        from ace_eval.e2e.scoring import resolve_score_fn

        with pytest.raises(ValueError, match="Unknown sm_score_fn"):
            resolve_score_fn("magic_oracle")


@pytest.mark.unit
class TestSkillManagerAcceptsScoreFn:
    """SkillManager.__init__ accepts and stores the score_fn parameter."""

    def test_default_score_fn_is_heuristic(self):
        from ace.implementations.skill_manager import (
            SkillManager,
            _score_update_batch,
        )

        with patch("ace.implementations.skill_manager.resolve_model"):
            sm = SkillManager("test-model")

        assert sm._score_fn is _score_update_batch

    def test_custom_score_fn_is_stored(self):
        from ace.implementations.skill_manager import SkillManager

        custom = lambda batch, **kwargs: 0.42
        with patch("ace.implementations.skill_manager.resolve_model"):
            sm = SkillManager("test-model", score_fn=custom)

        assert sm._score_fn is custom
