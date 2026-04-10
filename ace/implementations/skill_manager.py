"""SkillManager — transforms reflections into actionable skillbook updates.

Uses PydanticAI for structured output validation with automatic retry
and error feedback.
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from typing import Any, Callable, Union

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings

from ..core.context import SkillbookView
from ..core.outputs import ReflectorOutput, SkillManagerOutput
from ..core.skillbook import Skillbook, UpdateBatch
from ..providers.pydantic_ai import resolve_model
from .prompts import SKILL_MANAGER_PROMPT

logger = logging.getLogger(__name__)


ScoreFn = Callable[..., float]
"""Callable that scores one candidate UpdateBatch for best-of-N selection.

Scorers receive the candidate batch as the first positional argument and
accept keyword arguments ``existing_skill_contents``, ``reflections``,
``question_context``, and ``candidate_index``. Implementations may ignore
any kwargs they do not need via ``**_unused``.
"""


def _score_update_batch(
    batch: UpdateBatch,
    existing_skill_contents: tuple[str, ...] = (),
    **_unused: Any,
) -> float:
    """Score a candidate UpdateBatch by specificity, focus, and non-redundancy.

    Returns a composite score in [0, 1] used by best-of-N selection.
    """
    add_update_ops = [
        op for op in batch.operations if op.type.upper() in ("ADD", "UPDATE")
    ]
    if not add_update_ops:
        return 0.0

    # Specificity: prefer concrete, detailed content (capped at 500 chars)
    specificity_scores = []
    for op in add_update_ops:
        content = op.content or ""
        specificity_scores.append(min(len(content), 500) / 500)
    specificity = sum(specificity_scores) / len(specificity_scores)

    # Focus: prefer fewer, targeted operations over kitchen-sink batches
    n_ops = len(batch.operations)
    focus = 1.0 if n_ops <= 3 else max(0.0, 1.0 - (n_ops - 3) * 0.15)

    # Non-redundancy: penalize overlap with existing skillbook skills
    max_overlap = 0.0
    for op in add_update_ops:
        content = op.content or ""
        for existing in existing_skill_contents:
            ratio = SequenceMatcher(None, content, existing).ratio()
            if ratio > max_overlap:
                max_overlap = ratio
    non_redundancy = 1.0 - max_overlap

    return 0.4 * specificity + 0.3 * focus + 0.3 * non_redundancy


class SkillManager:
    """Transforms reflections into actionable skillbook updates.

    The SkillManager is the third ACE role. It analyzes the Reflector's
    output and decides how to update the skillbook — adding new
    strategies, updating existing ones, or removing harmful patterns.

    .. note::

        In ``ace``, deduplication is handled by a separate
        :class:`DeduplicateStep` in the pipeline. The SkillManager
        role only produces :class:`SkillManagerOutput`; it does not call
        a dedup manager itself.

    Args:
        model: Model identifier string. Supports any LiteLLM model
            or PydanticAI-native identifier.
        prompt_template: Custom prompt template (defaults to
            :data:`SKILL_MANAGER_PROMPT`).
        max_retries: Maximum retries for structured output validation.
        model_settings: Optional PydanticAI ``ModelSettings``.

    Example::

        sm = SkillManager("gpt-4o-mini")
        output = sm.update_skills(
            reflections=(reflection_output,),
            skillbook=skillbook,
            question_context="Math problem solving",
            progress="5/10 correct",
        )
        skillbook.apply_update(output.update)
    """

    def __init__(
        self,
        model: str,
        *,
        prompt_template: str = SKILL_MANAGER_PROMPT,
        max_retries: int = 3,
        model_settings: ModelSettings | None = None,
        n_candidates: int = 1,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self._prompt_template = prompt_template
        self._n_candidates = max(1, n_candidates)
        self._score_fn: ScoreFn = score_fn or _score_update_batch
        self._agent = PydanticAgent(
            resolve_model(model),
            output_type=SkillManagerOutput,
            retries=max_retries,
            model_settings=model_settings,
            defer_model_check=True,
        )

    def update_skills(
        self,
        *,
        reflections: tuple[ReflectorOutput, ...],
        skillbook: Union[SkillbookView, Skillbook],
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """Generate update operations based on the reflections.

        This method signature matches :class:`SkillManagerLike`.

        Args:
            reflections: Tuple of Reflector analyses (1-tuple for single,
                N-tuple for batch).
            skillbook: Current skillbook (needs ``as_prompt``, ``stats``).
            question_context: Description of the task domain.
            progress: Current progress summary (e.g. ``"5/10 correct"``).
            **kwargs: Accepted for protocol compatibility but not forwarded.

        Returns:
            :class:`SkillManagerOutput` containing the update operations.
        """
        reflections_data = [
            {
                "reasoning": r.reasoning,
                "error_identification": r.error_identification,
                "root_cause_analysis": r.root_cause_analysis,
                "correct_approach": r.correct_approach,
                "key_insight": r.key_insight,
                "extracted_learnings": [
                    l.model_dump() for l in r.extracted_learnings
                ],
            }
            for r in reflections
        ]

        prompt = self._prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflections=json.dumps(reflections_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        existing_skill_contents = tuple(
            s.content for s in skillbook.skills()
        )

        candidates: list[tuple[SkillManagerOutput, float]] = []
        for i in range(self._n_candidates):
            try:
                result = self._agent.run_sync(prompt)
                output = result.output
                usage = result.usage()
                output.raw = {
                    "usage": {
                        "prompt_tokens": usage.input_tokens or 0,
                        "completion_tokens": usage.output_tokens or 0,
                        "total_tokens": usage.total_tokens or 0,
                    },
                    "candidate_index": i,
                }
                score = self._score_fn(
                    output.update,
                    existing_skill_contents=existing_skill_contents,
                    reflections=reflections,
                    question_context=question_context,
                    candidate_index=i,
                )
                candidates.append((output, score))
            except Exception:
                logger.warning(
                    "SkillManager candidate %d failed, skipping", i
                )
                continue

        if not candidates:
            raise RuntimeError("All SkillManager candidates failed")

        best_output, best_score = max(candidates, key=lambda x: x[1])
        selected_index = best_output.raw.get("candidate_index", -1)

        if self._n_candidates > 1:
            logger.info(
                "SkillManager best-of-%d: selected candidate %d "
                "(score=%.3f) from %d candidates",
                self._n_candidates,
                selected_index,
                best_score,
                len(candidates),
            )
            best_output.raw["best_of_n"] = {
                "n_candidates": self._n_candidates,
                "n_generated": len(candidates),
                "scores": [s for _, s in candidates],
                "selected_score": best_score,
                "selected_index": selected_index,
                "candidates": [
                    {
                        "candidate_index": c.raw.get("candidate_index", i),
                        "score": s,
                        "update": c.update.to_json(),
                    }
                    for i, (c, s) in enumerate(candidates)
                ],
            }

        return best_output
