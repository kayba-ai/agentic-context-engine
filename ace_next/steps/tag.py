"""TagStep — applies skill tags from the Reflector's output to the skillbook."""

from __future__ import annotations

import logging

from ..core.skillbook import Skillbook

from ..core.context import ACEStepContext

logger = logging.getLogger(__name__)


class TagStep:
    """Tag skills on the real Skillbook based on the reflection's skill_tags.

    Side-effect step — mutates ``self.skillbook`` (the real Skillbook,
    injected via constructor).  ``max_workers = 1`` serialises writes.

    Hallucinated skill IDs are logged at WARNING rather than aborting.
    """

    requires = frozenset({"reflection"})
    provides = frozenset()

    max_workers = 1

    def __init__(self, skillbook: Skillbook) -> None:
        self.skillbook = skillbook

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        for skill_tag in ctx.reflection.skill_tags:
            try:
                self.skillbook.tag_skill(skill_tag.id, skill_tag.tag)
            except (ValueError, KeyError):
                logger.warning(
                    "TagStep: skill_id %r not found, skipping tag %r",
                    skill_tag.id,
                    skill_tag.tag,
                )
        return ctx
