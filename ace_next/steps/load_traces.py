"""LoadTracesStep — generic file-to-trace loader for JSONL files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..core.context import ACEStepContext

logger = logging.getLogger(__name__)


class LoadTracesStep:
    """Read a JSONL file from disk and place parsed events on ``ctx.trace``.

    Reads the file at ``ctx.sample`` (a ``str`` or ``Path``), parses each
    line as JSON, and places the resulting ``list[dict]`` on ``ctx.trace``.
    Unparseable lines are silently skipped.

    If the file is empty or missing, ``ctx.trace`` is set to an empty list.
    """

    requires = frozenset({"sample"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        path = Path(ctx.sample)
        events: list[dict] = []

        if not path.exists():
            logger.warning("Trace file not found: %s", path)
            return ctx.replace(trace=events)

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read trace file %s: %s", path, exc)
            return ctx.replace(trace=events)

        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug(
                    "Skipping unparseable line %d in %s", line_num, path.name
                )
                continue

        return ctx.replace(trace=events)
