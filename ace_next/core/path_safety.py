"""Path safety utilities — prevent directory traversal in file writes."""

from __future__ import annotations

import os
from pathlib import Path


def safe_resolve(path: str | Path) -> Path:
    """Resolve a path and reject directory-traversal sequences.

    Uses ``os.path.realpath`` to normalise the path, then verifies that no
    ``..`` component survived resolution.  This prevents writes to unintended
    locations when the caller-supplied *path* contains traversal sequences
    such as ``../``.

    Raises:
        ValueError: If the resolved path still contains ``..`` components.
    """
    resolved = Path(os.path.realpath(path))
    # After realpath, ".." should be gone.  Belt-and-suspenders check:
    if ".." in resolved.parts:
        raise ValueError(
            f"Path traversal detected: resolved path {resolved} "
            f"still contains '..' components."
        )
    return resolved


def safe_resolve_within(path: str | Path, base_dir: str | Path) -> Path:
    """Resolve *path* and ensure it stays within *base_dir*.

    Both *path* and *base_dir* are normalised via :func:`safe_resolve`
    before the containment check.

    Raises:
        ValueError: If the resolved *path* escapes *base_dir*.
    """
    resolved_base = safe_resolve(base_dir)
    resolved_path = safe_resolve(path)
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: {resolved_path} is not inside "
            f"the expected base directory {resolved_base}."
        ) from None
    return resolved_path
