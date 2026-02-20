"""Pipeline error types."""

from __future__ import annotations


class PipelineOrderError(Exception):
    """A step requires a field that no earlier step provides."""


class PipelineConfigError(Exception):
    """Invalid pipeline wiring.

    Examples:
    - More than one ``async_boundary = True`` step in the same pipeline.
    - An ``async_boundary = True`` step inside a Branch child.
    """


class BranchError(Exception):
    """One or more branch pipelines failed.

    All branches always run to completion before this is raised.
    ``failures`` contains the full list of exceptions — one per failed branch.
    """

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        super().__init__(
            f"{len(failures)} branch(es) failed: "
            + "; ".join(type(e).__name__ for e in failures)
        )
