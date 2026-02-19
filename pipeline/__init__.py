"""Generic pipeline engine — no ACE concepts, zero upward dependencies.

Public surface::

    from pipeline import (
        Pipeline,
        Branch,
        MergeStrategy,
        StepProtocol,
        StepContext,
        SampleResult,
        PipelineOrderError,
        PipelineConfigError,
        BranchError,
    )
"""

from .branch import Branch, MergeStrategy
from .context import StepContext
from .errors import BranchError, PipelineConfigError, PipelineOrderError
from .pipeline import Pipeline
from .protocol import SampleResult, StepProtocol

__all__ = [
    "Pipeline",
    "Branch",
    "MergeStrategy",
    "StepProtocol",
    "StepContext",
    "SampleResult",
    "PipelineOrderError",
    "PipelineConfigError",
    "BranchError",
]
