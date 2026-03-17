"""Generic pipeline engine, create and Compose pipelines, control execution mode.

Public surface::

    from pipeline import (
        Pipeline,
        Branch,
        MergeStrategy,
        StepProtocol,
        PipelineHook,
        StepContext,
        SampleResult,
        CancellationToken,
        PipelineOrderError,
        PipelineConfigError,
        PipelineCancelled,
        BranchError,
    )
"""

from .branch import Branch, MergeStrategy
from .context import StepContext
from .errors import BranchError, CancellationToken, PipelineCancelled, PipelineConfigError, PipelineOrderError
from .pipeline import Pipeline
from .protocol import PipelineHook, SampleResult, StepProtocol

__all__ = [
    "Pipeline",
    "Branch",
    "MergeStrategy",
    "StepProtocol",
    "PipelineHook",
    "StepContext",
    "SampleResult",
    "CancellationToken",
    "PipelineOrderError",
    "PipelineConfigError",
    "PipelineCancelled",
    "BranchError",
]
