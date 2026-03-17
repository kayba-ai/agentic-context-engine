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
        cancel_token_var,
        PipelineOrderError,
        PipelineConfigError,
        PipelineCancelled,
        BranchError,
    )
"""

from .branch import Branch, MergeStrategy
from .context import StepContext
from .errors import BranchError, CancellationToken, PipelineCancelled, PipelineConfigError, PipelineOrderError, cancel_token_var
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
    "cancel_token_var",
    "PipelineOrderError",
    "PipelineConfigError",
    "PipelineCancelled",
    "BranchError",
]
