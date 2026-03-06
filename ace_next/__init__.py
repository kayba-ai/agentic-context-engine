"""ACE next — pipeline-based rewrite of the ACE framework.

All public symbols are lazily imported to keep ``import ace_next`` fast.
Direct attribute access (``ace_next.ACE``, ``from ace_next import ACE``)
works as before — the underlying module is loaded on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static analysis / IDE autocomplete — never executed at runtime.
    from pipeline import Branch, MergeStrategy, Pipeline, SampleResult, StepProtocol

    from .core import (
        ACEStepContext,
        EnvironmentResult,
        Sample,
        SimpleEnvironment,
        Skill,
        Skillbook,
        SkillbookView,
        TaskEnvironment,
        UpdateBatch,
        UpdateOperation,
    )
    from .deduplication import DeduplicationManager, SimilarityDetector
    from .implementations import Agent, Reflector, SkillManager
    from .integrations import wrap_skillbook_context
    from .protocols import DeduplicationConfig
    from .providers import (
        ACEModelConfig,
        InstructorClient,
        LiteLLMClient,
        ModelConfig,
        wrap_with_instructor,
    )
    from .rr import RRConfig, RRStep
    from .runners import (
        ACE,
        ACELiteLLM,
        ACERunner,
        BrowserUse,
        ClaudeCode,
        LangChain,
        TraceAnalyser,
    )
    from .steps import (
        AgentStep,
        ApplyStep,
        CheckpointStep,
        DeduplicateStep,
        EvaluateStep,
        ExportSkillbookMarkdownStep,
        LoadTracesStep,
        ObservabilityStep,
        PersistStep,
        ReflectStep,
        TagStep,
        UpdateStep,
        learning_tail,
    )
    from .steps.opik import OPIK_AVAILABLE, OpikStep, register_opik_litellm_callback

# ---- lazy import mapping: name -> (module_path, attribute) ----------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Pipeline engine (re-exported from pipeline/)
    "Pipeline": ("pipeline", "Pipeline"),
    "Branch": ("pipeline", "Branch"),
    "MergeStrategy": ("pipeline", "MergeStrategy"),
    "StepProtocol": ("pipeline", "StepProtocol"),
    "SampleResult": ("pipeline", "SampleResult"),
    # ACE context
    "ACEStepContext": ("ace_next.core", "ACEStepContext"),
    "SkillbookView": ("ace_next.core", "SkillbookView"),
    # Core data types
    "Skill": ("ace_next.core", "Skill"),
    "Skillbook": ("ace_next.core", "Skillbook"),
    "UpdateOperation": ("ace_next.core", "UpdateOperation"),
    "UpdateBatch": ("ace_next.core", "UpdateBatch"),
    "Sample": ("ace_next.core", "Sample"),
    "EnvironmentResult": ("ace_next.core", "EnvironmentResult"),
    "TaskEnvironment": ("ace_next.core", "TaskEnvironment"),
    "SimpleEnvironment": ("ace_next.core", "SimpleEnvironment"),
    # Implementations
    "Agent": ("ace_next.implementations", "Agent"),
    "Reflector": ("ace_next.implementations", "Reflector"),
    "SkillManager": ("ace_next.implementations", "SkillManager"),
    # Deduplication
    "DeduplicationConfig": ("ace_next.protocols", "DeduplicationConfig"),
    "DeduplicationManager": ("ace_next.deduplication", "DeduplicationManager"),
    "SimilarityDetector": ("ace_next.deduplication", "SimilarityDetector"),
    # Integrations
    "wrap_skillbook_context": ("ace_next.integrations", "wrap_skillbook_context"),
    # LLM providers + config
    "LiteLLMClient": ("ace_next.providers", "LiteLLMClient"),
    "InstructorClient": ("ace_next.providers", "InstructorClient"),
    "wrap_with_instructor": ("ace_next.providers", "wrap_with_instructor"),
    "ModelConfig": ("ace_next.providers", "ModelConfig"),
    "ACEModelConfig": ("ace_next.providers", "ACEModelConfig"),
    # Runners
    "ACE": ("ace_next.runners", "ACE"),
    "ACELiteLLM": ("ace_next.runners", "ACELiteLLM"),
    "ACERunner": ("ace_next.runners", "ACERunner"),
    "BrowserUse": ("ace_next.runners", "BrowserUse"),
    "ClaudeCode": ("ace_next.runners", "ClaudeCode"),
    "LangChain": ("ace_next.runners", "LangChain"),
    "TraceAnalyser": ("ace_next.runners", "TraceAnalyser"),
    # Steps
    "AgentStep": ("ace_next.steps", "AgentStep"),
    "EvaluateStep": ("ace_next.steps", "EvaluateStep"),
    "ReflectStep": ("ace_next.steps", "ReflectStep"),
    "TagStep": ("ace_next.steps", "TagStep"),
    "UpdateStep": ("ace_next.steps", "UpdateStep"),
    "ApplyStep": ("ace_next.steps", "ApplyStep"),
    "DeduplicateStep": ("ace_next.steps", "DeduplicateStep"),
    "CheckpointStep": ("ace_next.steps", "CheckpointStep"),
    "LoadTracesStep": ("ace_next.steps", "LoadTracesStep"),
    "ExportSkillbookMarkdownStep": ("ace_next.steps", "ExportSkillbookMarkdownStep"),
    "ObservabilityStep": ("ace_next.steps", "ObservabilityStep"),
    "PersistStep": ("ace_next.steps", "PersistStep"),
    "learning_tail": ("ace_next.steps", "learning_tail"),
    # Recursive Reflector
    "RRStep": ("ace_next.rr", "RRStep"),
    "RRConfig": ("ace_next.rr", "RRConfig"),
    # Observability
    "OpikStep": ("ace_next.steps.opik", "OpikStep"),
    "OPIK_AVAILABLE": ("ace_next.steps.opik", "OPIK_AVAILABLE"),
    "register_opik_litellm_callback": (
        "ace_next.steps.opik",
        "register_opik_litellm_callback",
    ),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        # Cache on the module so __getattr__ is only called once per name.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'ace_next' has no attribute {name!r}")


__all__ = [
    # Pipeline composition
    "Pipeline",
    "Branch",
    "MergeStrategy",
    "StepProtocol",
    "SampleResult",
    # ACE context
    "ACEStepContext",
    "SkillbookView",
    # Core data types
    "Skill",
    "Skillbook",
    "UpdateOperation",
    "UpdateBatch",
    # Environments
    "Sample",
    "EnvironmentResult",
    "TaskEnvironment",
    "SimpleEnvironment",
    # Implementations
    "Agent",
    "Reflector",
    "SkillManager",
    # LLM providers + config
    "LiteLLMClient",
    "InstructorClient",
    "wrap_with_instructor",
    "ModelConfig",
    "ACEModelConfig",
    # Runners
    "ACE",
    "ACELiteLLM",
    "ACERunner",
    "BrowserUse",
    "ClaudeCode",
    "LangChain",
    "TraceAnalyser",
    # Steps
    "AgentStep",
    "EvaluateStep",
    "ReflectStep",
    "TagStep",
    "UpdateStep",
    "ApplyStep",
    "DeduplicateStep",
    "CheckpointStep",
    "LoadTracesStep",
    "ExportSkillbookMarkdownStep",
    "ObservabilityStep",
    "PersistStep",
    "learning_tail",
    # Recursive Reflector
    "RRStep",
    "RRConfig",
    # Deduplication
    "DeduplicationConfig",
    "DeduplicationManager",
    "SimilarityDetector",
    # Observability
    "OpikStep",
    "OPIK_AVAILABLE",
    "register_opik_litellm_callback",
    # Utilities
    "wrap_skillbook_context",
]
