"""ACE — Agentic Context Engine.

All public symbols are lazily imported to keep ``import ace`` fast.
Direct attribute access (``ace.ACE``, ``from ace import ACE``)
works — the underlying module is loaded on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static analysis / IDE autocomplete — never executed at runtime.
    from pipeline import Branch, MergeStrategy, Pipeline, SampleResult, StepProtocol

    from .core import (
        ACEStepContext,
        EnvironmentResult,
        InsightSource,
        Sample,
        SimpleEnvironment,
        Skill,
        Skillbook,
        SkillbookView,
        TaskEnvironment,
        TraceIdentity,
        UpdateBatch,
        UpdateOperation,
    )
    from .deduplication import DeduplicationManager, SimilarityDetector
    from .implementations import Agent, Reflector, SkillManager
    from .integrations import wrap_skillbook_context
    from .protocols import DeduplicationConfig
    from .providers import ACEModelConfig, ModelConfig
    from .steps.rr import RRConfig, RRStep
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
        UpdateStep,
        learning_tail,
    )

# ---- lazy import mapping: name -> (module_path, attribute) ----------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Pipeline engine (re-exported from pipeline/)
    "Pipeline": ("pipeline", "Pipeline"),
    "Branch": ("pipeline", "Branch"),
    "MergeStrategy": ("pipeline", "MergeStrategy"),
    "StepProtocol": ("pipeline", "StepProtocol"),
    "SampleResult": ("pipeline", "SampleResult"),
    # ACE context
    "ACEStepContext": ("ace.core", "ACEStepContext"),
    "SkillbookView": ("ace.core", "SkillbookView"),
    # Core data types
    "InsightSource": ("ace.core", "InsightSource"),
    "Skill": ("ace.core", "Skill"),
    "Skillbook": ("ace.core", "Skillbook"),
    "TraceIdentity": ("ace.core", "TraceIdentity"),
    "UpdateOperation": ("ace.core", "UpdateOperation"),
    "UpdateBatch": ("ace.core", "UpdateBatch"),
    "Sample": ("ace.core", "Sample"),
    "EnvironmentResult": ("ace.core", "EnvironmentResult"),
    "TaskEnvironment": ("ace.core", "TaskEnvironment"),
    "SimpleEnvironment": ("ace.core", "SimpleEnvironment"),
    # Implementations
    "Agent": ("ace.implementations", "Agent"),
    "Reflector": ("ace.implementations", "Reflector"),
    "SkillManager": ("ace.implementations", "SkillManager"),
    # Deduplication
    "DeduplicationConfig": ("ace.protocols", "DeduplicationConfig"),
    "DeduplicationManager": ("ace.deduplication", "DeduplicationManager"),
    "SimilarityDetector": ("ace.deduplication", "SimilarityDetector"),
    # Integrations
    "wrap_skillbook_context": ("ace.integrations", "wrap_skillbook_context"),
    # Config
    "ModelConfig": ("ace.providers", "ModelConfig"),
    "ACEModelConfig": ("ace.providers", "ACEModelConfig"),
    # Runners
    "ACE": ("ace.runners", "ACE"),
    "ACELiteLLM": ("ace.runners", "ACELiteLLM"),
    "ACERunner": ("ace.runners", "ACERunner"),
    "BrowserUse": ("ace.runners", "BrowserUse"),
    "ClaudeCode": ("ace.runners", "ClaudeCode"),
    "LangChain": ("ace.runners", "LangChain"),
    "TraceAnalyser": ("ace.runners", "TraceAnalyser"),
    # Steps
    "AgentStep": ("ace.steps", "AgentStep"),
    "EvaluateStep": ("ace.steps", "EvaluateStep"),
    "ReflectStep": ("ace.steps", "ReflectStep"),
    "UpdateStep": ("ace.steps", "UpdateStep"),
    "ApplyStep": ("ace.steps", "ApplyStep"),
    "DeduplicateStep": ("ace.steps", "DeduplicateStep"),
    "CheckpointStep": ("ace.steps", "CheckpointStep"),
    "LoadTracesStep": ("ace.steps", "LoadTracesStep"),
    "ExportSkillbookMarkdownStep": ("ace.steps", "ExportSkillbookMarkdownStep"),
    "ObservabilityStep": ("ace.steps", "ObservabilityStep"),
    "PersistStep": ("ace.steps", "PersistStep"),
    "learning_tail": ("ace.steps", "learning_tail"),
    # Recursive Reflector
    "RRStep": ("ace.steps.rr", "RRStep"),
    "RRConfig": ("ace.steps.rr", "RRConfig"),
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
    raise AttributeError(f"module 'ace' has no attribute {name!r}")


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
    "InsightSource",
    "Skill",
    "Skillbook",
    "TraceIdentity",
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
    # Config
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
    # Utilities
    "wrap_skillbook_context",
]
