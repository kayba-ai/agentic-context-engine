"""PydanticAI-based Recursive Reflector agent factories.

Each factory creates a PydanticAI agent with the appropriate tools
registered via the registrars in ``tools.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings

from ace.core.outputs import ReflectorOutput
from ace.providers.pydantic_ai import resolve_model

from .config import RecursiveConfig
from .prompts import SUBAGENT_SYSTEM
from .tools import (
    RRDeps,
    SubAgentDeps,
    register_analysis_tools,
    register_execute_code,
    register_orchestration_tools,
    register_output_validator,
    register_subagent_execute_code,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Agent factories
# ------------------------------------------------------------------


def create_rr_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create the PydanticAI agent for recursive reflection (single-trace).

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the reflector.
        config: RR configuration (timeouts, limits).
        model_settings: PydanticAI model settings.

    Returns:
        Configured PydanticAI agent with tools.
    """
    resolved = resolve_model(model)

    agent: PydanticAgent[RRDeps, ReflectorOutput] = PydanticAgent(
        resolved,
        output_type=ReflectorOutput,
        system_prompt=system_prompt
        or (
            "You are a trace analyst with tools. "
            "Analyze agent execution traces and extract learnings. "
            "Use execute_code to explore data, analyze for LLM reasoning, "
            "then produce your final structured output."
        ),
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    register_execute_code(agent)
    register_analysis_tools(agent)
    register_output_validator(agent)

    return agent


def create_orchestrator_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
    spawn_worker_fn: Callable[..., tuple[ReflectorOutput, RRDeps]] | None = None,
    slice_batch_fn: Callable[[Any, list[int]], Any] | None = None,
    get_batch_items_fn: Callable[[Any], list[Any] | None] | None = None,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create the orchestrator agent for batch RR with manager/worker tools.

    In addition to the standard tools (execute_code, analyze, batch_analyze),
    the orchestrator agent has ``spawn_analysis`` and ``collect_results``.

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the orchestrator.
        config: RR configuration.
        model_settings: PydanticAI model settings.
        spawn_worker_fn: Callback to run a worker RR session.
        slice_batch_fn: Callback to slice batch traces by indices.
        get_batch_items_fn: Callback to extract batch items from traces.

    Returns:
        Configured orchestrator agent.
    """
    cfg = config or RecursiveConfig()
    resolved = resolve_model(model)

    agent: PydanticAgent[RRDeps, ReflectorOutput] = PydanticAgent(
        resolved,
        output_type=ReflectorOutput,
        system_prompt=system_prompt,
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    register_execute_code(agent)
    register_analysis_tools(agent)

    if spawn_worker_fn is not None and slice_batch_fn is not None and get_batch_items_fn is not None:
        register_orchestration_tools(
            agent,
            cfg=cfg,
            spawn_worker_fn=spawn_worker_fn,
            slice_batch_fn=slice_batch_fn,
            get_batch_items_fn=get_batch_items_fn,
        )

    register_output_validator(agent)

    return agent


def create_worker_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
    enable_subagent: bool = False,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create a worker agent for delegated RR sessions.

    Workers have ``execute_code`` and optionally ``analyze``/``batch_analyze``.
    They never have orchestration tools (spawn_analysis, collect_results).

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the worker.
        config: RR configuration.
        model_settings: PydanticAI model settings.
        enable_subagent: Whether to include analyze/batch_analyze tools.

    Returns:
        Configured worker agent.
    """
    resolved = resolve_model(model)

    agent: PydanticAgent[RRDeps, ReflectorOutput] = PydanticAgent(
        resolved,
        output_type=ReflectorOutput,
        system_prompt=system_prompt,
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    register_execute_code(agent)

    if enable_subagent:
        register_analysis_tools(agent)

    register_output_validator(agent)

    return agent


# ------------------------------------------------------------------
# Sub-agent factory
# ------------------------------------------------------------------


def create_sub_agent(
    model: str,
    *,
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
) -> PydanticAgent[SubAgentDeps, str]:
    """Create the sub-agent for ``analyze`` / ``batch_analyze`` tools.

    The sub-agent has its own ``execute_code`` tool backed by an isolated
    sandbox snapshot.  It can explore trace data directly, so the main
    agent doesn't need to serialize data into tool parameters.

    Args:
        model: LiteLLM or PydanticAI model string.
        config: RR configuration for sub-agent settings.
        model_settings: Override model settings.

    Returns:
        PydanticAI agent with execute_code tool, producing text output.
    """
    cfg = config or RecursiveConfig()
    resolved = resolve_model(model)

    settings = model_settings or ModelSettings(
        temperature=cfg.subagent_temperature,
        max_tokens=cfg.subagent_max_tokens,
    )

    sub_agent: PydanticAgent[SubAgentDeps, str] = PydanticAgent(
        resolved,
        output_type=str,
        system_prompt=SUBAGENT_SYSTEM,
        model_settings=settings,
        defer_model_check=True,
        deps_type=SubAgentDeps,
    )

    register_subagent_execute_code(sub_agent)

    return sub_agent
