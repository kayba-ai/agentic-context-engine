"""PydanticAI-based Recursive Reflector agent factories.

Each factory creates a PydanticAI agent with the appropriate tools
registered via the registrars in ``tools.py``.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings

from ace.core.outputs import ReflectorOutput
from ace.providers.pydantic_ai import resolve_model

from .config import RecursiveConfig
from .tools import (
    RRDeps,
    register_execute_code,
    register_output_validator,
    register_recurse_tool,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Agent factory
# ------------------------------------------------------------------


def create_rr_agent(
    model: str,
    *,
    system_prompt: str = "",
    config: RecursiveConfig | None = None,
    model_settings: ModelSettings | None = None,
    depth: int = 0,
    max_depth: int = 2,
) -> PydanticAgent[RRDeps, ReflectorOutput]:
    """Create the PydanticAI agent for recursive reflection.

    Args:
        model: LiteLLM or PydanticAI model string.
        system_prompt: System prompt for the reflector.
        config: RR configuration (timeouts, limits).
        model_settings: PydanticAI model settings.
        depth: Current recursion depth (0 = root).
        max_depth: Maximum recursion depth. At max_depth, recurse tool is not registered.

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
            "Use execute_code to explore data, "
            "then produce your final structured output."
        ),
        retries=3,
        model_settings=model_settings,
        defer_model_check=True,
        deps_type=RRDeps,
    )

    register_execute_code(agent)
    if depth < max_depth:
        register_recurse_tool(
            agent,
            create_agent_fn=create_rr_agent,
        )
    register_output_validator(agent)

    return agent
