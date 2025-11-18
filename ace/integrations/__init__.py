"""
ACE integrations with external agentic frameworks.

This module provides integration adapters for popular agentic frameworks,
allowing them to leverage ACE's learning capabilities.

Available Integrations:
    - browser-use: ACEAgent - Self-improving browser automation

Pattern:
    All integrations follow the same pattern:
    1. External framework executes task (no ACE Generator)
    2. ACE injects playbook context beforehand (via wrap_playbook_context)
    3. ACE learns from execution afterward (Reflector + Curator)

Example:
    from ace.integrations import ACEAgent
    from browser_use import ChatBrowserUse

    agent = ACEAgent(llm=ChatBrowserUse())
    await agent.run(task="Find top HN post")
    agent.save_playbook("hn_expert.json")
"""

from .base import wrap_playbook_context

# Import browser-use integration if available
try:
    from .browser_use import ACEAgent, BROWSER_USE_AVAILABLE
except ImportError:
    ACEAgent = None  # type: ignore
    BROWSER_USE_AVAILABLE = False

__all__ = [
    "wrap_playbook_context",
    "ACEAgent",
    "BROWSER_USE_AVAILABLE",
]
