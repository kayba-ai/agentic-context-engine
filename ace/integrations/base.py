"""
Base classes and utilities for ACE integrations with external agentic frameworks.

This module provides the foundation for integrating ACE learning capabilities
with external agentic systems like browser-use, LangChain, CrewAI, etc.

Integration Pattern:
    1. External framework executes task (no ACE Generator)
    2. ACE injects playbook context beforehand (via wrap_playbook_context)
    3. ACE learns from execution afterward (Reflector + Curator)

Example:
    from ace.integrations.base import wrap_playbook_context
    from ace import Playbook

    playbook = Playbook()
    # ... playbook gets populated with bullets ...

    # Inject context into external agent's task
    task_with_context = f"{task}\\n\\n{wrap_playbook_context(playbook)}"
"""

from ..playbook import Playbook


def wrap_playbook_context(playbook: Playbook) -> str:
    """
    Wrap playbook bullets with explanation for external agents.

    This helper formats learned strategies from the playbook with instructions
    on how to apply them. It uses the same explanation as the ACE Generator
    so that external agents (browser-use, custom agents, etc.) can understand
    and leverage learned knowledge.

    The formatted output includes:
    - Header explaining these are learned strategies
    - List of bullets with success rates (helpful/harmful scores)
    - Usage instructions on how to apply strategies
    - Reminder that these are patterns, not rigid rules

    Args:
        playbook: Playbook with learned strategies

    Returns:
        Formatted text explaining playbook and listing strategies.
        Returns empty string if playbook has no bullets.

    Example:
        >>> playbook = Playbook()
        >>> playbook.add_bullet("general", "Always verify inputs")
        >>> context = wrap_playbook_context(playbook)
        >>> enhanced_task = f"{task}\\n\\n{context}"
    """
    bullets = playbook.bullets()

    if not bullets:
        return ""

    # Get formatted bullets from playbook
    bullet_text = playbook.as_prompt()

    # Wrap with explanation (extracted from Generator v2.1 prompt)
    wrapped = f"""
## 📚 Available Strategic Knowledge (Learned from Experience)

The following strategies have been learned from previous task executions.
Each bullet shows its success rate based on helpful/harmful feedback:

{bullet_text}

**How to use these strategies:**
- Review bullets relevant to your current task
- Prioritize strategies with high success rates (helpful > harmful)
- Apply strategies when they match your context
- Adapt general strategies to your specific situation
- Learn from both successful patterns and failure avoidance

**Important:** These are learned patterns, not rigid rules. Use judgment.
"""
    return wrapped


__all__ = ["wrap_playbook_context"]
