"""
Base classes and utilities for ACE integrations with external agentic frameworks.

This module provides the foundation for integrating ACE learning capabilities
with external agentic systems like browser-use, LangChain, CrewAI, and custom agents.

## When to Use Integrations vs Full ACE Pipeline

### Use INTEGRATIONS (this module) when:
- You have an existing agentic system (browser-use, LangChain, custom agent)
- The external agent handles task execution
- You want ACE to learn from that agent's results
- Example: Browser automation, LangChain chains, API-based agents

### Use FULL ACE PIPELINE when:
- Building a new agent from scratch
- Want ACE Generator to handle task execution
- Simple Q&A, classification, reasoning tasks
- Example: Question answering, data extraction, summarization

## Integration Pattern (Three Steps)

The integration pattern allows external agents to benefit from ACE learning
without replacing their execution logic:

    1. INJECT: Add playbook context to agent's input (optional)
       → wrap_playbook_context(playbook) formats learned strategies

    2. EXECUTE: External agent runs normally
       → Your framework handles the task (browser-use, LangChain, etc.)

    3. LEARN: ACE analyzes results and updates playbook
       → Reflector: Analyzes what worked/failed
       → Curator: Updates playbook with new strategies

## Why No ACE Generator?

Integrations bypass ACE's Generator because:
- External frameworks have their own execution logic
- They may use tools, browsers, or specialized workflows
- ACE focuses on LEARNING from their results, not replacing them

## Basic Example

```python
from ace.integrations.base import wrap_playbook_context
from ace import Playbook, Reflector, Curator, LiteLLMClient
from ace.roles import GeneratorOutput

# Setup
playbook = Playbook()
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)
reflector = Reflector(llm)
curator = Curator(llm)

# 1. INJECT: Add learned strategies to task (optional)
task = "Process user request"
if playbook.bullets():
    task_with_context = f"{task}\\n\\n{wrap_playbook_context(playbook)}"
else:
    task_with_context = task

# 2. EXECUTE: Your agent runs
result = your_agent.execute(task_with_context)

# 3. LEARN: ACE learns from results
generator_output = GeneratorOutput(
    reasoning=f"Task: {task}",
    final_answer=result.output,
    bullet_ids=[],  # External agents don't cite bullets
    raw={"success": result.success}
)

reflection = reflector.reflect(
    question=task,
    generator_output=generator_output,
    playbook=playbook,
    feedback=f"Task {'succeeded' if result.success else 'failed'}"
)

curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context=f"task: {task}",
    progress=f"Executing: {task}"
)

playbook.apply_delta(curator_output.delta)
playbook.save_to_file("learned.json")
```

## See Also

- Reference implementation: ace/integrations/browser_use.py
- Full integration guide: docs/INTEGRATION_GUIDE.md
- Out-of-box wrappers: ACELiteLLM, ACEAgent (browser-use), ACELangChain
"""

from ..playbook import Playbook
from ..prompts_v2_1 import wrap_playbook_for_external_agent


def wrap_playbook_context(playbook: Playbook) -> str:
    """
    Wrap playbook bullets with explanation for external agents.

    This helper formats learned strategies from the playbook with instructions
    on how to apply them. Delegates to the canonical implementation in
    prompts_v2_1 to ensure consistency across all ACE components.

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

    Examples:
        Basic usage with any agent:
        >>> playbook = Playbook()
        >>> playbook.add_bullet("general", "Always verify inputs")
        >>> context = wrap_playbook_context(playbook)
        >>> enhanced_task = f"{task}\\n\\n{context}"
        >>> result = your_agent.execute(enhanced_task)

        With browser-use:
        >>> from browser_use import Agent
        >>> task = "Find top HN post"
        >>> enhanced_task = f"{task}\\n\\n{wrap_playbook_context(playbook)}"
        >>> agent = Agent(task=enhanced_task, llm=llm)
        >>> await agent.run()

        With LangChain:
        >>> from langchain.chains import LLMChain
        >>> context = wrap_playbook_context(playbook)
        >>> chain.run(input=task, context=context)

        With API-based agents:
        >>> payload = {
        >>>     "task": task,
        >>>     "strategies": wrap_playbook_context(playbook)
        >>> }
        >>> response = api_client.post("/execute", json=payload)

        Conditional injection (skip if empty):
        >>> if playbook.bullets():
        >>>     task = f"{task}\\n\\n{wrap_playbook_context(playbook)}"
        >>> # task unchanged if no learned strategies yet

    Integration Patterns:
        1. String Concatenation (most common):
           enhanced_task = f"{task}\\n\\n{context}"

        2. Dict/Kwargs Injection:
           chain.run(input=task, learned_strategies=context)

        3. System Message Injection:
           messages = [
               {"role": "system", "content": context},
               {"role": "user", "content": task}
           ]

        4. Tool Description Enhancement:
           tool.description += f"\\n\\nLearned patterns: {context}"

    Note:
        This function delegates to wrap_playbook_for_external_agent() in
        prompts_v2_1 module, which is the single source of truth for
        playbook presentation. Kept here for backward compatibility and
        convenience.

    See Also:
        - ace/integrations/browser_use.py: Reference implementation
        - docs/INTEGRATION_GUIDE.md: Full integration guide
    """
    return wrap_playbook_for_external_agent(playbook)


__all__ = ["wrap_playbook_context"]
