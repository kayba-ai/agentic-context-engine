"""
ACE integrations with external agentic frameworks.

This module provides integration adapters for popular agentic frameworks,
allowing them to leverage ACE's learning capabilities.

Available Integrations:
    - LiteLLM: ACELiteLLM - Quick-start agent for simple tasks
    - browser-use: ACEAgent - Self-improving browser automation
    - LangChain: ACELangChain - Complex workflows with learning

Pattern:
    All integrations follow the same pattern:
    1. External framework executes task (or ACE Generator for LiteLLM)
    2. ACE injects playbook context beforehand (via wrap_playbook_context)
    3. ACE learns from execution afterward (Reflector + Curator)

Example:
    # LiteLLM (quick start)
    from ace.integrations import ACELiteLLM
    agent = ACELiteLLM(model="gpt-4o-mini")
    answer = agent.ask("What is 2+2?")

    # Browser-use
    from ace.integrations import ACEAgent
    from browser_use import ChatBrowserUse
    agent = ACEAgent(llm=ChatBrowserUse())
    await agent.run(task="Find top HN post")

    # LangChain
    from ace.integrations import ACELangChain
    from langchain_openai import ChatOpenAI
    chain = ChatOpenAI(temperature=0)
    ace_chain = ACELangChain(runnable=chain)
    result = ace_chain.invoke("What is ACE?")
"""

from .base import wrap_playbook_context

# Import LiteLLM integration (always available if ace-framework installed)
try:
    from .litellm import ACELiteLLM
except ImportError:
    ACELiteLLM = None  # type: ignore

# Import browser-use integration if available
try:
    from .browser_use import ACEAgent, BROWSER_USE_AVAILABLE
except ImportError:
    ACEAgent = None  # type: ignore
    BROWSER_USE_AVAILABLE = False

# Import LangChain integration if available
try:
    from .langchain import ACELangChain, LANGCHAIN_AVAILABLE
except ImportError:
    ACELangChain = None  # type: ignore
    LANGCHAIN_AVAILABLE = False

__all__ = [
    "wrap_playbook_context",
    "ACELiteLLM",
    "ACEAgent",
    "ACELangChain",
    "BROWSER_USE_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
]
