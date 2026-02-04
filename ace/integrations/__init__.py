"""
ACE integrations with external agentic frameworks.

This module provides integration adapters for popular agentic frameworks,
allowing them to leverage ACE's learning capabilities.

Available Integrations:
    - LiteLLM: ACELiteLLM - Quick-start agent for simple tasks
    - browser-use: ACEAgent - Self-improving browser automation
    - LangChain: ACELangChain - Complex workflows with learning
    - Claude Code: ACEClaudeCode - Claude Code CLI with learning
    - OpenCode: ACEOpenCode - OpenCode server API with learning

Pattern:
    All integrations follow the same pattern:
    1. External framework executes task (or ACE Agent for LiteLLM)
    2. ACE injects skillbook context beforehand (via wrap_skillbook_context)
    3. ACE learns from execution afterward (Reflector + SkillManager)

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

    # Claude Code
    from ace.integrations import ACEClaudeCode
    agent = ACEClaudeCode(working_dir="./my_project")
    result = agent.run(task="Add unit tests")
    agent.save_skillbook("learned.json")

    # OpenCode
    from ace.integrations import ACEOpenCode
    agent = ACEOpenCode(working_dir="./my_project")
    result = agent.run(task="Add unit tests")
    agent.save_skillbook("learned.json")
"""

from .base import wrap_skillbook_context

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

# Import Claude Code integration if available
try:
    from .claude_code import ACEClaudeCode, CLAUDE_CODE_AVAILABLE
except ImportError:
    ACEClaudeCode = None  # type: ignore
    CLAUDE_CODE_AVAILABLE = False

# Import OpenCode integration if available
try:
    from .opencode import ACEOpenCode, OPENCODE_AVAILABLE
except ImportError:
    ACEOpenCode = None  # type: ignore
    OPENCODE_AVAILABLE = False

__all__ = [
    "wrap_skillbook_context",
    "ACELiteLLM",
    "ACEAgent",
    "ACELangChain",
    "ACEClaudeCode",
    "ACEOpenCode",
    "BROWSER_USE_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "CLAUDE_CODE_AVAILABLE",
    "OPENCODE_AVAILABLE",
]
