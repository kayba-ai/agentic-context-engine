#!/usr/bin/env python3
"""
LangChain AgentExecutor with ACE meso-level learning example.

This example demonstrates how to wrap a LangChain AgentExecutor with ACELangChain.
When using AgentExecutor, ACE automatically captures the full reasoning trace
(thoughts, actions, observations) for meso-level learning.

INSIGHT LEVELS:
- Micro: Full ACE loop with TaskEnvironment - learns from ground truth/feedback
- Meso: AgentExecutor - learns from agent reasoning traces (no external feedback)

Meso-level learning captures:
- Agent thoughts (reasoning)
- Tool calls (actions)
- Tool outputs (observations)
- Final answer

Requirements:
    pip install ace-framework[langchain]
    # or: pip install langchain langchain-anthropic

Environment:
    export ANTHROPIC_API_KEY="your-api-key"
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check availability
try:
    from ace.integrations import ACELangChain, LANGCHAIN_AVAILABLE

    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not installed.")
        print("Install with: pip install ace-framework[langchain]")
        exit(1)
except ImportError:
    print("ACE framework not installed.")
    print("Install with: pip install ace-framework")
    exit(1)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


# Define custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def get_word_length(word: str) -> int:
    """Get the length of a word."""
    return len(word)


def example_agent_executor_meso():
    """
    Example 1: AgentExecutor with meso-level learning.

    This uses a real AgentExecutor, so ACE captures:
    - Agent's reasoning (thoughts)
    - Tool calls (actions)
    - Tool outputs (observations)
    - Final answer

    This produces much richer learning than simple chains!
    """
    print("\n" + "=" * 60)
    print("Example 1: AgentExecutor with Meso-Level Learning")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # Create tools
    tools = [add, multiply, get_word_length]

    # Create LLM
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")

    # Create prompt for tool-calling agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that uses tools to solve problems. "
                "Think step by step and use tools as needed.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Wrap with ACE - meso-level learning is automatic for AgentExecutor!
    ace_agent = ACELangChain(
        runnable=agent_executor,
        ace_model="claude-sonnet-4-5-20250929",
        is_learning=True,
    )

    print("\nNote: ACE automatically uses meso-level learning for AgentExecutor.")
    print("This captures the full reasoning trace for better insights.\n")

    # Test questions that require tool use
    questions = [
        "What is 23 + 45?",
        "Calculate 12 * 7",
        "How many characters are in the word 'anthropic'?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*40}")
        print(f"Question {i}: {question}")
        print("=" * 40)
        result = ace_agent.invoke({"input": question})
        print(f"\nFinal Answer: {result}")

    print(
        f"\n\nLearned {len(ace_agent.skillbook.skills())} strategies from agent traces"
    )

    # Show what was learned
    if ace_agent.skillbook.skills():
        print("\nLearned strategies (from meso-level analysis):")
        for i, skill in enumerate(ace_agent.skillbook.skills()[:3], 1):
            print(f"  {i}. {skill.content[:80]}...")

    ace_agent.save_skillbook("agent_executor_learned.json")
    print("\nSkillbook saved to: agent_executor_learned.json")


def example_simple_chain_vs_agent():
    """
    Example 2: Compare simple chain vs AgentExecutor learning.

    Shows the difference in what ACE sees:
    - Simple chain: Only sees input/output (basic learning)
    - AgentExecutor: Sees full reasoning trace (meso-level learning)

    Note: For true micro-level learning (with ground truth feedback),
    use OfflineACE or OnlineACE with a TaskEnvironment.
    """
    print("\n" + "=" * 60)
    print("Example 2: Simple Chain vs AgentExecutor")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    tools = [add, multiply]

    # --- Simple chain: basic learning ---
    print("\n--- SIMPLE CHAIN (basic learning) ---")
    simple_prompt = ChatPromptTemplate.from_template(
        "Calculate: {input}. Show your work."
    )
    simple_chain = simple_prompt | llm

    ace_simple = ACELangChain(
        runnable=simple_chain, ace_model="claude-sonnet-4-5-20250929"
    )
    result_simple = ace_simple.invoke({"input": "What is 5 * 6?"})
    print(f"Result: {result_simple.content}")
    print(f"Strategies learned: {len(ace_simple.skillbook.skills())}")
    print("ACE sees: request → response only")

    # --- AgentExecutor: meso-level learning ---
    print("\n--- AGENT EXECUTOR (meso-level learning) ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use tools to solve math problems."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    ace_agent = ACELangChain(
        runnable=agent_executor, ace_model="claude-sonnet-4-5-20250929"
    )
    result_agent = ace_agent.invoke({"input": "What is 5 * 6?"})
    print(f"Result: {result_agent}")
    print(f"Strategies learned: {len(ace_agent.skillbook.skills())}")
    print("ACE sees: thoughts → tool calls → observations → answer")

    print("\n--- KEY DIFFERENCE ---")
    print("AgentExecutor gives ACE visibility into the agent's reasoning process.")
    print(
        "This enables learning from HOW the agent solved the problem, not just WHAT it returned."
    )


async def example_async_agent():
    """Example 3: Async AgentExecutor with meso-level learning."""
    print("\n" + "=" * 60)
    print("Example 3: Async AgentExecutor")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    tools = [add, multiply, get_word_length]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant with tools."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # Wrap with ACE
    ace_agent = ACELangChain(
        runnable=agent_executor, ace_model="claude-sonnet-4-5-20250929"
    )

    # Async execution with meso-level learning
    questions = [
        "Add 100 and 200",
        "What is 15 times 4?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = await ace_agent.ainvoke({"input": question})
        print(f"Answer: {result}")

    print(f"\nLearned {len(ace_agent.skillbook.skills())} strategies (async meso)")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LangChain AgentExecutor + ACE Meso-Level Learning")
    print("=" * 60)
    print("\nThis demonstrates how ACE learns from agent reasoning traces.")
    print("AgentExecutor provides much richer learning than simple chains!")

    # Run sync examples
    example_agent_executor_meso()
    example_simple_chain_vs_agent()

    # Run async example
    import asyncio

    if os.getenv("ANTHROPIC_API_KEY"):
        asyncio.run(example_async_agent())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Use AgentExecutor for meso-level learning (richer insights)")
    print("- Simple chains only get micro-level learning")
    print("- ACE automatically detects AgentExecutor and enables meso mode")
    print("\nSee docs/COMPLETE_GUIDE_TO_ACE.md for more on insight levels")


if __name__ == "__main__":
    main()
