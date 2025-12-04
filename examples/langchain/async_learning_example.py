#!/usr/bin/env python3
"""
ACELangChain Async Learning Example

Demonstrates async learning with ACELangChain:
- Wraps a simple LangChain chain
- Shows async invoke with background learning
- Learning happens in thread pool (doesn't block event loop)

Requires: pip install langchain-anthropic
"""

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Check for LangChain
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ace import ACELangChain


async def run_async_langchain():
    """Run ACELangChain with async_learning=True."""
    print("\n" + "=" * 60)
    print("ACELangChain ASYNC MODE")
    print("=" * 60)

    # Create a simple LangChain chain
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer this question concisely: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    # Wrap with ACE for learning
    ace_chain = ACELangChain(
        runnable=chain,
        ace_model="claude-sonnet-4-5-20250929",
        async_learning=True,  # Learning in background!
    )

    # Test questions
    questions = [
        "What is 2+2?",
        "What color is the sky?",
        "Capital of France?",
        "What is 10-3?",
        "How many days in a week?",
    ]

    print(f"\nProcessing {len(questions)} questions with async learning...")
    print("(Learning runs in background thread - doesn't block)")

    start = time.time()
    answers = []

    for i, question in enumerate(questions, 1):
        q_start = time.time()

        # Use async invoke
        answer = await ace_chain.ainvoke({"question": question})

        q_time = time.time() - q_start
        answers.append(answer)

        print(f"\n  Q{i}: {question}")
        print(f"  A{i}: {answer[:50]}..." if len(answer) > 50 else f"  A{i}: {answer}")
        print(f"  ⏱️  {q_time:.2f}s (learning in background)")

    results_time = time.time() - start
    print(f"\n✅ All answers returned in: {results_time:.2f}s")

    # Check learning stats
    stats = ace_chain.learning_stats
    print(f"\nLearning stats (before wait):")
    print(f"  - Tasks submitted: {stats['tasks_submitted']}")
    print(f"  - Pending: {stats['pending']}")
    print(f"  - Completed: {stats['completed']}")

    # Wait for learning to complete
    print("\n⏳ Waiting for background learning to complete...")
    wait_start = time.time()
    await ace_chain.wait_for_learning(timeout=60.0)
    wait_time = time.time() - wait_start

    total_time = time.time() - start

    # Final stats
    final_stats = ace_chain.learning_stats
    print(f"\nFinal learning stats:")
    print(f"  - Completed: {final_stats['completed']}")
    print(f"  - Strategies learned: {len(ace_chain.skillbook.skills())}")

    print(f"\nResults:")
    print(f"  - Samples processed: {len(questions)}")
    print(f"  - Results returned in: {results_time:.2f}s")
    print(f"  - Learning wait time: {wait_time:.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Strategies learned: {len(ace_chain.skillbook.skills())}")

    # Show complete skillbook
    print(f"\n📚 COMPLETE SKILLBOOK:")
    if ace_chain.skillbook.skills():
        print(str(ace_chain.skillbook))
    else:
        print("(empty)")

    return results_time, total_time


async def run_sync_langchain():
    """Run ACELangChain with async_learning=False (for comparison)."""
    print("\n" + "=" * 60)
    print("ACELangChain SYNC MODE (for comparison)")
    print("=" * 60)

    # Create a simple LangChain chain
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer this question concisely: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    # Wrap with ACE - sync mode
    ace_chain = ACELangChain(
        runnable=chain,
        ace_model="claude-sonnet-4-5-20250929",
        async_learning=False,  # Blocking mode
    )

    questions = [
        "What is 2+2?",
        "What color is the sky?",
        "Capital of France?",
        "What is 10-3?",
        "How many days in a week?",
    ]

    print(f"\nProcessing {len(questions)} questions with sync learning...")
    print("(Learning blocks after each question)")

    start = time.time()

    for i, question in enumerate(questions, 1):
        q_start = time.time()
        answer = await ace_chain.ainvoke({"question": question})
        q_time = time.time() - q_start

        print(f"\n  Q{i}: {question}")
        print(f"  A{i}: {answer[:50]}..." if len(answer) > 50 else f"  A{i}: {answer}")
        print(f"  ⏱️  {q_time:.2f}s (includes learning)")

    total_time = time.time() - start
    print(f"\nResults:")
    print(f"  - Samples processed: {len(questions)}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Strategies learned: {len(ace_chain.skillbook.skills())}")

    return total_time


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not installed. Run: pip install langchain-anthropic")
        return

    print("=" * 60)
    print("ACELangChain ASYNC LEARNING DEMO")
    print("=" * 60)
    print("\nThis demo shows async learning with ACELangChain.")
    print("In async mode, answers return immediately while learning")
    print("runs in a background thread (doesn't block event loop).")

    # Run sync demo (fewer questions to save time)
    sync_time = await run_sync_langchain()

    # Run async demo
    async_results_time, async_total_time = await run_async_langchain()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSync mode total time:    {sync_time:.2f}s")
    print(f"Async mode results time: {async_results_time:.2f}s")
    print(f"Async mode total time:   {async_total_time:.2f}s")

    speedup = sync_time / async_results_time if async_results_time > 0 else 1.0
    print(f"\n✅ Async mode returned results {speedup:.1f}x faster!")


if __name__ == "__main__":
    asyncio.run(main())
