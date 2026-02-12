#!/usr/bin/env python3
"""
ACE + mini-swe-agent integration example.

This demonstrates using ACEMiniSWE to learn from SWE-bench-style coding tasks.
mini-swe-agent is a 100-line agent from Princeton/Stanford that scores 74%+
on SWE-bench verified.

Usage:
    # Basic usage
    uv run python examples/mini-swe/ace_mini_swe_example.py

    # With specific model
    uv run python examples/mini-swe/ace_mini_swe_example.py --model claude-sonnet-4-20250514

    # Save learned skillbook
    uv run python examples/mini-swe/ace_mini_swe_example.py --save-skillbook swe_skillbook.json

Requirements:
    - mini-swe-agent: uv add mini-swe-agent --group demos
    - API key: ANTHROPIC_API_KEY or OPENAI_API_KEY in environment
"""

import argparse
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from ace.integrations import ACEMiniSWE, MINI_SWE_AVAILABLE

# Load environment variables
load_dotenv()


def create_sample_project(tmpdir: Path) -> str:
    """Create a sample Python project with a bug to fix."""
    # Create a simple Python file with a bug
    utils_py = tmpdir / "utils.py"
    utils_py.write_text(
        '''"""Simple utility functions."""


def add_numbers(a, b):
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    # BUG: Should be a + b, not a - b
    return a - b


def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}!"
'''
    )

    # Create a test file
    test_utils_py = tmpdir / "test_utils.py"
    test_utils_py.write_text(
        '''"""Tests for utils module."""

from utils import add_numbers, greet


def test_add_numbers():
    """Test add_numbers function."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(0, 0) == 0
    assert add_numbers(-1, 1) == 0


def test_greet():
    """Test greet function."""
    assert greet("World") == "Hello, World!"


if __name__ == "__main__":
    test_add_numbers()
    test_greet()
    print("All tests passed!")
'''
    )

    return str(tmpdir)


def main():
    parser = argparse.ArgumentParser(
        description="ACE + mini-swe-agent integration example"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model for mini-swe-agent execution (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--ace-model",
        default="gpt-4o-mini",
        help="Model for ACE learning (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--save-skillbook",
        type=str,
        help="Path to save learned skillbook",
    )
    parser.add_argument(
        "--load-skillbook",
        type=str,
        help="Path to load existing skillbook",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=15,
        help="Maximum agent steps (default: 15)",
    )
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable ACE learning (execution only)",
    )

    args = parser.parse_args()

    # Check for mini-swe-agent
    if not MINI_SWE_AVAILABLE:
        print("mini-swe-agent not installed.")
        print("Install with: uv add mini-swe-agent --group demos")
        return

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        return

    print(f"Using model: {args.model}")
    print(f"ACE learning model: {args.ace_model}")
    print(f"Learning enabled: {not args.no_learning}")
    print()

    # Create ACEMiniSWE agent
    agent = ACEMiniSWE(
        model=args.model,
        ace_model=args.ace_model,
        skillbook_path=args.load_skillbook,
        is_learning=not args.no_learning,
        step_limit=args.step_limit,
        cost_limit=1.0,  # Lower cost limit for demo
    )

    if args.load_skillbook:
        print(f"Loaded skillbook from: {args.load_skillbook}")
        print(f"Starting with {len(agent.skillbook.skills())} strategies")
        print()

    # Create temporary project with bug
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = create_sample_project(Path(tmpdir))
        print(f"Created sample project in: {project_dir}")
        print()

        # Define the task
        task = """Fix the bug in utils.py.

The test file test_utils.py has failing tests. Run the tests to see what's broken,
then fix the bug in utils.py to make the tests pass.

Steps:
1. Run the tests to see the failure
2. Read utils.py to find the bug
3. Fix the bug
4. Run tests again to verify the fix"""

        print("Task:")
        print(task)
        print()
        print("=" * 60)
        print("Running mini-swe-agent with ACE learning...")
        print("=" * 60)
        print()

        # Run the task
        try:
            result = agent.run(task=task, working_dir=project_dir)

            print()
            print("=" * 60)
            print("RESULT")
            print("=" * 60)
            print(f"Status: {result.status}")
            print(f"Success: {result.success}")
            print(f"Steps: {result.steps}")
            print()
            print("Final message:")
            print(result.message[:500] if result.message else "(no message)")
            print()

            # Show execution trace summary
            print("Execution trace summary:")
            for i, msg in enumerate(result.trace):
                role = msg.get("role", "")
                content = msg.get("content", "")[:100]
                print(f"  {i+1}. [{role}] {content}...")
            print()

        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback

            traceback.print_exc()

    # Show learned strategies
    if agent.skillbook.skills():
        print()
        print("=" * 60)
        print("LEARNED STRATEGIES")
        print("=" * 60)
        for skill in agent.skillbook.skills():
            print(f"\n[{skill.id}] ({skill.helpful}/{skill.harmful})")
            print(f"  Section: {skill.section}")
            print(f"  Content: {skill.content[:200]}...")
    else:
        print("\nNo strategies learned yet.")

    # Save skillbook if requested
    if args.save_skillbook:
        agent.save_skillbook(args.save_skillbook)
        print(f"\nSaved skillbook to: {args.save_skillbook}")

    print()
    print(f"Agent: {agent}")


if __name__ == "__main__":
    main()
