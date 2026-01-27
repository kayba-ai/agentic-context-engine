"""
Direct test script for running ACE on AppWorld benchmark.

This bypasses HAL harness and tests the ACE agent directly against
the AppWorld HTTP servers.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ace_agent.main import run, process_task, APPWORLD_ENV_URL

import httpx


def get_test_task_ids(limit: int = 2) -> list[str]:
    """Get task IDs from the test_normal split."""
    datasets_file = Path.home() / ".appworld" / "data" / "datasets" / "test_normal.txt"
    if not datasets_file.exists():
        raise FileNotFoundError(f"AppWorld datasets file not found: {datasets_file}")

    with open(datasets_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]

    return task_ids[:limit]


def get_task_info(task_id: str) -> dict:
    """Get basic task info for display (instruction loaded by agent during execution)."""
    # The agent will load the full instruction during process_task
    # Here we just return a placeholder for display
    return {
        "task_id": task_id,
    }


def main():
    """Run ACE on AppWorld tasks."""
    print("=" * 60)
    print("ACE AppWorld Benchmark Test")
    print("=" * 60)

    # Check server connectivity
    print("\nChecking AppWorld server connectivity...")
    try:
        client = httpx.Client(timeout=5.0)
        resp = client.get(f"{APPWORLD_ENV_URL}/")
        print(f"  Environment server: OK (status {resp.status_code})")
        client.close()
    except Exception as e:
        print(f"  Environment server: FAILED ({e})")
        print("\nPlease start AppWorld servers:")
        print("  docker run -d --name appworld-env -p 8000:8000 \\")
        print("    -v ~/.appworld/data:/appworld/data -w /appworld \\")
        print("    -e SERVER_TYPE=environment ghcr.io/stonybrooknlp/appworld")
        return

    # Get task IDs
    print("\nLoading test tasks...")
    try:
        task_ids = get_test_task_ids(limit=2)
        print(f"  Found {len(task_ids)} tasks: {task_ids}")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("\n  Please download AppWorld data first:")
        print("    pip install appworld && appworld download data")
        return

    # Prepare input for ACE agent
    print("\nPreparing task data...")
    input_data = {}
    for task_id in task_ids:
        task_data = get_task_info(task_id)
        input_data[task_id] = task_data
        print(f"  {task_id}: ready")

    if not input_data:
        print("\nNo tasks found. Exiting.")
        return

    # Run ACE agent
    print("\n" + "=" * 60)
    print("Running ACE Agent on AppWorld Tasks")
    print("=" * 60)

    # Default to Claude if ANTHROPIC_API_KEY is set, otherwise GPT
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("ACE_MODEL", "claude-3-5-haiku-latest")
    else:
        model = os.environ.get("ACE_MODEL", "gpt-4o-mini")
    print(f"\nModel: {model}")
    print(f"Tasks: {len(input_data)}")

    try:
        results = run(
            input=input_data,
            model=model,
            max_interactions="5",  # Limit interactions for testing
        )

        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        for task_id, result in results.items():
            print(f"\n--- Task: {task_id} ---")
            if result.startswith("Error:"):
                print(f"Status: FAILED")
                print(f"Error: {result}")
            else:
                print(f"Status: COMPLETED")
                print(
                    f"Code:\n{result[:500]}..."
                    if len(result) > 500
                    else f"Code:\n{result}"
                )

        # Summary
        successful = sum(1 for r in results.values() if not r.startswith("Error:"))
        print("\n" + "=" * 60)
        print(f"Summary: {successful}/{len(results)} tasks completed")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running ACE agent: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
