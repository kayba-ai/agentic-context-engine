"""
Direct test script for running ACE on AppWorld benchmark with full learning.

This bypasses HAL harness and tests the ACE agent directly against
the AppWorld HTTP servers, with Reflector and SkillManager enabled.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ace_agent.main import process_task, APPWORLD_ENV_URL

import httpx

# ACE components for full learning pipeline
from ace import Skillbook, Agent, Reflector, SkillManager
from ace.roles import AgentOutput
from ace.llm_providers.litellm_client import LiteLLMClient


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

    # Initialize ACE components with full learning pipeline
    print("\n" + "=" * 60)
    print("Running ACE Agent on AppWorld Tasks (with Learning)")
    print("=" * 60)

    # Default to Claude if ANTHROPIC_API_KEY is set, otherwise GPT
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("ACE_MODEL", "claude-3-5-haiku-latest")
    else:
        model = os.environ.get("ACE_MODEL", "gpt-4o-mini")
    print(f"\nModel: {model}")
    print(f"Tasks: {len(input_data)}")
    print("Learning: ENABLED (Reflector + SkillManager)")

    try:
        # Initialize ACE components
        llm = LiteLLMClient(model=model)
        skillbook = Skillbook()
        agent = Agent(llm)
        reflector = Reflector(llm)
        skill_manager = SkillManager(llm)

        # Process each task with learning
        results = {}
        max_interactions = 5

        for task_id, task_data in input_data.items():
            print(f"\n--- Processing Task: {task_id} ---")

            try:
                # 1. Agent generates and executes code
                result = process_task(
                    task_id=task_id,
                    task_data=task_data,
                    agent=agent,
                    skillbook=skillbook,
                    max_interactions=max_interactions,
                )
                results[task_id] = result

                # 2. Build execution feedback from result
                is_success = not result.startswith("Error:")
                feedback = f"Task {task_id}: {'Completed successfully' if is_success else 'Failed with error'}"
                if not is_success:
                    feedback += f"\nError details: {result}"

                print(f"  Execution: {'SUCCESS' if is_success else 'FAILED'}")
                print("  Running Reflector...")

                # 3. Reflector analyzes execution
                # Wrap the result in an AgentOutput object
                agent_output = AgentOutput(
                    reasoning=f"Generated code to complete AppWorld task {task_id}",
                    final_answer=result,
                    skill_ids=[],
                )

                reflection = reflector.reflect(
                    question=f"AppWorld Task: {task_id}",
                    agent_output=agent_output,
                    feedback=feedback,
                    skillbook=skillbook,
                )
                reflection_summary = (
                    reflection.reasoning[:100] + "..."
                    if len(reflection.reasoning) > 100
                    else reflection.reasoning
                )
                print(f"  Reflection: {reflection_summary}")

                print("  Running SkillManager...")

                # 4. SkillManager updates skillbook
                skill_output = skill_manager.update_skills(
                    reflection=reflection,
                    skillbook=skillbook,
                    question_context=f"AppWorld task: {task_id}",
                    progress=f"Task {'completed' if is_success else 'failed'}",
                )
                skillbook.apply_update(skill_output.update)
                print(
                    f"  Updates applied: {len(skill_output.update.operations)} operations"
                )

            except Exception as e:
                print(f"  Task error: {e}")
                results[task_id] = f"Error: {e}"

        # Print results
        print("\n" + "=" * 60)
        print("Execution Results")
        print("=" * 60)

        for task_id, result in results.items():
            print(f"\n--- Task: {task_id} ---")
            if result.startswith("Error:"):
                print("Status: FAILED")
                print(f"Error: {result}")
            else:
                print("Status: COMPLETED")
                print(
                    f"Code:\n{result[:500]}..."
                    if len(result) > 500
                    else f"Code:\n{result}"
                )

        # Summary
        successful = sum(1 for r in results.values() if not r.startswith("Error:"))
        print("\n" + "=" * 60)
        print(f"Execution Summary: {successful}/{len(results)} tasks completed")
        print("=" * 60)

        # Display learned skillbook insights
        print("\n" + "=" * 60)
        print("Learned Skillbook Insights")
        print("=" * 60)

        if skillbook.skills():
            print(str(skillbook))  # Human-readable markdown format
            stats = skillbook.stats()
            print(f"\nStats: {stats}")
        else:
            print("\nNo skills learned during this session.")
            print("(This may happen if tasks completed without generating insights)")

    except Exception as e:
        print(f"\nError running ACE agent: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
