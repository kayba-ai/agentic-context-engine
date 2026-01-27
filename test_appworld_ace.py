"""
Direct test script for running ACE on AppWorld benchmark with full learning.

This bypasses HAL harness and tests the ACE agent directly against
the AppWorld HTTP servers, with Reflector and SkillManager enabled.

Usage:
    uv run python test_appworld_ace.py
    uv run python test_appworld_ace.py --model gpt-4o-mini --limit 5
    uv run python test_appworld_ace.py --limit 10 --save-skillbook appworld_skills.json
"""

import argparse
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ACE on AppWorld benchmark with full learning pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with defaults (2 tasks)
  %(prog)s --model gpt-4o-mini --limit 5      # Run 5 tasks with GPT-4o-mini
  %(prog)s --limit 10 --save-skillbook s.json # Save learned skills
  %(prog)s --quiet                            # Suppress progress output
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (default: gpt-4o-mini or claude-3-5-haiku-latest)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of tasks to run (default: 2)",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=5,
        help="Max interaction steps per task (default: 5)",
    )
    parser.add_argument(
        "--save-skillbook",
        type=str,
        metavar="PATH",
        help="Save learned skillbook to file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def log(msg: str, quiet: bool = False):
    """Print message if not in quiet mode."""
    if not quiet:
        print(msg)


def main():
    """Run ACE on AppWorld tasks."""
    args = parse_args()
    quiet = args.quiet

    log("=" * 60, quiet)
    log("ACE AppWorld Benchmark Test", quiet)
    log("=" * 60, quiet)

    # Check server connectivity
    log("\nChecking AppWorld server connectivity...", quiet)
    try:
        client = httpx.Client(timeout=5.0)
        resp = client.get(f"{APPWORLD_ENV_URL}/")
        log(f"  Environment server: OK (status {resp.status_code})", quiet)
        client.close()
    except Exception as e:
        print(f"  Environment server: FAILED ({e})")
        print("\nPlease start AppWorld servers:")
        print("  docker run -d --name appworld-env -p 8000:8000 \\")
        print("    -v ~/.appworld/data:/appworld/data -w /appworld \\")
        print("    -e SERVER_TYPE=environment ghcr.io/stonybrooknlp/appworld")
        return

    # Get task IDs
    log("\nLoading test tasks...", quiet)
    try:
        task_ids = get_test_task_ids(limit=args.limit)
        log(f"  Found {len(task_ids)} tasks: {task_ids}", quiet)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print("\n  Please download AppWorld data first:")
        print("    pip install appworld && appworld download data")
        return

    # Prepare input for ACE agent
    log("\nPreparing task data...", quiet)
    input_data = {}
    for task_id in task_ids:
        task_data = get_task_info(task_id)
        input_data[task_id] = task_data
        log(f"  {task_id}: ready", quiet)

    if not input_data:
        print("\nNo tasks found. Exiting.")
        return

    # Initialize ACE components with full learning pipeline
    log("\n" + "=" * 60, quiet)
    log("Running ACE Agent on AppWorld Tasks (with Learning)", quiet)
    log("=" * 60, quiet)

    # Determine model: CLI arg > env var > auto-detect
    if args.model:
        model = args.model
    elif os.environ.get("ACE_MODEL"):
        model = os.environ["ACE_MODEL"]
    elif os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        model = "claude-3-5-haiku-latest"
    else:
        model = "gpt-4o-mini"

    log(f"\nModel: {model}", quiet)
    log(f"Tasks: {len(input_data)}", quiet)
    log("Learning: ENABLED (Reflector + SkillManager)", quiet)

    try:
        # Initialize ACE components
        llm = LiteLLMClient(model=model)
        skillbook = Skillbook()
        agent = Agent(llm)
        reflector = Reflector(llm)
        skill_manager = SkillManager(llm)

        # Process each task with learning
        results = {}

        for task_id, task_data in input_data.items():
            log(f"\n--- Processing Task: {task_id} ---", quiet)

            try:
                # 1. Agent generates and executes code
                result = process_task(
                    task_id=task_id,
                    task_data=task_data,
                    agent=agent,
                    skillbook=skillbook,
                    max_interactions=args.max_interactions,
                )
                results[task_id] = result

                # 2. Build execution feedback from result
                is_success = not result.startswith("Error:")
                feedback = f"Task {task_id}: {'Completed successfully' if is_success else 'Failed with error'}"
                if not is_success:
                    feedback += f"\nError details: {result}"

                log(f"  Execution: {'SUCCESS' if is_success else 'FAILED'}", quiet)
                log("  Running Reflector...", quiet)

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
                log(f"  Reflection: {reflection_summary}", quiet)

                log("  Running SkillManager...", quiet)

                # 4. SkillManager updates skillbook
                skill_output = skill_manager.update_skills(
                    reflection=reflection,
                    skillbook=skillbook,
                    question_context=f"AppWorld task: {task_id}",
                    progress=f"Task {'completed' if is_success else 'failed'}",
                )
                skillbook.apply_update(skill_output.update)
                log(
                    f"  Updates applied: {len(skill_output.update.operations)} operations",
                    quiet,
                )

            except Exception as e:
                log(f"  Task error: {e}", quiet)
                results[task_id] = f"Error: {e}"

        # Print results
        log("\n" + "=" * 60, quiet)
        log("Execution Results", quiet)
        log("=" * 60, quiet)

        for task_id, result in results.items():
            log(f"\n--- Task: {task_id} ---", quiet)
            if result.startswith("Error:"):
                log("Status: FAILED", quiet)
                log(f"Error: {result}", quiet)
            else:
                log("Status: COMPLETED", quiet)
                log(
                    (
                        f"Code:\n{result[:500]}..."
                        if len(result) > 500
                        else f"Code:\n{result}"
                    ),
                    quiet,
                )

        # Summary (always print, even in quiet mode)
        successful = sum(1 for r in results.values() if not r.startswith("Error:"))
        print("\n" + "=" * 60)
        print(f"Execution Summary: {successful}/{len(results)} tasks completed")
        print("=" * 60)

        # Display learned skillbook insights
        log("\n" + "=" * 60, quiet)
        log("Learned Skillbook Insights", quiet)
        log("=" * 60, quiet)

        if skillbook.skills():
            log(str(skillbook), quiet)  # Human-readable markdown format
            stats = skillbook.stats()
            log(f"\nStats: {stats}", quiet)
        else:
            log("\nNo skills learned during this session.", quiet)
            log(
                "(This may happen if tasks completed without generating insights)",
                quiet,
            )

        # Save skillbook if requested
        if args.save_skillbook:
            skillbook.save_to_file(args.save_skillbook)
            print(f"\nSkillbook saved to: {args.save_skillbook}")

    except Exception as e:
        print(f"\nError running ACE agent: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
