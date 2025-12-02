#!/usr/bin/env python3
"""
ACE + Claude Code: Continuous Learning Loop

This script demonstrates using ACEClaudeCode to run Claude Code in a loop,
learning from each task execution. Tasks are read from a TODO.md file.

Usage:
    python ace_loop.py                    # Interactive mode
    AUTO_MODE=true python ace_loop.py     # Fully automatic
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from ace.integrations import ACEClaudeCode

load_dotenv()

# Configuration
DEMO_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("ACE_DEMO_DATA_DIR", str(DEMO_DIR / ".data")))
AUTO_MODE = os.getenv("AUTO_MODE", "false").lower() == "true"
ACE_MODEL = os.getenv("ACE_MODEL", "claude-sonnet-4-5-20250929")

# Paths
WORKSPACE_DIR = DEMO_DIR / "workspace"  # Separate git repo
PLAYBOOK_PATH = DATA_DIR / "playbooks" / "ace_typescript.json"
LOGS_DIR = DATA_DIR / "logs"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "playbooks").mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Verify workspace is a git repository
if not (WORKSPACE_DIR / ".git").exists():
    print("❌ Error: Workspace is not a git repository!")
    print(f"   Run ./reset_workspace.sh to initialize workspace")
    sys.exit(1)


def parse_next_task_from_todo(workspace_dir: Path) -> str | None:
    """
    Parse TODO.md to find next unchecked task.

    Returns:
        Next unchecked task description, or None if all complete
    """
    todo_paths = [workspace_dir / ".agent" / "TODO.md", workspace_dir / "TODO.md"]

    todo_path = None
    for path in todo_paths:
        if path.exists():
            todo_path = path
            print(f"   Found TODO.md at: {path}")
            break

    if not todo_path:
        print(f"   No TODO.md found")
        return None

    content = todo_path.read_text()

    # Look for unchecked tasks: [ ] or - [ ]
    pattern = r"^[\s\-]*\[ \]\s+(.+)$"

    # Skip vague category tasks
    category_indicators = [
        "phase",
        "step",
        "stage",
        "setup",
        "initialization",
        "eslint",
        "linting",
        "ci/cd",
        "configuration",
    ]

    for line in content.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            task = match.group(1).strip()
            task_lower = task.lower()

            # Skip category-level tasks
            if any(ind in task_lower for ind in category_indicators):
                continue

            # Skip markdown headers
            if task.startswith("#"):
                continue

            # Accept any non-empty task (file paths or descriptive tasks)
            print(f"   Found task: {task[:60]}...")
            return task

    return None


def build_task_prompt(task: str, context: str = "") -> str:
    """
    Build rich prompt for Claude Code with workspace context and instructions.

    This restores the detailed prompting from the original claude_code_environment.py
    that guides Claude Code on workspace structure, priorities, and response format.
    """
    return f"""TASK:
{task}

CONTEXT:
{context if context else 'None'}

WORKSPACE STRUCTURE:
- source/ - Python source code to translate
- target/ - TypeScript output directory
- specs/ - project.md (spec) and rules.md (coding standards)
- .agent/ - your scratchpad (TODO.md, PLAN.md, notes)

CRITICAL REQUIREMENTS:
1. Apply relevant strategies from the playbook context (injected above)
2. Test every new feature/change BEFORE committing (run tests, check compilation)
3. Make atomic git commits after each working unit (commit message should explain what was done and why)
4. You are working in a dedicated git repository - commit freely after testing
5. Focus on actual work, not elaborate documentation or setup

GIT WORKFLOW:
- After completing a logical unit of work (e.g., translate one file, fix one bug)
- Run tests to verify it works
- Commit with: git add <files> && git commit -m "Clear message"
- Each task may result in one or more commits

FOCUS YOUR EFFORT ON:
- Reading source files and understanding the implementation
- Writing quality code following the specs
- Writing/updating tests (aim for 20% of effort)
- Fixing compilation/test errors

DO NOT spend excessive time on:
- Elaborate documentation (README, guides)
- Complex linting/CI configurations
- Empty directory structures

RESPONSE FORMAT:
1. ## Approach - explain your plan and why
2. ## Implementation - do the actual work
3. ## Summary - what was accomplished

AFTER COMPLETING THIS TASK:
1. Update TODO.md: Mark this task as [x] (change [ ] to [x])
2. STOP - do not continue to next tasks (the loop will call you again)
"""


def run_npm_command(
    workspace_dir: Path, command: str, description: str
) -> tuple[bool, str]:
    """
    Run npm command and capture output.

    Args:
        workspace_dir: Directory to run command in
        command: npm script name (e.g., "build", "test")
        description: Human-readable description

    Returns:
        Tuple of (success, output)
    """
    try:
        print(f"   Running: npm run {command}")
        result = subprocess.run(
            ["npm", "run", command],
            cwd=workspace_dir / "target",  # TypeScript project is in target/
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after 300 seconds"
    except Exception as e:
        return False, f"Error running command: {str(e)}"


def extract_tsc_errors(output: str, max_lines: int = 50) -> str:
    """Extract TypeScript compilation errors from tsc output."""
    lines = output.split("\n")
    error_lines = [line for line in lines if "error TS" in line or ".ts(" in line]
    if len(error_lines) > max_lines:
        error_lines = error_lines[:max_lines] + [
            f"\n... ({len(error_lines) - max_lines} more errors)"
        ]
    return "\n".join(error_lines) if error_lines else output


def extract_jest_errors(output: str, max_lines: int = 50) -> str:
    """Extract Jest test errors from output."""
    lines = output.split("\n")
    # Look for FAIL markers and error messages
    error_lines = []
    in_error = False
    for line in lines:
        if "FAIL" in line or "Error:" in line or "Expected" in line:
            in_error = True
        if in_error:
            error_lines.append(line)
            if len(error_lines) >= max_lines:
                error_lines.append(f"... (output truncated)")
                break
    return "\n".join(error_lines) if error_lines else output


def validate_typescript_compilation(workspace_dir: Path) -> tuple[bool, str]:
    """Run tsc --noEmit to check TypeScript compiles."""
    success, output = run_npm_command(workspace_dir, "build", "TypeScript compilation")
    if success:
        return True, "✅ TypeScript compilation successful"
    else:
        errors = extract_tsc_errors(output)
        return False, f"❌ TypeScript compilation failed:\n{errors}"


def validate_unit_tests(workspace_dir: Path) -> tuple[bool, str]:
    """Run Jest unit tests."""
    success, output = run_npm_command(workspace_dir, "test", "Unit tests")
    if success:
        return True, "✅ All unit tests passed"
    else:
        errors = extract_jest_errors(output)
        return False, f"❌ Unit tests failed:\n{errors}"


def validate_example(workspace_dir: Path, example_name: str) -> tuple[bool, str]:
    """Run example file."""
    success, output = run_npm_command(
        workspace_dir, f"example:{example_name}", f"{example_name} example"
    )
    if success:
        return True, f"✅ {example_name} example ran successfully"
    else:
        # Truncate long output
        if len(output) > 500:
            output = output[:500] + "\n... (output truncated)"
        return False, f"❌ {example_name} example failed:\n{output}"


def run_full_validation(workspace_dir: Path) -> tuple[bool, str]:
    """
    Run all validation checks in sequence.

    Returns:
        Tuple of (success, message)
    """
    results = []

    # 1. TypeScript compilation
    print(f"\n🔍 Step 1/4: TypeScript Compilation")
    success, msg = validate_typescript_compilation(workspace_dir)
    results.append((success, msg))
    print(f"   {msg.split(':')[0]}")  # Print status only
    if not success:
        return False, msg

    # 2. Unit tests
    print(f"\n🔍 Step 2/4: Unit Tests")
    success, msg = validate_unit_tests(workspace_dir)
    results.append((success, msg))
    print(f"   {msg.split(':')[0]}")
    if not success:
        return False, msg

    # 3. Simple example
    print(f"\n🔍 Step 3/4: Simple Example")
    success, msg = validate_example(workspace_dir, "simple")
    results.append((success, msg))
    print(f"   {msg.split(':')[0]}")
    if not success:
        return False, msg

    # 4. Seahorse example
    print(f"\n🔍 Step 4/4: Seahorse Example")
    success, msg = validate_example(workspace_dir, "seahorse")
    results.append((success, msg))
    print(f"   {msg.split(':')[0]}")
    if not success:
        return False, msg

    # All passed!
    summary = "\n".join([msg for _, msg in results])
    return True, f"🎉 ALL VALIDATION CHECKS PASSED!\n\n{summary}"


def print_playbook_summary(playbook):
    """Show playbook as automatic prompt engineering."""
    bullets = list(playbook.bullets())
    new_count = len([b for b in bullets if b.helpful + b.harmful <= 1])

    print("\n" + "=" * 70)
    print("📚 PLAYBOOK LEARNING SUMMARY")
    print("=" * 70)
    print(f"\n✅ Learned {new_count} new strategies during this run")
    print("💡 THIS IS AUTOMATIC PROMPT ENGINEERING")
    print("   No manual iteration needed - ACE learned from execution\n")

    # Show top strategies
    sorted_bullets = sorted(bullets, key=lambda b: b.helpful - b.harmful, reverse=True)
    print("Top 5 Helpful Strategies:")
    for i, bullet in enumerate(sorted_bullets[:5], 1):
        score = f"[+{bullet.helpful} -{bullet.harmful}]"
        content = (
            bullet.content[:60] + "..." if len(bullet.content) > 60 else bullet.content
        )
        print(f"{i}. {score} {content}")

    print(f"\n🎯 Next run will start with {len(bullets)} strategies")
    print("   Expect 60-80% improvement in tasks/time/cost\n")


def print_improvement_analysis(current_run, previous_runs):
    """Show before/after if this is run 2+."""
    if len(previous_runs) == 0:
        return

    run1 = previous_runs[0]  # First run for comparison

    print("\n" + "=" * 70)
    print("📊 ACE LEARNING IMPROVEMENT ANALYSIS")
    print("=" * 70)

    # Calculate improvement
    task_reduction = 0
    if run1["task_count"] > 0:
        task_reduction = int((1 - current_run["task_count"] / run1["task_count"]) * 100)

    failure_reduction = 0
    if run1["total_failures"] > 0:
        failure_reduction = int(
            (1 - current_run["total_failures"] / run1["total_failures"]) * 100
        )

    print(f"\nRun 1 (Blind - No Strategies):")
    print(f"├─ Tasks: {run1['task_count']} tasks")
    print(f"├─ Validation failures: {run1['total_failures']} attempts")
    print(f"└─ Strategies learned: {run1['strategies_count']}")

    print(f"\nRun 2+ (With Learned Strategies):")
    print(f"├─ Tasks: {current_run['task_count']} tasks ({task_reduction}% reduction!)")
    print(
        f"├─ Validation failures: {current_run['total_failures']} attempts ({failure_reduction}% fewer!)"
    )
    print(f"└─ Strategies applied: {current_run['strategies_count']}")

    print("\n🎯 WHAT THIS MEANS:")
    print("- Manual prompt engineering = WEEKS of iteration")
    print("- ACE = ONE RUN learns, next run applies automatically")
    print("- This is the future: no prompt engineering, just learning\n")


def main():
    """Main orchestration function with continuous loop."""
    print("\n ACE + Claude Code")
    print("=" * 70)

    print(f"\n Initializing (model: {ACE_MODEL})...")
    print(f"   Mode: {'AUTOMATIC' if AUTO_MODE else 'INTERACTIVE'}")

    # Initialize ACEClaudeCode
    agent = ACEClaudeCode(
        working_dir=str(WORKSPACE_DIR),
        ace_model=ACE_MODEL,
        playbook_path=str(PLAYBOOK_PATH) if PLAYBOOK_PATH.exists() else None,
    )

    print(f" Playbook: {len(list(agent.playbook.bullets()))} strategies")
    print(f" Workspace: {WORKSPACE_DIR}")

    # Show current git branch
    try:
        current_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"📍 Branch: {current_branch}")
    except Exception:
        pass  # Ignore if git command fails

    # Read project spec for context
    spec_file = WORKSPACE_DIR / "specs" / "project.md"
    context = ""
    if spec_file.exists():
        context = f"Project specification:\n{spec_file.read_text()[:1000]}..."

    # Initial confirmation
    print("\n" + "=" * 70)
    if not AUTO_MODE:
        response = input(" Start learning loop? (y/n): ")
        if response.lower() != "y":
            print(" Cancelled")
            return

    task_count = 0
    results = []
    validation_attempts = 0
    total_validation_failures = 0  # NEW
    MAX_VALIDATION_ATTEMPTS = 5
    MAX_TOTAL_FAILURES = 10  # NEW

    # Load previous run metrics for comparison
    runs_file = DATA_DIR / "runs.json"
    previous_runs = []
    if runs_file.exists():
        try:
            previous_runs = json.loads(runs_file.read_text())
        except Exception:
            previous_runs = []

    # PHASE 1: TRANSLATION LOOP (existing behavior)
    print("\n" + "=" * 70)
    print("📝 PHASE 1: TRANSLATION")
    print("=" * 70)

    while True:
        task_count += 1

        # Determine next task
        if task_count == 1:
            task = """Create .agent/TODO.md with Python→TypeScript translation tasks.

Use checkbox format with clear descriptions. Examples:
- [ ] Translate source/ace/playbook.py to TypeScript (convert dataclasses to interfaces)
- [ ] Translate source/ace/roles.py to TypeScript (use Vercel AI SDK)
- [ ] Create package.json with TypeScript dependencies

List all Python files from source/ace/ that need translation.
Group by module (Core, Integrations, LLM Providers, etc.)."""
            print(f"\n Task {task_count} (bootstrap): Create TODO.md")
        else:
            task = parse_next_task_from_todo(WORKSPACE_DIR)
            if not task:
                print(f"\n✅ All translation tasks complete!")
                break
            print(f"\n Task {task_count}: {task[:60]}...")

        # Interactive confirmation
        if not AUTO_MODE and task_count > 1:
            response = input(" Process this task? (y/n/q): ").strip().lower()
            if response == "q":
                break
            elif response != "y":
                continue

        # Execute task
        print(f"\n{'=' * 70}")
        print(f" EXECUTING TASK {task_count}")
        print("=" * 70 + "\n")

        # Build rich prompt with workspace context (for non-bootstrap tasks)
        if task_count > 1:
            task_prompt = build_task_prompt(task, context)
        else:
            task_prompt = task  # Bootstrap task doesn't need wrapper

        result = agent.run(task=task_prompt, context="")
        results.append(result)

        # Summary
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n Task {task_count}: {status}")
        print(f" Playbook: {len(list(agent.playbook.bullets()))} strategies")

        # Save after each task
        agent.save_playbook(str(PLAYBOOK_PATH))

        if not AUTO_MODE:
            input("\nPress Enter to continue...")

    # PHASE 2 & 3: VALIDATION + FIX LOOP
    print("\n" + "=" * 70)
    print("🧪 PHASE 2 & 3: VALIDATION + FIX LOOP")
    print("=" * 70 + "\n")

    # Check if target/package.json exists (TypeScript project setup)
    if not (WORKSPACE_DIR / "target" / "package.json").exists():
        print("⚠️  No TypeScript project found in target/ - skipping validation")
        print("   (Validation requires target/package.json with npm scripts)")
    else:
        while validation_attempts < MAX_VALIDATION_ATTEMPTS:
            validation_attempts += 1
            print(
                f"\n🔍 Validation Attempt {validation_attempts}/{MAX_VALIDATION_ATTEMPTS}"
            )

            # Run all validation checks
            success, validation_output = run_full_validation(WORKSPACE_DIR)

            if success:
                # SUCCESS!
                print("\n" + validation_output)
                print("\n" + "=" * 70)
                print("🎉 TRANSLATION SUCCESSFUL!")
                print("=" * 70)
                break
            else:
                # FAILED - Feed errors back to Claude Code
                total_validation_failures += 1
                print("\n" + validation_output)
                print(
                    f"\n❌ Validation failed (total failures: {total_validation_failures}/{MAX_TOTAL_FAILURES})"
                )

                # Check if environment is complaining too much
                if total_validation_failures >= MAX_TOTAL_FAILURES:
                    print("\n" + "=" * 70)
                    print("🛑 CANCELLING - ENVIRONMENT COMPLAINED TOO MUCH")
                    print("=" * 70)
                    print(
                        f"\n{total_validation_failures} validation failures detected."
                    )
                    print(
                        "This suggests fundamental issues with the translation approach."
                    )
                    print(
                        "\nPlaybook has learned from failures - next run will be better."
                    )
                    break

                if validation_attempts >= MAX_VALIDATION_ATTEMPTS:
                    print(
                        f"\n❌ Maximum validation attempts ({MAX_VALIDATION_ATTEMPTS}) reached"
                    )
                    break

                # Create fix prompt
                fix_prompt = f"""VALIDATION FAILED - FIX REQUIRED

The TypeScript translation has validation errors. Please analyze and fix them.

{validation_output}

INSTRUCTIONS:
1. Read the error messages carefully
2. Identify which files need fixes
3. Make the necessary corrections
4. Test your changes (run the failing command to verify fix)
5. Commit your fixes with: git add . && git commit -m "Fix validation errors"
6. Respond with a summary of what you fixed

Focus only on fixing the specific errors shown above.
Do NOT move on to other tasks or improvements.
"""

                # Feed back to Claude Code
                print(f"\n🔧 Feeding errors back to Claude Code for fixes...")
                if not AUTO_MODE:
                    response = (
                        input("   Continue with fix attempt? (y/n): ").strip().lower()
                    )
                    if response != "y":
                        print("   Skipping fix loop")
                        break

                fix_result = agent.run(task=fix_prompt, context="")
                results.append(fix_result)
                agent.save_playbook(str(PLAYBOOK_PATH))

                # Loop back to validation

    # Final summary
    print("\n" + "=" * 70)
    print(" COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in results if r.success)
    print(f"\nTranslation tasks: {successful}/{len(results)} successful")
    print(f"Validation attempts: {validation_attempts}")
    print(f"Total validation failures: {total_validation_failures}")

    # Show playbook learning summary
    print_playbook_summary(agent.playbook)

    # Save run metrics
    try:
        current_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        current_branch = "unknown"

    run_metadata = {
        "run_id": current_branch,
        "timestamp": datetime.now().isoformat(),
        "task_count": len(results),
        "validation_attempts": validation_attempts,
        "total_failures": total_validation_failures,
        "strategies_count": len(list(agent.playbook.bullets())),
        "success": validation_attempts > 0
        and total_validation_failures < MAX_TOTAL_FAILURES,
    }

    # Save to runs.json
    if runs_file.exists():
        runs = json.loads(runs_file.read_text())
    else:
        runs = []
    runs.append(run_metadata)
    runs_file.write_text(json.dumps(runs, indent=2))

    # Show improvement if run 2+
    if len(runs) > 1:
        print_improvement_analysis(run_metadata, runs[:-1])

    print(f"Playbook saved to: {PLAYBOOK_PATH}")
    print(f"Run metrics saved to: {runs_file}")


if __name__ == "__main__":
    main()
