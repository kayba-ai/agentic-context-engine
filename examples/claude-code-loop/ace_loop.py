#!/usr/bin/env python3
"""
ACE + Claude Code: TypeScript Translation using Integration Pattern

This script uses ACE's Integration Pattern where Claude Code acts as the agent:
1. INJECT: Playbook context added to Claude Code's prompt
2. EXECUTE: Claude Code runs with learned strategies
3. LEARN: Reflector and Curator learn from Claude Code's execution

Key points:
- No PassthroughGenerator needed - Claude Code IS the agent
- Claude Code receives playbook as context (like browser-use integration)
- Reflector receives only Claude Code's output, not pre-made strategy lists
- Eliminates bias from seeing all strategies with helpful counts
- Continuous loop: Reads TODO.md after each task and creates new samples
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ACE imports
from ace import (
    LiteLLMClient,
    Reflector,
    Curator,
    Sample,
    Playbook,
    EnvironmentResult,
)
from ace.prompts_v2_1 import PromptManager
from dataclasses import dataclass

# Local imports
from claude_code_environment import ClaudeCodeEnvironment


# CONFIGURATION
AUTO_MODE = os.getenv("AUTO_MODE", "false").lower() == "true"  # Set to true for fully automatic


def parse_next_task_from_todo(workspace_dir: Path) -> str | None:
    """
    Parse TODO.md to find next unchecked task.

    Skips category-level tasks and finds specific, actionable items.
    Prefers tasks that mention specific files or components.

    Returns:
        Next unchecked task description, or None if all complete
    """
    # Check both locations for TODO.md
    todo_paths = [
        workspace_dir / ".agent" / "TODO.md",
        workspace_dir / "TODO.md"  # Fallback if Claude Code puts it in wrong place
    ]

    todo_path = None
    for path in todo_paths:
        if path.exists():
            todo_path = path
            print(f"   📄 Found TODO.md at: {path}")
            break

    if not todo_path:
        print(f"   ⚠️  No TODO.md found in .agent/ or workspace/")
        return None

    # Read TODO.md
    content = todo_path.read_text()
    print(f"   📝 TODO.md has {len(content.split(chr(10)))} lines")

    # Look for unchecked tasks: [ ] or - [ ]
    # Pattern matches: "- [ ] Task description" or "[ ] Task description"
    pattern = r'^[\s\-]*\[ \]\s+(.+)$'

    # Category indicators (skip these - they're too vague)
    category_indicators = [
        'phase', 'step', 'stage', 'setup', 'initialization',
        'project initialization', 'core module', 'provider integration',
        'tests and documentation', 'final packaging',
        # Add infrastructure/meta-work terms:
        'eslint', 'linting', 'ci/cd', 'github actions', 'dependabot',
        'configuration', 'build scripts', 'documentation',
        'issue templates', 'pr templates', 'readme', 'migration guide',
        'architecture document', 'project structure', 'workflow',
        'prettier', 'formatting', 'code style', 'commit hooks'
    ]

    # Count tasks for debugging
    unchecked_count = 0
    checked_count = 0

    for line in content.split('\n'):
        if re.match(r'^[\s\-]*\[x\]\s+', line.strip(), re.IGNORECASE):
            checked_count += 1

        match = re.match(pattern, line.strip())
        if match:
            unchecked_count += 1
            task = match.group(1).strip()

            # Skip category-level tasks (too vague)
            task_lower = task.lower()
            is_category = any(indicator in task_lower for indicator in category_indicators)

            # Prefer specific tasks (mention files, classes, or have arrows →)
            is_specific = ('→' in task or
                          '.py' in task or
                          '.ts' in task or
                          'class' in task_lower or
                          'interface' in task_lower or
                          'translate' in task_lower)

            # Skip vague category tasks, prefer specific ones
            if is_specific or not is_category:
                # Additional check: skip if it's ONLY a category name
                if len(task.split()) > 2 or is_specific:  # More than 2 words or specific
                    print(f"   ✅ Found task: {task[:60]}...")
                    return task

    print(f"   📊 Task status: {checked_count} completed, {unchecked_count} unchecked")
    if unchecked_count == 0:
        print(f"   ℹ️  No unchecked tasks found. All tasks might be marked [x] or not in checkbox format")
    return None


def main():
    """Main orchestration function with continuous loop."""
    print("\n🚀 ACE + Claude Code")
    print("=" * 70)

    # Configuration
    MODEL = os.getenv("ACE_MODEL", "claude-sonnet-4-5-20250929")
    WORKSPACE_DIR = Path(__file__).parent / "workspace"
    PLAYBOOK_PATH = Path(__file__).parent / "playbooks" / "ace_typescript.json"

    print(f"\n🧠 Initializing ACE (model: {MODEL})...")
    print(f"   Mode: {'AUTOMATIC' if AUTO_MODE else 'INTERACTIVE (semi-automatic)'}")

    # Initialize LLM (for Reflector and Curator only)
    llm = LiteLLMClient(model=MODEL, temperature=0.2, max_tokens=2048)

    # Initialize ACE components
    prompt_manager = PromptManager()

    # Use standard v2.1 prompts (same as browseruse example)
    reflector = Reflector(llm, prompt_template=prompt_manager.get_reflector_prompt())
    print("   ✅ Reflector ready")

    curator = Curator(llm, prompt_template=prompt_manager.get_curator_prompt())
    print("   ✅ Curator ready")

    # Load or create playbook
    playbook = Playbook()
    if PLAYBOOK_PATH.exists():
        playbook = Playbook.load_from_file(str(PLAYBOOK_PATH))
        print(f"📚 Loaded playbook: {len(list(playbook.bullets()))} strategies")
    else:
        print("📚 Starting with empty playbook")
        PLAYBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initialize environment (Integration Pattern: Claude Code is the agent)
    environment = ClaudeCodeEnvironment(workspace_dir=str(WORKSPACE_DIR))
    print(f"✅ Environment initialized: {WORKSPACE_DIR}")

    # Read project spec for context
    spec_file = WORKSPACE_DIR / "specs" / "project.md"
    if spec_file.exists():
        spec_content = spec_file.read_text()
        context = f"Project specification:\n{spec_content[:1000]}..."
    else:
        context = "Translate ACE framework from Python to TypeScript following best practices."

    # Initial confirmation
    print("\n" + "=" * 70)
    print("🔄 READY TO START")
    print("=" * 70)

    if not AUTO_MODE:
        response = input("▶️  Start learning loop? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return

    # Task counter
    task_count = 0
    all_results = []

    # CONTINUOUS LOOP
    while True:
        task_count += 1

        # Determine next task
        if task_count == 1:
            # First task: Bootstrap
            task = """Create .agent/TODO.md with Python-to-TypeScript translation tasks.

CRITICAL REQUIREMENTS:
1. Create the file at .agent/TODO.md (your working directory is already 'workspace')
2. Use markdown checkbox format: - [ ] for each task (NOT [x] or plain text)
3. List specific Python files from source/ that need translation
4. Focus on actual code translation tasks, not project setup
5. Each task should be specific and actionable (e.g., "Translate ace/playbook.py to TypeScript")

Example format:
- [ ] Translate ace/playbook.py → target/src/playbook.ts
- [ ] Translate ace/delta.py → target/src/delta.ts
- [ ] Translate ace/llm.py → target/src/llm.ts
- [ ] Create unit tests for translated modules

DO NOT mark tasks as complete [x] - leave them all as [ ] for the loop to process."""
            print(f"\n📋 Task {task_count} (bootstrap): {task}")
        else:
            # Subsequent tasks: Read from TODO.md
            task = parse_next_task_from_todo(WORKSPACE_DIR)

            if not task:
                print(f"\n✅ No more tasks in TODO.md - all complete!")
                break

            print(f"\n📋 Task {task_count} from TODO.md:")
            print(f"   {task}")

        # Show current playbook size
        num_strategies = len(list(playbook.bullets()))
        print(f"📚 Current playbook: {num_strategies} strategies")

        # Interactive mode: Ask user
        if not AUTO_MODE and task_count > 1:
            print("")
            response = input("▶️  Process this task? (y/n/q): ").strip().lower()

            if response == 'q':
                print("\n👋 Quitting. Run again to resume from this point.")
                break
            elif response == 'n':
                print("⏭️  Skipping this task...")
                # TODO: Mark as skipped in TODO.md
                continue
            elif response != 'y':
                print("❓ Invalid input. Skipping...")
                continue

        # Execute task
        print(f"\n{'=' * 70}")
        print(f"🚀 EXECUTING TASK {task_count}")
        print("=" * 70 + "\n")

        # Create sample for this task
        sample = Sample(
            question=task,
            context=context,
            ground_truth=None
        )

        # Integration Pattern: INJECT → EXECUTE → LEARN
        print(f"\n🔄 Step 1: INJECT - Adding playbook context to Claude Code prompt")
        print(f"   📚 {len(list(playbook.bullets()))} strategies available")

        # EXECUTE: Run Claude Code with playbook context
        print(f"\n🔄 Step 2: EXECUTE - Running Claude Code with learned strategies")
        env_result, generator_output = environment.evaluate(sample, playbook)

        # LEARN: Reflector analyzes execution
        print(f"\n🔄 Step 3: LEARN - Analyzing execution feedback")
        reflection = reflector.reflect(
            question=sample.question,
            generator_output=generator_output,  # Minimal output, no playbook counts
            playbook=playbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            max_refinement_rounds=2,
        )

        # Curator updates playbook
        print(f"   🧠 Reflector analysis complete")
        curator_output = curator.curate(
            reflection=reflection,
            playbook=playbook,
            question_context=f"task: {sample.question}",
            progress=f"Executing: {sample.question}"
        )
        print(f"   📝 Curator generated {len(curator_output.delta.operations)} playbook operations")

        # Apply delta to playbook
        playbook.apply_delta(curator_output.delta)

        # Save playbook after each task
        playbook.save_to_file(str(PLAYBOOK_PATH))
        print(f"\n💾 Playbook saved to {PLAYBOOK_PATH}")

        # Store result for summary
        @dataclass
        class TaskResult:
            environment_result: EnvironmentResult

        result = TaskResult(environment_result=env_result)
        all_results.append(result)

        # Show task summary
        success = env_result.metrics.get("success", False)
        new_strategies = len(list(playbook.bullets())) - num_strategies

        print(f"\n✅ Task {task_count} completed: {'SUCCESS' if success else 'FAILED'}")
        print(f"📚 Playbook now has {len(list(playbook.bullets()))} strategies (+{new_strategies} new)")

        if not AUTO_MODE:
            input("\nPress Enter to continue to next task...")

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 ALL TASKS COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in all_results if r.environment_result.metrics.get("success", False))
    print(f"\nTotal tasks processed: {len(all_results)}")
    print(f"Successful: {successful}/{len(all_results)}")
    print(f"Final playbook: {len(list(playbook.bullets()))} strategies")

    # Show top strategies
    if list(playbook.bullets()):
        print(f"\n🎯 Top 10 Learned Strategies:")
        print("-" * 70)
        sorted_bullets = sorted(
            playbook.bullets(),
            key=lambda b: b.helpful - b.harmful,
            reverse=True
        )
        for i, bullet in enumerate(sorted_bullets[:10], 1):
            impact = f"+{bullet.helpful} helpful, -{bullet.harmful} harmful"
            print(f"{i}. [{bullet.id}] {bullet.content[:80]}...")
            print(f"   Impact: {impact}")

    print(f"\n📂 Workspace: {WORKSPACE_DIR}")
    print(f"📚 Playbook: {PLAYBOOK_PATH}")
    print(f"\n💡 To run in fully automatic mode, set environment variable:")
    print(f"   export AUTO_MODE=true")
    print(f"   python ace_loop.py")


if __name__ == "__main__":
    main()
