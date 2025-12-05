#!/usr/bin/env python3
"""
ACE + Claude Code: Simple Learning Loop

Claude Code works autonomously until done, using .agent/ as scratchpad.
After each session, ACE learning runs, then fresh session continues.

Stopping conditions:
- Stall detected (N consecutive sessions with no code changes)
- Manual interruption

Usage:
    python ace_loop.py                    # Interactive mode
    AUTO_MODE=true python ace_loop.py     # Fully automatic
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from ace.integrations import ACEClaudeCode

load_dotenv()

# Configuration
AUTO_MODE = os.getenv("AUTO_MODE", "false").lower() == "true"
ACE_MODEL = os.getenv("ACE_MODEL", "claude-sonnet-4-5-20250929")
DEMO_DIR = Path(__file__).parent
WORKSPACE_DIR = DEMO_DIR / "workspace"
DATA_DIR = Path(os.getenv("ACE_DEMO_DATA_DIR", str(DEMO_DIR / ".data")))
SKILLBOOK_PATH = DATA_DIR / "skillbooks" / "ace_typescript.json"


def get_commit_count(workspace_dir: Path) -> int:
    """Get current commit count in target repo."""
    target_dir = workspace_dir / "target"
    if not (target_dir / ".git").exists():
        return 0
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=target_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return int(result.stdout.strip())
    return 0


def are_recent_commits_code_changes(workspace_dir: Path, n: int = 3) -> bool:
    """Check if recent N commits contain actual code changes (not just docs)."""
    target_dir = workspace_dir / "target"
    if not (target_dir / ".git").exists():
        return True

    result = subprocess.run(
        ["git", "log", f"-{n}", "--name-only", "--format="],
        cwd=target_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return True

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    code_extensions = {".ts", ".tsx", ".js"}
    doc_files = {"README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"}

    for f in files:
        ext = Path(f).suffix.lower()
        basename = Path(f).name
        if ext in code_extensions:
            return True
        if ext == ".json" and basename not in {"package.json", "package-lock.json"}:
            return True
        if basename not in doc_files and ext != ".md":
            return True
    return False


def build_prompt() -> str:
    """Build the task prompt."""
    return """Your job is to port ACE framework (Python) to TypeScript and maintain the repository.

You have access to source/ (Python) and target/ (TypeScript git repo).

Make a commit after every single file edit.

Use .agent/ directory as scratchpad for your work. Store long term plans and todo lists there.

Key substitution: LiteLLM (Python) → Vercel AI SDK (TypeScript)

The .env file contains API keys for running examples.

Spend 80% of time on porting, 20% on testing. When porting is complete, improve code quality and fix any issues.
"""


def main():
    """Main orchestration: simple learning loop."""
    print("\n" + "=" * 60)
    print(" ACE + Claude Code")
    print("=" * 60)

    print(f"\n Initializing (model: {ACE_MODEL})...")
    print(f"   Mode: {'AUTOMATIC' if AUTO_MODE else 'INTERACTIVE'}")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SKILLBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initialize ACEClaudeCode
    agent = ACEClaudeCode(
        working_dir=str(WORKSPACE_DIR),
        ace_model=ACE_MODEL,
        skillbook_path=str(SKILLBOOK_PATH) if SKILLBOOK_PATH.exists() else None,
    )

    print(f"   Skillbook: {len(list(agent.skillbook.skills()))} strategies")
    print(f"   Workspace: {WORKSPACE_DIR}")

    # Initial confirmation
    if not AUTO_MODE:
        print("\n" + "-" * 60)
        response = input(" Start learning loop? (y/n): ")
        if response.lower() != "y":
            print(" Cancelled")
            return

    session_count = 0
    total_commits = 0
    results = []
    stall_count = 0
    MAX_STALLS = 4

    # SIMPLE LOOP
    while True:
        session_count += 1
        initial_commits = get_commit_count(WORKSPACE_DIR)

        print(f"\n{'=' * 60}")
        print(f" SESSION {session_count}")
        print(f"   Skillbook: {len(list(agent.skillbook.skills()))} strategies")
        print("=" * 60)

        # Interactive confirmation
        if not AUTO_MODE and session_count > 1:
            response = input("\n Continue? (y/n): ").strip().lower()
            if response != "y":
                break

        # Run Claude - let it work until done
        prompt = build_prompt()
        result = agent.run(task=prompt, context="")
        results.append(result)

        # Check progress
        final_commits = get_commit_count(WORKSPACE_DIR)
        commits_made = final_commits - initial_commits
        total_commits += commits_made

        print(f"\n   Commits: {commits_made} | Total: {total_commits}")
        print(f"   Skillbook: {len(list(agent.skillbook.skills()))} strategies")

        # Save skillbook
        agent.save_skillbook(str(SKILLBOOK_PATH))

        # Stall detection
        if commits_made == 0:
            stall_count += 1
            print(f"   Warning: No commits ({stall_count}/{MAX_STALLS})")
            if stall_count >= MAX_STALLS:
                print("\n STALLED - no progress. Stopping.")
                break
        elif not are_recent_commits_code_changes(WORKSPACE_DIR, n=commits_made):
            stall_count += 1
            print(f"   Warning: Doc-only commits ({stall_count}/{MAX_STALLS})")
            if stall_count >= MAX_STALLS:
                print("\n STALLED - doc-only changes. Stopping.")
                break
        else:
            stall_count = 0  # Reset on real progress

    # Final summary
    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60)
    print(f"\nSessions: {len(results)}")
    print(f"Commits: {total_commits}")
    print(f"Skillbook: {len(list(agent.skillbook.skills()))} strategies")

    skills = list(agent.skillbook.skills())
    if skills:
        print(f"\n Top Strategies:")
        sorted_skills = sorted(
            skills, key=lambda s: s.helpful - s.harmful, reverse=True
        )
        for i, skill in enumerate(sorted_skills[:5], 1):
            print(f"  {i}. {skill.content[:65]}...")

    print(f"\n Saved: {SKILLBOOK_PATH}")


if __name__ == "__main__":
    main()
