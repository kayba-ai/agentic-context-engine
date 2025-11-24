#!/usr/bin/env python3
"""
Run all Ollama agent demos sequentially.

This script runs all 10 real-world use case agents to demonstrate
ACE learning capabilities across different domains.

Agents demonstrated:
Development & DevOps:
1. Code Review Agent - Security and best practices
2. Test Case Generator - Unit test generation
3. SQL Query Generator - Natural language to SQL
4. Git Commit Message Generator - Conventional commits
5. Troubleshooting Assistant - System diagnostics

Operations & Support:
6. Email/Ticket Classifier - Support automation
7. Bug Report Analyzer - Issue triage

Data & Analytics:
8. Data Analysis Agent - Insights and patterns

Security:
9. Security Log Analyzer - Threat detection

Documentation:
10. Technical Writer - Documentation generation

Prerequisites:
- Ollama installed and running
- Model pulled (llama3.1:8b recommended)
"""

import sys
import subprocess
import time
from pathlib import Path


DEMOS = [
    # Development & DevOps
    {
        "name": "Code Review Agent",
        "script": "code_review_agent.py",
        "description": "Reviews code for bugs, security issues, and best practices",
        "emoji": "🔍"
    },
    {
        "name": "Test Case Generator",
        "script": "test_generator_agent.py",
        "description": "Generates comprehensive unit tests from code",
        "emoji": "🧪"
    },
    {
        "name": "SQL Query Generator",
        "script": "sql_query_agent.py",
        "description": "Generates SQL queries from natural language",
        "emoji": "🗄️"
    },
    {
        "name": "Git Commit Message Generator",
        "script": "commit_message_agent.py",
        "description": "Generates conventional commit messages from diffs",
        "emoji": "📝"
    },
    {
        "name": "Troubleshooting Assistant",
        "script": "troubleshooting_agent.py",
        "description": "Diagnoses system issues and suggests fixes",
        "emoji": "🔧"
    },
    # Operations & Support
    {
        "name": "Email/Ticket Classifier",
        "script": "email_classifier_agent.py",
        "description": "Categorizes and routes support tickets",
        "emoji": "📧"
    },
    {
        "name": "Bug Report Analyzer",
        "script": "bug_report_agent.py",
        "description": "Parses bug reports and extracts structured information",
        "emoji": "🐛"
    },
    # Data & Analytics
    {
        "name": "Data Analysis Agent",
        "script": "data_analysis_agent.py",
        "description": "Analyzes data and generates insights",
        "emoji": "📊"
    },
    # Security
    {
        "name": "Security Log Analyzer",
        "script": "security_log_agent.py",
        "description": "Analyzes security logs and detects threats",
        "emoji": "🔐"
    },
    # Documentation
    {
        "name": "Technical Writer Agent",
        "script": "technical_writer_agent.py",
        "description": "Converts technical content to documentation",
        "emoji": "📝"
    },
]


def check_ollama():
    """Check if Ollama is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_demo(demo: dict) -> bool:
    """Run a single demo script."""
    print("\n" + "=" * 70)
    print(f"{demo['emoji']} {demo['name']}")
    print(f"   {demo['description']}")
    print("=" * 70)

    script_path = Path(__file__).parent / demo['script']

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            timeout=300  # 5 minute timeout per demo
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  Demo timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        return False


def main():
    print("🚀 ACE Framework - Ollama Agent Demos")
    print("=" * 70)
    print("Running all 10 real-world use case agents sequentially")
    print("This will take approximately 20-30 minutes")
    print("=" * 70)

    # Check Ollama
    if not check_ollama():
        print("\n❌ Ollama not available!")
        print("\nSetup:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull a model: ollama pull llama3.1:8b")
        print("3. Verify: ollama list")
        return 1

    print("\n✅ Ollama is available\n")

    # Run all demos
    results = {}
    start_time = time.time()

    for i, demo in enumerate(DEMOS, 1):
        print(f"\n[{i}/{len(DEMOS)}] Running {demo['name']}...")
        success = run_demo(demo)
        results[demo['name']] = success

        if not success:
            print(f"\n⚠️  {demo['name']} failed or timed out")

        # Pause between demos
        if i < len(DEMOS):
            print("\n⏸️  Waiting 5 seconds before next demo...")
            time.sleep(5)

    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for success in results.values() if success)

    print("\n\n" + "=" * 70)
    print("📊 Demo Summary")
    print("=" * 70)

    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {demo_name}")

    print("\n" + "=" * 70)
    print(f"Completed {successful}/{len(DEMOS)} demos successfully")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)

    # Show saved playbooks
    print("\n💾 Saved Playbooks:")
    playbook_dir = Path(__file__).parent
    for playbook in playbook_dir.glob("*_playbook.json"):
        size_kb = playbook.stat().st_size / 1024
        print(f"   • {playbook.name} ({size_kb:.1f} KB)")

    print("\n✅ All demos completed!")
    print("📖 Load these playbooks to reuse learned knowledge")

    return 0 if successful == len(DEMOS) else 1


if __name__ == "__main__":
    sys.exit(main())
