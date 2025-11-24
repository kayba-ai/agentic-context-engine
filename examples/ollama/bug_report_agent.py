#!/usr/bin/env python3
"""
Bug Report Analyzer Agent - Parses bug reports and extracts key information.

This agent learns to extract structured data from bug reports, identify
duplicates, classify severity, and suggest component assignment.

Use Cases:
- GitHub issue triage
- JIRA ticket validation
- Duplicate bug detection
- Severity/priority assignment

Model Recommendation: ollama/llama3.1:8b or ollama/mistral:7b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class BugReportEnvironment(TaskEnvironment):
    """Evaluates bug report analysis quality."""

    def __init__(self, expected_analysis: dict[str, dict]):
        """
        Args:
            expected_analysis: Dict mapping bug IDs to expected analysis
        """
        self.expected_analysis = expected_analysis

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if analysis identified key information correctly."""
        bug_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_analysis.get(bug_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation criteria available",
                ground_truth="Valid analysis"
            )

        severity = expected.get("severity", "").lower()
        component = expected.get("component", "").lower()
        required_info = expected.get("required_info", [])
        wrong_severity = expected.get("wrong_severity", [])

        answer_lower = answer.lower()

        # Check analysis accuracy
        has_severity = severity in answer_lower
        has_component = component in answer_lower
        has_required = sum(1 for info in required_info if info.lower() in answer_lower)
        has_wrong = any(ws.lower() in answer_lower for ws in wrong_severity)

        success = (
            has_severity and
            has_component and
            has_required >= len(required_info) * 0.6 and
            not has_wrong
        )

        feedback = ""
        if has_severity:
            feedback += f"✓ Correct severity ({severity}). "
        else:
            feedback += f"✗ Wrong severity (expected {severity}). "

        if has_component:
            feedback += f"✓ Correct component ({component}). "
        else:
            feedback += f"✗ Wrong component (expected {component}). "

        feedback += f"Identified {has_required}/{len(required_info)} key details. "

        if has_wrong:
            feedback += "✗ Incorrect severity classification! "
            success = False

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Severity: {severity}, Component: {component}"
        )


# Training data: bug reports with expected analysis
TRAINING_DATA = [
    {
        "id": "bug_1",
        "report": """
Title: App crashes when clicking Save button
Description: Every time I try to save a document, the entire application
crashes and closes. I lose all my unsaved work. This happens 100% of the
time on Windows 10, version 2.5.1.
Steps to reproduce:
1. Open any document
2. Make changes
3. Click Save button
4. App crashes
        """,
        "severity": "critical",
        "component": "ui",
        "required_info": ["reproducible", "data loss", "version", "platform"],
        "wrong_severity": ["low", "minor"]
    },
    {
        "id": "bug_2",
        "report": """
Title: Typo in settings menu
Description: In the Settings > Preferences dialog, "Notification" is
spelled as "Notifcation" (missing 'i'). Not a big deal but wanted to
report it.
        """,
        "severity": "trivial",
        "component": "ui",
        "required_info": ["cosmetic", "ui text", "location"],
        "wrong_severity": ["critical", "high"]
    },
    {
        "id": "bug_3",
        "report": """
Title: API returns 500 error on /users endpoint
Description: The /users endpoint is returning 500 Internal Server Error
intermittently. Happens about 30% of the time. Error logs show database
connection timeout. This is blocking our integration.
Environment: Production API v3.2
        """,
        "severity": "high",
        "component": "api",
        "required_info": ["api endpoint", "error code", "intermittent", "logs"],
        "wrong_severity": ["trivial", "low"]
    },
    {
        "id": "bug_4",
        "report": """
Title: Button color slightly off in dark mode
Description: In dark mode, the submit button appears to be #2A2A2A but
design specs say it should be #2B2B2B. The difference is barely visible
but want to maintain consistency.
        """,
        "severity": "minor",
        "component": "ui",
        "required_info": ["dark mode", "visual", "design spec"],
        "wrong_severity": ["critical", "blocker"]
    },
    {
        "id": "bug_5",
        "report": """
Title: URGENT: Payment processing completely broken
Description: ALL payment transactions are failing with "Gateway timeout".
No customer can complete checkout. This started 2 hours ago and we're
losing sales every minute. Monitoring shows payment service is down.
Impact: 100% of users, revenue loss.
        """,
        "severity": "blocker",
        "component": "payments",
        "required_info": ["100% impact", "revenue loss", "recent", "service down"],
        "wrong_severity": ["minor", "low"]
    },
    {
        "id": "bug_6",
        "report": """
Title: Memory leak in background process
Description: After running for 24+ hours, the background worker process
gradually consumes more memory until it hits the 2GB limit and crashes.
Restarting temporarily fixes it. Server logs show unclosed database
connections accumulating.
        """,
        "severity": "high",
        "component": "backend",
        "required_info": ["memory leak", "long-running", "resource leak", "database"],
        "wrong_severity": ["trivial"]
    },
]


def create_bug_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected analysis mapping."""
    samples = []
    expected_analysis = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Analyze this bug report:\n\n{item['report']}\n\n"
        question += "Provide: severity, component, required information, and reproducibility."

        samples.append(Sample(
            question=question,
            ground_truth=f"Severity: {item['severity']}, Component: {item['component']}"
        ))

        expected_analysis[item['id']] = {
            "severity": item['severity'],
            "component": item['component'],
            "required_info": item['required_info'],
            "wrong_severity": item.get('wrong_severity', [])
        }

    return samples, expected_analysis


def main():
    print("🐛 Bug Report Analyzer Agent with ACE Learning\n")
    print("=" * 70)

    # Check if Ollama is available
    try:
        from ace.llm_providers import LiteLLMClient
        test_client = LiteLLMClient(model="ollama/llama3.1:8b", max_tokens=10)
        test_client.complete("test")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("\nSetup:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull model: ollama pull llama3.1:8b")
        return 1

    playbook_path = Path(__file__).parent / "bug_report_playbook.json"

    # Create agent
    print("\n🤖 Creating Bug Report Analyzer Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=1536,
        temperature=0.3,
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, expected_analysis = create_bug_samples()
    environment = BugReportEnvironment(expected_analysis)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_bug = """
Title: Dashboard takes 30 seconds to load
Description: Our main dashboard page is taking 30 seconds to load when it
used to be instant. This started yesterday after the database migration.
Affects all users. Looking at the network tab, the /api/dashboard endpoint
is timing out frequently.
    """
    print(f"Bug report to analyze:\n{test_bug}")

    analysis_before = agent.ask(f"Analyze this bug report:\n{test_bug}")
    print(f"\n🔍 Analysis:\n{analysis_before}\n")

    # Train the agent
    print("\n🎓 Training on bug report analysis patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully analyzed {successful}/{len(results)} bug reports")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} bug analysis strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Bug report to analyze:\n{test_bug}")

    analysis_after = agent.ask(f"Analyze this bug report:\n{test_bug}")
    print(f"\n🔍 Analysis:\n{analysis_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Incomplete Bug Report",
            "report": """
Title: It doesn't work
Description: When I click the thing, nothing happens. Fix this please.
            """
        },
        {
            "name": "Security Vulnerability",
            "report": """
Title: SQL injection in search endpoint
Description: The search parameter in /api/search is not properly sanitized.
I was able to execute arbitrary SQL by passing: ' OR '1'='1
This exposes our entire database to unauthorized access.
POC included: [sanitized for security]
            """
        },
        {
            "name": "Feature vs Bug",
            "report": """
Title: Can't export data to PDF
Description: I tried to export my report to PDF format but there's no
button for that. Only see CSV and Excel export options. Really need
PDF export for my workflow.
            """
        },
        {
            "name": "Race Condition",
            "report": """
Title: Concurrent edits cause data corruption
Description: When two users edit the same document simultaneously, changes
from one user get lost or corrupted. Doesn't happen every time, maybe 20%
of cases. Found data inconsistencies in database after reproduced scenario.
Critical for our multi-user editing feature.
            """
        },
        {
            "name": "Mobile-Specific Issue",
            "report": """
Title: Login button not working on iPhone
Description: On iPhone Safari (iOS 17), the login button appears but
nothing happens when tapped. Works fine on Android and desktop browsers.
Console shows "touchend event not recognized" error.
Device: iPhone 14 Pro, iOS 17.1, Safari
            """
        },
    ]

    for test in test_cases:
        print(f"\n📋 Test Case: {test['name']}")
        print(f"Bug Report:\n{test['report']}")
        analysis = agent.ask(f"Analyze this bug report:\n{test['report']}")
        print(f"\n🔍 Analysis:\n{analysis}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved bug analysis knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on bug report analysis!")
    print("📖 Load this playbook for consistent bug triage and classification")

    return 0


if __name__ == "__main__":
    sys.exit(main())
