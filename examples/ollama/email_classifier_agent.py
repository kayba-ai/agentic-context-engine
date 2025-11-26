#!/usr/bin/env python3
"""
Email/Ticket Classifier Agent - Categorizes and routes support tickets.

This agent learns routing rules, priority classification, and customer
intent recognition to automate ticket triage and routing.

Use Cases:
- Customer support automation
- Bug report triage
- Feature request categorization
- Email inbox organization

Model Recommendation: ollama/llama3.1:8b or ollama/mistral:7b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class EmailClassificationEnvironment(TaskEnvironment):
    """Evaluates email classification accuracy."""

    def __init__(self, expected_classifications: dict[str, dict]):
        """
        Args:
            expected_classifications: Dict mapping email IDs to expected categories
        """
        self.expected_classifications = expected_classifications

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if classification matches expected category and priority."""
        email_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_classifications.get(email_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation criteria available",
                ground_truth="Valid classification"
            )

        expected_category = expected.get("category", "").lower()
        expected_priority = expected.get("priority", "").lower()
        expected_department = expected.get("department", "").lower()
        wrong_categories = [c.lower() for c in expected.get("wrong", [])]

        answer_lower = answer.lower()

        # Check classification accuracy
        has_category = expected_category in answer_lower
        has_priority = expected_priority in answer_lower
        has_department = expected_department in answer_lower if expected_department else True
        has_wrong = any(wc in answer_lower for wc in wrong_categories)

        success = has_category and has_priority and has_department and not has_wrong

        feedback = ""
        if has_category:
            feedback += f"✓ Correct category ({expected_category}). "
        else:
            feedback += f"✗ Wrong category (expected {expected_category}). "

        if has_priority:
            feedback += f"✓ Correct priority ({expected_priority}). "
        else:
            feedback += f"✗ Wrong priority (expected {expected_priority}). "

        if has_wrong:
            feedback += "✗ Contains incorrect classification! "
            success = False

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Category: {expected_category}, Priority: {expected_priority}"
        )


# Training data: support tickets with classifications
TRAINING_DATA = [
    {
        "id": "ticket_1",
        "content": """
Subject: Cannot login to account
Body: Hi, I've been trying to log in for the past hour but keep getting
"Invalid credentials" error even though I'm sure my password is correct.
This is blocking my work. Please help ASAP!
        """,
        "category": "authentication",
        "priority": "high",
        "department": "technical support",
        "wrong": ["billing", "feature request"]
    },
    {
        "id": "ticket_2",
        "content": """
Subject: Feature Idea: Dark Mode
Body: Hey team! Love your product. Would be cool to have a dark mode
option for the dashboard. Not urgent, just a nice-to-have for those
late-night work sessions :)
        """,
        "category": "feature request",
        "priority": "low",
        "department": "product",
        "wrong": ["bug", "urgent"]
    },
    {
        "id": "ticket_3",
        "content": """
Subject: URGENT: Payment failed but subscription canceled
Body: My credit card payment failed yesterday and now my entire account
is suspended! I have 50 users who can't access the system. This is
costing us money every hour. Please restore access immediately!
        """,
        "category": "billing",
        "priority": "critical",
        "department": "billing",
        "wrong": ["feature", "low priority"]
    },
    {
        "id": "ticket_4",
        "content": """
Subject: Dashboard showing wrong numbers
Body: The analytics dashboard is displaying incorrect revenue figures.
The numbers don't match our internal reports. This started happening
after yesterday's update. Can you investigate?
        """,
        "category": "bug",
        "priority": "high",
        "department": "technical support",
        "wrong": ["feature request", "low"]
    },
    {
        "id": "ticket_5",
        "content": """
Subject: Question about API rate limits
Body: Hi, I'm building an integration and wanted to understand the API
rate limits. The documentation mentions 100 req/min but I'm seeing
throttling at 80. Can you clarify?
        """,
        "category": "question",
        "priority": "medium",
        "department": "developer support",
        "wrong": ["bug", "critical"]
    },
    {
        "id": "ticket_6",
        "content": """
Subject: Security concern - possible data breach
Body: I noticed unusual activity in our account logs. Multiple failed
login attempts from unfamiliar IPs, followed by successful logins.
Need immediate security review of our account.
        """,
        "category": "security",
        "priority": "critical",
        "department": "security team",
        "wrong": ["low", "feature"]
    },
    {
        "id": "ticket_7",
        "content": """
Subject: Typo in documentation
Body: Found a small typo in the "Getting Started" guide. Page 3,
paragraph 2 says "setp" instead of "step". Just wanted to let you know.
        """,
        "category": "documentation",
        "priority": "low",
        "department": "documentation",
        "wrong": ["bug", "urgent", "high"]
    },
]


def create_classification_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected classifications mapping."""
    samples = []
    expected_classifications = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Classify this support ticket:\n\n{item['content']}\n\n"
        question += "Provide: category, priority, and suggested department."

        samples.append(Sample(
            question=question,
            ground_truth=f"Category: {item['category']}, Priority: {item['priority']}"
        ))

        expected_classifications[item['id']] = {
            "category": item['category'],
            "priority": item['priority'],
            "department": item.get('department', ''),
            "wrong": item.get('wrong', [])
        }

    return samples, expected_classifications


def main():
    print("📧 Email/Ticket Classifier Agent with ACE Learning\n")
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

    playbook_path = Path(__file__).parent / "email_classifier_playbook.json"

    # Create agent
    print("\n🤖 Creating Email/Ticket Classifier Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=1024,
        temperature=0.2,  # Low temp for consistent classification
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, expected_classifications = create_classification_samples()
    environment = EmailClassificationEnvironment(expected_classifications)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_ticket = """
Subject: Refund request - charged twice
Body: I noticed I was charged twice for my monthly subscription - once on
Nov 20 and again on Nov 21. My bank shows both charges are pending. Can
you please refund one of them?
    """
    print(f"Ticket to classify:\n{test_ticket}")

    classification_before = agent.ask(
        f"Classify this support ticket (category, priority, department):\n{test_ticket}"
    )
    print(f"\n🏷️  Classification:\n{classification_before}\n")

    # Train the agent
    print("\n🎓 Training on ticket classification patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully classified {successful}/{len(results)} tickets")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} classification strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Ticket to classify:\n{test_ticket}")

    classification_after = agent.ask(
        f"Classify this support ticket (category, priority, department):\n{test_ticket}"
    )
    print(f"\n🏷️  Classification:\n{classification_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Performance Issue",
            "ticket": """
Subject: App extremely slow since update
Body: Ever since the v2.5 update rolled out last week, our entire team
has been experiencing major slowdowns. Pages that used to load instantly
now take 10-15 seconds. This is impacting our productivity significantly.
            """
        },
        {
            "name": "Integration Request",
            "ticket": """
Subject: Slack integration?
Body: Do you have plans to integrate with Slack? Would love to get
notifications in our team channel when certain events happen. This would
be super useful for our workflow!
            """
        },
        {
            "name": "Account Lockout",
            "ticket": """
Subject: LOCKED OUT - need immediate access
Body: My account got locked after trying different passwords. I have a
client presentation in 30 minutes and need access to our dashboard
urgently. Please unlock my account ASAP!
            """
        },
        {
            "name": "Spam Report",
            "ticket": """
Subject: Unsubscribe me NOW
Body: I never signed up for this. Stop sending me emails or I'll report
you. UNSUBSCRIBE ME IMMEDIATELY!!!
            """
        },
        {
            "name": "Data Export Request",
            "ticket": """
Subject: GDPR data export request
Body: Under GDPR regulations, I'm requesting a complete export of all
my personal data stored in your system. Please provide this within the
legally required 30-day timeframe.
            """
        },
    ]

    for test in test_cases:
        print(f"\n📋 Test Case: {test['name']}")
        print(f"Ticket:\n{test['ticket']}")
        classification = agent.ask(
            f"Classify this support ticket (category, priority, department):\n{test['ticket']}"
        )
        print(f"\n🏷️  Classification:\n{classification}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved classification knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on ticket classification!")
    print("📖 Load this playbook for consistent routing and prioritization")

    return 0


if __name__ == "__main__":
    sys.exit(main())
