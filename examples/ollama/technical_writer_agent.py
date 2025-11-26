#!/usr/bin/env python3
"""
Technical Writer Agent - Converts technical content between formats.

This agent learns to write clear, consistent technical documentation by
understanding patterns in code-to-docs conversion, API documentation,
and technical writing best practices.

Use Cases:
- Generate README files from code
- Convert API specs to tutorials
- Create release notes from commits
- Learn company documentation style

Model Recommendation: ollama/llama3.1:8b or ollama/mistral:7b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class TechnicalWritingEnvironment(TaskEnvironment):
    """Evaluates technical writing quality based on required elements."""

    def __init__(self, expected_elements: dict[str, dict]):
        """
        Args:
            expected_elements: Dict mapping doc IDs to expected content elements
        """
        self.expected_elements = expected_elements

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if documentation includes required elements and good practices."""
        doc_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_elements.get(doc_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation criteria available",
                ground_truth="Valid documentation"
            )

        required = expected.get("required", [])
        best_practices = expected.get("best_practices", [])
        anti_patterns = expected.get("anti_patterns", [])

        # Check for required elements
        has_required = sum(1 for elem in required if elem.lower() in answer.lower())
        has_best_practices = sum(1 for bp in best_practices if bp.lower() in answer.lower())
        has_anti_patterns = any(ap.lower() in answer.lower() for ap in anti_patterns)

        # Success if most required elements present and follows best practices
        success = (
            has_required >= len(required) * 0.7 and
            has_best_practices >= len(best_practices) * 0.5 and
            not has_anti_patterns
        )

        feedback = f"Has {has_required}/{len(required)} required elements. "
        feedback += f"Follows {has_best_practices}/{len(best_practices)} best practices. "

        if has_anti_patterns:
            feedback += "Contains anti-patterns (jargon, assumptions, etc.). "
            success = False
        elif success:
            feedback += "Good documentation!"
        else:
            missing = [r for r in required if r.lower() not in answer.lower()]
            feedback += f"Missing: {', '.join(missing[:3])}"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Required: {', '.join(required)}"
        )


# Training data: technical content to documentation
TRAINING_DATA = [
    {
        "id": "doc_1",
        "task": "Create API documentation",
        "input": """
Function: authenticate(username, password)
Returns: {token: string, expires_at: timestamp}
Throws: AuthenticationError if credentials invalid
Rate limit: 10 requests per minute
        """,
        "required": ["authentication", "parameters", "returns", "errors", "rate limit", "example"],
        "best_practices": ["code example", "curl", "response format"],
        "anti_patterns": ["simply", "just", "easy"]
    },
    {
        "id": "doc_2",
        "task": "Write README for Python package",
        "input": """
Package: data-processor
Purpose: Process CSV files and generate analytics
Main function: process_csv(filepath, options)
Dependencies: pandas, numpy
Python: 3.8+
        """,
        "required": ["installation", "usage", "requirements", "example"],
        "best_practices": ["quick start", "pip install", "code block"],
        "anti_patterns": ["obviously", "simply run"]
    },
    {
        "id": "doc_3",
        "task": "Document error message",
        "input": """
Error: "DATABASE_CONNECTION_FAILED"
Cause: Cannot connect to PostgreSQL database
Common reasons: Wrong credentials, database not running, network issue
Solution: Check connection string and database status
        """,
        "required": ["error description", "cause", "solution", "example"],
        "best_practices": ["troubleshooting steps", "how to verify", "common mistakes"],
        "anti_patterns": ["should be obvious", "just check"]
    },
    {
        "id": "doc_4",
        "task": "Create changelog entry",
        "input": """
Version: 2.5.0
Changes: Added user authentication, fixed memory leak, improved performance
Breaking: Authentication now required for all endpoints
Migration: Add AUTH_TOKEN to environment variables
        """,
        "required": ["version", "added", "fixed", "breaking changes", "migration"],
        "best_practices": ["upgrade instructions", "impact", "examples"],
        "anti_patterns": ["minor changes", "various improvements"]
    },
    {
        "id": "doc_5",
        "task": "Document configuration option",
        "input": """
Config: MAX_UPLOAD_SIZE
Type: integer (bytes)
Default: 10485760 (10MB)
Purpose: Maximum file size for uploads
Valid range: 1MB to 100MB
        """,
        "required": ["name", "type", "default", "description", "example"],
        "best_practices": ["valid values", "when to change", "impact"],
        "anti_patterns": ["simply set", "just change"]
    },
    {
        "id": "doc_6",
        "task": "Write tutorial introduction",
        "input": """
Tutorial: Building a REST API with FastAPI
Goal: Create CRUD endpoints for a todo application
Prerequisites: Python 3.8+, basic HTTP knowledge
Duration: 30 minutes
        """,
        "required": ["what you'll learn", "prerequisites", "time", "outcome"],
        "best_practices": ["learning goals", "what you'll build", "next steps"],
        "anti_patterns": ["easy", "simple", "straightforward"]
    },
]


def create_writing_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected elements mapping."""
    samples = []
    expected_elements = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: {item['task']}\n\nInput:\n{item['input']}\n\nWrite clear documentation."

        samples.append(Sample(
            question=question,
            ground_truth=f"Should include: {', '.join(item['required'])}"
        ))

        expected_elements[item['id']] = {
            "required": item['required'],
            "best_practices": item['best_practices'],
            "anti_patterns": item.get('anti_patterns', [])
        }

    return samples, expected_elements


def main():
    print("📝 Technical Writer Agent with ACE Learning\n")
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

    playbook_path = Path(__file__).parent / "technical_writer_playbook.json"

    # Create agent
    print("\n🤖 Creating Technical Writer Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=2048,
        temperature=0.5,  # Some creativity for writing
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, expected_elements = create_writing_samples()
    environment = TechnicalWritingEnvironment(expected_elements)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_task = """
Create documentation for this function:

def send_email(to: str, subject: str, body: str, attachments: list = None):
    \"\"\"Send email via SMTP.\"\"\"
    # Implementation details...
    return {"status": "sent", "message_id": "abc123"}
    """
    print(test_task)

    doc_before = agent.ask(f"Write API documentation:\n{test_task}")
    print(f"\n📄 Documentation:\n{doc_before}\n")

    # Train the agent
    print("\n🎓 Training on technical writing patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully wrote {successful}/{len(results)} documentation pieces")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} writing strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(test_task)

    doc_after = agent.ask(f"Write API documentation:\n{test_task}")
    print(f"\n📄 Documentation:\n{doc_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "CLI Tool Documentation",
            "task": """
Write documentation for this CLI command:

Command: deploy --env <environment> --version <version> [--force]
Purpose: Deploy application to specified environment
Options:
  --env: production, staging, or development
  --version: Version tag to deploy
  --force: Skip confirmation prompts
Examples: deploy --env staging --version v2.1.0
            """
        },
        {
            "name": "Webhook Documentation",
            "task": """
Document this webhook:

Event: order.completed
Payload: {order_id, user_id, total, items[]}
Delivery: POST request to configured URL
Retries: 3 attempts with exponential backoff
Security: HMAC signature in X-Webhook-Signature header
            """
        },
        {
            "name": "Migration Guide",
            "task": """
Write migration guide:

From: v1.x API using /api/users/{id}
To: v2.x API using /api/v2/users/{id}
Changes: Now requires authentication, returns different format
Old: {id, name, email}
New: {user: {id, name, email, created_at}}
Timeline: v1 deprecated on 2025-01-01
            """
        }
    ]

    for test in test_cases:
        print(f"\n📋 Task: {test['name']}")
        print(test['task'])
        documentation = agent.ask(f"Write documentation:\n{test['task']}")
        print(f"\n📄 Documentation:\n{documentation}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved technical writing knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on technical writing best practices!")
    print("📖 Load this playbook for consistent documentation style")

    return 0


if __name__ == "__main__":
    sys.exit(main())
