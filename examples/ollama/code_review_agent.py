#!/usr/bin/env python3
"""
Code Review Agent - Reviews code for bugs, security issues, and best practices.

This agent learns from feedback to improve code review quality over time.
It can detect common issues, suggest improvements, and learn team-specific patterns.

Use Cases:
- Pre-commit code quality checks
- Security vulnerability scanning
- Style and best practice enforcement
- Learning team-specific coding patterns

Model Recommendation: ollama/qwen2.5:7b or ollama/llama3.1:8b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class CodeReviewEnvironment(TaskEnvironment):
    """Evaluates code review quality based on known issues."""

    def __init__(self, known_issues: dict[str, list[str]]):
        """
        Args:
            known_issues: Dict mapping code samples to list of issues
        """
        self.known_issues = known_issues

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if review identified known issues."""
        code_id = sample.question.split("```")[0].strip()
        expected_issues = self.known_issues.get(code_id, [])

        if not expected_issues:
            return EnvironmentResult(
                success=True,
                feedback="No known issues to check",
                ground_truth="Clean code"
            )

        # Check if review mentions key issues
        found_issues = sum(1 for issue in expected_issues if issue.lower() in answer.lower())
        success = found_issues >= len(expected_issues) * 0.7  # 70% threshold

        feedback = f"Found {found_issues}/{len(expected_issues)} known issues. "
        if success:
            feedback += "Good review!"
        else:
            feedback += f"Missed: {', '.join([i for i in expected_issues if i.lower() not in answer.lower()])}"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Should identify: {', '.join(expected_issues)}"
        )


# Training data: code samples with known issues
TRAINING_DATA = [
    {
        "id": "python_sql_injection",
        "code": '''
def get_user(username):
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    return db.execute(query)
''',
        "issues": ["SQL injection", "parameterized query", "security vulnerability"]
    },
    {
        "id": "python_resource_leak",
        "code": '''
def read_config():
    file = open("config.txt")
    data = file.read()
    return data
''',
        "issues": ["resource leak", "context manager", "file not closed"]
    },
    {
        "id": "python_exception_handling",
        "code": '''
try:
    result = risky_operation()
except:
    pass
''',
        "issues": ["bare except", "silent failure", "error handling"]
    },
    {
        "id": "python_mutable_default",
        "code": '''
def add_item(item, items=[]):
    items.append(item)
    return items
''',
        "issues": ["mutable default", "default argument", "shared state"]
    },
    {
        "id": "javascript_xss",
        "code": '''
function displayUserInput(input) {
    document.getElementById("output").innerHTML = input;
}
''',
        "issues": ["XSS vulnerability", "innerHTML", "sanitize input"]
    },
    {
        "id": "python_race_condition",
        "code": '''
if os.path.exists(filename):
    with open(filename) as f:
        data = f.read()
''',
        "issues": ["race condition", "TOCTOU", "file check"]
    },
]


def create_code_samples() -> tuple[list[Sample], dict]:
    """Create training samples and issue mapping."""
    samples = []
    known_issues = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}\n```python\n{item['code'].strip()}\n```\nReview this code."
        samples.append(Sample(
            question=question,
            ground_truth=f"Issues: {', '.join(item['issues'])}"
        ))
        known_issues[item['id']] = item['issues']

    return samples, known_issues


def main():
    print("🔍 Code Review Agent with ACE Learning\n")
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
        print("3. Verify: ollama list")
        return 1

    playbook_path = Path(__file__).parent / "code_review_playbook.json"

    # Create agent
    print("\n🤖 Creating Code Review Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",  # Good for code understanding
        max_tokens=1024,
        temperature=0.2,  # Lower temp for more consistent reviews
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, known_issues = create_code_samples()
    environment = CodeReviewEnvironment(known_issues)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_code = '''
def execute_command(cmd):
    import os
    os.system(cmd)
'''
    print("Code to review:")
    print(test_code)

    review_before = agent.ask(
        f"Review this code for security issues:\n```python\n{test_code}\n```"
    )
    print(f"\n🔍 Review: {review_before}\n")

    # Train the agent
    print("\n🎓 Training on code review patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully reviewed {successful}/{len(results)} code samples")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try: ollama pull llama3.1:8b")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} code review strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print("Code to review:")
    print(test_code)

    review_after = agent.ask(
        f"Review this code for security issues:\n```python\n{test_code}\n```"
    )
    print(f"\n🔍 Review: {review_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Hardcoded Credentials",
            "code": '''
API_KEY = "sk-1234567890abcdef"
def connect():
    return requests.get(url, headers={"Authorization": API_KEY})
'''
        },
        {
            "name": "Type Confusion",
            "code": '''
def calculate_total(prices):
    total = 0
    for price in prices:
        total = total + price  # What if price is a string?
    return total
'''
        },
        {
            "name": "Insecure Randomness",
            "code": '''
import random
def generate_token():
    return str(random.randint(1000000, 9999999))
'''
        }
    ]

    for test in test_cases:
        print(f"\n📋 Test: {test['name']}")
        print(f"Code:\n{test['code']}")
        review = agent.ask(f"Review this code:\n```python\n{test['code']}\n```")
        print(f"🔍 Review: {review}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved code review knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on common code issues!")
    print("📖 Load this playbook in future sessions for instant expertise")

    return 0


if __name__ == "__main__":
    sys.exit(main())
