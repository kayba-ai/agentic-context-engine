#!/usr/bin/env python3
"""
Test Case Generator Agent - Generates unit tests from code.

This agent learns testing patterns, edge cases, and team conventions
to generate comprehensive test cases for code.

Use Cases:
- Generate tests for legacy code
- Suggest missing test scenarios
- Learn project-specific test patterns
- Improve test quality over time

Model Recommendation: ollama/qwen2.5:7b or ollama/codellama:13b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class TestGenerationEnvironment(TaskEnvironment):
    """Evaluates test generation quality based on coverage criteria."""

    def __init__(self, expected_tests: dict[str, dict]):
        """
        Args:
            expected_tests: Dict mapping code IDs to expected test patterns
        """
        self.expected_tests = expected_tests

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if generated tests cover important scenarios."""
        code_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_tests.get(code_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation criteria available",
                ground_truth="Valid test"
            )

        test_scenarios = expected.get("scenarios", [])
        edge_cases = expected.get("edge_cases", [])
        patterns = expected.get("patterns", [])

        # Check test coverage
        has_scenarios = sum(1 for s in test_scenarios if s.lower() in answer.lower())
        has_edge_cases = sum(1 for e in edge_cases if e.lower() in answer.lower())
        has_patterns = sum(1 for p in patterns if p.lower() in answer.lower())

        total_expected = len(test_scenarios) + len(edge_cases)
        total_found = has_scenarios + has_edge_cases

        success = (
            has_scenarios >= len(test_scenarios) * 0.6 and
            has_edge_cases >= len(edge_cases) * 0.5 and
            has_patterns >= len(patterns) * 0.5
        )

        feedback = f"Covers {has_scenarios}/{len(test_scenarios)} main scenarios, "
        feedback += f"{has_edge_cases}/{len(edge_cases)} edge cases, "
        feedback += f"{has_patterns}/{len(patterns)} testing patterns. "

        if success:
            feedback += "Good test coverage!"
        else:
            missing = [s for s in test_scenarios if s.lower() not in answer.lower()]
            if missing:
                feedback += f"Missing: {', '.join(missing[:2])}"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Should test: {', '.join(test_scenarios)}"
        )


# Training data: code samples with expected test patterns
TRAINING_DATA = [
    {
        "id": "test_1",
        "code": """
def calculate_discount(price: float, discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
        """,
        "scenarios": ["valid discount", "boundary values", "exception handling"],
        "edge_cases": ["0% discount", "100% discount", "negative price", "zero price"],
        "patterns": ["pytest", "assert", "raises"]
    },
    {
        "id": "test_2",
        "code": """
def merge_dicts(dict1: dict, dict2: dict) -> dict:
    result = dict1.copy()
    result.update(dict2)
    return result
        """,
        "scenarios": ["basic merge", "overlapping keys", "empty dicts"],
        "edge_cases": ["both empty", "one empty", "dict2 overwrites dict1"],
        "patterns": ["assert equal", "multiple test cases"]
    },
    {
        "id": "test_3",
        "code": """
def fetch_user(user_id: int) -> dict:
    response = requests.get(f"/api/users/{user_id}")
    response.raise_for_status()
    return response.json()
        """,
        "scenarios": ["successful fetch", "error handling", "mocking"],
        "edge_cases": ["404 error", "500 error", "network timeout", "invalid JSON"],
        "patterns": ["mock", "patch", "requests"]
    },
    {
        "id": "test_4",
        "code": """
def is_palindrome(text: str) -> bool:
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]
        """,
        "scenarios": ["valid palindrome", "not palindrome", "case insensitive"],
        "edge_cases": ["empty string", "single char", "spaces", "punctuation"],
        "patterns": ["parametrize", "test data"]
    },
    {
        "id": "test_5",
        "code": """
class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item, quantity=1):
        self.items.append({"item": item, "quantity": quantity})

    def total(self):
        return sum(item["item"]["price"] * item["quantity"] for item in self.items)
        """,
        "scenarios": ["add items", "calculate total", "empty cart"],
        "edge_cases": ["multiple quantities", "zero quantity", "missing price"],
        "patterns": ["fixture", "setup", "teardown"]
    },
    {
        "id": "test_6",
        "code": """
async def process_batch(items: list) -> list:
    results = []
    for item in items:
        result = await process_item(item)
        results.append(result)
    return results
        """,
        "scenarios": ["successful processing", "async testing", "batch operations"],
        "edge_cases": ["empty list", "single item", "processing failure"],
        "patterns": ["pytest-asyncio", "async def", "await"]
    },
]


def create_test_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected tests mapping."""
    samples = []
    expected_tests = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Generate pytest unit tests for this code:\n\n{item['code']}"

        samples.append(Sample(
            question=question,
            ground_truth=f"Test scenarios: {', '.join(item['scenarios'])}"
        ))

        expected_tests[item['id']] = {
            "scenarios": item['scenarios'],
            "edge_cases": item['edge_cases'],
            "patterns": item['patterns']
        }

    return samples, expected_tests


def main():
    print("🧪 Test Case Generator Agent with ACE Learning\n")
    print("=" * 70)

    # Check if Ollama is available
    try:
        from ace.llm_providers import LiteLLMClient
        test_client = LiteLLMClient(model="ollama/qwen2.5:7b", max_tokens=10)
        test_client.complete("test")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("\nSetup:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull model: ollama pull qwen2.5:7b  (best for code)")
        print("   Alternative: ollama pull llama3.1:8b")
        return 1

    playbook_path = Path(__file__).parent / "test_generator_playbook.json"

    # Create agent
    print("\n🤖 Creating Test Case Generator Agent...")
    agent = ACELiteLLM(
        model="ollama/qwen2.5:7b",  # Excellent for code generation
        max_tokens=2048,
        temperature=0.3,
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, expected_tests = create_test_samples()
    environment = TestGenerationEnvironment(expected_tests)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_code = '''
def validate_email(email: str) -> bool:
    if '@' not in email:
        return False
    parts = email.split('@')
    return len(parts) == 2 and parts[0] and parts[1]
'''
    print(f"Code to test:\n{test_code}")

    tests_before = agent.ask(f"Generate pytest unit tests for this code:\n{test_code}")
    print(f"\n🧪 Generated Tests:\n{tests_before}\n")

    # Train the agent
    print("\n🎓 Training on test generation patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully generated tests for {successful}/{len(results)} code samples")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Ensure you have qwen2.5:7b or llama3.1:8b installed")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} testing strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Code to test:\n{test_code}")

    tests_after = agent.ask(f"Generate pytest unit tests for this code:\n{test_code}")
    print(f"\n🧪 Generated Tests:\n{tests_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Date Formatting Function",
            "code": """
from datetime import datetime

def format_date(date: datetime, format_type: str = "short") -> str:
    if format_type == "short":
        return date.strftime("%Y-%m-%d")
    elif format_type == "long":
        return date.strftime("%B %d, %Y")
    else:
        raise ValueError(f"Unknown format: {format_type}")
            """
        },
        {
            "name": "File Size Formatter",
            "code": """
def format_bytes(bytes_size: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_size)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"
            """
        },
        {
            "name": "API Rate Limiter",
            "code": """
from time import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def allow_request(self) -> bool:
        now = time()
        self.requests = [t for t in self.requests if now - t < self.window_seconds]

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
            """
        }
    ]

    for test in test_cases:
        print(f"\n📋 Test Case: {test['name']}")
        print(f"Code:\n{test['code']}")
        tests = agent.ask(f"Generate pytest unit tests for this code:\n{test['code']}")
        print(f"\n🧪 Generated Tests:\n{tests}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved test generation knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on testing patterns!")
    print("📖 Load this playbook for consistent test generation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
