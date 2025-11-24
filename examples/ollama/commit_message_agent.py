#!/usr/bin/env python3
"""
Git Commit Message Generator Agent - Generates conventional commit messages.

This agent learns project commit conventions, scope naming patterns, and
generates clear, consistent commit messages from code diffs.

Use Cases:
- Enforce commit standards
- Generate release notes
- Improve commit history quality
- Learn from past commits

Model Recommendation: ollama/llama3.1:8b or ollama/qwen2.5:7b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class CommitMessageEnvironment(TaskEnvironment):
    """Evaluates commit message quality against conventions."""

    def __init__(self, expected_commits: dict[str, dict]):
        """
        Args:
            expected_commits: Dict mapping change IDs to expected commit format
        """
        self.expected_commits = expected_commits

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if commit message follows conventions."""
        change_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_commits.get(change_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation criteria available",
                ground_truth="Valid commit message"
            )

        commit_type = expected.get("type", "").lower()
        scope = expected.get("scope", "").lower()
        keywords = expected.get("keywords", [])
        anti_patterns = expected.get("anti_patterns", [])

        answer_lower = answer.lower()

        # Check conventional commit format
        has_type = commit_type in answer_lower
        has_scope = scope in answer_lower if scope else True
        has_keywords = sum(1 for kw in keywords if kw.lower() in answer_lower)
        has_anti_patterns = any(ap.lower() in answer_lower for ap in anti_patterns)

        # Check format: type(scope): description
        has_colon = ":" in answer
        first_line = answer.split("\n")[0] if answer else ""
        proper_format = has_colon and len(first_line) < 72  # 72 char limit

        success = (
            has_type and
            has_scope and
            has_keywords >= len(keywords) * 0.5 and
            not has_anti_patterns and
            proper_format
        )

        feedback = ""
        if has_type:
            feedback += f"✓ Correct type ({commit_type}). "
        else:
            feedback += f"✗ Wrong type (expected {commit_type}). "

        if not proper_format:
            feedback += "✗ Doesn't follow format (type(scope): description). "

        if has_anti_patterns:
            feedback += "✗ Contains anti-patterns (vague language). "
            success = False

        if success:
            feedback += "Good commit message!"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Type: {commit_type}, Scope: {scope}"
        )


# Training data: code changes with expected commit messages
TRAINING_DATA = [
    {
        "id": "commit_1",
        "diff": """
diff --git a/src/auth/login.js b/src/auth/login.js
@@ -10,7 +10,7 @@ function validatePassword(password) {
-  return password.length >= 6;
+  return password.length >= 8 && /[0-9]/.test(password);
 }
        """,
        "type": "feat",
        "scope": "auth",
        "keywords": ["password", "validation", "security"],
        "anti_patterns": ["updated", "fixed", "changed"]
    },
    {
        "id": "commit_2",
        "diff": """
diff --git a/src/api/users.js b/src/api/users.js
@@ -45,3 +45,3 @@ async function getUser(id) {
-  return db.query(`SELECT * FROM users WHERE id = ${id}`);
+  return db.query('SELECT * FROM users WHERE id = ?', [id]);
 }
        """,
        "type": "fix",
        "scope": "api",
        "keywords": ["sql injection", "security", "parameterized"],
        "anti_patterns": ["updated", "changes"]
    },
    {
        "id": "commit_3",
        "diff": """
diff --git a/README.md b/README.md
@@ -5,0 +6,3 @@
+## Installation
+
+Run `npm install` to install dependencies.
        """,
        "type": "docs",
        "scope": "readme",
        "keywords": ["installation", "documentation"],
        "anti_patterns": ["added", "updated"]
    },
    {
        "id": "commit_4",
        "diff": """
diff --git a/src/utils/format.js b/src/utils/format.js
@@ -12,8 +12,5 @@ function formatDate(date) {
-  const month = date.getMonth() + 1;
-  const day = date.getDate();
-  const year = date.getFullYear();
-  return `${month}/${day}/${year}`;
+  return date.toLocaleDateString('en-US');
 }
        """,
        "type": "refactor",
        "scope": "utils",
        "keywords": ["date formatting", "simplify", "built-in"],
        "anti_patterns": ["fixed", "changed"]
    },
    {
        "id": "commit_5",
        "diff": """
diff --git a/package.json b/package.json
@@ -10,3 +10,3 @@
-    "react": "^17.0.0",
+    "react": "^18.0.0",
        """,
        "type": "chore",
        "scope": "deps",
        "keywords": ["upgrade", "react", "dependency"],
        "anti_patterns": ["updated package"]
    },
    {
        "id": "commit_6",
        "diff": """
diff --git a/src/components/Button.js b/src/components/Button.js
@@ -0,0 +1,15 @@
+export function Button({ onClick, children }) {
+  return (
+    <button onClick={onClick} className="btn">
+      {children}
+    </button>
+  );
+}
        """,
        "type": "feat",
        "scope": "ui",
        "keywords": ["button component", "react"],
        "anti_patterns": ["added file", "new file"]
    },
]


def create_commit_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected commit messages mapping."""
    samples = []
    expected_commits = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Generate a conventional commit message for this change:\n\n{item['diff']}"

        samples.append(Sample(
            question=question,
            ground_truth=f"Type: {item['type']}, Scope: {item['scope']}"
        ))

        expected_commits[item['id']] = {
            "type": item['type'],
            "scope": item.get('scope', ''),
            "keywords": item['keywords'],
            "anti_patterns": item.get('anti_patterns', [])
        }

    return samples, expected_commits


def main():
    print("📝 Git Commit Message Generator Agent with ACE Learning\n")
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

    playbook_path = Path(__file__).parent / "commit_message_playbook.json"

    # Create agent
    print("\n🤖 Creating Commit Message Generator Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=512,
        temperature=0.3,
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    print("\n📋 Conventional Commit Format: type(scope): description")
    print("Types: feat, fix, docs, style, refactor, test, chore")

    # Prepare training data
    samples, expected_commits = create_commit_samples()
    environment = CommitMessageEnvironment(expected_commits)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_diff = """
diff --git a/tests/test_auth.py b/tests/test_auth.py
@@ -0,0 +1,8 @@
+def test_login_with_valid_credentials():
+    result = login("user@test.com", "password123")
+    assert result.success is True
+    assert result.token is not None
    """
    print(f"Code diff:\n{test_diff}")

    commit_before = agent.ask(f"Generate a conventional commit message:\n{test_diff}")
    print(f"\n💬 Generated Commit:\n{commit_before}\n")

    # Train the agent
    print("\n🎓 Training on commit message patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully generated {successful}/{len(results)} commit messages")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} commit message strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Code diff:\n{test_diff}")

    commit_after = agent.ask(f"Generate a conventional commit message:\n{test_diff}")
    print(f"\n💬 Generated Commit:\n{commit_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Performance Optimization",
            "diff": """
diff --git a/src/data/cache.js b/src/data/cache.js
@@ -15,7 +15,10 @@ function getData(key) {
-  return fetch(`/api/data/${key}`).then(r => r.json());
+  const cached = localStorage.getItem(key);
+  if (cached) return Promise.resolve(JSON.parse(cached));
+  return fetch(`/api/data/${key}`).then(r => {
+    localStorage.setItem(key, JSON.stringify(r));
+    return r.json();
+  });
 }
            """
        },
        {
            "name": "Breaking Change",
            "diff": """
diff --git a/src/api/client.js b/src/api/client.js
@@ -8,7 +8,7 @@ export class APIClient {
-  constructor(apiKey) {
+  constructor(config) {
+    this.apiKey = config.apiKey;
+    this.baseUrl = config.baseUrl || 'https://api.example.com';
   }
            """
        },
        {
            "name": "Bug Fix with Issue Reference",
            "diff": """
diff --git a/src/cart/checkout.js b/src/cart/checkout.js
@@ -22,3 +22,4 @@ function calculateTotal(items) {
-  return items.reduce((sum, item) => sum + item.price, 0);
+  return items.reduce((sum, item) =>
+    sum + (item.price * item.quantity), 0);
 }
            """
        },
        {
            "name": "Test Addition",
            "diff": """
diff --git a/tests/integration/api_test.py b/tests/integration/api_test.py
@@ -0,0 +1,12 @@
+@pytest.mark.integration
+def test_api_rate_limiting():
+    for i in range(150):
+        response = client.get('/api/data')
+    assert response.status_code == 429
+    assert 'rate limit' in response.json()['error'].lower()
            """
        },
        {
            "name": "Configuration Change",
            "diff": """
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
@@ -10,3 +10,4 @@ jobs:
       - uses: actions/checkout@v3
+      - name: Cache dependencies
+        uses: actions/cache@v3
            """
        },
    ]

    for test in test_cases:
        print(f"\n📋 Test Case: {test['name']}")
        print(f"Diff:\n{test['diff']}")
        commit = agent.ask(f"Generate a conventional commit message:\n{test['diff']}")
        print(f"\n💬 Generated Commit:\n{commit}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved commit message knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on commit message conventions!")
    print("📖 Load this playbook for consistent commit messages")

    return 0


if __name__ == "__main__":
    sys.exit(main())
