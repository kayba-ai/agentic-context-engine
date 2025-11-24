#!/usr/bin/env python3
"""
Troubleshooting Assistant Agent - Diagnoses system issues and suggests fixes.

This agent learns from resolution outcomes to improve diagnostic accuracy
and solution recommendations over time.

Use Cases:
- Log file analysis
- Error message interpretation
- System diagnostics
- Learning environment-specific issues

Model Recommendation: ollama/llama3.1:8b or ollama/mistral:7b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class TroubleshootingEnvironment(TaskEnvironment):
    """Evaluates troubleshooting quality based on correct diagnosis."""

    def __init__(self, known_solutions: dict[str, dict]):
        """
        Args:
            known_solutions: Dict mapping issue IDs to root causes and solutions
        """
        self.known_solutions = known_solutions

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if troubleshooting identified root cause and valid solution."""
        issue_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.known_solutions.get(issue_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No known solution to validate",
                ground_truth="Valid troubleshooting"
            )

        root_cause = expected.get("root_cause", "")
        keywords = expected.get("keywords", [])
        wrong_diagnosis = expected.get("wrong_diagnosis", [])

        # Check if answer identifies root cause
        has_root_cause = root_cause.lower() in answer.lower()
        has_keywords = sum(1 for kw in keywords if kw.lower() in answer.lower())
        has_wrong = any(wd.lower() in answer.lower() for wd in wrong_diagnosis)

        # Success if root cause identified and no wrong diagnosis
        success = has_root_cause and has_keywords >= len(keywords) * 0.5 and not has_wrong

        feedback = ""
        if has_root_cause:
            feedback += "✓ Identified root cause. "
        else:
            feedback += f"✗ Missed root cause: {root_cause}. "

        if has_keywords >= len(keywords) * 0.5:
            feedback += f"✓ Found {has_keywords}/{len(keywords)} key solution elements. "
        else:
            feedback += f"✗ Only found {has_keywords}/{len(keywords)} solution elements. "

        if has_wrong:
            feedback += "✗ Contains incorrect diagnosis! "

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Root cause: {root_cause}. Solution: {', '.join(keywords)}"
        )


# Training data: common issues with root causes and solutions
TRAINING_DATA = [
    {
        "id": "issue_1",
        "symptom": "Application responds with '502 Bad Gateway' error intermittently",
        "logs": """
[2024-11-24 10:15:33] ERROR: Connection timeout to backend server
[2024-11-24 10:15:45] WARNING: Retrying connection... attempt 2/3
[2024-11-24 10:15:50] ERROR: Max retries exceeded
[2024-11-24 10:16:02] INFO: Successfully connected to backend
        """,
        "root_cause": "backend server timeout",
        "keywords": ["increase timeout", "backend server", "connection pool", "health check"],
        "wrong_diagnosis": ["DNS issue", "SSL certificate"]
    },
    {
        "id": "issue_2",
        "symptom": "Database queries extremely slow, application freezing",
        "logs": """
[2024-11-24 09:30:15] SLOW QUERY: SELECT * FROM users WHERE email LIKE '%@gmail.com%' (15234ms)
[2024-11-24 09:30:45] SLOW QUERY: SELECT * FROM orders WHERE status = 'pending' (8921ms)
[2024-11-24 09:31:12] WARNING: Connection pool exhausted, waiting for available connection
        """,
        "root_cause": "missing database index",
        "keywords": ["create index", "query optimization", "LIKE pattern", "full table scan"],
        "wrong_diagnosis": ["insufficient RAM", "disk space"]
    },
    {
        "id": "issue_3",
        "symptom": "Memory usage grows continuously until application crashes",
        "logs": """
[2024-11-24 08:00:00] INFO: Memory usage: 512MB
[2024-11-24 09:00:00] WARNING: Memory usage: 1.2GB
[2024-11-24 10:00:00] WARNING: Memory usage: 2.8GB
[2024-11-24 10:30:00] ERROR: OutOfMemoryError: Java heap space
[2024-11-24 10:30:01] INFO: Unclosed connections: 3421
        """,
        "root_cause": "memory leak",
        "keywords": ["connection leak", "close resources", "finally block", "resource management"],
        "wrong_diagnosis": ["increase heap size", "garbage collection"]
    },
    {
        "id": "issue_4",
        "symptom": "Authentication fails for some users randomly",
        "logs": """
[2024-11-24 14:22:10] INFO: Login attempt for user john@example.com - SUCCESS
[2024-11-24 14:22:45] ERROR: Login attempt for user sarah@example.com - Token validation failed
[2024-11-24 14:23:15] INFO: Login attempt for user sarah@example.com - SUCCESS
[2024-11-24 14:23:30] INFO: System time: 2024-11-24 14:23:30, Server time: 2024-11-24 14:18:15
        """,
        "root_cause": "clock skew",
        "keywords": ["NTP sync", "time synchronization", "token expiration", "clock drift"],
        "wrong_diagnosis": ["password hash", "session storage"]
    },
    {
        "id": "issue_5",
        "symptom": "File upload fails with 413 error for files larger than 2MB",
        "logs": """
[2024-11-24 16:45:12] INFO: Received upload request, file size: 2.3MB
[2024-11-24 16:45:13] ERROR: 413 Request Entity Too Large
[2024-11-24 16:45:13] DEBUG: nginx client_max_body_size: 2m
        """,
        "root_cause": "nginx upload size limit",
        "keywords": ["client_max_body_size", "nginx configuration", "increase limit", "reverse proxy"],
        "wrong_diagnosis": ["disk space", "file permissions"]
    },
    {
        "id": "issue_6",
        "symptom": "API rate limiting triggers even with low traffic",
        "logs": """
[2024-11-24 12:00:00] INFO: Request from IP 192.168.1.100 - Count: 95/100
[2024-11-24 12:00:05] WARNING: Rate limit exceeded for IP 192.168.1.100
[2024-11-24 12:00:10] INFO: All requests coming through same proxy IP
[2024-11-24 12:00:15] DEBUG: X-Forwarded-For header: 10.1.1.50, 192.168.1.100
        """,
        "root_cause": "proxy IP used for rate limiting",
        "keywords": ["X-Forwarded-For", "real client IP", "proxy configuration", "rate limit key"],
        "wrong_diagnosis": ["increase rate limit", "redis issue"]
    },
]


def create_troubleshooting_samples() -> tuple[list[Sample], dict]:
    """Create training samples and known solutions mapping."""
    samples = []
    known_solutions = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Troubleshoot this issue:\n\n"
        question += f"Symptom: {item['symptom']}\n\n"
        question += f"Logs:\n{item['logs']}\n\n"
        question += "What is the root cause and how to fix it?"

        samples.append(Sample(
            question=question,
            ground_truth=f"Root cause: {item['root_cause']}. Fix: {', '.join(item['keywords'])}"
        ))

        known_solutions[item['id']] = {
            "root_cause": item['root_cause'],
            "keywords": item['keywords'],
            "wrong_diagnosis": item.get('wrong_diagnosis', [])
        }

    return samples, known_solutions


def main():
    print("🔧 Troubleshooting Assistant Agent with ACE Learning\n")
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

    playbook_path = Path(__file__).parent / "troubleshooting_playbook.json"

    # Create agent
    print("\n🤖 Creating Troubleshooting Assistant...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=1536,
        temperature=0.3,
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, known_solutions = create_troubleshooting_samples()
    environment = TroubleshootingEnvironment(known_solutions)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_issue = """
Symptom: Website loads slowly, takes 15-20 seconds
Logs:
[2024-11-24 15:00:00] INFO: DNS lookup completed in 12ms
[2024-11-24 15:00:12] WARNING: SSL handshake timeout
[2024-11-24 15:00:18] INFO: Connection established
[2024-11-24 15:00:20] INFO: Page loaded
    """
    print(test_issue)

    diagnosis_before = agent.ask(f"Troubleshoot this issue:\n{test_issue}\nWhat is the root cause and solution?")
    print(f"\n🔍 Diagnosis: {diagnosis_before}\n")

    # Train the agent
    print("\n🎓 Training on troubleshooting patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully diagnosed {successful}/{len(results)} issues")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} troubleshooting strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(test_issue)

    diagnosis_after = agent.ask(f"Troubleshoot this issue:\n{test_issue}\nWhat is the root cause and solution?")
    print(f"\n🔍 Diagnosis: {diagnosis_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Docker Container Crash Loop",
            "issue": """
Symptom: Container restarts every 30 seconds
Logs:
[2024-11-24] Starting application...
[2024-11-24] ERROR: Cannot bind to port 8080: Address already in use
[2024-11-24] Application exited with code 1
[2024-11-24] Restarting...
            """
        },
        {
            "name": "Intermittent Network Failures",
            "issue": """
Symptom: API calls fail randomly with connection timeout
Logs:
[2024-11-24 11:00:00] INFO: Request to api.example.com - SUCCESS (150ms)
[2024-11-24 11:00:30] ERROR: Request to api.example.com - TIMEOUT (30000ms)
[2024-11-24 11:01:00] INFO: Request to api.example.com - SUCCESS (145ms)
[2024-11-24 11:01:15] ERROR: DNS resolution took 25000ms
            """
        },
        {
            "name": "High CPU Usage",
            "issue": """
Symptom: CPU at 100%, application unresponsive
Logs:
[2024-11-24] INFO: Processing 1000 items in loop
[2024-11-24] DEBUG: Executing regex: .*([a-zA-Z0-9]+).*{1,100} on large text
[2024-11-24] WARNING: Operation taking longer than expected
[2024-11-24] INFO: CPU usage per process: python 99.8%
            """
        }
    ]

    for test in test_cases:
        print(f"\n📋 Issue: {test['name']}")
        print(test['issue'])
        diagnosis = agent.ask(f"Troubleshoot this issue:\n{test['issue']}\nWhat is the root cause and solution?")
        print(f"\n🔍 Diagnosis: {diagnosis}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved troubleshooting knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on common system issues!")
    print("📖 Load this playbook for faster, more accurate diagnostics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
