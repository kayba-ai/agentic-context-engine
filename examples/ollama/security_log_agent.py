#!/usr/bin/env python3
"""
Security Log Analyzer Agent - Analyzes security logs and detects threats.

This agent learns to identify normal vs suspicious behavior, attack patterns,
reduce false positives, and provide actionable incident response guidance.

Use Cases:
- SIEM log analysis
- Intrusion detection
- Anomaly detection
- Automated incident response

Model Recommendation: ollama/llama3.1:8b or ollama/mistral:7b
"""

import sys
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class SecurityLogEnvironment(TaskEnvironment):
    """Evaluates security log analysis accuracy."""

    def __init__(self, expected_analysis: dict[str, dict]):
        """
        Args:
            expected_analysis: Dict mapping log IDs to expected threat classification
        """
        self.expected_analysis = expected_analysis

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if threat detection and severity are correct."""
        log_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_analysis.get(log_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation criteria available",
                ground_truth="Valid analysis"
            )

        threat_level = expected.get("threat_level", "").lower()
        threat_type = expected.get("threat_type", "").lower()
        indicators = expected.get("indicators", [])
        false_alarm = expected.get("false_alarm", False)

        answer_lower = answer.lower()

        # Check threat detection accuracy
        has_threat_level = threat_level in answer_lower
        has_threat_type = threat_type in answer_lower if threat_type else True
        has_indicators = sum(1 for ind in indicators if ind.lower() in answer_lower)

        # Check if false alarm correctly identified
        if false_alarm:
            success = "false positive" in answer_lower or "benign" in answer_lower or "normal" in answer_lower
            feedback = "✓ Correctly identified as false positive. " if success else "✗ Missed false positive. "
        else:
            success = (
                has_threat_level and
                has_threat_type and
                has_indicators >= len(indicators) * 0.5
            )

            feedback = ""
            if has_threat_level:
                feedback += f"✓ Correct threat level ({threat_level}). "
            else:
                feedback += f"✗ Wrong threat level (expected {threat_level}). "

            if has_threat_type:
                feedback += f"✓ Identified threat type ({threat_type}). "

            feedback += f"Found {has_indicators}/{len(indicators)} threat indicators. "

        if success:
            feedback += "Good threat analysis!"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Threat: {threat_level}, Type: {threat_type}"
        )


# Training data: security logs with threat classifications
TRAINING_DATA = [
    {
        "id": "log_1",
        "logs": """
[2024-11-24 03:15:22] AUTH FAILED: user=admin from=185.220.101.45
[2024-11-24 03:15:25] AUTH FAILED: user=root from=185.220.101.45
[2024-11-24 03:15:28] AUTH FAILED: user=administrator from=185.220.101.45
[2024-11-24 03:15:31] AUTH FAILED: user=postgres from=185.220.101.45
[2024-11-24 03:15:34] AUTH FAILED: user=oracle from=185.220.101.45
        """,
        "threat_level": "high",
        "threat_type": "brute force",
        "indicators": ["multiple failed attempts", "common usernames", "sequential", "same IP"],
        "false_alarm": False
    },
    {
        "id": "log_2",
        "logs": """
[2024-11-24 14:22:10] GET /api/users/123 200 45ms user_id=123
[2024-11-24 14:22:15] GET /api/users/123 200 42ms user_id=123
[2024-11-24 14:22:20] GET /api/users/123 200 43ms user_id=123
        """,
        "threat_level": "none",
        "threat_type": "",
        "indicators": ["normal api usage", "legitimate user"],
        "false_alarm": True
    },
    {
        "id": "log_3",
        "logs": """
[2024-11-24 09:45:12] GET /admin/../../etc/passwd 404
[2024-11-24 09:45:15] GET /admin/../../../etc/shadow 404
[2024-11-24 09:45:18] GET /cgi-bin/../../../../etc/passwd 404
[2024-11-24 09:45:21] GET /api/../../../database/config.yml 404
        """,
        "threat_level": "critical",
        "threat_type": "path traversal",
        "indicators": ["directory traversal", "system files", "attack pattern", "reconnaissance"],
        "false_alarm": False
    },
    {
        "id": "log_4",
        "logs": """
[2024-11-24 11:30:00] SQL Query: SELECT * FROM products WHERE id = 1
[2024-11-24 11:30:05] SQL Query: SELECT * FROM products WHERE id = 2' OR '1'='1
[2024-11-24 11:30:07] SQL ERROR: Syntax error near '1'='1'
[2024-11-24 11:30:10] SQL Query: SELECT * FROM users WHERE id = 1; DROP TABLE users--
        """,
        "threat_level": "critical",
        "threat_type": "sql injection",
        "indicators": ["sql injection", "malicious query", "drop table", "attack escalation"],
        "false_alarm": False
    },
    {
        "id": "log_5",
        "logs": """
[2024-11-24 16:00:00] User john@company.com uploaded report.pdf
[2024-11-24 16:00:15] User john@company.com uploaded invoice.pdf
[2024-11-24 16:00:30] User john@company.com uploaded contract.pdf
[2024-11-24 16:00:45] User john@company.com uploaded proposal.pdf
        """,
        "threat_level": "none",
        "threat_type": "",
        "indicators": ["legitimate file uploads", "business documents"],
        "false_alarm": True
    },
    {
        "id": "log_6",
        "logs": """
[2024-11-24 02:30:00] OUTBOUND CONNECTION: 192.168.1.100 -> 45.142.120.55:4444
[2024-11-24 02:30:05] DATA EXFIL: 2.5GB transferred to 45.142.120.55
[2024-11-24 02:30:10] PROCESS: powershell.exe -encodedCommand <base64>
[2024-11-24 02:30:15] FILE ACCESS: Multiple sensitive files accessed
        """,
        "threat_level": "critical",
        "threat_type": "data exfiltration",
        "indicators": ["unusual outbound", "large data transfer", "encoded command", "C2 communication"],
        "false_alarm": False
    },
    {
        "id": "log_7",
        "logs": """
[2024-11-24 10:15:00] SCANNER DETECTED: Nikto/2.1.6
[2024-11-24 10:15:01] GET /admin 404 from=203.0.113.50
[2024-11-24 10:15:02] GET /backup 404 from=203.0.113.50
[2024-11-24 10:15:03] GET /config 404 from=203.0.113.50
[2024-11-24 10:15:04] GET /.git/config 404 from=203.0.113.50
        """,
        "threat_level": "medium",
        "threat_type": "reconnaissance",
        "indicators": ["vulnerability scanner", "directory enumeration", "attack preparation"],
        "false_alarm": False
    },
]


def create_security_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected analysis mapping."""
    samples = []
    expected_analysis = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Analyze these security logs:\n\n{item['logs']}\n\n"
        question += "Classify threat level, identify attack type, and suggest response."

        samples.append(Sample(
            question=question,
            ground_truth=f"Threat: {item['threat_level']}, Type: {item.get('threat_type', 'none')}"
        ))

        expected_analysis[item['id']] = {
            "threat_level": item['threat_level'],
            "threat_type": item.get('threat_type', ''),
            "indicators": item['indicators'],
            "false_alarm": item.get('false_alarm', False)
        }

    return samples, expected_analysis


def main():
    print("🔐 Security Log Analyzer Agent with ACE Learning\n")
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

    playbook_path = Path(__file__).parent / "security_log_playbook.json"

    # Create agent
    print("\n🤖 Creating Security Log Analyzer Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=1536,
        temperature=0.2,  # Low temp for consistent threat detection
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, expected_analysis = create_security_samples()
    environment = SecurityLogEnvironment(expected_analysis)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_logs = """
[2024-11-24 08:30:00] LOGIN: user@example.com from=10.0.1.50
[2024-11-24 08:30:15] FILE ACCESS: /confidential/payroll.xlsx
[2024-11-24 08:30:20] FILE ACCESS: /confidential/salaries.pdf
[2024-11-24 08:30:25] FILE DOWNLOAD: payroll.xlsx (2.5MB)
[2024-11-24 08:30:30] EMAIL SENT: payroll.xlsx to external@competitor.com
    """
    print(f"Security logs:\n{test_logs}")

    analysis_before = agent.ask(
        f"Analyze these security logs (threat level, type, indicators, response):\n{test_logs}"
    )
    print(f"\n🔍 Security Analysis:\n{analysis_before}\n")

    # Train the agent
    print("\n🎓 Training on security threat patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully analyzed {successful}/{len(results)} security logs")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} security analysis strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Security logs:\n{test_logs}")

    analysis_after = agent.ask(
        f"Analyze these security logs (threat level, type, indicators, response):\n{test_logs}"
    )
    print(f"\n🔍 Security Analysis:\n{analysis_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "Privilege Escalation Attempt",
            "logs": """
[2024-11-24 13:00:00] USER: bob executed: sudo su -
[2024-11-24 13:00:05] AUTH: sudo authentication failed for bob
[2024-11-24 13:00:10] USER: bob executed: su root
[2024-11-24 13:00:15] AUTH: su authentication failed for bob
[2024-11-24 13:00:20] FILE: bob accessed /etc/sudoers (denied)
            """
        },
        {
            "name": "Cryptomining Activity",
            "logs": """
[2024-11-24 19:00:00] PROCESS: xmrig started by www-data
[2024-11-24 19:00:05] CPU USAGE: Process 12345 using 95% CPU
[2024-11-24 19:00:10] NETWORK: Outbound to pool.minexmr.com:3333
[2024-11-24 19:00:15] PROCESS: xmrig running for 2 hours
            """
        },
        {
            "name": "Phishing Email Detection",
            "logs": """
[2024-11-24 09:15:00] EMAIL RECEIVED: from=noreply@paypa1.com
[2024-11-24 09:15:05] LINK DETECTED: http://paypa1-secure.tk/login
[2024-11-24 09:15:10] USER: alice clicked link
[2024-11-24 09:15:15] CREDENTIALS SUBMITTED: username=alice@company.com
[2024-11-24 09:15:20] WARNING: Suspected phishing domain
            """
        },
        {
            "name": "Normal Admin Activity",
            "logs": """
[2024-11-24 15:00:00] ADMIN: sysadmin logged in from 10.0.0.100
[2024-11-24 15:00:30] ADMIN: sysadmin executed: systemctl restart nginx
[2024-11-24 15:01:00] ADMIN: sysadmin viewed /var/log/nginx/access.log
[2024-11-24 15:01:30] ADMIN: sysadmin logged out
            """
        },
        {
            "name": "Ransomware Indicators",
            "logs": """
[2024-11-24 04:00:00] PROCESS: unknown.exe started from %TEMP%
[2024-11-24 04:00:05] FILE: documents/report.docx renamed to report.docx.encrypted
[2024-11-24 04:00:10] FILE: photos/vacation.jpg renamed to vacation.jpg.encrypted
[2024-11-24 04:00:15] FILE CREATED: README_RANSOM.txt in multiple directories
[2024-11-24 04:00:20] MASS FILE ENCRYPTION: 1500 files affected
            """
        },
    ]

    for test in test_cases:
        print(f"\n📋 Test Case: {test['name']}")
        print(f"Logs:\n{test['logs']}")
        analysis = agent.ask(
            f"Analyze these security logs (threat level, type, indicators, response):\n{test['logs']}"
        )
        print(f"\n🔍 Security Analysis:\n{analysis}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved security analysis knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on threat detection patterns!")
    print("📖 Load this playbook for consistent security incident analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
