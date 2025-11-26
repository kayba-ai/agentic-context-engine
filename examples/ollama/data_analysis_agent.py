#!/usr/bin/env python3
"""
Data Analysis Agent - Analyzes data and generates insights.

This agent learns what types of analysis and insights are valuable for different
data types. It improves at identifying patterns, anomalies, and actionable insights.

Use Cases:
- Automated data profiling
- Anomaly detection
- Business metrics analysis
- Learning domain-specific patterns

Model Recommendation: ollama/llama3.1:8b or ollama/mistral:7b
"""

import sys
import json
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class DataAnalysisEnvironment(TaskEnvironment):
    """Evaluates data analysis quality based on expected insights."""

    def __init__(self, expected_insights: dict[str, list[str]]):
        """
        Args:
            expected_insights: Dict mapping dataset IDs to expected insights
        """
        self.expected_insights = expected_insights

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if analysis identified key insights."""
        # Extract dataset ID from question
        dataset_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_insights.get(dataset_id, [])

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No expected insights to validate",
                ground_truth="Valid analysis"
            )

        # Check if analysis mentions key insights
        found_insights = sum(1 for insight in expected if insight.lower() in answer.lower())
        success = found_insights >= len(expected) * 0.6  # 60% threshold

        feedback = f"Identified {found_insights}/{len(expected)} key insights. "
        if success:
            feedback += "Good analysis!"
        else:
            missed = [i for i in expected if i.lower() not in answer.lower()]
            feedback += f"Missed: {', '.join(missed[:3])}"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=f"Key insights: {', '.join(expected)}"
        )


# Training data: datasets with expected insights
TRAINING_DATA = [
    {
        "id": "sales_data",
        "data": {
            "records": [
                {"month": "Jan", "sales": 10000, "returns": 500},
                {"month": "Feb", "sales": 12000, "returns": 600},
                {"month": "Mar", "sales": 8000, "returns": 1200},
                {"month": "Apr", "sales": 15000, "returns": 400},
            ]
        },
        "insights": ["sales increasing", "March spike in returns", "return rate variation", "seasonal pattern"]
    },
    {
        "id": "user_engagement",
        "data": {
            "metrics": {
                "daily_active_users": [1000, 1050, 980, 1100, 1200],
                "session_duration_minutes": [15, 14, 10, 16, 18],
                "bounce_rate_percent": [45, 42, 55, 40, 38]
            }
        },
        "insights": ["DAU growth trend", "session duration correlates", "bounce rate improving", "day 3 anomaly"]
    },
    {
        "id": "website_errors",
        "data": {
            "errors": [
                {"hour": 0, "count": 5, "type": "404"},
                {"hour": 1, "count": 3, "type": "404"},
                {"hour": 2, "count": 45, "type": "500"},
                {"hour": 3, "count": 52, "type": "500"},
            ]
        },
        "insights": ["spike in 500 errors", "hours 2-3 incident", "server issue", "not user-caused"]
    },
    {
        "id": "customer_satisfaction",
        "data": {
            "survey_scores": [5, 5, 4, 2, 1, 5, 4, 1, 2, 5],
            "avg_score": 3.4,
            "response_rate": "15%"
        },
        "insights": ["bimodal distribution", "polarized feedback", "low response rate", "unhappy customers vocal"]
    },
    {
        "id": "api_performance",
        "data": {
            "endpoints": [
                {"path": "/api/users", "avg_ms": 45, "requests": 10000},
                {"path": "/api/search", "avg_ms": 1200, "requests": 5000},
                {"path": "/api/profile", "avg_ms": 30, "requests": 8000},
            ]
        },
        "insights": ["search endpoint slow", "performance bottleneck", "optimization needed", "high traffic path"]
    },
]


def create_analysis_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected insights mapping."""
    samples = []
    expected_insights = {}

    for item in TRAINING_DATA:
        data_str = json.dumps(item['data'], indent=2)
        question = f"{item['id']}: Analyze this data:\n{data_str}"

        samples.append(Sample(
            question=question,
            ground_truth=f"Key insights: {', '.join(item['insights'])}"
        ))
        expected_insights[item['id']] = item['insights']

    return samples, expected_insights


def main():
    print("📊 Data Analysis Agent with ACE Learning\n")
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

    playbook_path = Path(__file__).parent / "data_analysis_playbook.json"

    # Create agent
    print("\n🤖 Creating Data Analysis Agent...")
    agent = ACELiteLLM(
        model="ollama/llama3.1:8b",
        max_tokens=2048,
        temperature=0.4,  # Some creativity for insights
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    # Prepare training data
    samples, expected_insights = create_analysis_samples()
    environment = DataAnalysisEnvironment(expected_insights)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_data = {
        "page_views": [100, 105, 110, 400, 115, 120],
        "conversions": [5, 5, 6, 8, 6, 7]
    }
    print(f"Data: {json.dumps(test_data, indent=2)}")

    analysis_before = agent.ask(f"Analyze this data and provide insights:\n{json.dumps(test_data, indent=2)}")
    print(f"\n📊 Analysis: {analysis_before}\n")

    # Train the agent
    print("\n🎓 Training on data analysis patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully analyzed {successful}/{len(results)} datasets")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} analysis strategies:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Data: {json.dumps(test_data, indent=2)}")

    analysis_after = agent.ask(f"Analyze this data and provide insights:\n{json.dumps(test_data, indent=2)}")
    print(f"\n📊 Analysis: {analysis_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        {
            "name": "E-commerce Conversion Funnel",
            "data": {
                "visitors": 10000,
                "product_views": 5000,
                "add_to_cart": 1000,
                "checkout_started": 400,
                "completed_purchase": 250
            }
        },
        {
            "name": "System Resource Usage",
            "data": {
                "timestamps": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                "cpu_percent": [20, 18, 45, 88, 92, 35],
                "memory_gb": [4.2, 4.1, 5.5, 7.8, 8.2, 5.0]
            }
        },
        {
            "name": "Customer Churn Analysis",
            "data": {
                "cohort": "2024-Q1",
                "month_1_retention": 100,
                "month_2_retention": 75,
                "month_3_retention": 45,
                "month_4_retention": 42,
                "month_5_retention": 40
            }
        }
    ]

    for test in test_cases:
        print(f"\n📋 Dataset: {test['name']}")
        data_str = json.dumps(test['data'], indent=2)
        print(f"Data:\n{data_str}")
        analysis = agent.ask(f"Analyze this {test['name']} data:\n{data_str}")
        print(f"\n📊 Analysis: {analysis}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved data analysis knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on data analysis patterns!")
    print("📖 Load this playbook in future sessions for better insights")

    return 0


if __name__ == "__main__":
    sys.exit(main())
