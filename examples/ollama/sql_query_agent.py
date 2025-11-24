#!/usr/bin/env python3
"""
SQL Query Generator Agent - Generates SQL queries from natural language.

This agent learns to translate natural language requests into correct SQL queries,
improving accuracy and learning database-specific patterns over time.

Use Cases:
- Natural language to SQL translation
- Query optimization suggestions
- Database schema understanding
- Learning business-specific query patterns

Model Recommendation: ollama/qwen2.5:7b or ollama/codellama:13b
"""

import sys
import re
from pathlib import Path
from ace import ACELiteLLM, Sample, TaskEnvironment, EnvironmentResult


class SQLQueryEnvironment(TaskEnvironment):
    """Evaluates SQL query correctness against expected patterns."""

    def __init__(self, expected_queries: dict[str, dict]):
        """
        Args:
            expected_queries: Dict mapping query IDs to expected SQL patterns
        """
        self.expected_queries = expected_queries

    def evaluate(self, sample, answer: str) -> EnvironmentResult:
        """Check if generated SQL matches expected patterns."""
        # Extract query ID from question
        query_id = sample.question.split("\n")[0].split(":")[0].strip()
        expected = self.expected_queries.get(query_id, {})

        if not expected:
            return EnvironmentResult(
                success=True,
                feedback="No validation available",
                ground_truth="Valid SQL"
            )

        # Extract SQL from answer (look for SELECT, INSERT, UPDATE, DELETE)
        sql_pattern = r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP).*?(?=;|$|\n\n)'
        sql_matches = re.findall(sql_pattern, answer, re.IGNORECASE | re.DOTALL)

        if not sql_matches:
            return EnvironmentResult(
                success=False,
                feedback="No SQL query found in response",
                ground_truth=expected.get("query", "")
            )

        generated_sql = sql_matches[0].strip().upper()

        # Check for required keywords/patterns
        required = expected.get("required", [])
        found = sum(1 for keyword in required if keyword.upper() in generated_sql)
        success = found >= len(required) * 0.8  # 80% threshold

        # Check for forbidden patterns (common mistakes)
        forbidden = expected.get("forbidden", [])
        has_forbidden = any(pattern.upper() in generated_sql for pattern in forbidden)

        if has_forbidden:
            success = False

        feedback = f"Found {found}/{len(required)} required elements. "
        if has_forbidden:
            feedback += "Contains forbidden patterns (common mistakes). "
        if success:
            feedback += "Good query!"
        else:
            missing = [k for k in required if k.upper() not in generated_sql]
            if missing:
                feedback += f"Missing: {', '.join(missing[:3])}"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=expected.get("query", "")
        )


# Database schema for context
DATABASE_SCHEMA = """
Tables:
- users (id, username, email, created_at, status)
- orders (id, user_id, total, status, created_at, updated_at)
- products (id, name, price, stock, category)
- order_items (order_id, product_id, quantity, price)
"""

# Training data: natural language to SQL mappings
TRAINING_DATA = [
    {
        "id": "q1",
        "nl": "Find all active users",
        "query": "SELECT * FROM users WHERE status = 'active'",
        "required": ["SELECT", "FROM users", "WHERE", "status"],
        "forbidden": ["*"]  # Should specify columns
    },
    {
        "id": "q2",
        "nl": "Get total sales per month for 2024",
        "query": """SELECT
            DATE_TRUNC('month', created_at) as month,
            SUM(total) as total_sales
        FROM orders
        WHERE EXTRACT(YEAR FROM created_at) = 2024
        GROUP BY DATE_TRUNC('month', created_at)""",
        "required": ["SELECT", "SUM", "FROM orders", "GROUP BY", "WHERE"],
        "forbidden": ["COUNT(*)"]  # Common mistake: counting instead of summing
    },
    {
        "id": "q3",
        "nl": "Find users who never made an order",
        "query": """SELECT u.id, u.username
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE o.id IS NULL""",
        "required": ["SELECT", "LEFT JOIN", "WHERE", "IS NULL"],
        "forbidden": ["NOT IN", "NOT EXISTS"]  # Less efficient alternatives
    },
    {
        "id": "q4",
        "nl": "Top 5 products by revenue",
        "query": """SELECT
            p.name,
            SUM(oi.quantity * oi.price) as revenue
        FROM products p
        JOIN order_items oi ON p.id = oi.product_id
        GROUP BY p.id, p.name
        ORDER BY revenue DESC
        LIMIT 5""",
        "required": ["SELECT", "SUM", "JOIN", "GROUP BY", "ORDER BY", "LIMIT"],
        "forbidden": []
    },
    {
        "id": "q5",
        "nl": "Average order value by user",
        "query": """SELECT
            u.username,
            AVG(o.total) as avg_order_value
        FROM users u
        JOIN orders o ON u.id = o.user_id
        GROUP BY u.id, u.username
        HAVING COUNT(o.id) >= 3""",
        "required": ["SELECT", "AVG", "JOIN", "GROUP BY", "HAVING"],
        "forbidden": ["WHERE COUNT"]  # HAVING vs WHERE confusion
    },
    {
        "id": "q6",
        "nl": "Products low on stock (less than 10 items)",
        "query": "SELECT id, name, stock FROM products WHERE stock < 10 ORDER BY stock ASC",
        "required": ["SELECT", "FROM products", "WHERE stock", "ORDER BY"],
        "forbidden": []
    },
]


def create_sql_samples() -> tuple[list[Sample], dict]:
    """Create training samples and expected query mapping."""
    samples = []
    expected_queries = {}

    for item in TRAINING_DATA:
        question = f"{item['id']}: Generate SQL for: {item['nl']}\n\nSchema:\n{DATABASE_SCHEMA}"
        samples.append(Sample(
            question=question,
            ground_truth=item['query']
        ))
        expected_queries[item['id']] = {
            "query": item['query'],
            "required": item['required'],
            "forbidden": item.get('forbidden', [])
        }

    return samples, expected_queries


def main():
    print("🗄️  SQL Query Generator Agent with ACE Learning\n")
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
        print("2. Pull model: ollama pull qwen2.5:7b  (best for code/SQL)")
        print("   Alternative: ollama pull llama3.1:8b")
        return 1

    playbook_path = Path(__file__).parent / "sql_query_playbook.json"

    # Create agent
    print("\n🤖 Creating SQL Query Generator Agent...")
    agent = ACELiteLLM(
        model="ollama/qwen2.5:7b",  # Excellent for code and SQL
        max_tokens=1024,
        temperature=0.2,  # Low temp for deterministic SQL
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None
    )

    print(f"\n📋 Database Schema:")
    print(DATABASE_SCHEMA)

    # Prepare training data
    samples, expected_queries = create_sql_samples()
    environment = SQLQueryEnvironment(expected_queries)

    # Test before learning
    print("\n📝 Testing BEFORE learning:")
    print("-" * 70)
    test_request = "Find all orders over $100 placed in the last 30 days"
    print(f"Request: {test_request}")

    query_before = agent.ask(f"Generate SQL for: {test_request}\n\nSchema:\n{DATABASE_SCHEMA}")
    print(f"\n💻 Generated SQL:\n{query_before}\n")

    # Train the agent
    print("\n🎓 Training on SQL query patterns...")
    print(f"Training samples: {len(samples)}")
    print("-" * 70)

    try:
        results = agent.learn(samples, environment, epochs=2)
        successful = len([r for r in results if r.success])
        print(f"\n✅ Successfully generated {successful}/{len(results)} queries")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Ensure you have qwen2.5:7b or llama3.1:8b installed")
        return 1

    # Show learned strategies
    if agent.playbook.bullets():
        print(f"\n📚 Learned {len(agent.playbook.bullets())} SQL patterns:")
        for i, bullet in enumerate(agent.playbook.bullets()[:5], 1):
            score = f"(+{bullet.helpful}/-{bullet.harmful})"
            print(f"  {i}. {bullet.content[:80]}... {score}")

    # Test after learning
    print("\n\n📝 Testing AFTER learning:")
    print("-" * 70)
    print(f"Request: {test_request}")

    query_after = agent.ask(f"Generate SQL for: {test_request}\n\nSchema:\n{DATABASE_SCHEMA}")
    print(f"\n💻 Generated SQL:\n{query_after}\n")

    # Real-world test cases
    print("\n" + "=" * 70)
    print("🧪 Real-World Test Cases:")
    print("=" * 70)

    test_cases = [
        "Find users who placed more than 5 orders",
        "Calculate total revenue by product category",
        "Find orders that contain products from multiple categories",
        "Get the most popular products (by order frequency)",
        "Find users whose last order was more than 90 days ago",
        "Show daily new user registrations for the current month",
    ]

    for i, request in enumerate(test_cases, 1):
        print(f"\n📋 Request {i}: {request}")
        query = agent.ask(f"Generate SQL for: {request}\n\nSchema:\n{DATABASE_SCHEMA}")
        print(f"💻 Generated SQL:\n{query}\n")
        print("-" * 70)

    # Save learned knowledge
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved SQL query knowledge to: {playbook_path}")
    print("\n✅ Agent is now trained on SQL query patterns!")
    print("📖 Load this playbook for consistent SQL generation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
