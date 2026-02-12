"""Tests for benchmark environments."""

import pytest
from typing import Dict, Any

from ace import Sample

from benchmarks.base import BenchmarkConfig
from benchmarks.environments import (
    MultipleChoiceEnvironment,
    MathEnvironment,
    GenericBenchmarkEnvironment,
)


@pytest.fixture
def basic_config() -> BenchmarkConfig:
    """Create a basic config for testing."""
    return BenchmarkConfig(
        task="test",
        version="1.0",
        data={"source": "huggingface"},
        preprocessing={},
        metrics=[{"name": "accuracy", "weight": 1.0}],
    )


class TestMultipleChoiceEnvironment:
    """Tests for MultipleChoiceEnvironment."""

    def test_extract_answer_direct_letter(self, basic_config: BenchmarkConfig):
        """Test extracting direct letter answer."""
        env = MultipleChoiceEnvironment(basic_config)
        assert env._extract_answer("A") == "A"
        assert env._extract_answer("B") == "B"
        assert env._extract_answer("c") == "C"  # Lowercase

    def test_extract_answer_with_prefix(self, basic_config: BenchmarkConfig):
        """Test extracting answer with common prefixes."""
        env = MultipleChoiceEnvironment(basic_config)
        assert env._extract_answer("Answer: A") == "A"
        assert env._extract_answer("The answer is B") == "B"
        assert env._extract_answer("Choice: C") == "C"

    def test_extract_answer_with_parentheses(self, basic_config: BenchmarkConfig):
        """Test extracting answer with parentheses."""
        env = MultipleChoiceEnvironment(basic_config)
        assert env._extract_answer("(A)") == "A"
        assert env._extract_answer("A)") == "A"
        assert env._extract_answer("(B) is correct") == "B"

    def test_normalize_answer_letter(self, basic_config: BenchmarkConfig):
        """Test normalizing letter answers."""
        env = MultipleChoiceEnvironment(basic_config)
        assert env._normalize_answer("A") == "A"
        assert env._normalize_answer("a") == "A"
        assert env._normalize_answer("  B  ") == "B"

    def test_normalize_answer_numeric(self, basic_config: BenchmarkConfig):
        """Test normalizing numeric answers to letters."""
        env = MultipleChoiceEnvironment(basic_config)
        assert env._normalize_answer("0") == "A"
        assert env._normalize_answer("1") == "B"
        assert env._normalize_answer("2") == "C"

    def test_evaluate_correct_answer(
        self, basic_config: BenchmarkConfig, mock_agent_output
    ):
        """Test evaluation with correct answer."""
        env = MultipleChoiceEnvironment(basic_config)
        sample = Sample(question="Test?", ground_truth="A")
        output = mock_agent_output("A")

        result = env.evaluate(sample, output)

        assert result.metrics["accuracy"] == 1.0
        assert "Correct" in result.feedback

    def test_evaluate_incorrect_answer(
        self, basic_config: BenchmarkConfig, mock_agent_output
    ):
        """Test evaluation with incorrect answer."""
        env = MultipleChoiceEnvironment(basic_config)
        sample = Sample(question="Test?", ground_truth="A")
        output = mock_agent_output("B")

        result = env.evaluate(sample, output)

        assert result.metrics["accuracy"] == 0.0
        assert "Incorrect" in result.feedback


class TestMathEnvironment:
    """Tests for MathEnvironment."""

    def test_extract_number_with_hash(self, basic_config: BenchmarkConfig):
        """Test extracting number from #### format."""
        env = MathEnvironment(basic_config)
        assert env._extract_number("#### 42") == 42.0
        assert env._extract_number("#### 3.14") == 3.14
        assert env._extract_number("Solution: #### 100") == 100.0

    def test_extract_number_with_comma(self, basic_config: BenchmarkConfig):
        """Test extracting number with comma formatting."""
        env = MathEnvironment(basic_config)
        assert env._extract_number("#### 1,000") == 1000.0
        assert env._extract_number("#### 1,234,567") == 1234567.0

    def test_extract_number_from_text(self, basic_config: BenchmarkConfig):
        """Test extracting number from natural text."""
        env = MathEnvironment(basic_config)
        assert env._extract_number("The answer is 42") == 42.0
        assert env._extract_number("Result: 3.14159") == 3.14159

    def test_compute_math_metrics_exact_match(self, basic_config: BenchmarkConfig):
        """Test metrics computation for exact match."""
        env = MathEnvironment(basic_config)
        metrics = env._compute_math_metrics(42.0, 42.0)

        assert metrics["exact_match"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["within_1_percent"] == 1.0

    def test_compute_math_metrics_close_match(self, basic_config: BenchmarkConfig):
        """Test metrics computation for close but not exact match."""
        env = MathEnvironment(basic_config)
        metrics = env._compute_math_metrics(100.5, 100.0)  # 0.5% error

        assert metrics["exact_match"] == 0.0
        assert metrics["within_1_percent"] == 1.0
        assert metrics["within_5_percent"] == 1.0


class TestGenericBenchmarkEnvironment:
    """Tests for GenericBenchmarkEnvironment."""

    def test_evaluate_high_score(
        self, basic_config: BenchmarkConfig, mock_agent_output
    ):
        """Test evaluation with high score."""
        env = GenericBenchmarkEnvironment(basic_config)
        sample = Sample(question="Test", ground_truth="expected answer")
        output = mock_agent_output("expected answer")

        result = env.evaluate(sample, output)

        assert result.metrics["accuracy"] == 1.0
        assert "Good performance" in result.feedback

    def test_evaluate_low_score(self, basic_config: BenchmarkConfig, mock_agent_output):
        """Test evaluation with low score."""
        env = GenericBenchmarkEnvironment(basic_config)
        sample = Sample(question="Test", ground_truth="expected answer")
        output = mock_agent_output("completely wrong")

        result = env.evaluate(sample, output)

        assert result.metrics["accuracy"] == 0.0
        assert "Low performance" in result.feedback
