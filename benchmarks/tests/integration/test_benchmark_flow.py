"""Integration tests for end-to-end benchmark flow."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Iterator, Dict, Any

from ace import Sample

from benchmarks import BenchmarkTaskManager, BenchmarkConfig
from benchmarks.environments import GenericBenchmarkEnvironment


class TestBenchmarkManagerIntegration:
    """Integration tests for BenchmarkTaskManager."""

    def test_manager_initialization(self, temp_tasks_dir: Path):
        """Test manager initializes and discovers configs."""
        manager = BenchmarkTaskManager(tasks_dir=temp_tasks_dir)
        benchmarks = manager.list_benchmarks()

        assert "test_benchmark" in benchmarks

    def test_get_config(self, temp_tasks_dir: Path):
        """Test retrieving benchmark config."""
        manager = BenchmarkTaskManager(tasks_dir=temp_tasks_dir)
        config = manager.get_config("test_benchmark")

        assert config.task == "test_benchmark"
        assert config.version == "1.0"
        assert config.data["source"] == "huggingface"

    def test_get_benchmark_environment(self, temp_tasks_dir: Path):
        """Test retrieving benchmark environment."""
        manager = BenchmarkTaskManager(tasks_dir=temp_tasks_dir)
        env = manager.get_benchmark("test_benchmark")

        # Default should be GenericBenchmarkEnvironment for unknown types
        assert isinstance(env, GenericBenchmarkEnvironment)

    def test_validate_config(self, temp_tasks_dir: Path):
        """Test config validation."""
        manager = BenchmarkTaskManager(tasks_dir=temp_tasks_dir)
        errors = manager.validate_config("test_benchmark")

        assert len(errors) == 0

    def test_validate_config_missing_benchmark(self, temp_tasks_dir: Path):
        """Test validation fails for missing benchmark."""
        manager = BenchmarkTaskManager(tasks_dir=temp_tasks_dir)
        errors = manager.validate_config("nonexistent_benchmark")

        assert len(errors) > 0
        assert "Unknown benchmark" in errors[0]


class TestEnvironmentEvaluation:
    """Integration tests for environment evaluation."""

    def test_generic_environment_evaluation(
        self, basic_benchmark_config: Dict[str, Any]
    ):
        """Test generic environment can evaluate samples."""
        config = BenchmarkConfig.from_dict(basic_benchmark_config)
        env = GenericBenchmarkEnvironment(config)

        sample = Sample(
            question="What is 2+2?",
            ground_truth="4",
        )

        # Create mock agent output
        mock_output = MagicMock()
        mock_output.final_answer = "4"

        result = env.evaluate(sample, mock_output)

        assert result.metrics["accuracy"] == 1.0
        assert "Good performance" in result.feedback


class TestEndToEndFlow:
    """Tests for complete benchmark evaluation flow."""

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""

        class MockLoader:
            def supports_source(self, source: str) -> bool:
                return source == "mock"

            def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
                for i in range(5):
                    yield {
                        "question": f"Question {i}?",
                        "answer": f"Answer {i}",
                        "context": f"Context {i}",
                    }

        return MockLoader()

    def test_sample_creation(self):
        """Test creating samples from raw data."""
        raw_data = {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "context": "Geography trivia",
        }

        sample = Sample(
            question=raw_data["question"],
            ground_truth=raw_data["answer"],
            context=raw_data["context"],
        )

        assert sample.question == "What is the capital of France?"
        assert sample.ground_truth == "Paris"
        assert sample.context == "Geography trivia"

    def test_metrics_computation(self, basic_benchmark_config: Dict[str, Any]):
        """Test metrics are computed correctly."""
        config = BenchmarkConfig.from_dict(basic_benchmark_config)
        env = GenericBenchmarkEnvironment(config)

        # Test exact match
        sample = Sample(question="Q", ground_truth="exact answer")
        mock_output = MagicMock()
        mock_output.final_answer = "exact answer"

        result = env.evaluate(sample, mock_output)
        assert result.metrics.get("accuracy", 0) == 1.0

        # Test no match
        mock_output.final_answer = "wrong answer"
        result = env.evaluate(sample, mock_output)
        assert result.metrics.get("accuracy", 0) == 0.0
