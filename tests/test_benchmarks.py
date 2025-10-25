"""
Tests for the ACE benchmarking system.

Tests configuration loading, environment evaluation, and end-to-end benchmark execution.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ace import Sample, EnvironmentResult

# Import benchmark components
try:
    from benchmarks.base import BenchmarkConfig, BenchmarkSample, BenchmarkEnvironment
    from benchmarks.manager import BenchmarkTaskManager
    from benchmarks.environments import GenericBenchmarkEnvironment, FiNEREnvironment
    BENCHMARKS_AVAILABLE = True
except ImportError:
    BENCHMARKS_AVAILABLE = False


@unittest.skipUnless(BENCHMARKS_AVAILABLE, "Benchmarks module not available")
class TestBenchmarkConfig(unittest.TestCase):
    """Test benchmark configuration loading and validation."""

    def test_config_from_dict(self):
        """Test creating BenchmarkConfig from dictionary."""
        config_dict = {
            "task": "test_task",
            "version": "1.0",
            "data": {"source": "test", "dataset": "test_data"},
            "preprocessing": {"question_template": "{text}"},
            "metrics": [{"name": "accuracy"}],
            "metadata": {"description": "Test benchmark"}
        }

        config = BenchmarkConfig.from_dict(config_dict)

        self.assertEqual(config.task, "test_task")
        self.assertEqual(config.version, "1.0")
        self.assertEqual(config.data["source"], "test")
        self.assertEqual(config.preprocessing["question_template"], "{text}")
        self.assertEqual(config.metrics[0]["name"], "accuracy")
        self.assertEqual(config.metadata["description"], "Test benchmark")

    def test_config_minimal(self):
        """Test minimal config without optional fields."""
        config_dict = {
            "task": "minimal",
            "version": "1.0",
            "data": {"source": "test"},
            "preprocessing": {},
            "metrics": []
        }

        config = BenchmarkConfig.from_dict(config_dict)
        self.assertEqual(config.task, "minimal")
        self.assertIsNone(config.metadata)


class TestBenchmarkSample(unittest.TestCase):
    """Test BenchmarkSample functionality (now just Sample)."""

    def test_benchmark_sample_creation(self):
        """Test creating BenchmarkSample with core fields."""
        sample = BenchmarkSample(
            question="What is 2+2?",
            ground_truth="4",
            context="Math problem"
        )

        self.assertEqual(sample.question, "What is 2+2?")
        self.assertEqual(sample.ground_truth, "4")
        self.assertEqual(sample.context, "Math problem")

    def test_benchmark_sample_defaults(self):
        """Test BenchmarkSample with minimal required fields."""
        sample = BenchmarkSample(
            question="Test question",
            ground_truth="Test answer"
        )

        self.assertEqual(sample.question, "Test question")
        self.assertEqual(sample.ground_truth, "Test answer")
        # BenchmarkSample is now just Sample, so no extra fields


@unittest.skipUnless(BENCHMARKS_AVAILABLE, "Benchmarks module not available")
class TestBenchmarkEnvironment(unittest.TestCase):
    """Test benchmark environment evaluation."""

    def setUp(self):
        """Set up test environment."""
        self.config = BenchmarkConfig(
            task="test_env",
            version="1.0",
            data={"source": "test"},
            preprocessing={},
            metrics=[
                {"name": "accuracy"},
                {"name": "f1"}
            ]
        )

    def test_generic_environment_evaluation(self):
        """Test generic environment evaluation."""
        env = GenericBenchmarkEnvironment(self.config)

        sample = Sample(
            question="What is the capital of France?",
            ground_truth="Paris"
        )

        # Mock generator output
        mock_output = Mock()
        mock_output.final_answer = "Paris"

        result = env.evaluate(sample, mock_output)

        self.assertIsInstance(result, EnvironmentResult)
        self.assertIn("Good performance", result.feedback)
        self.assertEqual(result.metrics["accuracy"], 1.0)
        self.assertEqual(result.ground_truth, "Paris")

    def test_generic_environment_partial_match(self):
        """Test generic environment with partial match."""
        env = GenericBenchmarkEnvironment(self.config)

        sample = Sample(
            question="What is the capital of France?",
            ground_truth="Paris France"
        )

        mock_output = Mock()
        mock_output.final_answer = "Paris"

        result = env.evaluate(sample, mock_output)

        # Should have some F1 score but not exact match
        self.assertEqual(result.metrics["accuracy"], 0.0)  # No exact match
        self.assertGreater(result.metrics["f1"], 0.0)  # But some F1 overlap
        self.assertIn("Low performance", result.feedback)

    def test_compute_f1_score(self):
        """Test F1 score computation."""
        env = GenericBenchmarkEnvironment(self.config)

        # Perfect match
        f1 = env._compute_f1("hello world", "hello world")
        self.assertEqual(f1, 1.0)

        # Partial overlap
        f1 = env._compute_f1("hello world", "hello there")
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

        # No overlap
        f1 = env._compute_f1("hello", "goodbye")
        self.assertEqual(f1, 0.0)

        # Empty strings
        f1 = env._compute_f1("", "")
        self.assertEqual(f1, 1.0)


@unittest.skipUnless(BENCHMARKS_AVAILABLE, "Benchmarks module not available")
class TestFiNEREnvironment(unittest.TestCase):
    """Test FiNER-specific environment."""

    def setUp(self):
        """Set up FiNER environment."""
        self.config = BenchmarkConfig(
            task="finer",
            version="1.0",
            data={"source": "test"},
            preprocessing={},
            metrics=[{"name": "f1"}, {"name": "precision"}, {"name": "recall"}]
        )
        self.env = FiNEREnvironment(self.config)

    def test_extract_entities_json(self):
        """Test entity extraction from JSON format."""
        prediction = '[{"text": "Apple Inc.", "label": "ORG"}, {"text": "Tim Cook", "label": "PERSON"}]'

        sample = Sample(question="Test", ground_truth="")
        entities = self.env._extract_entities(prediction, sample)

        expected = {("Apple Inc.", "ORG"), ("Tim Cook", "PERSON")}
        self.assertEqual(entities, expected)

    def test_extract_entities_text(self):
        """Test entity extraction from free text."""
        prediction = "PERSON: John Smith\nORGANIZATION: Microsoft Corp\nLOCATION: New York"

        sample = Sample(question="Test", ground_truth="")
        entities = self.env._extract_entities(prediction, sample)

        self.assertIn(("John Smith", "PERSON"), entities)
        self.assertIn(("Microsoft Corp", "ORGANIZATION"), entities)
        self.assertIn(("New York", "LOCATION"), entities)

    def test_ner_metrics_calculation(self):
        """Test NER metrics calculation."""
        predicted = {("Apple", "ORG"), ("Cook", "PERSON"), ("Wrong", "MISC")}
        gold = {("Apple", "ORG"), ("Tim Cook", "PERSON")}

        metrics = self.env._compute_ner_metrics(predicted, gold)

        # Only "Apple" is correctly identified
        self.assertEqual(metrics["precision"], 1/3)  # 1 correct out of 3 predicted
        self.assertEqual(metrics["recall"], 1/2)     # 1 correct out of 2 gold
        self.assertAlmostEqual(metrics["f1"], 2 * (1/3) * (1/2) / ((1/3) + (1/2)), places=3)
        self.assertEqual(metrics["exact_match"], 0.0)  # Not exact match


@unittest.skipUnless(BENCHMARKS_AVAILABLE, "Benchmarks module not available")
class TestBenchmarkTaskManager(unittest.TestCase):
    """Test benchmark task manager functionality."""

    def test_manager_initialization(self):
        """Test manager initializes without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BenchmarkTaskManager(tasks_dir=Path(temp_dir))
            self.assertIsInstance(manager, BenchmarkTaskManager)
            self.assertEqual(len(manager.list_benchmarks()), 0)

    def test_config_discovery(self):
        """Test YAML config discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir)

            # Create test config
            config_content = """
task: test_discovery
version: 1.0
data:
  source: test
preprocessing: {}
metrics:
  - name: accuracy
"""
            (tasks_dir / "test_discovery.yaml").write_text(config_content)

            manager = BenchmarkTaskManager(tasks_dir=tasks_dir)
            benchmarks = manager.list_benchmarks()

            self.assertIn("test_discovery", benchmarks)
            config = manager.get_config("test_discovery")
            self.assertEqual(config.task, "test_discovery")

    def test_config_validation(self):
        """Test config validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir)

            # Valid config
            valid_config = """
task: valid_test
version: 1.0
data:
  source: huggingface
  dataset: test
preprocessing:
  question_template: "{text}"
metrics:
  - name: accuracy
"""
            (tasks_dir / "valid_test.yaml").write_text(valid_config)

            # Invalid config (missing required fields)
            invalid_config = """
task: invalid_test
data:
  source: unknown_source
"""
            (tasks_dir / "invalid_test.yaml").write_text(invalid_config)

            manager = BenchmarkTaskManager(tasks_dir=tasks_dir)

            # Valid config should have no errors (huggingface is a known source)
            errors = manager.validate_config("valid_test")
            self.assertEqual(len(errors), 0)  # Should be valid

            # Invalid config should have errors
            errors = manager.validate_config("invalid_test")
            self.assertGreater(len(errors), 0)

    def test_unknown_benchmark_error(self):
        """Test error handling for unknown benchmark."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = BenchmarkTaskManager(tasks_dir=Path(temp_dir))

            with self.assertRaises(ValueError):
                manager.get_config("nonexistent_benchmark")


class TestBenchmarkIntegration(unittest.TestCase):
    """Integration tests for the benchmarking system."""

    @patch('benchmarks.loaders.huggingface.HuggingFaceLoader.load')
    def test_end_to_end_mock(self, mock_load):
        """Test end-to-end benchmark execution with mocked data."""
        if not BENCHMARKS_AVAILABLE:
            self.skipTest("Benchmarks module not available")

        # Mock data
        mock_data = [
            {
                "question": "What is 2+2?",
                "ground_truth": "4",
                "context": "Simple math"
            },
            {
                "question": "What is 3+3?",
                "ground_truth": "6",
                "context": "Simple math"
            }
        ]
        mock_load.return_value = iter(mock_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir)

            # Create simple test config
            config_content = """
task: math_test
version: 1.0
data:
  source: huggingface
  dataset: mock_math
  split: test
preprocessing:
  question_template: "Q: {question} A:"
  ground_truth_field: ground_truth
metrics:
  - name: accuracy
metadata:
  description: "Simple math test"
"""
            (tasks_dir / "math_test.yaml").write_text(config_content)

            manager = BenchmarkTaskManager(tasks_dir=tasks_dir)

            # Verify config loads
            config = manager.get_config("math_test")
            self.assertEqual(config.task, "math_test")

            # Verify data loading
            data = list(manager.load_benchmark_data("math_test"))
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["question"], "What is 2+2?")

            # Verify environment creation
            env = manager.get_benchmark("math_test")
            self.assertIsInstance(env, BenchmarkEnvironment)


if __name__ == "__main__":
    unittest.main()