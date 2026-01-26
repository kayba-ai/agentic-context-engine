"""Pytest fixtures for benchmark tests."""

import pytest
from pathlib import Path
from typing import Dict, Any

from ace import Sample


@pytest.fixture
def sample_mmlu_data() -> Dict[str, Any]:
    """Sample MMLU data for testing."""
    return {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": 1,
        "subject": "geography",
    }


@pytest.fixture
def sample_gsm8k_data() -> Dict[str, Any]:
    """Sample GSM8K data for testing."""
    return {
        "question": "If John has 5 apples and gives 2 away, how many does he have?",
        "answer": "John has 5 apples. He gives away 2. 5 - 2 = 3. #### 3",
    }


@pytest.fixture
def sample_arc_data() -> Dict[str, Any]:
    """Sample ARC data for testing."""
    return {
        "question": "What is the smallest unit of matter?",
        "choices": {
            "text": ["Atom", "Molecule", "Cell", "Organism"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
    }


@pytest.fixture
def sample_hellaswag_data() -> Dict[str, Any]:
    """Sample HellaSwag data for testing."""
    return {
        "ctx": "A person is playing the piano.",
        "endings": [
            "They press the keys to make music.",
            "They go swimming in the ocean.",
            "They cook dinner in the kitchen.",
            "They ride a bicycle outside.",
        ],
        "label": "0",
    }


@pytest.fixture
def sample_sample() -> Sample:
    """Sample ACE Sample for testing."""
    return Sample(
        question="What is 2 + 2?",
        ground_truth="4",
        context="This is a math problem.",
    )


@pytest.fixture
def temp_tasks_dir(tmp_path: Path) -> Path:
    """Create temporary tasks directory with sample configs."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    # Create sample YAML config
    config = tasks_dir / "test_benchmark.yaml"
    config.write_text(
        """
task: test_benchmark
version: "1.0"
data:
  source: huggingface
  dataset_path: squad
  split: validation
preprocessing:
  question_template: "{question}"
  ground_truth_field: answers
metrics:
  - name: exact_match
    weight: 1.0
metadata:
  description: Test benchmark
"""
    )

    return tasks_dir


@pytest.fixture
def mock_agent_output():
    """Mock agent output for testing."""

    class MockAgentOutput:
        def __init__(self, answer: str = "Test answer"):
            self.final_answer = answer

    return MockAgentOutput


@pytest.fixture
def basic_benchmark_config() -> Dict[str, Any]:
    """Basic benchmark config for testing."""
    return {
        "task": "test_benchmark",
        "version": "1.0",
        "data": {"source": "huggingface", "dataset_path": "squad"},
        "preprocessing": {},
        "metrics": [{"name": "accuracy", "weight": 1.0}],
        "metadata": {"description": "Test benchmark"},
    }
