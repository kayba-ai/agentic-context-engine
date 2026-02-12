"""Tests for benchmark processors."""

import pytest
from typing import Dict, Any

from benchmarks.processors import (
    MultipleChoiceProcessor,
    GSM8KProcessor,
    _validate_required,
    _validate_type,
)


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_required_success(self):
        """Test validation passes with all required fields."""
        data = {"question": "Q?", "answer": "A", "context": "C"}
        # Should not raise
        _validate_required(data, ["question", "answer"], "test")

    def test_validate_required_missing_field(self):
        """Test validation fails with missing field."""
        data = {"question": "Q?"}
        with pytest.raises(ValueError, match="Missing required field 'answer'"):
            _validate_required(data, ["question", "answer"], "test")

    def test_validate_required_none_value(self):
        """Test validation fails with None value."""
        data = {"question": "Q?", "answer": None}
        with pytest.raises(ValueError, match="Field 'answer' is None"):
            _validate_required(data, ["question", "answer"], "test")

    def test_validate_type_success(self):
        """Test type validation passes with correct type."""
        _validate_type(42, int, "answer")
        _validate_type("hello", str, "name")
        _validate_type([1, 2, 3], list, "items")

    def test_validate_type_failure(self):
        """Test type validation fails with incorrect type."""
        with pytest.raises(TypeError, match="Expected int for 'answer'"):
            _validate_type("not an int", int, "answer")


class TestMultipleChoiceProcessor:
    """Tests for MultipleChoiceProcessor."""

    def test_process_mmlu_valid(self, sample_mmlu_data: Dict[str, Any]):
        """Test MMLU processing with valid data."""
        processor = MultipleChoiceProcessor(benchmark_type="mmlu")
        result = list(processor.process_samples([sample_mmlu_data]))[0]

        assert result.ground_truth == "B"
        assert "Paris" in result.question
        assert result.metadata["subject"] == "geography"

    def test_process_mmlu_missing_answer(self):
        """Test MMLU processing fails with missing answer."""
        processor = MultipleChoiceProcessor(benchmark_type="mmlu")
        invalid_data = {"question": "Test?", "choices": ["A", "B"]}

        with pytest.raises(ValueError, match="Missing required field"):
            list(processor.process_samples([invalid_data]))

    def test_process_mmlu_invalid_answer_index(self):
        """Test MMLU processing fails with out-of-range answer index."""
        processor = MultipleChoiceProcessor(benchmark_type="mmlu")
        invalid_data = {
            "question": "Test?",
            "choices": ["A", "B"],
            "answer": 5,  # Out of range
        }

        with pytest.raises(ValueError, match="out of range"):
            list(processor.process_samples([invalid_data]))

    def test_process_hellaswag_valid(self, sample_hellaswag_data: Dict[str, Any]):
        """Test HellaSwag processing with valid data."""
        processor = MultipleChoiceProcessor(benchmark_type="hellaswag")
        result = list(processor.process_samples([sample_hellaswag_data]))[0]

        assert result.ground_truth == "A"
        assert "piano" in result.question

    def test_process_hellaswag_empty_endings(self):
        """Test HellaSwag processing fails with empty endings."""
        processor = MultipleChoiceProcessor(benchmark_type="hellaswag")
        invalid_data = {"ctx": "Context", "endings": [], "label": 0}

        with pytest.raises(ValueError, match="empty endings"):
            list(processor.process_samples([invalid_data]))

    def test_process_arc_valid(self, sample_arc_data: Dict[str, Any]):
        """Test ARC processing with valid data."""
        processor = MultipleChoiceProcessor(benchmark_type="arc")
        result = list(processor.process_samples([sample_arc_data]))[0]

        assert result.ground_truth == "A"
        assert "Atom" in result.question


class TestGSM8KProcessor:
    """Tests for GSM8KProcessor."""

    def test_extract_final_answer_with_hash(self, sample_gsm8k_data: Dict[str, Any]):
        """Test extracting answer from GSM8K format with ####."""
        processor = GSM8KProcessor()
        result = list(processor.process_samples([sample_gsm8k_data]))[0]

        assert result.ground_truth == "3"

    def test_extract_final_answer_without_hash(self):
        """Test extracting answer without #### marker."""
        processor = GSM8KProcessor()
        data = {"question": "Q?", "answer": "The answer is 42."}
        result = list(processor.process_samples([data]))[0]

        assert result.ground_truth == "42"

    def test_extract_final_answer_with_comma(self):
        """Test extracting answer with comma formatting."""
        processor = GSM8KProcessor()
        data = {"question": "Q?", "answer": "#### 1,000"}
        result = list(processor.process_samples([data]))[0]

        assert result.ground_truth == "1000"
