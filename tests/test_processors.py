"""
Tests for benchmark data processors.

Tests the data transformation logic for various benchmark types including
FiNER (NER), multiple choice, GSM8K math, and simple Q&A processors.
"""

import unittest
from unittest.mock import Mock

import pytest

try:
    from benchmarks.processors import (
        FiNERProcessor,
        MultipleChoiceProcessor,
        GSM8KProcessor,
        SimpleQAProcessor,
        TruthfulQAProcessor,
        WinoGrandeProcessor,
        get_processor,
    )
    from ace import Sample

    PROCESSORS_AVAILABLE = True
except ImportError:
    PROCESSORS_AVAILABLE = False


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestFiNERProcessor(unittest.TestCase):
    """Test FiNER NER processor for token-level data."""

    def setUp(self):
        """Set up FiNER processor."""
        self.processor = FiNERProcessor()

    def test_label_map(self):
        """Test BIO label mapping."""
        self.assertEqual(self.processor.label_map[0], "O")
        self.assertEqual(self.processor.label_map[1], "B-PER")
        self.assertEqual(self.processor.label_map[2], "I-PER")
        self.assertEqual(self.processor.label_map[3], "B-LOC")
        self.assertEqual(self.processor.label_map[5], "B-ORG")

    def test_reconstruct_sentence_basic(self):
        """Test basic sentence reconstruction."""
        tokens = ["Hello", "world", "!"]
        result = self.processor._reconstruct_sentence(tokens)
        self.assertEqual(result, "Hello world!")

    def test_reconstruct_sentence_empty(self):
        """Test empty token list."""
        result = self.processor._reconstruct_sentence([])
        self.assertEqual(result, "")

    def test_reconstruct_sentence_punctuation(self):
        """Test punctuation handling in sentence reconstruction."""
        tokens = ["Apple", "Inc", ".", "is", "great", "."]
        result = self.processor._reconstruct_sentence(tokens)
        self.assertEqual(result, "Apple Inc. is great.")

    def test_is_punctuation(self):
        """Test punctuation detection."""
        self.assertTrue(self.processor._is_punctuation("."))
        self.assertTrue(self.processor._is_punctuation(","))
        self.assertTrue(self.processor._is_punctuation("!"))
        self.assertFalse(self.processor._is_punctuation("word"))
        self.assertFalse(self.processor._is_punctuation(""))

    def test_extract_entities_simple(self):
        """Test entity extraction with simple BIO sequence."""
        tokens = ["Apple", "Inc", ".", "announced"]
        labels = ["B-ORG", "I-ORG", "O", "O"]

        entities = self.processor._extract_entities(tokens, labels)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["text"], "Apple Inc")
        self.assertEqual(entities[0]["label"], "ORG")

    def test_extract_entities_multiple(self):
        """Test extraction of multiple entities."""
        tokens = ["Tim", "Cook", "leads", "Apple", "."]
        labels = ["B-PER", "I-PER", "O", "B-ORG", "O"]

        entities = self.processor._extract_entities(tokens, labels)

        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["text"], "Tim Cook")
        self.assertEqual(entities[0]["label"], "PER")
        self.assertEqual(entities[1]["text"], "Apple")
        self.assertEqual(entities[1]["label"], "ORG")

    def test_extract_entities_no_entities(self):
        """Test extraction when no entities present."""
        tokens = ["Hello", "world"]
        labels = ["O", "O"]

        entities = self.processor._extract_entities(tokens, labels)
        self.assertEqual(len(entities), 0)

    def test_format_entities_as_string(self):
        """Test entity formatting for ground truth."""
        entities = [
            {"text": "Apple", "label": "ORG"},
            {"text": "Tim Cook", "label": "PER"},
        ]

        result = self.processor._format_entities_as_string(entities)
        self.assertEqual(result, "Apple (ORG); Tim Cook (PER)")

    def test_format_entities_empty(self):
        """Test formatting when no entities."""
        result = self.processor._format_entities_as_string([])
        self.assertEqual(result, "No named entities found.")

    def test_process_token_stream(self):
        """Test full token stream processing."""
        token_stream = [
            {"doc_idx": 0, "sent_idx": 0, "gold_token": "Apple", "gold_label": 5},
            {"doc_idx": 0, "sent_idx": 0, "gold_token": "Inc", "gold_label": 6},
            {"doc_idx": 0, "sent_idx": 0, "gold_token": ".", "gold_label": 0},
        ]

        samples = list(self.processor.process_token_stream(iter(token_stream)))

        self.assertEqual(len(samples), 1)
        self.assertIsInstance(samples[0], Sample)
        self.assertIn("Apple Inc", samples[0].question)
        self.assertEqual(samples[0].ground_truth, "Apple Inc (ORG)")


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestMultipleChoiceProcessor(unittest.TestCase):
    """Test multiple choice processor for MMLU, HellaSwag, ARC."""

    def test_process_mmlu(self):
        """Test MMLU sample processing."""
        processor = MultipleChoiceProcessor(benchmark_type="mmlu")

        sample_data = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
            "subject": "geography",
        }

        samples = list(processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("What is the capital of France?", sample.question)
        self.assertIn("A) London", sample.question)
        self.assertIn("B) Paris", sample.question)
        self.assertEqual(sample.ground_truth, "B")
        self.assertEqual(sample.metadata["subject"], "geography")

    def test_process_hellaswag(self):
        """Test HellaSwag sample processing."""
        processor = MultipleChoiceProcessor(benchmark_type="hellaswag")

        sample_data = {
            "ctx": "A woman is cooking in the kitchen.",
            "endings": [
                "She burns the food.",
                "She makes a delicious meal.",
                "She leaves the house.",
                "She goes to sleep.",
            ],
            "label": "1",
            "activity_label": "cooking",
        }

        samples = list(processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("A woman is cooking", sample.question)
        self.assertIn("A) She burns the food", sample.question)
        self.assertEqual(sample.ground_truth, "B")

    def test_process_arc(self):
        """Test ARC sample processing."""
        processor = MultipleChoiceProcessor(benchmark_type="arc")

        sample_data = {
            "question": "What causes seasons?",
            "choices": {
                "text": ["Earth's tilt", "Distance from sun", "Moon phases", "Wind"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        }

        samples = list(processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("What causes seasons?", sample.question)
        self.assertEqual(sample.ground_truth, "A")

    def test_process_generic(self):
        """Test generic multiple choice processing."""
        processor = MultipleChoiceProcessor(benchmark_type="generic")

        sample_data = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }

        samples = list(processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].ground_truth, "B")

    def test_letter_map(self):
        """Test letter mapping."""
        processor = MultipleChoiceProcessor()
        self.assertEqual(processor.letter_map[0], "A")
        self.assertEqual(processor.letter_map[3], "D")

    def test_format_multiple_choice(self):
        """Test multiple choice formatting."""
        processor = MultipleChoiceProcessor()
        result = processor._format_multiple_choice("Test?", ["A", "B", "C"])

        self.assertIn("Question: Test?", result)
        self.assertIn("A) A", result)
        self.assertIn("B) B", result)
        self.assertIn("C) C", result)


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestGSM8KProcessor(unittest.TestCase):
    """Test GSM8K math problem processor."""

    def setUp(self):
        """Set up GSM8K processor."""
        self.processor = GSM8KProcessor()

    def test_extract_final_answer_with_delimiter(self):
        """Test extraction with #### delimiter."""
        answer = "Step 1: 2+2=4\nStep 2: 4+4=8\n#### 8"
        result = self.processor._extract_final_answer(answer)
        self.assertEqual(result, "8")

    def test_extract_final_answer_with_commas(self):
        """Test extraction handles comma-separated numbers."""
        answer = "The total is #### 1,234"
        result = self.processor._extract_final_answer(answer)
        self.assertEqual(result, "1234")

    def test_extract_final_answer_fallback(self):
        """Test fallback to last number when no #### present."""
        answer = "The answer is 42."
        result = self.processor._extract_final_answer(answer)
        self.assertEqual(result, "42")

    def test_extract_final_answer_negative(self):
        """Test extraction of negative numbers."""
        answer = "#### -15"
        result = self.processor._extract_final_answer(answer)
        self.assertEqual(result, "-15")

    def test_process_samples(self):
        """Test full sample processing."""
        sample_data = {
            "question": "If Alice has 3 apples and Bob gives her 2 more, how many does she have?",
            "answer": "Alice starts with 3 apples. Bob gives her 2 more. 3 + 2 = 5. #### 5",
        }

        samples = list(self.processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("3 apples", sample.question)
        self.assertEqual(sample.ground_truth, "5")
        self.assertIn("#### 5", sample.metadata["full_solution"])


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestSimpleQAProcessor(unittest.TestCase):
    """Test simple Q&A processor."""

    def setUp(self):
        """Set up SimpleQA processor."""
        self.processor = SimpleQAProcessor()

    def test_process_squad_format(self):
        """Test SQuAD-style answer format."""
        sample_data = {
            "question": "What is the capital of France?",
            "context": "France is a country in Europe. Its capital is Paris.",
            "answers": {"text": ["Paris"], "answer_start": [45]},
        }

        samples = list(self.processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("capital of France", sample.question)
        self.assertEqual(sample.ground_truth, "Paris")

    def test_process_list_answers(self):
        """Test list-style answer format."""
        sample_data = {
            "question": "Who wrote Hamlet?",
            "context": "Shakespeare wrote many plays.",
            "answers": ["Shakespeare"],
        }

        samples = list(self.processor.process_samples(iter([sample_data])))
        self.assertEqual(samples[0].ground_truth, "Shakespeare")

    def test_process_string_answer(self):
        """Test string answer format."""
        sample_data = {
            "question": "What is 2+2?",
            "context": "Math question",
            "answers": "4",
        }

        samples = list(self.processor.process_samples(iter([sample_data])))
        self.assertEqual(samples[0].ground_truth, "4")


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestTruthfulQAProcessor(unittest.TestCase):
    """Test TruthfulQA processor."""

    def setUp(self):
        """Set up TruthfulQA processor."""
        self.processor = TruthfulQAProcessor()

    def test_process_samples(self):
        """Test TruthfulQA sample processing."""
        sample_data = {
            "question": "What happens if you step on a crack?",
            "mc1_targets": {
                "choices": [
                    "Nothing happens",
                    "Your mother's back breaks",
                    "You fall down",
                ],
                "labels": [1, 0, 0],
            },
        }

        samples = list(self.processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("step on a crack", sample.question)
        self.assertEqual(sample.ground_truth, "A")  # Index 0 has label 1


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestWinoGrandeProcessor(unittest.TestCase):
    """Test WinoGrande processor."""

    def setUp(self):
        """Set up WinoGrande processor."""
        self.processor = WinoGrandeProcessor()

    def test_process_samples(self):
        """Test WinoGrande sample processing."""
        sample_data = {
            "sentence": "The trophy doesn't fit in the suitcase because _ is too big.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "1",
        }

        samples = list(self.processor.process_samples(iter([sample_data])))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("trophy", sample.question)
        self.assertIn("1) the trophy", sample.question)
        self.assertIn("2) the suitcase", sample.question)
        self.assertEqual(sample.ground_truth, "1")


@unittest.skipUnless(PROCESSORS_AVAILABLE, "Processors module not available")
@pytest.mark.unit
class TestGetProcessor(unittest.TestCase):
    """Test processor factory function."""

    def test_get_finer_processor(self):
        """Test getting FiNER processor."""
        processor = get_processor("finer_ord")
        self.assertIsInstance(processor, FiNERProcessor)

    def test_get_mmlu_processor(self):
        """Test getting MMLU processor."""
        processor = get_processor("mmlu")
        self.assertIsInstance(processor, MultipleChoiceProcessor)
        self.assertEqual(processor.benchmark_type, "mmlu")

    def test_get_gsm8k_processor(self):
        """Test getting GSM8K processor."""
        processor = get_processor("gsm8k")
        self.assertIsInstance(processor, GSM8KProcessor)

    def test_get_simple_qa_processor(self):
        """Test getting SimpleQA processor."""
        processor = get_processor("simple_qa")
        self.assertIsInstance(processor, SimpleQAProcessor)

    def test_get_unknown_processor(self):
        """Test getting unknown processor returns None."""
        processor = get_processor("unknown_benchmark")
        self.assertIsNone(processor)

    def test_arc_processors(self):
        """Test ARC easy and challenge use same processor type."""
        easy = get_processor("arc_easy")
        challenge = get_processor("arc_challenge")
        self.assertIsInstance(easy, MultipleChoiceProcessor)
        self.assertIsInstance(challenge, MultipleChoiceProcessor)
        self.assertEqual(easy.benchmark_type, "arc")
        self.assertEqual(challenge.benchmark_type, "arc")


if __name__ == "__main__":
    unittest.main()
