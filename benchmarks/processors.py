"""
Dataset-specific processors for handling different data granularities.

This module provides processors that convert raw dataset formats into
properly structured samples for evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Iterator, Any, Tuple

from ace import Sample


class FiNERProcessor:
    """
    Processor for FiNER dataset that groups token-level data into sentences.

    FiNER dataset stores individual tokens as rows with BIO labels.
    This processor reconstructs sentences and converts BIO tags to entity spans.
    """

    def __init__(self):
        self.label_map = {
            0: "O",  # Outside
            1: "B-PER",  # Begin Person
            2: "I-PER",  # Inside Person
            3: "B-LOC",  # Begin Location
            4: "I-LOC",  # Inside Location
            5: "B-ORG",  # Begin Organization
            6: "I-ORG",  # Inside Organization
        }

    def process_token_stream(
        self, token_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """
        Process token stream and yield sentence-level samples.

        Args:
            token_stream: Iterator yielding individual token dictionaries

        Yields:
            Sample objects for each sentence
        """
        # Group tokens by document and sentence
        doc_sentences = defaultdict(lambda: defaultdict(list))

        for token_data in token_stream:
            doc_idx = token_data["doc_idx"]
            sent_idx = token_data["sent_idx"]

            doc_sentences[doc_idx][sent_idx].append(
                {"token": token_data["gold_token"], "label": token_data["gold_label"]}
            )

        # Process each sentence
        sample_id = 0
        for doc_idx in sorted(doc_sentences.keys()):
            document = doc_sentences[doc_idx]

            for sent_idx in sorted(document.keys()):
                sentence_tokens = document[sent_idx]

                # Reconstruct sentence text
                tokens = [item["token"] for item in sentence_tokens]
                labels = [
                    self.label_map.get(item["label"], "O") for item in sentence_tokens
                ]

                sentence_text = self._reconstruct_sentence(tokens)
                entities = self._extract_entities(tokens, labels)

                yield Sample(
                    question=f"Identify named entities in the following financial text:\n\n{sentence_text}",
                    ground_truth=self._format_entities_as_string(entities),
                )

                sample_id += 1

    def _reconstruct_sentence(self, tokens: List[str]) -> str:
        """
        Reconstruct sentence from tokens, handling punctuation properly.
        """
        if not tokens:
            return ""

        result = []
        for i, token in enumerate(tokens):
            # Add space before token unless it's punctuation or first token
            if i > 0 and not self._is_punctuation(token):
                result.append(" ")
            result.append(token)

        return "".join(result)

    def _is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation that shouldn't have space before it."""
        punctuation = {".", ",", "!", "?", ";", ":", "'", '"', ")", "]", "}", "%"}
        return token in punctuation

    def _extract_entities(
        self, tokens: List[str], labels: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Convert BIO labels to entity spans.

        Args:
            tokens: List of tokens
            labels: List of BIO labels

        Returns:
            List of entity dictionaries with text, label, start, end
        """
        entities = []
        current_entity = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, tokens))

                # Start new entity
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    "type": entity_type,
                    "start_idx": i,
                    "end_idx": i,
                    "token_indices": [i],
                }

            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if entity_type == current_entity["type"]:
                    current_entity["end_idx"] = i
                    current_entity["token_indices"].append(i)
                else:
                    # Type mismatch - finalize previous and start new
                    entities.append(self._finalize_entity(current_entity, tokens))
                    current_entity = {
                        "type": entity_type,
                        "start_idx": i,
                        "end_idx": i,
                        "token_indices": [i],
                    }

            elif label == "O":
                # Outside - finalize current entity if exists
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, tokens))
                    current_entity = None

        # Finalize last entity if exists
        if current_entity:
            entities.append(self._finalize_entity(current_entity, tokens))

        return entities

    def _finalize_entity(
        self, entity_info: Dict[str, Any], tokens: List[str]
    ) -> Dict[str, Any]:
        """Convert entity info to final entity dictionary."""
        entity_tokens = [tokens[i] for i in entity_info["token_indices"]]
        entity_text = self._reconstruct_entity_text(entity_tokens)

        return {
            "text": entity_text,
            "label": entity_info["type"],
            "start_idx": entity_info["start_idx"],
            "end_idx": entity_info["end_idx"],
            "tokens": entity_tokens,
        }

    def _reconstruct_entity_text(self, entity_tokens: List[str]) -> str:
        """Reconstruct entity text from tokens."""
        if not entity_tokens:
            return ""

        # Simple reconstruction - join with spaces, but handle subwords
        result = []
        for i, token in enumerate(entity_tokens):
            if i > 0 and not token.startswith("'") and not self._is_punctuation(token):
                result.append(" ")
            result.append(token)

        return "".join(result)

    def _format_entities_as_string(self, entities: List[Dict[str, Any]]) -> str:
        """Format entities as string for ground truth comparison."""
        if not entities:
            return "No named entities found."

        entity_strs = []
        for entity in entities:
            entity_strs.append(f"{entity['text']} ({entity['label']})")

        return "; ".join(entity_strs)


class XBRLMathProcessor:
    """Processor for XBRL-Math dataset - handles numerical reasoning problems."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process XBRL-Math samples - may need restructuring based on actual format."""
        sample_id = 0

        for sample_data in sample_stream:
            yield Sample(
                question=sample_data.get("question", ""),
                context=sample_data.get("context", ""),
                ground_truth=str(sample_data.get("answer", "")),
            )
            sample_id += 1


class AppWorldProcessor:
    """Processor for AppWorld dataset - handles agent tasks."""

    def process_tasks(self, task_stream: Iterator[Dict[str, Any]]) -> Iterator[Sample]:
        """Process AppWorld tasks."""
        for task_data in task_stream:
            yield Sample(
                question=task_data["instruction"],
                context=f"Available APIs: {task_data['api_docs']}",
                ground_truth="Task completion successful",
            )


class SWEBenchProcessor:
    """Processor for SWE-bench dataset."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process SWE-bench instances into samples."""
        for data in sample_stream:
            yield Sample(
                question=self._format_question(data),
                ground_truth=data.get("patch", ""),
                context=self._format_context(data),
                metadata={
                    "instance_id": data.get("instance_id", ""),
                    "repo": data.get("repo", ""),
                    "base_commit": data.get("base_commit", ""),
                    "test_patch": data.get("test_patch", ""),
                    "hints_text": data.get("hints_text", ""),
                    "created_at": data.get("created_at", ""),
                },
            )

    def _format_question(self, data: Dict[str, Any]) -> str:
        """Format the problem statement as a question."""
        return f"""Repository: {data.get('repo', 'unknown')}

Issue: {data.get('problem_statement', '')}

Base Commit: {data.get('base_commit', '')}

Please analyze this issue and provide a patch (in diff format) that resolves it."""

    def _format_context(self, data: Dict[str, Any]) -> str:
        """Format additional context."""
        parts = []

        if data.get("hints_text"):
            parts.append(f"Hints: {data['hints_text']}")

        if data.get("test_patch"):
            parts.append(f"Test patch (for reference):\n{data['test_patch']}")

        return "\n\n".join(parts) if parts else ""


class LettaProcessor:
    """Processor for Letta benchmark - handles memory and conversation tasks."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process Letta tasks into samples."""
        for data in sample_stream:
            yield Sample(
                question=self._format_question(data),
                ground_truth=data.get("expected_response", ""),
                context=data.get("conversation_history", ""),
                metadata={
                    "task_id": data.get("task_id", ""),
                    "task_type": data.get("task_type", "memory"),
                    "relevant_memories": data.get("relevant_memories", []),
                    "difficulty": data.get("difficulty", "unknown"),
                    "memory_depth": data.get("memory_depth", 0),
                    "conversation_turns": data.get("conversation_turns", 0),
                },
            )

    def _format_question(self, data: Dict[str, Any]) -> str:
        """Format the conversation query as a question."""
        history = data.get("conversation_history", "")
        query = data.get("query", "")

        if history:
            return f"""{history}

Current query: {query}

Respond based on the conversation history and any relevant memories."""
        return query


class MultipleChoiceProcessor:
    """
    Processor for multiple-choice benchmarks (MMLU, HellaSwag, ARC, etc.).

    Handles the transformation of HuggingFace dataset formats into standardized
    ACE Sample format for multiple-choice questions.
    """

    def __init__(self, benchmark_type: str = "generic"):
        """
        Initialize processor with benchmark-specific settings.

        Args:
            benchmark_type: One of 'mmlu', 'hellaswag', 'arc', 'generic'
        """
        self.benchmark_type = benchmark_type
        self.letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        self.reverse_letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process multiple-choice samples into ACE format."""
        for sample_data in sample_stream:
            if self.benchmark_type == "mmlu":
                yield self._process_mmlu(sample_data)
            elif self.benchmark_type == "hellaswag":
                yield self._process_hellaswag(sample_data)
            elif self.benchmark_type == "arc":
                yield self._process_arc(sample_data)
            else:
                yield self._process_generic(sample_data)

    def _process_mmlu(self, data: Dict[str, Any]) -> Sample:
        """Process MMLU sample format."""
        question = data.get("question", "")
        choices = data.get("choices", [])
        answer_idx = data.get("answer", 0)
        subject = data.get("subject", "unknown")

        # Format question with choices
        formatted_question = self._format_multiple_choice(question, choices)

        # Convert numeric answer to letter
        ground_truth = self.letter_map.get(answer_idx, str(answer_idx))

        return Sample(
            question=formatted_question,
            ground_truth=ground_truth,
            context=f"Subject: {subject}",
            metadata={
                "subject": subject,
                "num_choices": len(choices),
                "answer_index": answer_idx,
            },
        )

    def _process_hellaswag(self, data: Dict[str, Any]) -> Sample:
        """Process HellaSwag sample format."""
        ctx = data.get("ctx", data.get("context", ""))
        endings = data.get("endings", [])
        label = data.get("label", 0)

        # HellaSwag label is often a string
        if isinstance(label, str):
            label = int(label) if label.isdigit() else 0

        # Format question
        formatted_question = f"Context: {ctx}\n\nWhich ending makes the most sense?\n\n"
        for i, ending in enumerate(endings):
            formatted_question += f"{self.letter_map[i]}) {ending}\n"
        formatted_question += "\nAnswer with just the letter."

        ground_truth = self.letter_map.get(label, str(label))

        return Sample(
            question=formatted_question,
            ground_truth=ground_truth,
            metadata={
                "activity_label": data.get("activity_label", ""),
                "num_choices": len(endings),
                "label_index": label,
            },
        )

    def _process_arc(self, data: Dict[str, Any]) -> Sample:
        """Process ARC (AI2 Reasoning Challenge) sample format."""
        question = data.get("question", "")
        choices_data = data.get("choices", {})
        answer_key = data.get("answerKey", "A")

        # ARC has nested choices structure
        if isinstance(choices_data, dict):
            choice_texts = choices_data.get("text", [])
            choice_labels = choices_data.get("label", [])
        else:
            # Handle list format
            choice_texts = (
                [c.get("text", "") for c in choices_data] if choices_data else []
            )
            choice_labels = (
                [c.get("label", "") for c in choices_data] if choices_data else []
            )

        # Format question with choices
        formatted_question = f"Question: {question}\n\n"
        for i, (label, text) in enumerate(zip(choice_labels, choice_texts)):
            formatted_question += f"{label}) {text}\n"
        formatted_question += "\nAnswer with just the letter."

        return Sample(
            question=formatted_question,
            ground_truth=answer_key,
            metadata={
                "num_choices": len(choice_texts),
                "choice_labels": choice_labels,
            },
        )

    def _process_generic(self, data: Dict[str, Any]) -> Sample:
        """Process generic multiple-choice format."""
        question = data.get("question", data.get("prompt", ""))
        choices = data.get("choices", data.get("options", []))
        answer = data.get("answer", data.get("label", 0))

        formatted_question = self._format_multiple_choice(question, choices)

        # Handle both numeric and letter answers
        if isinstance(answer, int):
            ground_truth = self.letter_map.get(answer, str(answer))
        else:
            ground_truth = str(answer)

        return Sample(
            question=formatted_question,
            ground_truth=ground_truth,
        )

    def _format_multiple_choice(self, question: str, choices: List[str]) -> str:
        """Format a multiple choice question with lettered options."""
        formatted = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            letter = self.letter_map.get(i, str(i))
            formatted += f"{letter}) {choice}\n"
        formatted += "\nAnswer with just the letter (A, B, C, or D)."
        return formatted


class GSM8KProcessor:
    """Processor for GSM8K math word problems."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process GSM8K samples into ACE format."""
        for data in sample_stream:
            question = data.get("question", "")
            answer = data.get("answer", "")

            # GSM8K answers have format "#### NUMBER"
            # Extract just the final number
            final_answer = self._extract_final_answer(answer)

            yield Sample(
                question=f"Solve this math problem step by step:\n\n{question}\n\nProvide your final numerical answer.",
                ground_truth=final_answer,
                context=answer,  # Full solution as context for evaluation
                metadata={
                    "full_solution": answer,
                },
            )

    def _extract_final_answer(self, answer: str) -> str:
        """Extract the final numerical answer from GSM8K format."""
        import re

        # Look for #### pattern
        match = re.search(r"####\s*(.+)", answer)
        if match:
            return match.group(1).strip().replace(",", "")

        # Fallback: find last number in answer
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        if numbers:
            return numbers[-1].replace(",", "")

        return answer.strip()


class SimpleQAProcessor:
    """Processor for simple QA datasets like SQuAD."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process QA samples into ACE format."""
        for data in sample_stream:
            question = data.get("question", "")
            context = data.get("context", "")
            answers = data.get("answers", {})

            # SQuAD answers format: {"text": [...], "answer_start": [...]}
            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
                ground_truth = answer_texts[0] if answer_texts else ""
            elif isinstance(answers, list):
                ground_truth = answers[0] if answers else ""
            else:
                ground_truth = str(answers)

            yield Sample(
                question=f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                ground_truth=ground_truth,
                context=context,
                metadata={
                    "all_answers": (
                        answer_texts if isinstance(answers, dict) else [ground_truth]
                    ),
                },
            )


class TruthfulQAProcessor:
    """Processor for TruthfulQA multiple choice format."""

    def __init__(self):
        self.letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process TruthfulQA samples into ACE format."""
        for data in sample_stream:
            question = data.get("question", "")
            mc1_targets = data.get("mc1_targets", {})

            choices = mc1_targets.get("choices", [])
            labels = mc1_targets.get("labels", [])

            # Find the correct answer (label = 1)
            correct_idx = labels.index(1) if 1 in labels else 0
            ground_truth = self.letter_map.get(correct_idx, "A")

            # Format question with choices
            formatted_question = f"Question: {question}\n\n"
            for i, choice in enumerate(choices[:5]):  # Limit to 5 choices
                letter = self.letter_map.get(i, str(i))
                formatted_question += f"{letter}) {choice}\n"
            formatted_question += (
                "\nWhich answer is truthfully correct? Answer with just the letter."
            )

            yield Sample(
                question=formatted_question,
                ground_truth=ground_truth,
                metadata={
                    "num_choices": len(choices),
                    "correct_index": correct_idx,
                },
            )


class WinoGrandeProcessor:
    """Processor for WinoGrande commonsense reasoning format."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process WinoGrande samples into ACE format."""
        for data in sample_stream:
            sentence = data.get("sentence", "")
            option1 = data.get("option1", "")
            option2 = data.get("option2", "")
            answer = data.get("answer", "1")

            formatted_question = f"""Complete the sentence by choosing the correct option:

{sentence}

Options:
1) {option1}
2) {option2}

Answer with just the number (1 or 2)."""

            yield Sample(
                question=formatted_question,
                ground_truth=str(answer),
                metadata={
                    "option1": option1,
                    "option2": option2,
                },
            )


def get_processor(benchmark_name: str):
    """Get appropriate processor for benchmark."""
    processors = {
        "finer_ord": FiNERProcessor(),
        "xbrl_math": XBRLMathProcessor(),
        "appworld": AppWorldProcessor(),
        "swe_bench": SWEBenchProcessor(),
        "letta_bench": LettaProcessor(),
        # Multiple choice benchmarks
        "mmlu": MultipleChoiceProcessor(benchmark_type="mmlu"),
        "hellaswag": MultipleChoiceProcessor(benchmark_type="hellaswag"),
        "arc_easy": MultipleChoiceProcessor(benchmark_type="arc"),
        "arc_challenge": MultipleChoiceProcessor(benchmark_type="arc"),
        "truthfulqa": TruthfulQAProcessor(),
        "winogrande": WinoGrandeProcessor(),
        # Math benchmarks
        "gsm8k": GSM8KProcessor(),
        "simple_math": GSM8KProcessor(),  # Alias
        # QA benchmarks
        "simple_qa": SimpleQAProcessor(),
    }

    return processors.get(benchmark_name)
