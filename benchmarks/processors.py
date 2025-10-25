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
            0: 'O',      # Outside
            1: 'B-PER',  # Begin Person
            2: 'I-PER',  # Inside Person
            3: 'B-LOC',  # Begin Location
            4: 'I-LOC',  # Inside Location
            5: 'B-ORG',  # Begin Organization
            6: 'I-ORG',  # Inside Organization
        }

    def process_token_stream(self, token_stream: Iterator[Dict[str, Any]]) -> Iterator[Sample]:
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
            doc_idx = token_data['doc_idx']
            sent_idx = token_data['sent_idx']

            doc_sentences[doc_idx][sent_idx].append({
                'token': token_data['gold_token'],
                'label': token_data['gold_label']
            })

        # Process each sentence
        sample_id = 0
        for doc_idx in sorted(doc_sentences.keys()):
            document = doc_sentences[doc_idx]

            for sent_idx in sorted(document.keys()):
                sentence_tokens = document[sent_idx]

                # Reconstruct sentence text
                tokens = [item['token'] for item in sentence_tokens]
                labels = [self.label_map.get(item['label'], 'O') for item in sentence_tokens]

                sentence_text = self._reconstruct_sentence(tokens)
                entities = self._extract_entities(tokens, labels)

                yield Sample(
                    question=f"Identify named entities in the following financial text:\n\n{sentence_text}",
                    ground_truth=self._format_entities_as_string(entities)
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
        punctuation = {'.', ',', '!', '?', ';', ':', "'", '"', ')', ']', '}', '%'}
        return token in punctuation

    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
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
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, tokens))

                # Start new entity
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    'type': entity_type,
                    'start_idx': i,
                    'end_idx': i,
                    'token_indices': [i]
                }

            elif label.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if entity_type == current_entity['type']:
                    current_entity['end_idx'] = i
                    current_entity['token_indices'].append(i)
                else:
                    # Type mismatch - finalize previous and start new
                    entities.append(self._finalize_entity(current_entity, tokens))
                    current_entity = {
                        'type': entity_type,
                        'start_idx': i,
                        'end_idx': i,
                        'token_indices': [i]
                    }

            elif label == 'O':
                # Outside - finalize current entity if exists
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, tokens))
                    current_entity = None

        # Finalize last entity if exists
        if current_entity:
            entities.append(self._finalize_entity(current_entity, tokens))

        return entities

    def _finalize_entity(self, entity_info: Dict[str, Any], tokens: List[str]) -> Dict[str, Any]:
        """Convert entity info to final entity dictionary."""
        entity_tokens = [tokens[i] for i in entity_info['token_indices']]
        entity_text = self._reconstruct_entity_text(entity_tokens)

        return {
            'text': entity_text,
            'label': entity_info['type'],
            'start_idx': entity_info['start_idx'],
            'end_idx': entity_info['end_idx'],
            'tokens': entity_tokens
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

    def process_samples(self, sample_stream: Iterator[Dict[str, Any]]) -> Iterator[Sample]:
        """Process XBRL-Math samples - may need restructuring based on actual format."""
        sample_id = 0

        for sample_data in sample_stream:
            yield Sample(
                question=sample_data.get('question', ''),
                context=sample_data.get('context', ''),
                ground_truth=str(sample_data.get('answer', ''))
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
                ground_truth="Task completion successful"
            )


def get_processor(benchmark_name: str):
    """Get appropriate processor for benchmark."""
    processors = {
        'finer_ord': FiNERProcessor(),
        'xbrl_math': XBRLMathProcessor(),
        'appworld': AppWorldProcessor()
    }

    return processors.get(benchmark_name)