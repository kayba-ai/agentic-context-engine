"""Shared pytest fixtures for ACE test suite.

This module provides reusable fixtures to eliminate code duplication
across test files and ensure consistent test data.
"""

import json
from typing import Any, Dict, Type, TypeVar
import pytest
from pydantic import BaseModel

from ace import Playbook, Sample, LLMClient
from ace.llm import LLMResponse

T = TypeVar("T", bound=BaseModel)


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.

    Includes complete_structured() to prevent auto-wrapping with Instructor.

    Usage:
        @pytest.fixture
        def llm(mock_llm_client):
            mock_llm_client.set_response('{"answer": "42"}')
            return mock_llm_client
    """

    def __init__(self):
        super().__init__(model="mock")
        self._responses = []
        self._call_history = []

    def set_response(self, response: str) -> None:
        """Queue a response for the next complete() call."""
        self._responses.append(response)

    def set_responses(self, responses: list[str]) -> None:
        """Queue multiple responses."""
        self._responses.extend(responses)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return queued response or raise if none available."""
        self._call_history.append({"prompt": prompt, "kwargs": kwargs})

        if not self._responses:
            raise RuntimeError(
                "MockLLMClient has no queued responses. "
                "Use set_response() or set_responses() first."
            )

        response = self._responses.pop(0)
        return LLMResponse(text=response)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Mock structured output - parses JSON and validates with Pydantic.

        This prevents roles from auto-wrapping with real Instructor.
        """
        self._call_history.append(
            {"prompt": prompt, "response_model": response_model, "kwargs": kwargs}
        )

        if not self._responses:
            raise RuntimeError(
                "MockLLMClient has no queued responses. "
                "Use set_response() or set_responses() first."
            )

        response = self._responses.pop(0)

        # Parse JSON and validate with Pydantic model
        data = json.loads(response)
        return response_model.model_validate(data)

    @property
    def call_history(self) -> list[Dict[str, Any]]:
        """Get history of all complete() calls."""
        return self._call_history

    def reset(self) -> None:
        """Clear all responses and history."""
        self._responses = []
        self._call_history = []


@pytest.fixture
def mock_llm_client():
    """
    Provides a fresh MockLLMClient for each test.

    Example:
        def test_something(mock_llm_client):
            mock_llm_client.set_response('{"result": "success"}')
            # use in your test
    """
    return MockLLMClient()


@pytest.fixture
def empty_playbook():
    """Provides an empty Playbook instance."""
    return Playbook()


@pytest.fixture
def sample_playbook():
    """
    Provides a Playbook with sample bullets for testing.

    Sections:
        - general: 2 bullets
        - math: 2 bullets
        - reasoning: 1 bullet
    """
    playbook = Playbook()

    # General section
    playbook.add_bullet(
        "general",
        "Be clear and concise in your answers",
        metadata={"helpful": 5, "harmful": 0},
    )
    playbook.add_bullet(
        "general",
        "Always provide context and examples",
        metadata={"helpful": 3, "harmful": 0},
    )

    # Math section
    playbook.add_bullet(
        "math", "Show your work step by step", metadata={"helpful": 8, "harmful": 0}
    )
    playbook.add_bullet(
        "math",
        "Verify calculations before presenting final answer",
        metadata={"helpful": 6, "harmful": 0},
    )

    # Reasoning section
    playbook.add_bullet(
        "reasoning",
        "Break complex problems into smaller parts",
        metadata={"helpful": 7, "harmful": 0},
    )

    return playbook


@pytest.fixture
def sample_question():
    """Provides a sample question for testing."""
    return "What is 2 + 2?"


@pytest.fixture
def sample_context():
    """Provides sample context text for testing."""
    return "This is a simple arithmetic problem."


@pytest.fixture
def sample_correct_answer():
    """Provides a sample correct answer."""
    return "4"


@pytest.fixture
def sample_incorrect_answer():
    """Provides a sample incorrect answer for testing error paths."""
    return "5"


@pytest.fixture
def sample_training_sample(sample_question, sample_correct_answer):
    """
    Provides a Sample instance for training/testing.

    Can be used with OfflineAdapter or OnlineAdapter.
    """
    return Sample(
        question=sample_question,
        response=sample_correct_answer,
        context="",
        metadata={"type": "test"},
    )


@pytest.fixture
def generator_valid_json():
    """Provides valid JSON response for Generator."""
    return json.dumps(
        {
            "reasoning": "This is a simple addition problem.",
            "final_answer": "4",
            "bullet_ids": [],
        }
    )


@pytest.fixture
def reflector_valid_json():
    """Provides valid JSON response for Reflector with helpful tags."""
    return json.dumps(
        {
            "analysis": "The answer is correct.",
            "bullet_tags": [
                {"bullet_id": "test_id_1", "tag": "helpful"},
                {"bullet_id": "test_id_2", "tag": "neutral"},
            ],
        }
    )


@pytest.fixture
def curator_add_operation_json():
    """Provides valid JSON for Curator ADD operation."""
    return json.dumps(
        {
            "reasoning": "Need to add a new strategy.",
            "delta_batch": {
                "operations": [
                    {
                        "type": "ADD",
                        "section": "general",
                        "content": "New helpful strategy",
                    }
                ]
            },
        }
    )


@pytest.fixture
def curator_tag_operation_json():
    """Provides valid JSON for Curator TAG operation."""
    return json.dumps(
        {
            "reasoning": "Updating bullet statistics.",
            "delta_batch": {
                "operations": [
                    {
                        "type": "TAG",
                        "section": "general",
                        "bullet_id": "test_bullet_id",
                        "metadata": {"helpful": 1},
                    }
                ]
            },
        }
    )


# Test markers configuration
pytest_configure_done = False


def pytest_configure(config):
    """Register custom markers."""
    global pytest_configure_done
    if not pytest_configure_done:
        config.addinivalue_line(
            "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
        )
        config.addinivalue_line(
            "markers", "integration: marks tests as integration tests"
        )
        config.addinivalue_line("markers", "unit: marks tests as unit tests")
        config.addinivalue_line(
            "markers",
            "requires_api: marks tests requiring external API keys (skipped in CI)",
        )
        pytest_configure_done = True
