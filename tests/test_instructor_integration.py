"""Integration tests for Instructor-based structured output validation.

This module tests the Instructor integration that provides automatic Pydantic
validation and intelligent retry for Generator, Reflector, and Curator roles.
"""

import unittest
from typing import Any, Type, TypeVar
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from ace import Generator, Reflector, Curator, Playbook
from ace.llm import LLMClient, LLMResponse
from ace.roles import GeneratorOutput, ReflectorOutput, CuratorOutput, BulletTag
from ace.delta import DeltaBatch

T = TypeVar("T", bound=BaseModel)


class MockInstructorClient(LLMClient):
    """
    Mock LLM client that simulates Instructor's complete_structured method.

    This allows testing the Instructor code path without actual API calls.
    """

    def __init__(self, model: str = "mock-instructor"):
        super().__init__(model=model)
        self._structured_responses = []
        self._call_history = []
        self._should_fail_validation = False
        self._validation_retries = 0

    def set_structured_response(self, response: BaseModel) -> None:
        """Queue a structured response (Pydantic model instance)."""
        self._structured_responses.append(response)

    def set_validation_failure(self, retries_before_success: int = 0) -> None:
        """
        Simulate validation failures.

        Args:
            retries_before_success: Number of retries before returning valid response
        """
        self._should_fail_validation = True
        self._validation_retries = retries_before_success

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Standard completion (not used with Instructor)."""
        raise NotImplementedError("MockInstructorClient should use complete_structured")

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        Mock implementation of Instructor's structured output completion.

        Args:
            prompt: The prompt sent to the LLM
            response_model: Pydantic model class to validate against
            **kwargs: Additional parameters

        Returns:
            Instance of response_model with validated data

        Raises:
            ValidationError: If validation fails after retries
        """
        self._call_history.append(
            {"prompt": prompt, "response_model": response_model, "kwargs": kwargs}
        )

        # Simulate validation retry logic
        if self._should_fail_validation and self._validation_retries > 0:
            self._validation_retries -= 1
            raise ValidationError.from_exception_data(
                "Mock validation error",
                [
                    {
                        "type": "mock_error",
                        "loc": ("field",),
                        "msg": "Mock validation failure",
                    }
                ],
            )

        # Return queued structured response
        if not self._structured_responses:
            raise RuntimeError(
                "MockInstructorClient has no queued structured responses. "
                "Use set_structured_response() first."
            )

        response = self._structured_responses.pop(0)

        # Validate it matches the expected response_model type
        if not isinstance(response, response_model):
            raise TypeError(
                f"Queued response is {type(response).__name__}, "
                f"but expected {response_model.__name__}"
            )

        return response

    @property
    def call_history(self):
        """Get history of all complete_structured() calls."""
        return self._call_history

    def reset(self):
        """Clear all responses and history."""
        self._structured_responses = []
        self._call_history = []
        self._should_fail_validation = False
        self._validation_retries = 0


@pytest.mark.unit
class TestInstructorIntegrationGenerator(unittest.TestCase):
    """Test Generator with Instructor client."""

    def setUp(self):
        """Set up test fixtures."""
        self.instructor_llm = MockInstructorClient()
        self.generator = Generator(self.instructor_llm)
        self.playbook = Playbook()

    def test_generator_uses_complete_structured(self):
        """Test that Generator uses complete_structured when available."""
        # Queue a valid GeneratorOutput
        expected_output = GeneratorOutput(
            reasoning="Step 1: Add 2 + 2 = 4", final_answer="4", bullet_ids=[], raw={}
        )
        self.instructor_llm.set_structured_response(expected_output)

        # Generate
        result = self.generator.generate(
            question="What is 2+2?", context="Show your work", playbook=self.playbook
        )

        # Verify result
        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.reasoning, "Step 1: Add 2 + 2 = 4")

        # Verify complete_structured was called
        self.assertEqual(len(self.instructor_llm.call_history), 1)
        call = self.instructor_llm.call_history[0]
        self.assertEqual(call["response_model"], GeneratorOutput)

    def test_generator_extracts_bullet_ids_from_reasoning(self):
        """Test that Generator extracts bullet IDs even with Instructor."""
        # Create output with bullet citations in reasoning
        output_with_citations = GeneratorOutput(
            reasoning="Following [general-00001] and [math-00042], I calculated 2+2=4",
            final_answer="4",
            bullet_ids=[],  # Empty initially
            raw={},
        )
        self.instructor_llm.set_structured_response(output_with_citations)

        result = self.generator.generate(
            question="What is 2+2?", context="", playbook=self.playbook
        )

        # Verify bullet_ids were extracted
        self.assertEqual(result.bullet_ids, ["general-00001", "math-00042"])

    def test_generator_pydantic_validation(self):
        """Test that Pydantic validates required fields."""
        # This test verifies the output model structure
        with self.assertRaises(ValidationError):
            GeneratorOutput(
                # Missing required fields: reasoning, final_answer
                bullet_ids=[],
                raw={},
            )

    def test_generator_with_playbook_context(self):
        """Test Generator includes playbook in prompt when using Instructor."""
        self.playbook.add_bullet("math", "Show your work step by step")

        expected_output = GeneratorOutput(
            reasoning="Following the playbook, I'll show my work...",
            final_answer="4",
            bullet_ids=[],
            raw={},
        )
        self.instructor_llm.set_structured_response(expected_output)

        result = self.generator.generate(
            question="What is 2+2?", context="", playbook=self.playbook
        )

        # Verify the prompt included playbook
        call = self.instructor_llm.call_history[0]
        self.assertIn("math", call["prompt"])


@pytest.mark.unit
class TestInstructorIntegrationReflector(unittest.TestCase):
    """Test Reflector with Instructor client."""

    def setUp(self):
        """Set up test fixtures."""
        self.instructor_llm = MockInstructorClient()
        self.reflector = Reflector(self.instructor_llm)
        self.playbook = Playbook()
        self.generator_output = GeneratorOutput(
            reasoning="2 + 2 = 4", final_answer="4", bullet_ids=[], raw={}
        )

    def test_reflector_uses_complete_structured(self):
        """Test that Reflector uses complete_structured when available."""
        expected_output = ReflectorOutput(
            reasoning="The answer is correct",
            error_identification="",
            root_cause_analysis="",
            correct_approach="The approach was sound",
            key_insight="Addition was performed correctly",
            bullet_tags=[],
            raw={},
        )
        self.instructor_llm.set_structured_response(expected_output)

        result = self.reflector.reflect(
            question="What is 2+2?",
            generator_output=self.generator_output,
            playbook=self.playbook,
            ground_truth="4",
            feedback="Correct!",
        )

        # Verify result
        self.assertEqual(result.key_insight, "Addition was performed correctly")

        # Verify complete_structured was called
        self.assertEqual(len(self.instructor_llm.call_history), 1)
        call = self.instructor_llm.call_history[0]
        self.assertEqual(call["response_model"], ReflectorOutput)

    def test_reflector_bullet_tagging(self):
        """Test that Reflector can tag bullets as helpful/harmful."""
        # Add a bullet and get its auto-generated ID
        bullet = self.playbook.add_bullet("math", "Show your work")
        bullet_id = bullet.id

        expected_output = ReflectorOutput(
            reasoning="The strategy was effective",
            error_identification="",
            root_cause_analysis="",
            correct_approach="Continue using this approach",
            key_insight="Showing work helps avoid errors",
            bullet_tags=[BulletTag(id=bullet_id, tag="helpful")],
            raw={},
        )
        self.instructor_llm.set_structured_response(expected_output)

        result = self.reflector.reflect(
            question="What is 2+2?",
            generator_output=self.generator_output,
            playbook=self.playbook,
            ground_truth="4",
        )

        # Verify bullet tags
        self.assertEqual(len(result.bullet_tags), 1)
        self.assertEqual(result.bullet_tags[0].id, bullet_id)
        self.assertEqual(result.bullet_tags[0].tag, "helpful")

    def test_reflector_pydantic_validation(self):
        """Test that Pydantic validates required fields in ReflectorOutput."""
        with self.assertRaises(ValidationError):
            ReflectorOutput(
                # Missing required fields: reasoning, correct_approach, key_insight
                error_identification="",
                bullet_tags=[],
            )


@pytest.mark.unit
class TestInstructorIntegrationCurator(unittest.TestCase):
    """Test Curator with Instructor client."""

    def setUp(self):
        """Set up test fixtures."""
        self.instructor_llm = MockInstructorClient()
        self.curator = Curator(self.instructor_llm)
        self.playbook = Playbook()
        self.reflection = ReflectorOutput(
            reasoning="The approach was good",
            error_identification="",
            root_cause_analysis="",
            correct_approach="Keep doing this",
            key_insight="This strategy works well",
            bullet_tags=[],
            raw={},
        )

    def test_curator_uses_complete_structured(self):
        """Test that Curator uses complete_structured when available."""
        delta_batch = DeltaBatch.from_json(
            {
                "reasoning": "Adding a new strategy",
                "operations": [
                    {"type": "ADD", "section": "general", "content": "Be concise"}
                ],
            }
        )

        expected_output = CuratorOutput(delta=delta_batch, raw={})
        self.instructor_llm.set_structured_response(expected_output)

        result = self.curator.curate(
            reflection=self.reflection,
            playbook=self.playbook,
            question_context="Math problems",
            progress="1/10 correct",
        )

        # Verify result
        self.assertEqual(len(result.delta.operations), 1)
        self.assertEqual(result.delta.operations[0].type, "ADD")

        # Verify complete_structured was called
        self.assertEqual(len(self.instructor_llm.call_history), 1)
        call = self.instructor_llm.call_history[0]
        self.assertEqual(call["response_model"], CuratorOutput)

    def test_curator_multiple_operations(self):
        """Test Curator with multiple delta operations."""
        delta_batch = DeltaBatch.from_json(
            {
                "reasoning": "Multiple updates needed",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "math",
                        "content": "Double-check calculations",
                    },
                    {
                        "type": "TAG",
                        "bullet_id": "math-00001",
                        "metadata": {"helpful": 1},
                    },
                ],
            }
        )

        expected_output = CuratorOutput(delta=delta_batch, raw={})
        self.instructor_llm.set_structured_response(expected_output)

        result = self.curator.curate(
            reflection=self.reflection,
            playbook=self.playbook,
            question_context="Math",
            progress="5/10",
        )

        # Verify operations
        self.assertEqual(len(result.delta.operations), 2)
        self.assertEqual(result.delta.operations[0].type, "ADD")
        self.assertEqual(result.delta.operations[1].type, "TAG")

    def test_curator_pydantic_validation(self):
        """Test that Pydantic validates CuratorOutput structure."""
        with self.assertRaises(ValidationError):
            CuratorOutput(
                # Missing required field: delta
                raw={}
            )


@pytest.mark.unit
class TestInstructorDetection(unittest.TestCase):
    """Test detection of Instructor capabilities."""

    def test_hasattr_detection(self):
        """Test that hasattr correctly detects complete_structured."""
        instructor_client = MockInstructorClient()
        non_instructor_client = Mock(spec=LLMClient)

        # Instructor client should have the method
        self.assertTrue(hasattr(instructor_client, "complete_structured"))

        # Regular LLM client should not
        self.assertFalse(hasattr(non_instructor_client, "complete_structured"))

    def test_pydantic_models_are_serializable(self):
        """Test that output models can be serialized/deserialized."""
        # GeneratorOutput
        gen_output = GeneratorOutput(
            reasoning="test",
            final_answer="42",
            bullet_ids=["id-1"],
            raw={"key": "value"},
        )
        json_str = gen_output.model_dump_json()
        self.assertIn("test", json_str)
        self.assertIn("42", json_str)

        # ReflectorOutput
        ref_output = ReflectorOutput(
            reasoning="test",
            error_identification="none",
            root_cause_analysis="n/a",
            correct_approach="good",
            key_insight="works",
            bullet_tags=[BulletTag(id="test-1", tag="helpful")],
            raw={},
        )
        json_str = ref_output.model_dump_json()
        self.assertIn("works", json_str)

        # CuratorOutput
        delta = DeltaBatch.from_json({"reasoning": "test", "operations": []})
        cur_output = CuratorOutput(delta=delta, raw={})
        json_str = cur_output.model_dump_json()
        self.assertIn("operations", json_str)


@pytest.mark.unit
class TestInstructorClaudeParameterResolution(unittest.TestCase):
    """
    REGRESSION TESTS: Ensure InstructorClient applies Claude parameter resolution.

    Previously, InstructorClient bypassed the _resolve_sampling_params() function
    from LiteLLMClient, causing both temperature and top_p to be sent to Claude models.
    This resulted in Anthropic API errors:
    "temperature and top_p cannot both be specified for this model"

    This broke ACE learning (Reflector/Curator failed, 0 strategies learned).
    Fixed in commit 9740603.
    """

    @patch("ace.llm_providers.instructor_client.completion")
    def test_instructor_client_applies_claude_parameter_resolution(
        self, mock_completion
    ):
        """
        Test that InstructorClient uses _resolve_sampling_params for Claude models.

        This is the core regression test - if this fails, Claude learning will break.
        """
        from ace.llm_providers.litellm_client import LiteLLMClient, LiteLLMConfig
        from ace.llm_providers.instructor_client import InstructorClient
        from ace.roles import GeneratorOutput
        import instructor

        # Create a LiteLLM client with Claude model
        config = LiteLLMConfig(
            model="claude-3-sonnet-20240229",
            temperature=0.0,
            top_p=0.9,  # Explicitly set to simulate old bug
        )
        base_llm = LiteLLMClient(config=config)
        instructor_client = InstructorClient(base_llm)

        # Mock the response
        mock_response = GeneratorOutput(
            reasoning="Test",
            final_answer="42",
            bullet_ids=[],
            raw={},
        )

        # Patch instructor's create to capture call params
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return mock_response

        with patch.object(
            instructor_client.client.chat.completions,
            "create",
            side_effect=capture_create,
        ):
            try:
                instructor_client.complete_structured(
                    prompt="Test prompt",
                    response_model=GeneratorOutput,
                )
            except Exception:
                pass  # We just want to capture the params

        # THE KEY ASSERTION: For Claude, should NOT have both temperature and top_p
        has_temperature = "temperature" in captured_params
        has_top_p = "top_p" in captured_params

        # Either temperature OR top_p, but not both
        self.assertFalse(
            has_temperature and has_top_p,
            f"InstructorClient sent BOTH temperature and top_p to Claude model! "
            f"This causes Anthropic API errors. "
            f"Captured params: {captured_params}",
        )

    def test_resolve_sampling_params_is_called_for_claude(self):
        """Test that _resolve_sampling_params is actually called in InstructorClient."""
        from ace.llm_providers.litellm_client import LiteLLMClient, LiteLLMConfig
        from ace.llm_providers.instructor_client import InstructorClient
        from ace.roles import GeneratorOutput

        # Create a LiteLLM client with Claude model
        config = LiteLLMConfig(
            model="claude-haiku-4-5-20251001",
            temperature=0.7,
            top_p=0.9,
        )
        base_llm = LiteLLMClient(config=config)
        instructor_client = InstructorClient(base_llm)

        # Patch _resolve_sampling_params to track if it's called
        with patch.object(
            LiteLLMClient,
            "_resolve_sampling_params",
            wraps=LiteLLMClient._resolve_sampling_params,
        ) as mock_resolve:
            # Also patch the actual completion to prevent API call
            with patch.object(instructor_client.client.chat.completions, "create"):
                try:
                    instructor_client.complete_structured(
                        prompt="Test",
                        response_model=GeneratorOutput,
                    )
                except Exception:
                    pass  # We just want to verify _resolve_sampling_params was called

            # Verify it was called for Claude model
            mock_resolve.assert_called_once()
            call_args = mock_resolve.call_args
            self.assertIn(
                "claude", call_args[0][1].lower()
            )  # model name should contain "claude"
