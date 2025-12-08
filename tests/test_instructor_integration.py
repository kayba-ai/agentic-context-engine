"""Integration tests for Instructor-based structured output validation.

This module tests the Instructor integration that provides automatic Pydantic
validation and intelligent retry for Agent, Reflector, and SkillManager roles.
"""

import unittest
from typing import Any, Type, TypeVar
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from ace import Agent, Reflector, SkillManager, Skillbook
from ace.llm import LLMClient, LLMResponse
from ace.roles import AgentOutput, ReflectorOutput, SkillManagerOutput, SkillTag
from ace.updates import UpdateBatch

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
class TestInstructorIntegrationAgent(unittest.TestCase):
    """Test Agent with Instructor client."""

    def setUp(self):
        """Set up test fixtures."""
        self.instructor_llm = MockInstructorClient()
        self.agent = Agent(self.instructor_llm)
        self.skillbook = Skillbook()

    def test_agent_uses_complete_structured(self):
        """Test that Agent uses complete_structured when available."""
        # Queue a valid AgentOutput
        expected_output = AgentOutput(
            reasoning="Step 1: Add 2 + 2 = 4", final_answer="4", skill_ids=[], raw={}
        )
        self.instructor_llm.set_structured_response(expected_output)

        # Generate
        result = self.agent.generate(
            question="What is 2+2?", context="Show your work", skillbook=self.skillbook
        )

        # Verify result
        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.reasoning, "Step 1: Add 2 + 2 = 4")

        # Verify complete_structured was called
        self.assertEqual(len(self.instructor_llm.call_history), 1)
        call = self.instructor_llm.call_history[0]
        self.assertEqual(call["response_model"], AgentOutput)

    def test_agent_extracts_skill_ids_from_reasoning(self):
        """Test that Agent extracts skill IDs even with Instructor."""
        # Create output with skill citations in reasoning
        output_with_citations = AgentOutput(
            reasoning="Following [general-00001] and [math-00042], I calculated 2+2=4",
            final_answer="4",
            skill_ids=[],  # Empty initially
            raw={},
        )
        self.instructor_llm.set_structured_response(output_with_citations)

        result = self.agent.generate(
            question="What is 2+2?", context="", skillbook=self.skillbook
        )

        # Verify skill_ids were extracted
        self.assertEqual(result.skill_ids, ["general-00001", "math-00042"])

    def test_agent_pydantic_validation(self):
        """Test that Pydantic validates required fields."""
        # This test verifies the output model structure
        with self.assertRaises(ValidationError):
            AgentOutput(
                # Missing required fields: reasoning, final_answer
                skill_ids=[],
                raw={},
            )

    def test_agent_with_skillbook_context(self):
        """Test Agent includes skillbook in prompt when using Instructor."""
        self.skillbook.add_skill("math", "Show your work step by step")

        expected_output = AgentOutput(
            reasoning="Following the skillbook, I'll show my work...",
            final_answer="4",
            skill_ids=[],
            raw={},
        )
        self.instructor_llm.set_structured_response(expected_output)

        result = self.agent.generate(
            question="What is 2+2?", context="", skillbook=self.skillbook
        )

        # Verify the prompt included skillbook
        call = self.instructor_llm.call_history[0]
        self.assertIn("math", call["prompt"])


@pytest.mark.unit
class TestInstructorIntegrationReflector(unittest.TestCase):
    """Test Reflector with Instructor client."""

    def setUp(self):
        """Set up test fixtures."""
        self.instructor_llm = MockInstructorClient()
        self.reflector = Reflector(self.instructor_llm)
        self.skillbook = Skillbook()
        self.agent_output = AgentOutput(
            reasoning="2 + 2 = 4", final_answer="4", skill_ids=[], raw={}
        )

    def test_reflector_uses_complete_structured(self):
        """Test that Reflector uses complete_structured when available."""
        expected_output = ReflectorOutput(
            reasoning="The answer is correct",
            error_identification="",
            root_cause_analysis="",
            correct_approach="The approach was sound",
            key_insight="Addition was performed correctly",
            skill_tags=[],
            raw={},
        )
        self.instructor_llm.set_structured_response(expected_output)

        result = self.reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        # Verify result
        self.assertEqual(result.key_insight, "Addition was performed correctly")

        # Verify complete_structured was called
        self.assertEqual(len(self.instructor_llm.call_history), 1)
        call = self.instructor_llm.call_history[0]
        self.assertEqual(call["response_model"], ReflectorOutput)

    def test_reflector_skill_tagging(self):
        """Test that Reflector can tag skills as helpful/harmful."""
        # Add a skill and get its auto-generated ID
        skill = self.skillbook.add_skill("math", "Show your work")
        skill_id = skill.id

        expected_output = ReflectorOutput(
            reasoning="The strategy was effective",
            error_identification="",
            root_cause_analysis="",
            correct_approach="Continue using this approach",
            key_insight="Showing work helps avoid errors",
            skill_tags=[SkillTag(id=skill_id, tag="helpful")],
            raw={},
        )
        self.instructor_llm.set_structured_response(expected_output)

        result = self.reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
        )

        # Verify skill tags
        self.assertEqual(len(result.skill_tags), 1)
        self.assertEqual(result.skill_tags[0].id, skill_id)
        self.assertEqual(result.skill_tags[0].tag, "helpful")

    def test_reflector_pydantic_validation(self):
        """Test that Pydantic validates required fields in ReflectorOutput."""
        with self.assertRaises(ValidationError):
            ReflectorOutput(
                # Missing required fields: reasoning, correct_approach, key_insight
                error_identification="",
                skill_tags=[],
            )


@pytest.mark.unit
class TestInstructorIntegrationSkillManager(unittest.TestCase):
    """Test SkillManager with Instructor client."""

    def setUp(self):
        """Set up test fixtures."""
        self.instructor_llm = MockInstructorClient()
        self.skill_manager = SkillManager(self.instructor_llm)
        self.skillbook = Skillbook()
        self.reflection = ReflectorOutput(
            reasoning="The approach was good",
            error_identification="",
            root_cause_analysis="",
            correct_approach="Keep doing this",
            key_insight="This strategy works well",
            skill_tags=[],
            raw={},
        )

    def test_skill_manager_uses_complete_structured(self):
        """Test that SkillManager uses complete_structured when available."""
        update_batch = UpdateBatch.from_json(
            {
                "reasoning": "Adding a new strategy",
                "operations": [
                    {"type": "ADD", "section": "general", "content": "Be concise"}
                ],
            }
        )

        expected_output = SkillManagerOutput(update=update_batch, raw={})
        self.instructor_llm.set_structured_response(expected_output)

        result = self.skill_manager.update_skills(
            reflection=self.reflection,
            skillbook=self.skillbook,
            question_context="Math problems",
            progress="1/10 correct",
        )

        # Verify result
        self.assertEqual(len(result.update.operations), 1)
        self.assertEqual(result.update.operations[0].type, "ADD")

        # Verify complete_structured was called
        self.assertEqual(len(self.instructor_llm.call_history), 1)
        call = self.instructor_llm.call_history[0]
        self.assertEqual(call["response_model"], SkillManagerOutput)

    def test_skill_manager_multiple_operations(self):
        """Test SkillManager with multiple update operations."""
        update_batch = UpdateBatch.from_json(
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
                        "skill_id": "math-00001",
                        "metadata": {"helpful": 1},
                    },
                ],
            }
        )

        expected_output = SkillManagerOutput(update=update_batch, raw={})
        self.instructor_llm.set_structured_response(expected_output)

        result = self.skill_manager.update_skills(
            reflection=self.reflection,
            skillbook=self.skillbook,
            question_context="Math",
            progress="5/10",
        )

        # Verify operations
        self.assertEqual(len(result.update.operations), 2)
        self.assertEqual(result.update.operations[0].type, "ADD")
        self.assertEqual(result.update.operations[1].type, "TAG")

    def test_skill_manager_pydantic_validation(self):
        """Test that Pydantic validates SkillManagerOutput structure."""
        with self.assertRaises(ValidationError):
            SkillManagerOutput(
                # Missing required field: update
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
        # AgentOutput
        gen_output = AgentOutput(
            reasoning="test",
            final_answer="42",
            skill_ids=["id-1"],
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
            skill_tags=[SkillTag(id="test-1", tag="helpful")],
            raw={},
        )
        json_str = ref_output.model_dump_json()
        self.assertIn("works", json_str)

        # SkillManagerOutput
        update = UpdateBatch.from_json({"reasoning": "test", "operations": []})
        sm_output = SkillManagerOutput(update=update, raw={})
        json_str = sm_output.model_dump_json()
        self.assertIn("operations", json_str)


@pytest.mark.unit
class TestInstructorClaudeParameterResolution(unittest.TestCase):
    """
    REGRESSION TESTS: Ensure InstructorClient applies Claude parameter resolution.

    Previously, InstructorClient bypassed the _resolve_sampling_params() function
    from LiteLLMClient, causing both temperature and top_p to be sent to Claude models.
    This resulted in Anthropic API errors:
    "temperature and top_p cannot both be specified for this model"

    This broke ACE learning (Reflector/SkillManager failed, 0 strategies learned).
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
        from ace.roles import AgentOutput
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
        mock_response = AgentOutput(
            reasoning="Test",
            final_answer="42",
            skill_ids=[],
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
                    response_model=AgentOutput,
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
        from ace.roles import AgentOutput

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
                        response_model=AgentOutput,
                    )
                except Exception:
                    pass  # We just want to verify _resolve_sampling_params was called

            # Verify it was called for Claude model
            mock_resolve.assert_called_once()
            call_args = mock_resolve.call_args
            self.assertIn(
                "claude", call_args[0][1].lower()
            )  # model name should contain "claude"


@pytest.mark.unit
class TestInstructorClientCredentialForwarding(unittest.TestCase):
    """
    REGRESSION TESTS: Ensure InstructorClient forwards authentication parameters.

    GitHub Issue #44: When users pass api_key and base_url to ACELiteLLM,
    these credentials must be forwarded to the Instructor client.

    Previously, InstructorClient.complete_structured() extracted model/temperature/max_tokens
    from the wrapped LLM's config but NEVER extracted api_key, api_base, extra_headers,
    or ssl_verify. This caused AuthenticationError for custom OpenAI-compatible endpoints.
    """

    @patch("ace.llm_providers.instructor_client.completion")
    def test_instructor_forwards_api_key(self, mock_completion):
        """Test that InstructorClient forwards api_key to LiteLLM calls."""
        from ace.llm_providers.litellm_client import LiteLLMClient
        from ace.llm_providers.instructor_client import InstructorClient

        base_llm = LiteLLMClient(model="openai/custom-model", api_key="sk-test-key-123")
        instructor_client = InstructorClient(base_llm)

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return AgentOutput(
                reasoning="Test", final_answer="42", skill_ids=[], raw={}
            )

        with patch.object(
            instructor_client.client.chat.completions,
            "create",
            side_effect=capture_create,
        ):
            try:
                instructor_client.complete_structured(
                    prompt="Test", response_model=AgentOutput
                )
            except Exception:
                pass

        self.assertEqual(
            captured_params.get("api_key"),
            "sk-test-key-123",
            "InstructorClient must forward api_key from wrapped LLM config",
        )

    @patch("ace.llm_providers.instructor_client.completion")
    def test_instructor_forwards_api_base(self, mock_completion):
        """Test that InstructorClient forwards api_base to LiteLLM calls."""
        from ace.llm_providers.litellm_client import LiteLLMClient
        from ace.llm_providers.instructor_client import InstructorClient

        base_llm = LiteLLMClient(
            model="openai/custom-model",
            api_key="sk-test",
            api_base="https://custom.endpoint.com/v1",
        )
        instructor_client = InstructorClient(base_llm)

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return AgentOutput(
                reasoning="Test", final_answer="42", skill_ids=[], raw={}
            )

        with patch.object(
            instructor_client.client.chat.completions,
            "create",
            side_effect=capture_create,
        ):
            try:
                instructor_client.complete_structured(
                    prompt="Test", response_model=AgentOutput
                )
            except Exception:
                pass

        self.assertEqual(
            captured_params.get("api_base"),
            "https://custom.endpoint.com/v1",
            "InstructorClient must forward api_base from wrapped LLM config",
        )

    @patch("ace.llm_providers.instructor_client.completion")
    def test_instructor_forwards_extra_headers(self, mock_completion):
        """Test that InstructorClient forwards extra_headers to LiteLLM calls."""
        from ace.llm_providers.litellm_client import LiteLLMClient
        from ace.llm_providers.instructor_client import InstructorClient

        headers = {"X-Custom-Header": "custom-value"}
        base_llm = LiteLLMClient(model="gpt-4", extra_headers=headers)
        instructor_client = InstructorClient(base_llm)

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return AgentOutput(
                reasoning="Test", final_answer="42", skill_ids=[], raw={}
            )

        with patch.object(
            instructor_client.client.chat.completions,
            "create",
            side_effect=capture_create,
        ):
            try:
                instructor_client.complete_structured(
                    prompt="Test", response_model=AgentOutput
                )
            except Exception:
                pass

        self.assertEqual(
            captured_params.get("extra_headers"),
            headers,
            "InstructorClient must forward extra_headers from wrapped LLM config",
        )

    @patch("ace.llm_providers.instructor_client.completion")
    def test_instructor_forwards_ssl_verify(self, mock_completion):
        """Test that InstructorClient forwards ssl_verify to LiteLLM calls."""
        from ace.llm_providers.litellm_client import LiteLLMClient
        from ace.llm_providers.instructor_client import InstructorClient

        base_llm = LiteLLMClient(model="gpt-4", ssl_verify=False)
        instructor_client = InstructorClient(base_llm)

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            return AgentOutput(
                reasoning="Test", final_answer="42", skill_ids=[], raw={}
            )

        with patch.object(
            instructor_client.client.chat.completions,
            "create",
            side_effect=capture_create,
        ):
            try:
                instructor_client.complete_structured(
                    prompt="Test", response_model=AgentOutput
                )
            except Exception:
                pass

        self.assertEqual(
            captured_params.get("ssl_verify"),
            False,
            "InstructorClient must forward ssl_verify from wrapped LLM config",
        )
