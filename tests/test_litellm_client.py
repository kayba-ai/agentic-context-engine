"""Basic tests for LiteLLM client integration."""

import unittest
from unittest.mock import patch, MagicMock
import logging

import pytest


@pytest.mark.unit
class TestLiteLLMClient(unittest.IsolatedAsyncioTestCase):
    """Test LiteLLM client functionality."""

    def test_import(self):
        """Test that LiteLLM client can be imported."""
        try:
            from ace.llm_providers import LiteLLMClient

            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import LiteLLMClient")

    @patch("ace.llm_providers.litellm_client.completion")
    def test_basic_completion(self, mock_completion):
        """Test basic completion functionality."""
        from ace.llm_providers import LiteLLMClient

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.model = "gpt-3.5-turbo"
        mock_completion.return_value = mock_response

        # Create client and test
        client = LiteLLMClient(model="gpt-3.5-turbo")
        response = client.complete("Test prompt")

        self.assertEqual(response.text, "Test response")
        self.assertIn("usage", response.raw)

    def test_parameter_filtering(self):
        """Test that ACE-specific parameters are filtered."""
        from ace.llm_providers import LiteLLMClient

        with patch("ace.llm_providers.litellm_client.completion") as mock:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test"))]
            mock_response.usage = None
            mock_response.model = "test"
            mock.return_value = mock_response

            client = LiteLLMClient(model="test")
            client.complete("Test", refinement_round=1, max_refinement_rounds=3)

            # Check that filtered params aren't in the call
            call_kwargs = mock.call_args[1]
            self.assertNotIn("refinement_round", call_kwargs)
            self.assertNotIn("max_refinement_rounds", call_kwargs)


class TestClaudeParameterResolution(unittest.IsolatedAsyncioTestCase):
    """Test Claude-specific parameter resolution functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from ace.llm_providers import LiteLLMClient

        self.mock_response = MagicMock()
        self.mock_response.choices = [
            MagicMock(message=MagicMock(content="Test response"))
        ]
        self.mock_response.usage = None
        self.mock_response.model = "claude-3-sonnet-20240229"

    @patch("ace.llm_providers.litellm_client.completion")
    def test_claude_temperature_priority_default(self, mock_completion):
        """Test default temperature priority for Claude models."""
        from ace.llm_providers import LiteLLMClient

        mock_completion.return_value = self.mock_response

        client = LiteLLMClient(model="claude-3-sonnet-20240229", temperature=0.7)
        client.complete("Test prompt", top_p=0.9)

        # Should exclude top_p when temperature is present (default priority)
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertNotIn("top_p", call_kwargs)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_claude_fallback_priority_with_default_temperature(self, mock_completion):
        """Test fallback priority when temperature is 0 and top_p provided."""
        from ace.llm_providers import LiteLLMClient

        mock_completion.return_value = self.mock_response

        # Temperature is 0 (default), but top_p is provided
        client = LiteLLMClient(model="claude-3-sonnet-20240229")
        client.complete("Test prompt", top_p=0.9)

        # With fallback logic: temperature=0 should allow top_p
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["top_p"], 0.9)
        self.assertNotIn("temperature", call_kwargs)  # Should be removed in fallback

    @patch("ace.llm_providers.litellm_client.completion")
    def test_claude_prefer_top_p_strategy(self, mock_completion):
        """Test prefer_top_p strategy for Claude models."""
        from ace.llm_providers import LiteLLMClient, LiteLLMConfig

        mock_completion.return_value = self.mock_response

        config = LiteLLMConfig(
            model="claude-3-sonnet-20240229", temperature=0.7, sampling_priority="top_p"
        )
        client = LiteLLMClient(config=config)
        client.complete("Test prompt", top_p=0.9)

        # Should exclude temperature when top_p priority is set
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["top_p"], 0.9)
        self.assertNotIn("temperature", call_kwargs)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_claude_prefer_top_k_strategy(self, mock_completion):
        """Test prefer_top_k strategy for Claude models."""
        from ace.llm_providers import LiteLLMClient, LiteLLMConfig

        mock_completion.return_value = self.mock_response

        config = LiteLLMConfig(
            model="claude-3-sonnet-20240229", temperature=0.7, sampling_priority="top_k"
        )
        client = LiteLLMClient(config=config)
        client.complete("Test prompt", top_p=0.9, top_k=50)

        # Should exclude temperature and top_p when top_k priority is set
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 50)
        self.assertNotIn("temperature", call_kwargs)
        self.assertNotIn("top_p", call_kwargs)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_non_claude_model_no_filtering(self, mock_completion):
        """Test that non-Claude models don't get parameter filtering."""
        from ace.llm_providers import LiteLLMClient

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_completion.return_value = mock_response

        client = LiteLLMClient(model="gpt-4", temperature=0.7)
        client.complete("Test prompt", top_p=0.9)

        # Should keep both parameters for non-Claude models
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["top_p"], 0.9)

    def test_invalid_sampling_priority(self):
        """Test that invalid sampling priority raises ValueError."""
        from ace.llm_providers import LiteLLMClient

        with self.assertRaises(ValueError) as context:
            LiteLLMClient._resolve_sampling_params(
                {"temperature": 0.7, "top_p": 0.9},
                "claude-3-sonnet-20240229",
                "invalid_priority",
            )

        self.assertIn("Invalid sampling_priority", str(context.exception))

    @patch("ace.llm_providers.litellm_client.acompletion")
    async def test_async_claude_parameter_resolution(self, mock_acompletion):
        """Test that async completion also applies Claude parameter resolution."""
        from ace.llm_providers import LiteLLMClient

        mock_acompletion.return_value = self.mock_response

        client = LiteLLMClient(model="claude-3-sonnet-20240229", temperature=0.7)
        await client.acomplete("Test prompt", top_p=0.9)

        # Should exclude top_p in async method too (temperature priority)
        call_kwargs = mock_acompletion.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertNotIn("top_p", call_kwargs)

    def test_resolve_sampling_params_edge_cases(self):
        """Test edge cases in parameter resolution."""
        from ace.llm_providers import LiteLLMClient

        # Test with None values
        result = LiteLLMClient._resolve_sampling_params(
            {"temperature": None, "top_p": 0.9}, "claude-3-sonnet-20240229"
        )
        self.assertEqual(result["top_p"], 0.9)
        self.assertNotIn("temperature", result)

        # Test with only temperature=0
        result = LiteLLMClient._resolve_sampling_params(
            {"temperature": 0.0}, "claude-3-sonnet-20240229"
        )
        self.assertEqual(result["temperature"], 0.0)
        self.assertNotIn("top_p", result)

        # Test non-Claude model (should pass through unchanged)
        result = LiteLLMClient._resolve_sampling_params(
            {"temperature": 0.7, "top_p": 0.9}, "gpt-4"
        )
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["top_p"], 0.9)


class TestLiteLLMConfigDefaults(unittest.TestCase):
    """Test LiteLLMConfig default values prevent parameter conflicts."""

    def test_top_p_default_is_none(self):
        """
        REGRESSION TEST: top_p must default to None, not 0.9.

        When top_p defaults to 0.9, Claude models receive both temperature and top_p,
        causing Anthropic API errors:
        "temperature and top_p cannot both be specified for this model"

        This broke ACE learning (Reflector/Curator failed, 0 strategies learned).
        Fixed in commit 9740603.
        """
        from ace.llm_providers.litellm_client import LiteLLMConfig

        config = LiteLLMConfig(model="claude-3-sonnet-20240229")
        self.assertIsNone(
            config.top_p,
            "top_p must default to None to prevent temperature+top_p conflicts for Claude models. "
            "Previous default of 0.9 caused Anthropic API errors.",
        )

    def test_config_with_explicit_top_p(self):
        """Test that explicitly setting top_p still works."""
        from ace.llm_providers.litellm_client import LiteLLMConfig

        config = LiteLLMConfig(model="gpt-4", top_p=0.9)
        self.assertEqual(config.top_p, 0.9)


if __name__ == "__main__":
    unittest.main()
