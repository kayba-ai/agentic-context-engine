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

        This broke ACE learning (Reflector/SkillManager failed, 0 strategies learned).
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


@pytest.mark.unit
class TestACELiteLLMConfiguration(unittest.TestCase):
    """Test ACELiteLLM configuration parameter passing."""

    def _mock_response(self):
        """Create mock LiteLLM response."""
        mock = MagicMock()
        mock.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"reasoning":"test","skill_ids":[],"final_answer":"ok"}'
                )
            )
        ]
        mock.usage = None
        mock.model = "gpt-4"
        return mock

    @patch("ace.llm_providers.litellm_client.completion")
    def test_api_key_passed_to_client(self, mock_completion):
        """Test api_key parameter is passed to LiteLLMClient."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        agent = ACELiteLLM(model="gpt-4", api_key="test-key-123")
        self.assertEqual(agent.llm.config.api_key, "test-key-123")

    @patch("ace.llm_providers.litellm_client.completion")
    def test_base_url_maps_to_api_base(self, mock_completion):
        """Test base_url maps to api_base in config."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        agent = ACELiteLLM(model="openai/local", base_url="http://localhost:1234/v1")
        self.assertEqual(agent.llm.config.api_base, "http://localhost:1234/v1")

    @patch("ace.llm_providers.litellm_client.completion")
    def test_extra_headers_passed(self, mock_completion):
        """Test extra_headers parameter is passed through."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        headers = {"X-Custom": "value", "X-Tenant-ID": "team-alpha"}
        agent = ACELiteLLM(model="gpt-4", extra_headers=headers)
        self.assertEqual(agent.llm.config.extra_headers, headers)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_ssl_verify_false(self, mock_completion):
        """Test ssl_verify=False is passed through."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        agent = ACELiteLLM(model="gpt-4", ssl_verify=False)
        self.assertEqual(agent.llm.config.ssl_verify, False)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_ssl_verify_path(self, mock_completion):
        """Test ssl_verify with CA bundle path."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        agent = ACELiteLLM(model="gpt-4", ssl_verify="/path/to/ca.pem")
        self.assertEqual(agent.llm.config.ssl_verify, "/path/to/ca.pem")

    @patch("ace.llm_providers.litellm_client.completion")
    def test_kwargs_passed_through(self, mock_completion):
        """Test **llm_kwargs are passed to LiteLLMClient."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        agent = ACELiteLLM(model="gpt-4", timeout=120, max_retries=5)
        self.assertEqual(agent.llm.config.timeout, 120)
        self.assertEqual(agent.llm.config.max_retries, 5)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_backward_compatibility(self, mock_completion):
        """Test that existing code without new params still works."""
        mock_completion.return_value = self._mock_response()

        from ace.integrations import ACELiteLLM

        agent = ACELiteLLM(model="gpt-4o-mini", max_tokens=1024, temperature=0.5)
        self.assertEqual(agent.model, "gpt-4o-mini")
        self.assertEqual(agent.llm.config.max_tokens, 1024)
        self.assertEqual(agent.llm.config.temperature, 0.5)
        # New params should be None by default (api_key may be picked up from env vars)
        self.assertIsNone(agent.llm.config.extra_headers)
        self.assertIsNone(agent.llm.config.ssl_verify)


@pytest.mark.unit
class TestLiteLLMClientDirectConfig(unittest.TestCase):
    """Test LiteLLMClient direct configuration."""

    def _mock_response(self):
        """Create mock LiteLLM response."""
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock.usage = None
        mock.model = "gpt-4"
        return mock

    @patch("ace.llm_providers.litellm_client.completion")
    def test_extra_headers_in_call_params(self, mock_completion):
        """Test extra_headers is included in LiteLLM call_params."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        headers = {"X-Custom": "value"}
        client = LiteLLMClient(model="gpt-4", extra_headers=headers)
        client.complete("Test prompt")

        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["extra_headers"], headers)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_ssl_verify_false_in_call_params(self, mock_completion):
        """Test ssl_verify=False is included in LiteLLM call_params."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        client = LiteLLMClient(model="gpt-4", ssl_verify=False)
        client.complete("Test prompt")

        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["ssl_verify"], False)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_ssl_verify_path_in_call_params(self, mock_completion):
        """Test ssl_verify path is included in LiteLLM call_params."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        client = LiteLLMClient(model="gpt-4", ssl_verify="/path/to/ca.pem")
        client.complete("Test prompt")

        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["ssl_verify"], "/path/to/ca.pem")

    @patch("ace.llm_providers.litellm_client.completion")
    def test_ssl_verify_none_not_in_call_params(self, mock_completion):
        """Test ssl_verify=None is NOT included in call_params."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        client = LiteLLMClient(model="gpt-4")  # ssl_verify defaults to None
        client.complete("Test prompt")

        call_kwargs = mock_completion.call_args[1]
        self.assertNotIn("ssl_verify", call_kwargs)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_reasoning_effort_passed_through(self, mock_completion):
        """Test reasoning_effort parameter is passed to LiteLLM (Issue #48)."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        client = LiteLLMClient(model="gpt-5", reasoning_effort="low")
        client.complete("Test prompt")

        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["reasoning_effort"], "low")

    @patch("ace.llm_providers.litellm_client.completion")
    def test_extra_params_multiple_values(self, mock_completion):
        """Test multiple extra_params are passed to LiteLLM."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        client = LiteLLMClient(
            model="gpt-5",
            reasoning_effort="high",
            budget_tokens=10000,
        )
        client.complete("Test prompt")

        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["reasoning_effort"], "high")
        self.assertEqual(call_kwargs["budget_tokens"], 10000)

    @patch("ace.llm_providers.litellm_client.completion")
    def test_extra_params_stored_in_config(self, mock_completion):
        """Test extra_params are stored in config.extra_params."""
        mock_completion.return_value = self._mock_response()

        from ace.llm_providers import LiteLLMClient

        client = LiteLLMClient(model="gpt-5", reasoning_effort="medium")

        self.assertEqual(client.config.extra_params, {"reasoning_effort": "medium"})


if __name__ == "__main__":
    unittest.main()
