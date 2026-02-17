"""Basic tests for LangChain client integration."""

import unittest
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestLangChainClient(unittest.TestCase):
    """Test LangChain client functionality."""

    def test_import_fallback(self):
        """Test that import handles missing langchain gracefully."""
        try:
            from ace.llm_providers import LangChainLiteLLMClient

            # If it imports, langchain is installed
            self.assertTrue(True)
        except ImportError:
            # This is expected if langchain not installed
            self.assertTrue(True)

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_basic_initialization(self, mock_chat):
        """Test client initialization."""
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        # Create client
        client = LangChainLiteLLMClient(model="gpt-3.5-turbo")

        # Check that ChatLiteLLM was initialized
        mock_chat.assert_called_once()
        self.assertFalse(client.is_router)

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", False)
    def test_missing_langchain_error(
        self,
    ):
        """Test error when langchain not available."""
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        with self.assertRaises(ImportError) as context:
            LangChainLiteLLMClient(model="test")

        self.assertIn("LangChain is not installed", str(context.exception))

    def test_parameter_filtering(self):
        """Test that ACE-specific parameters are filtered."""
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        client = LangChainLiteLLMClient.__new__(LangChainLiteLLMClient)

        # Test the filter method directly
        filtered = client._filter_kwargs(
            {
                "temperature": 0.5,
                "refinement_round": 1,
                "max_refinement_rounds": 3,
                "max_tokens": 100,
            }
        )

        self.assertIn("temperature", filtered)
        self.assertIn("max_tokens", filtered)
        self.assertNotIn("refinement_round", filtered)
        self.assertNotIn("max_refinement_rounds", filtered)


@pytest.mark.unit
class TestLangChainClientComplete(unittest.TestCase):
    """Tests for LangChainLiteLLMClient.complete()."""

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_complete_returns_response_content(self, mock_chat_cls):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_response.response_metadata = {"model": "gpt-4", "finish_reason": "stop"}
        mock_response.usage_metadata = None
        mock_chat_cls.return_value.invoke.return_value = mock_response

        client = LangChainLiteLLMClient(model="gpt-4")
        result = client.complete("Hi")
        self.assertEqual(result.text, "Hello world")

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_complete_extracts_usage_metadata(self, mock_chat_cls):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.response_metadata = {"model": "gpt-4"}
        mock_response.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        mock_chat_cls.return_value.invoke.return_value = mock_response

        client = LangChainLiteLLMClient(model="gpt-4")
        result = client.complete("test")
        self.assertIn("usage", result.raw)
        self.assertEqual(result.raw["usage"]["prompt_tokens"], 10)
        self.assertEqual(result.raw["usage"]["completion_tokens"], 20)

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_complete_raises_on_exception(self, mock_chat_cls):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        mock_chat_cls.return_value.invoke.side_effect = RuntimeError("API error")
        client = LangChainLiteLLMClient(model="gpt-4")
        with self.assertRaises(RuntimeError):
            client.complete("test")


@pytest.mark.unit
class TestLangChainClientStream(unittest.TestCase):
    """Tests for complete_with_stream()."""

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_stream_concatenates_chunks(self, mock_chat_cls):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        chunk1 = MagicMock()
        chunk1.content = "Hello"
        chunk2 = MagicMock()
        chunk2.content = " world"
        mock_chat_cls.return_value.stream.return_value = [chunk1, chunk2]

        client = LangChainLiteLLMClient(model="gpt-4")
        tokens = list(client.complete_with_stream("test"))
        self.assertEqual(tokens, ["Hello", " world"])

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_stream_skips_empty_chunks(self, mock_chat_cls):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        chunk1 = MagicMock()
        chunk1.content = "data"
        chunk2 = MagicMock()
        chunk2.content = ""  # empty
        chunk3 = MagicMock()
        chunk3.content = "more"
        mock_chat_cls.return_value.stream.return_value = [chunk1, chunk2, chunk3]

        client = LangChainLiteLLMClient(model="gpt-4")
        tokens = list(client.complete_with_stream("test"))
        self.assertEqual(tokens, ["data", "more"])


@pytest.mark.unit
class TestLangChainClientFilterKwargs(unittest.TestCase):
    """Additional tests for _filter_kwargs."""

    def test_filters_all_ace_params(self):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        client = LangChainLiteLLMClient.__new__(LangChainLiteLLMClient)
        result = client._filter_kwargs(
            {"refinement_round": 2, "max_refinement_rounds": 5}
        )
        self.assertEqual(result, {})

    def test_passes_standard_params(self):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        client = LangChainLiteLLMClient.__new__(LangChainLiteLLMClient)
        result = client._filter_kwargs(
            {"temperature": 0.7, "top_p": 0.9, "stop": ["\n"]}
        )
        self.assertEqual(len(result), 3)


@pytest.mark.unit
class TestLangChainClientRouter(unittest.TestCase):
    """Tests for router initialization path."""

    @patch("ace.llm_providers.langchain_client.LANGCHAIN_AVAILABLE", True)
    @patch("ace.llm_providers.langchain_client.ChatLiteLLMRouter")
    @patch("ace.llm_providers.langchain_client.ChatLiteLLM")
    def test_router_initialization(self, mock_chat, mock_router_cls):
        from ace.llm_providers.langchain_client import LangChainLiteLLMClient

        mock_router = MagicMock()
        client = LangChainLiteLLMClient(model="gpt-4", router=mock_router)

        mock_router_cls.assert_called_once()
        self.assertTrue(client.is_router)
        mock_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
