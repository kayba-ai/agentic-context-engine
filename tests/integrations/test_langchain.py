"""Tests for LangChain integration (ACELangChain)."""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Skip all tests if langchain not available
pytest.importorskip("langchain_core")

from ace.integrations import ACELangChain, LANGCHAIN_AVAILABLE
from ace import Skillbook, Skill, LiteLLMClient


class TestLangChainAvailability:
    """Test LangChain availability flag."""

    def test_langchain_available(self):
        """LANGCHAIN_AVAILABLE should be True when langchain-core is installed."""
        assert LANGCHAIN_AVAILABLE is True


class TestACELangChainInitialization:
    """Test ACELangChain initialization."""

    def test_basic_initialization(self):
        """Should initialize with minimal parameters."""
        # Create a mock runnable
        mock_runnable = Mock()

        agent = ACELangChain(runnable=mock_runnable)

        assert agent.runnable is mock_runnable
        assert agent.is_learning is True  # Default
        assert agent.skillbook is not None
        assert agent.reflector is not None
        assert agent.skill_manager is not None
        assert agent.output_parser is not None

    def test_with_skillbook_path(self):
        """Should load existing skillbook from path."""
        mock_runnable = Mock()

        # Create temp skillbook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"skills": []}')
            skillbook_path = f.name

        try:
            agent = ACELangChain(runnable=mock_runnable, skillbook_path=skillbook_path)
            assert agent.skillbook is not None
        finally:
            Path(skillbook_path).unlink()

    def test_with_learning_disabled(self):
        """Should respect is_learning parameter."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        assert agent.is_learning is False

    def test_with_custom_ace_model(self):
        """Should accept custom ace_model parameter."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable, ace_model="gpt-4")

        assert agent.llm is not None
        assert agent.llm.model == "gpt-4"

    def test_with_custom_output_parser(self):
        """Should accept custom output_parser."""
        mock_runnable = Mock()

        def custom_parser(result):
            return f"custom: {result}"

        agent = ACELangChain(runnable=mock_runnable, output_parser=custom_parser)

        assert agent.output_parser is custom_parser
        assert agent.output_parser("test") == "custom: test"


class TestContextInjection:
    """Test _inject_context method."""

    def test_empty_skillbook_returns_unchanged(self):
        """Should return input unchanged when skillbook is empty."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable)

        # String input
        result = agent._inject_context("test input")
        assert result == "test input"

        # Dict input
        result = agent._inject_context({"input": "test"})
        assert result == {"input": "test"}

    def test_string_input_appends_context(self):
        """Should append skillbook context to string input."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable)

        # Add a skill
        agent.skillbook.add_skill("general", "Test strategy")

        result = agent._inject_context("What is ACE?")

        assert isinstance(result, str)
        assert "What is ACE?" in result
        assert "Test strategy" in result

    def test_dict_with_input_key_enhances_input_field(self):
        """Should enhance 'input' field in dict."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable)

        # Add a skill
        agent.skillbook.add_skill("general", "Test strategy")

        original = {"input": "Question", "other": "data"}
        result = agent._inject_context(original)

        assert isinstance(result, dict)
        assert "other" in result
        assert result["other"] == "data"
        assert "Question" in result["input"]
        assert "Test strategy" in result["input"]

    def test_dict_without_input_key_adds_skillbook_context(self):
        """Should add skillbook_context key to dict."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable)

        # Add a skill
        agent.skillbook.add_skill("general", "Test strategy")

        original = {"question": "What?", "data": "value"}
        result = agent._inject_context(original)

        assert isinstance(result, dict)
        assert "question" in result
        assert result["question"] == "What?"
        assert "skillbook_context" in result
        assert "Test strategy" in result["skillbook_context"]


class TestOutputParser:
    """Test _default_output_parser method."""

    def test_string_input_returns_as_is(self):
        """Should return string unchanged."""
        result = ACELangChain._default_output_parser("simple string")
        assert result == "simple string"

    def test_object_with_content_attr(self):
        """Should extract .content attribute from LangChain messages."""
        mock_message = Mock()
        mock_message.content = "message content"

        result = ACELangChain._default_output_parser(mock_message)
        assert result == "message content"

    def test_dict_with_output_key(self):
        """Should extract common output keys from dict."""
        result = ACELangChain._default_output_parser({"output": "the answer"})
        assert result == "the answer"

        result = ACELangChain._default_output_parser({"answer": "the answer"})
        assert result == "the answer"

        result = ACELangChain._default_output_parser({"result": "the answer"})
        assert result == "the answer"

    def test_dict_without_common_keys(self):
        """Should convert entire dict to string if no common keys."""
        input_dict = {"custom": "value", "data": 123}
        result = ACELangChain._default_output_parser(input_dict)
        assert "custom" in result
        assert "value" in result

    def test_other_types_convert_to_string(self):
        """Should convert other types to string."""
        result = ACELangChain._default_output_parser(42)
        assert result == "42"

        result = ACELangChain._default_output_parser([1, 2, 3])
        assert "1" in result and "2" in result


class TestInvokeMethod:
    """Test invoke() method."""

    def test_invoke_calls_runnable(self):
        """Should call runnable.invoke with enhanced input."""
        mock_runnable = Mock()
        mock_runnable.invoke.return_value = Mock(content="response")

        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        result = agent.invoke("test input")

        mock_runnable.invoke.assert_called_once()
        assert result.content == "response"

    def test_invoke_with_learning_disabled_skips_learning(self):
        """Should not call learning methods when disabled."""
        mock_runnable = Mock()
        mock_runnable.invoke.return_value = "response"

        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        with patch.object(agent, "_learn") as mock_learn:
            agent.invoke("test input")
            mock_learn.assert_not_called()

    def test_invoke_with_dict_input(self):
        """Should handle dict input."""
        mock_runnable = Mock()
        mock_runnable.invoke.return_value = {"output": "answer"}

        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        result = agent.invoke({"input": "question"})

        mock_runnable.invoke.assert_called_once()
        assert result == {"output": "answer"}


class TestAsyncInvokeMethod:
    """Test ainvoke() method."""

    @pytest.mark.asyncio
    async def test_ainvoke_calls_runnable(self):
        """Should call runnable.ainvoke with enhanced input."""
        mock_runnable = Mock()
        mock_runnable.ainvoke = AsyncMock(return_value=Mock(content="response"))

        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        result = await agent.ainvoke("test input")

        mock_runnable.ainvoke.assert_called_once()
        assert result.content == "response"

    @pytest.mark.asyncio
    async def test_ainvoke_with_learning_disabled(self):
        """Should not call learning methods when disabled."""
        mock_runnable = Mock()
        mock_runnable.ainvoke = AsyncMock(return_value="response")

        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        with patch.object(agent, "_learn") as mock_learn:
            await agent.ainvoke("test input")
            mock_learn.assert_not_called()


class TestLearningControl:
    """Test learning enable/disable methods."""

    def test_enable_disable_learning(self):
        """Should toggle learning flag."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable, is_learning=True)

        assert agent.is_learning is True

        agent.disable_learning()
        assert agent.is_learning is False

        agent.enable_learning()
        assert agent.is_learning is True


class TestSkillbookOperations:
    """Test skillbook save/load methods."""

    def test_save_skillbook(self):
        """Should save skillbook to file."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable)

        agent.skillbook.add_skill("general", "Test skill")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            agent.save_skillbook(temp_path)

            # Verify file exists and is valid JSON
            loaded_skillbook = Skillbook.load_from_file(temp_path)
            assert len(loaded_skillbook.skills()) == 1
            assert loaded_skillbook.skills()[0].content == "Test skill"
        finally:
            Path(temp_path).unlink()

    def test_load_skillbook(self):
        """Should load skillbook from file."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable)

        # Create and save a skillbook
        temp_skillbook = Skillbook()
        temp_skillbook.add_skill("general", "Loaded skill")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            temp_skillbook.save_to_file(temp_path)

            # Load it
            agent.load_skillbook(temp_path)

            assert len(agent.skillbook.skills()) == 1
            assert agent.skillbook.skills()[0].content == "Loaded skill"
        finally:
            Path(temp_path).unlink()


class TestReprMethod:
    """Test __repr__ method."""

    def test_repr_includes_key_info(self):
        """Should include runnable type, strategies count, and learning status."""
        mock_runnable = Mock()
        mock_runnable.__class__.__name__ = "TestRunnable"

        agent = ACELangChain(runnable=mock_runnable, is_learning=True)
        agent.skillbook.add_skill("general", "Test")

        repr_str = repr(agent)

        assert "TestRunnable" in repr_str
        assert "strategies=1" in repr_str
        assert "enabled" in repr_str

    def test_repr_with_learning_disabled(self):
        """Should show disabled in repr when learning is off."""
        mock_runnable = Mock()
        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        repr_str = repr(agent)

        assert "disabled" in repr_str


class TestErrorHandling:
    """Test error handling in invoke and learning."""

    def test_invoke_propagates_runnable_errors(self):
        """Should propagate errors from runnable execution."""
        mock_runnable = Mock()
        mock_runnable.invoke.side_effect = ValueError("Test error")

        agent = ACELangChain(runnable=mock_runnable, is_learning=False)

        with pytest.raises(ValueError, match="Test error"):
            agent.invoke("test")

    def test_invoke_learns_from_failure_when_enabled(self):
        """Should call _learn_from_failure when runnable raises error."""
        mock_runnable = Mock()
        mock_runnable.invoke.side_effect = ValueError("Test error")

        agent = ACELangChain(runnable=mock_runnable, is_learning=True)

        with patch.object(agent, "_learn_from_failure") as mock_learn_failure:
            with pytest.raises(ValueError):
                agent.invoke("test input")

            mock_learn_failure.assert_called_once_with("test input", "Test error")

    def test_learning_errors_dont_crash(self):
        """Should continue execution even if learning fails."""
        mock_runnable = Mock()
        mock_runnable.invoke.return_value = "response"

        agent = ACELangChain(runnable=mock_runnable, is_learning=True)

        # Make reflector.reflect raise an error (simulates learning failure)
        with patch.object(
            agent.reflector, "reflect", side_effect=Exception("Learning failed")
        ):
            # Should not raise, should return result (error caught in _learn)
            result = agent.invoke("test")
            assert result == "response"


class TestBackwardsCompatibility:
    """Test imports and backward compatibility."""

    def test_can_import_from_ace(self):
        """Should be able to import ACELangChain from ace package."""
        from ace import ACELangChain as ImportedACELangChain

        assert ImportedACELangChain is not None

    def test_can_import_from_integrations(self):
        """Should be able to import from ace.integrations."""
        from ace.integrations import ACELangChain as ImportedACELangChain

        assert ImportedACELangChain is not None

    def test_can_check_availability(self):
        """Should be able to check LANGCHAIN_AVAILABLE flag."""
        from ace import LANGCHAIN_AVAILABLE as flag

        assert flag is True
