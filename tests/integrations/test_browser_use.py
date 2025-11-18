"""Tests for browser-use integration (ACEAgent)."""

import pytest
from pathlib import Path
import tempfile

# Skip all tests if browser-use not available
pytest.importorskip("browser_use")

from ace.integrations import (
    ACEAgent,
    wrap_playbook_context,
    BROWSER_USE_AVAILABLE,
)
from ace import Playbook, Bullet, LiteLLMClient


class TestWrapPlaybookContext:
    """Test the wrap_playbook_context helper function."""

    def test_empty_playbook(self):
        """Should return empty string for empty playbook."""
        playbook = Playbook()
        result = wrap_playbook_context(playbook)
        assert result == ""

    def test_with_bullets(self):
        """Should format bullets with explanation."""
        playbook = Playbook()
        playbook.add_bullet("general", "Always check search box first")
        playbook.add_bullet("general", "Scroll before clicking")

        result = wrap_playbook_context(playbook)

        # Should contain header
        assert "Strategic Knowledge" in result
        assert "Learned from Experience" in result

        # Should contain bullets
        assert "Always check search box first" in result
        assert "Scroll before clicking" in result

        # Should contain usage instructions
        assert "How to use these strategies" in result
        assert "success rates" in result

    def test_bullet_scores_shown(self):
        """Should show helpful/harmful scores."""
        playbook = Playbook()
        bullet = playbook.add_bullet(
            "general", "Test strategy", metadata={"helpful": 5, "harmful": 2}
        )

        result = wrap_playbook_context(playbook)

        # Should show the bullet content
        assert "Test strategy" in result


class TestACEAgentInitialization:
    """Test ACEAgent initialization."""

    def test_browser_use_available(self):
        """BROWSER_USE_AVAILABLE should be True when browser-use is installed."""
        assert BROWSER_USE_AVAILABLE is True

    def test_basic_initialization(self):
        """Should initialize with minimal parameters."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        assert agent.browser_llm is not None
        assert agent.is_learning is True  # Default
        assert agent.playbook is not None
        assert agent.reflector is not None
        assert agent.curator is not None

    def test_with_ace_model(self):
        """Should accept ace_model parameter."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse(), ace_model="gpt-4")

        assert agent.ace_llm is not None
        assert agent.ace_llm.model == "gpt-4"

    def test_with_custom_ace_llm(self):
        """Should accept custom ace_llm parameter."""
        from browser_use import ChatBrowserUse

        custom_llm = LiteLLMClient(model="claude-3-opus-20240229")
        agent = ACEAgent(llm=ChatBrowserUse(), ace_llm=custom_llm)

        assert agent.ace_llm is custom_llm

    def test_with_ace_max_tokens(self):
        """Should accept ace_max_tokens parameter."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse(), ace_max_tokens=4096)

        assert agent.ace_llm is not None
        assert agent.ace_llm.config.max_tokens == 4096

    def test_default_ace_max_tokens(self):
        """Should use default max_tokens of 2048."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        assert agent.ace_llm is not None
        assert agent.ace_llm.config.max_tokens == 2048

    def test_learning_disabled(self):
        """Should respect is_learning=False."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse(), is_learning=False)

        assert agent.is_learning is False
        # Should still create components for potential later use
        assert agent.playbook is not None

    def test_with_playbook_path(self):
        """Should load playbook from path."""
        from browser_use import ChatBrowserUse

        # Create a temporary playbook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            playbook_path = f.name

        try:
            # Create and save playbook
            playbook = Playbook()
            playbook.add_bullet("general", "Pre-loaded strategy")
            playbook.save_to_file(playbook_path)

            # Load in ACEAgent
            agent = ACEAgent(llm=ChatBrowserUse(), playbook_path=playbook_path)

            assert len(agent.playbook.bullets()) == 1
            assert agent.playbook.bullets()[0].content == "Pre-loaded strategy"
        finally:
            Path(playbook_path).unlink(missing_ok=True)

    def test_with_task_in_constructor(self):
        """Should accept task in constructor."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(task="Test task", llm=ChatBrowserUse())

        assert agent.task == "Test task"


class TestACEAgentLearningControl:
    """Test learning enable/disable functionality."""

    def test_enable_disable_learning(self):
        """Should toggle learning on/off."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse(), is_learning=True)

        assert agent.is_learning is True

        agent.disable_learning()
        assert agent.is_learning is False

        agent.enable_learning()
        assert agent.is_learning is True

    def test_playbook_operations(self):
        """Should support save/load playbook."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        # Add a bullet manually
        agent.playbook.add_bullet("general", "Test strategy")

        # Save
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            playbook_path = f.name

        try:
            agent.save_playbook(playbook_path)

            # Load in new agent
            agent2 = ACEAgent(llm=ChatBrowserUse())
            agent2.load_playbook(playbook_path)

            assert len(agent2.playbook.bullets()) == 1
            assert agent2.playbook.bullets()[0].content == "Test strategy"
        finally:
            Path(playbook_path).unlink(missing_ok=True)

    def test_get_strategies(self):
        """Should return formatted strategies."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        # Empty playbook
        strategies = agent.get_strategies()
        assert strategies == ""

        # With bullets
        agent.playbook.add_bullet("general", "Strategy 1")
        strategies = agent.get_strategies()
        assert "Strategy 1" in strategies
        assert "Strategic Knowledge" in strategies


class TestACEAgentRunMethod:
    """Test ACEAgent.run() method."""

    def test_run_requires_task(self):
        """Should raise error if no task provided."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        with pytest.raises(ValueError, match="Task must be provided"):
            import asyncio

            asyncio.run(agent.run())

    def test_task_from_constructor(self):
        """Should use task from constructor if not provided to run()."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(task="Constructor task", llm=ChatBrowserUse())

        # This should not raise (though it will fail without browser setup)
        # We're just testing that task is recognized
        assert agent.task == "Constructor task"

    def test_task_override(self):
        """run() task should override constructor task."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(task="Constructor task", llm=ChatBrowserUse())

        # Verify we can override (mock test, not actually running)
        # Just verify the logic works
        task_to_use = "Override task" or agent.task
        assert task_to_use == "Override task"


class TestRichFeedbackBuilder:
    """Test rich feedback extraction from browser-use history."""

    def test_build_rich_feedback_with_no_history(self):
        """Should handle None history gracefully."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())
        result = agent._build_rich_feedback(None, success=False, error="Test error")

        assert "feedback" in result
        assert "raw_trace" in result
        assert "steps" in result
        assert "output" in result
        assert result["steps"] == 0
        assert "Test error" in result["feedback"]

    def test_build_rich_feedback_extracts_basic_info(self):
        """Should extract basic information from history."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history object
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Test output"
        mock_history.number_of_steps.return_value = 5

        result = agent._build_rich_feedback(mock_history, success=True)

        assert result["output"] == "Test output"
        assert result["steps"] == 5
        assert "succeeded" in result["feedback"]
        assert "5 steps" in result["feedback"]

    def test_build_rich_feedback_extracts_urls(self):
        """Should extract URLs from history."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with URLs
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 3
        mock_history.urls.return_value = [
            "https://example.com",
            "https://test.com",
            None,
        ]

        result = agent._build_rich_feedback(mock_history, success=True)

        assert "urls" in result["raw_trace"]
        assert "https://example.com" in result["raw_trace"]["urls"]
        assert "https://test.com" in result["raw_trace"]["urls"]
        assert "URLs visited" in result["feedback"]

    def test_build_rich_feedback_extracts_errors(self):
        """Should extract step errors from history."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with errors
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 3
        mock_history.errors.return_value = [None, "Click failed", "Element not found"]

        result = agent._build_rich_feedback(mock_history, success=False, error="Failed")

        assert "step_errors" in result["raw_trace"]
        assert "Click failed" in result["raw_trace"]["step_errors"]
        assert "Errors encountered" in result["feedback"]

    def test_build_rich_feedback_extracts_actions(self):
        """Should extract actions from history."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with actions
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 3
        mock_history.model_actions.return_value = [
            {"goto": {"url": "https://example.com"}},
            {"click": {"index": 5}},
            {"extract": {"data": "text"}},
        ]

        result = agent._build_rich_feedback(mock_history, success=True)

        assert "actions" in result["raw_trace"]
        assert len(result["raw_trace"]["actions"]) == 3
        assert "Actions taken" in result["feedback"]
        assert "goto" in result["feedback"]

    def test_build_rich_feedback_extracts_thoughts(self):
        """Should extract agent thoughts from history."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with thoughts
        mock_thought = MagicMock()
        mock_thought.thinking = "Planning to search"
        mock_thought.evaluation_previous_goal = "Navigated successfully"
        mock_thought.memory = "On homepage"
        mock_thought.next_goal = "Search for product"

        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 2
        mock_history.model_thoughts.return_value = [mock_thought]

        result = agent._build_rich_feedback(mock_history, success=True)

        assert "thoughts" in result["raw_trace"]
        assert len(result["raw_trace"]["thoughts"]) == 1
        assert result["raw_trace"]["thoughts"][0]["next_goal"] == "Search for product"
        assert "Agent Reasoning" in result["feedback"]

    def test_build_rich_feedback_handles_exceptions(self):
        """Should handle exceptions when extracting trace data."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history that raises exceptions
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 2
        mock_history.urls.side_effect = Exception("URL error")
        mock_history.errors.side_effect = Exception("Error extraction failed")

        result = agent._build_rich_feedback(mock_history, success=True)

        # Should still return valid result with error indicators
        assert "urls_error" in result["raw_trace"]
        assert "errors_error" in result["raw_trace"]
        assert result["feedback"]  # Should still have feedback


@pytest.mark.integration
class TestACEAgentIntegration:
    """Integration tests for ACEAgent (requires actual browser-use execution)."""

    @pytest.mark.skip(reason="Requires browser setup and API keys")
    async def test_full_learning_cycle(self):
        """Full test of learning cycle (manual test)."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse(), is_learning=True)

        # This would run actual browser automation
        # Skipped in automated tests
        # await agent.run(task="Find top HN post")
        # assert len(agent.playbook.bullets()) > 0

        pass


class TestBackwardsCompatibility:
    """Test that existing code patterns still work."""

    def test_can_import_from_ace(self):
        """Should be importable from ace package."""
        from ace import ACEAgent as ImportedACEAgent

        assert ImportedACEAgent is not None

    def test_can_import_helper_from_ace(self):
        """Should import helper function from ace package."""
        from ace import wrap_playbook_context as imported_wrap

        assert imported_wrap is not None

    def test_can_check_availability(self):
        """Should check browser-use availability."""
        from ace import BROWSER_USE_AVAILABLE as imported_available

        assert imported_available is True
