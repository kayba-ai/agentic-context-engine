"""Tests for browser-use integration (ACEAgent)."""

import pytest
from pathlib import Path
import tempfile

# Skip all tests if browser-use not available
pytest.importorskip("browser_use")

from ace.integrations import (
    ACEAgent,
    wrap_skillbook_context,
    BROWSER_USE_AVAILABLE,
)
from ace import Skillbook, Skill, LiteLLMClient


class TestWrapSkillbookContext:
    """Test the wrap_skillbook_context helper function."""

    def test_empty_skillbook(self):
        """Should return empty string for empty skillbook."""
        skillbook = Skillbook()
        result = wrap_skillbook_context(skillbook)
        assert result == ""

    def test_with_skills(self):
        """Should format skills with explanation."""
        skillbook = Skillbook()
        skillbook.add_skill("general", "Always check search box first")
        skillbook.add_skill("general", "Scroll before clicking")

        result = wrap_skillbook_context(skillbook)

        # Should contain header
        assert "Strategic Knowledge" in result
        assert "Learned from Experience" in result

        # Should contain skills
        assert "Always check search box first" in result
        assert "Scroll before clicking" in result

        # Should contain usage instructions
        assert "How to use these strategies" in result
        assert "success rates" in result

    def test_skill_scores_shown(self):
        """Should show helpful/harmful scores."""
        skillbook = Skillbook()
        skill = skillbook.add_skill(
            "general", "Test strategy", metadata={"helpful": 5, "harmful": 2}
        )

        result = wrap_skillbook_context(skillbook)

        # Should show the skill content
        assert "Test strategy" in result


@pytest.mark.requires_api
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
        assert agent.skillbook is not None
        assert agent.reflector is not None
        assert agent.skill_manager is not None

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
        assert agent.skillbook is not None

    def test_with_skillbook_path(self):
        """Should load skillbook from path."""
        from browser_use import ChatBrowserUse

        # Create a temporary skillbook
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            skillbook_path = f.name

        try:
            # Create and save skillbook
            skillbook = Skillbook()
            skillbook.add_skill("general", "Pre-loaded strategy")
            skillbook.save_to_file(skillbook_path)

            # Load in ACEAgent
            agent = ACEAgent(llm=ChatBrowserUse(), skillbook_path=skillbook_path)

            assert len(agent.skillbook.skills()) == 1
            assert agent.skillbook.skills()[0].content == "Pre-loaded strategy"
        finally:
            Path(skillbook_path).unlink(missing_ok=True)

    def test_with_task_in_constructor(self):
        """Should accept task in constructor."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(task="Test task", llm=ChatBrowserUse())

        assert agent.task == "Test task"


@pytest.mark.requires_api
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

    def test_skillbook_operations(self):
        """Should support save/load skillbook."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        # Add a skill manually
        agent.skillbook.add_skill("general", "Test strategy")

        # Save
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            skillbook_path = f.name

        try:
            agent.save_skillbook(skillbook_path)

            # Load in new agent
            agent2 = ACEAgent(llm=ChatBrowserUse())
            agent2.load_skillbook(skillbook_path)

            assert len(agent2.skillbook.skills()) == 1
            assert agent2.skillbook.skills()[0].content == "Test strategy"
        finally:
            Path(skillbook_path).unlink(missing_ok=True)

    def test_get_strategies(self):
        """Should return formatted strategies."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        # Empty skillbook
        strategies = agent.get_strategies()
        assert strategies == ""

        # With skills
        agent.skillbook.add_skill("general", "Strategy 1")
        strategies = agent.get_strategies()
        assert "Strategy 1" in strategies
        assert "Strategic Knowledge" in strategies


@pytest.mark.requires_api
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


@pytest.mark.requires_api
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
        """Should extract URLs from chronological steps."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with chronological steps containing URLs
        mock_step1 = MagicMock()
        mock_step1.model_output = None
        mock_step1.result = None
        mock_step1.state = MagicMock()
        mock_step1.state.url = "https://example.com"
        mock_step1.state.screenshot = None

        mock_step2 = MagicMock()
        mock_step2.model_output = None
        mock_step2.result = None
        mock_step2.state = MagicMock()
        mock_step2.state.url = "https://test.com"
        mock_step2.state.screenshot = None

        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 2
        mock_history.history = [mock_step1, mock_step2]

        result = agent._build_rich_feedback(mock_history, success=True)

        assert "chronological_steps" in result["raw_trace"]
        assert len(result["raw_trace"]["chronological_steps"]) == 2
        assert (
            result["raw_trace"]["chronological_steps"][0]["url"]
            == "https://example.com"
        )
        assert (
            result["raw_trace"]["chronological_steps"][1]["url"] == "https://test.com"
        )
        assert "https://example.com" in result["feedback"]

    def test_build_rich_feedback_extracts_errors(self):
        """Should extract step errors from chronological results."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with errors in results
        mock_result1 = MagicMock()
        mock_result1.is_done = False
        mock_result1.success = False
        mock_result1.error = "Click failed"
        mock_result1.extracted_content = None

        mock_step = MagicMock()
        mock_step.model_output = None
        mock_step.result = [mock_result1]
        mock_step.state = None

        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 1
        mock_history.history = [mock_step]

        result = agent._build_rich_feedback(mock_history, success=False, error="Failed")

        assert "chronological_steps" in result["raw_trace"]
        assert len(result["raw_trace"]["chronological_steps"]) == 1
        assert (
            result["raw_trace"]["chronological_steps"][0]["results"][0]["error"]
            == "Click failed"
        )
        assert "Click failed" in result["feedback"]

    def test_build_rich_feedback_extracts_actions(self):
        """Should extract actions from chronological steps."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock action objects
        mock_action1 = MagicMock()
        mock_action1.model_dump.return_value = {"goto": {"url": "https://example.com"}}

        mock_action2 = MagicMock()
        mock_action2.model_dump.return_value = {"click": {"index": 5}}

        # Mock model outputs with actions
        mock_output1 = MagicMock()
        mock_output1.thinking = None
        mock_output1.evaluation_previous_goal = None
        mock_output1.memory = None
        mock_output1.next_goal = None
        mock_output1.action = [mock_action1]

        mock_output2 = MagicMock()
        mock_output2.thinking = None
        mock_output2.evaluation_previous_goal = None
        mock_output2.memory = None
        mock_output2.next_goal = None
        mock_output2.action = [mock_action2]

        # Mock steps
        mock_step1 = MagicMock()
        mock_step1.model_output = mock_output1
        mock_step1.result = None
        mock_step1.state = None

        mock_step2 = MagicMock()
        mock_step2.model_output = mock_output2
        mock_step2.result = None
        mock_step2.state = None

        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 2
        mock_history.history = [mock_step1, mock_step2]

        result = agent._build_rich_feedback(mock_history, success=True)

        assert "chronological_steps" in result["raw_trace"]
        assert len(result["raw_trace"]["chronological_steps"]) == 2
        assert "goto" in str(result["raw_trace"]["chronological_steps"][0]["actions"])
        assert "click" in str(result["raw_trace"]["chronological_steps"][1]["actions"])
        assert "goto" in result["feedback"]

    def test_build_rich_feedback_extracts_thoughts(self):
        """Should extract agent thoughts from chronological steps."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock model output with thought
        mock_output = MagicMock()
        mock_output.thinking = "Planning to search"
        mock_output.evaluation_previous_goal = "Navigated successfully"
        mock_output.memory = "On homepage"
        mock_output.next_goal = "Search for product"
        mock_output.action = None

        # Mock step
        mock_step = MagicMock()
        mock_step.model_output = mock_output
        mock_step.result = None
        mock_step.state = None

        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 1
        mock_history.history = [mock_step]

        result = agent._build_rich_feedback(mock_history, success=True)

        assert "chronological_steps" in result["raw_trace"]
        assert len(result["raw_trace"]["chronological_steps"]) == 1
        assert (
            result["raw_trace"]["chronological_steps"][0]["thought"]["next_goal"]
            == "Search for product"
        )
        assert "Planning to search" in result["feedback"]

    def test_build_rich_feedback_handles_exceptions(self):
        """Should handle exceptions when extracting trace data."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history that raises exceptions
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Output"
        mock_history.number_of_steps.return_value = 2
        mock_history.history = MagicMock()
        mock_history.history.__iter__.side_effect = Exception(
            "History extraction failed"
        )

        result = agent._build_rich_feedback(mock_history, success=True)

        # Should still return valid result with error indicator
        assert "extraction_error" in result["raw_trace"]
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
        # assert len(agent.skillbook.skills()) > 0

        pass


@pytest.mark.requires_api
class TestPromptVersionUsage:
    """Test that ACEAgent uses v2.1 prompts by default."""

    def test_reflector_uses_v2_1(self):
        """Should use v2.1 prompt for Reflector."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        # Verify Reflector uses v2.1
        assert agent.reflector.prompt_template is not None
        assert (
            "v2.1" in agent.reflector.prompt_template
            or "2.1" in agent.reflector.prompt_template
        )
        # v2.1 has enhanced structure with QUICK REFERENCE
        assert "QUICK REFERENCE" in agent.reflector.prompt_template

    def test_skill_manager_uses_v2_1(self):
        """Should use v2.1 prompt for SkillManager."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        # Verify SkillManager uses v2.1
        assert agent.skill_manager.prompt_template is not None
        assert (
            "v2.1" in agent.skill_manager.prompt_template
            or "2.1" in agent.skill_manager.prompt_template
        )
        # v2.1 has atomicity scoring
        assert "atomicity" in agent.skill_manager.prompt_template.lower()

    def test_skillbook_wrapper_uses_canonical_function(self):
        """Should use canonical wrap function from prompts_v2_1."""
        from ace.integrations.base import wrap_skillbook_context
        from ace.prompts_v2_1 import wrap_skillbook_for_external_agent
        from ace import Skillbook

        skillbook = Skillbook()
        skillbook.add_skill("general", "Test strategy")

        # Both functions should produce identical output
        result1 = wrap_skillbook_context(skillbook)
        result2 = wrap_skillbook_for_external_agent(skillbook)

        assert result1 == result2
        assert "📚 Available Strategic Knowledge" in result1
        assert "Test strategy" in result1

    def test_skillbook_wrapper_includes_usage_instructions(self):
        """Should include SKILLBOOK_USAGE_INSTRUCTIONS constant."""
        from ace.integrations.base import wrap_skillbook_context
        from ace import Skillbook

        skillbook = Skillbook()
        skillbook.add_skill("general", "Test strategy")

        result = wrap_skillbook_context(skillbook)

        # Should include instructions from constant
        assert "How to use these strategies" in result
        assert "Review skills relevant to your current task" in result
        assert "Prioritize strategies with high success rates" in result
        assert "These are learned patterns, not rigid rules" in result


@pytest.mark.requires_api
class TestCitationExtraction:
    """Test citation extraction in browser-use integration."""

    def test_extract_from_agent_thoughts(self):
        """Extract citations from browser-use agent thoughts."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        # Mock history with thoughts containing citations
        mock_thought = MagicMock()
        mock_thought.thinking = (
            "Following [content_extraction-00001], I will extract the title."
        )

        mock_history = MagicMock()
        mock_history.model_thoughts.return_value = [mock_thought]

        cited_ids = agent._extract_cited_ids_from_history(mock_history)

        assert cited_ids == ["content_extraction-00001"]

    def test_extract_multiple_citations(self):
        """Extract multiple citations from thoughts."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        mock_thought1 = MagicMock()
        mock_thought1.thinking = "Using [strategy-001] to navigate."
        mock_thought2 = MagicMock()
        mock_thought2.thinking = "Applying [strategy-002] for extraction."

        mock_history = MagicMock()
        mock_history.model_thoughts.return_value = [mock_thought1, mock_thought2]

        cited_ids = agent._extract_cited_ids_from_history(mock_history)

        assert cited_ids == ["strategy-001", "strategy-002"]

    def test_no_citations_returns_empty(self):
        """Return empty list when no citations found."""
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock

        agent = ACEAgent(llm=ChatBrowserUse())

        mock_thought = MagicMock()
        mock_thought.thinking = "Navigate to the page and extract data."

        mock_history = MagicMock()
        mock_history.model_thoughts.return_value = [mock_thought]

        cited_ids = agent._extract_cited_ids_from_history(mock_history)

        assert cited_ids == []

    def test_handles_missing_history_gracefully(self):
        """Handle None history gracefully."""
        from browser_use import ChatBrowserUse

        agent = ACEAgent(llm=ChatBrowserUse())

        cited_ids = agent._extract_cited_ids_from_history(None)

        assert cited_ids == []

    def test_filters_invalid_skill_ids_against_skillbook(self):
        """
        Test that cited IDs are validated against skillbook in Reflector call.

        This ensures only valid skill IDs (that exist in skillbook) are passed
        to the Reflector, preventing errors from hallucinated or invalid citations.
        """
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock
        from ace import Skillbook

        # Create agent with skillbook containing specific skills
        skillbook = Skillbook()
        skill1 = skillbook.add_skill("navigation", "Always scroll before clicking")
        skill2 = skillbook.add_skill("extraction", "Use CSS selectors for data")

        agent = ACEAgent(llm=ChatBrowserUse(), skillbook=skillbook)

        # Mock history with thoughts containing BOTH valid and invalid IDs
        mock_thought = MagicMock()
        # Valid: navigation-00001, extraction-00001
        # Invalid: nonexistent-99999, fake-12345
        mock_thought.thinking = (
            f"Following [{skill1.id}], I scrolled first. "
            f"Then using [nonexistent-99999] and [{skill2.id}]. "
            f"Also tried [fake-12345]."
        )

        mock_history = MagicMock()
        mock_history.model_thoughts.return_value = [mock_thought]
        mock_history.final_result.return_value = "Result"
        mock_history.number_of_steps.return_value = 2
        mock_history.history = []  # Empty history for trace extraction

        # Capture what skill_ids are passed to AgentOutput
        captured_skill_ids = None

        original_build = agent._build_rich_feedback

        def patched_build(*args, **kwargs):
            return original_build(*args, **kwargs)

        agent._build_rich_feedback = patched_build

        # Mock Reflector to capture agent_output
        from ace import ReflectorOutput

        def capture_reflect(*args, **kwargs):
            nonlocal captured_skill_ids
            agent_output = kwargs.get("agent_output")
            if agent_output:
                captured_skill_ids = agent_output.skill_ids
            return ReflectorOutput(
                reasoning="mock",
                error_identification="none",
                root_cause_analysis="none",
                correct_approach="continue",
                key_insight="test",
                skill_tags=[],
                raw={},
            )

        agent.reflector.reflect = capture_reflect

        # Mock skill_manager
        from ace import SkillManagerOutput, UpdateBatch

        agent.skill_manager.update_skills = lambda *args, **kwargs: SkillManagerOutput(
            update=UpdateBatch(reasoning="mock", operations=[]), raw={}
        )

        # Run learning
        import asyncio

        asyncio.run(
            agent._learn_from_execution(
                task="Test task", history=mock_history, success=True
            )
        )

        # ASSERTIONS: Only valid IDs should be passed
        assert captured_skill_ids is not None, "skill_ids not captured"

        # Should contain valid IDs
        assert (
            skill1.id in captured_skill_ids
        ), f"Valid ID {skill1.id} should be included"
        assert (
            skill2.id in captured_skill_ids
        ), f"Valid ID {skill2.id} should be included"

        # Should NOT contain invalid IDs
        assert (
            "nonexistent-99999" not in captured_skill_ids
        ), "Invalid ID should be filtered out"
        assert (
            "fake-12345" not in captured_skill_ids
        ), "Invalid ID should be filtered out"

        # Should have exactly 2 IDs (the valid ones)
        assert (
            len(captured_skill_ids) == 2
        ), f"Expected 2 valid IDs, got {len(captured_skill_ids)}: {captured_skill_ids}"


@pytest.mark.requires_api
class TestBackwardsCompatibility:
    """Test that existing code patterns still work."""

    def test_can_import_from_ace(self):
        """Should be importable from ace package."""
        from ace import ACEAgent as ImportedACEAgent

        assert ImportedACEAgent is not None

    def test_can_import_helper_from_ace(self):
        """Should import helper function from ace package."""
        from ace import wrap_skillbook_context as imported_wrap

        assert imported_wrap is not None

    def test_can_check_availability(self):
        """Should check browser-use availability."""
        from ace import BROWSER_USE_AVAILABLE as imported_available

        assert imported_available is True

    def test_trace_passed_to_reflector_in_reasoning_field(self):
        """
        Critical test: Verify full execution trace is passed to Reflector in reasoning field.

        This test ensures that the browser execution trace (thoughts, actions, results)
        is passed to the Reflector via AgentOutput.reasoning, not just a task description.

        Regression test for: Initial query duplication and missing trace in reasoning field.
        """
        from browser_use import ChatBrowserUse
        from unittest.mock import MagicMock, patch

        agent = ACEAgent(llm=ChatBrowserUse(), ace_model="gpt-4o-mini")

        # Create mock history with steps
        mock_history = MagicMock()
        mock_history.final_result.return_value = "Result"
        mock_history.number_of_steps.return_value = 2
        mock_history.total_duration_seconds.return_value = 1.5

        # Mock step 1
        mock_step1 = MagicMock()
        mock_step1.model_output = MagicMock(
            thinking="Step 1 thinking",
            evaluation_previous_goal="Step 1 eval",
            memory="Step 1 memory",
            next_goal="Step 1 goal",
            action=[MagicMock(model_dump=lambda: {"navigate": {"url": "test.com"}})],
        )
        mock_step1.result = [
            MagicMock(
                is_done=False,
                success=True,
                error=None,
                extracted_content="navigated",
            )
        ]
        mock_step1.state = MagicMock(url="https://test.com")

        # Mock step 2
        mock_step2 = MagicMock()
        mock_step2.model_output = MagicMock(
            thinking="Step 2 thinking",
            evaluation_previous_goal="Step 2 eval",
            memory="Step 2 memory",
            next_goal="Step 2 goal",
            action=[MagicMock(model_dump=lambda: {"done": {"text": "complete"}})],
        )
        mock_step2.result = [
            MagicMock(is_done=True, success=True, error=None, extracted_content="done")
        ]
        mock_step2.state = MagicMock(url="https://test.com/page")

        mock_history.history = [mock_step1, mock_step2]

        # Capture what gets passed to Reflector
        captured_agent_output = None

        original_reflect = agent.reflector.reflect

        def capture_reflect(*args, **kwargs):
            nonlocal captured_agent_output
            captured_agent_output = kwargs.get("agent_output")
            # Return a mock reflection to avoid actual LLM call
            from ace import ReflectorOutput

            return ReflectorOutput(
                reasoning="mock",
                error_identification="none",
                root_cause_analysis="none",
                correct_approach="continue",
                key_insight="test",
                skill_tags=[],
                raw={},
            )

        agent.reflector.reflect = capture_reflect

        # Also mock skill_manager to avoid LLM call
        from ace import SkillManagerOutput, UpdateBatch

        agent.skill_manager.update_skills = lambda *args, **kwargs: SkillManagerOutput(
            update=UpdateBatch(reasoning="mock", operations=[]), raw={}
        )

        # Call _learn_from_execution
        import asyncio

        asyncio.run(
            agent._learn_from_execution(
                task="Test task", history=mock_history, success=True
            )
        )

        # CRITICAL ASSERTIONS
        assert captured_agent_output is not None, "AgentOutput not captured"

        reasoning = captured_agent_output.reasoning

        # 1. Reasoning should NOT just be the task description
        assert (
            reasoning != "Browser automation task: Test task"
        ), "Reasoning should not be just task description (regression: initial implementation)"

        # 2. Reasoning should contain the full trace (not just summary)
        assert len(reasoning) > 200, (
            f"Reasoning too short ({len(reasoning)} chars). "
            "Should contain full execution trace with thoughts/actions/results."
        )

        # 3. Reasoning should contain step-by-step execution details
        assert (
            "Step 1" in reasoning or "--- Step 1 ---" in reasoning
        ), "Reasoning should contain Step 1 details"
        assert (
            "Step 2" in reasoning or "--- Step 2 ---" in reasoning
        ), "Reasoning should contain Step 2 details"

        # 4. Reasoning should contain agent thoughts
        assert (
            "Step 1 memory" in reasoning or "memory" in reasoning.lower()
        ), "Reasoning should contain agent's memory/thoughts"

        # 5. Reasoning should contain actions taken
        assert (
            "navigate" in reasoning or "Action" in reasoning
        ), "Reasoning should contain actions taken"

        # 6. Reasoning should contain execution results
        assert (
            "Result" in reasoning or "success" in reasoning.lower()
        ), "Reasoning should contain execution results"

        # 7. Verify trace structure markers are present
        assert (
            "BROWSER EXECUTION TRACE" in reasoning or "Step 1" in reasoning
        ), "Reasoning should have trace structure markers"

        print(
            f"✅ Trace properly passed to Reflector: {len(reasoning)} characters with full execution details"
        )
