"""Unit tests for TraceContext factory methods and conversation parsing."""

import re
import unittest

import pytest

from ace.reflector.trace_context import TraceContext, TraceStep


@pytest.mark.unit
class TestTraceStepMethods(unittest.TestCase):
    """Test TraceStep __str__, content, and preview methods."""

    def test_str_with_thought_and_observation(self):
        """Test __str__ shows thought and observation."""
        step = TraceStep(
            index=0,
            action="search_api",
            thought="Searching for user data",
            observation="Found 5 results",
        )
        result = str(step)

        self.assertIn("Step 0 [search_api]", result)
        self.assertIn("Thought: Searching for user data", result)
        self.assertIn("Observation: Found 5 results", result)

    def test_str_truncates_long_content(self):
        """Test __str__ truncates content over 200 chars."""
        long_thought = "x" * 250
        step = TraceStep(
            index=1,
            action="test",
            thought=long_thought,
            observation="short",
        )
        result = str(step)

        # Should truncate to 200 + "..."
        self.assertIn("x" * 200 + "...", result)
        self.assertNotIn("x" * 250, result)

    def test_str_with_empty_fields(self):
        """Test __str__ omits empty thought/observation."""
        step = TraceStep(
            index=0,
            action="navigate",
            thought="",
            observation="Page loaded",
        )
        result = str(step)

        self.assertIn("Step 0 [navigate]", result)
        self.assertNotIn("Thought:", result)
        self.assertIn("Observation: Page loaded", result)

    def test_content_property_combines_thought_and_observation(self):
        """Test content property joins thought and observation."""
        step = TraceStep(
            index=0,
            action="test",
            thought="First part",
            observation="Second part",
        )
        self.assertEqual(step.content, "First part\nSecond part")

    def test_content_property_with_only_thought(self):
        """Test content property with only thought."""
        step = TraceStep(
            index=0,
            action="test",
            thought="Only thought",
            observation="",
        )
        self.assertEqual(step.content, "Only thought")

    def test_content_property_with_only_observation(self):
        """Test content property with only observation."""
        step = TraceStep(
            index=0,
            action="test",
            thought="",
            observation="Only observation",
        )
        self.assertEqual(step.content, "Only observation")

    def test_content_property_empty(self):
        """Test content property with both empty."""
        step = TraceStep(
            index=0,
            action="test",
            thought="",
            observation="",
        )
        self.assertEqual(step.content, "")

    def test_preview_short_content(self):
        """Test preview returns full content when short."""
        step = TraceStep(
            index=0,
            action="test",
            thought="Short thought",
            observation="Short obs",
        )
        result = step.preview(max_len=300)
        self.assertEqual(result, "Short thought\nShort obs")

    def test_preview_truncates_long_content(self):
        """Test preview truncates content over max_len."""
        step = TraceStep(
            index=0,
            action="test",
            thought="x" * 400,
            observation="y" * 100,
        )
        result = step.preview(max_len=300)

        self.assertTrue(result.startswith("x" * 300))
        self.assertIn("more chars", result)
        # Total content is 400 + 1 (newline) + 100 = 501
        self.assertIn("201 more chars", result)

    def test_preview_custom_max_len(self):
        """Test preview with custom max_len."""
        step = TraceStep(
            index=0,
            action="test",
            thought="a" * 100,
            observation="",
        )
        result = step.preview(max_len=50)

        self.assertTrue(result.startswith("a" * 50))
        self.assertIn("50 more chars", result)


@pytest.mark.unit
class TestParseConversationMarkers(unittest.TestCase):
    """Test _parse_conversation_markers static method."""

    def test_basic_markers(self):
        """Test parsing basic [assistant]/[user] markers."""
        text = (
            "[assistant] I'll help you with that.\n"
            "[user] Thanks, can you also check the logs?\n"
            "[assistant] Sure, here are the logs."
        )
        messages = TraceContext._parse_conversation_markers(text)

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertIn("help you", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("check the logs", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("logs", messages[2]["content"])

    def test_empty_text(self):
        """Test that empty text returns empty list."""
        messages = TraceContext._parse_conversation_markers("")
        self.assertEqual(messages, [])

    def test_no_markers(self):
        """Test that text without markers returns empty list."""
        messages = TraceContext._parse_conversation_markers(
            "Just some plain text without any markers."
        )
        self.assertEqual(messages, [])

    def test_empty_content_skipped(self):
        """Test that markers with empty content are skipped."""
        text = "[assistant] \n[user] Hello\n[assistant]  "
        messages = TraceContext._parse_conversation_markers(text)

        # Only the [user] Hello message should be included
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")

    def test_multiline_content(self):
        """Test parsing multiline content between markers."""
        text = (
            "[assistant] Here is the code:\n"
            "def hello():\n"
            "    print('hello')\n"
            "[user] Looks good!"
        )
        messages = TraceContext._parse_conversation_markers(text)

        self.assertEqual(len(messages), 2)
        self.assertIn("def hello():", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Looks good!")

    def test_case_insensitive_markers(self):
        """Test that markers are case-insensitive."""
        text = "[Assistant] Hello\n[User] Hi there"
        messages = TraceContext._parse_conversation_markers(text)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[1]["role"], "user")


@pytest.mark.unit
class TestFromConversationHistory(unittest.TestCase):
    """Test from_conversation_history factory method."""

    def test_string_content(self):
        """Test messages with string content."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        trace = TraceContext.from_conversation_history(messages)

        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0].action, "user")
        self.assertIn("2+2", trace[0].thought)
        self.assertEqual(trace[1].action, "assistant")
        self.assertIn("answer is 4", trace[1].thought)

    def test_block_content_with_text(self):
        """Test messages with list-of-blocks content containing text."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for that."},
                ],
            },
        ]
        trace = TraceContext.from_conversation_history(messages)

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].action, "assistant")
        self.assertIn("search for that", trace[0].thought)

    def test_block_content_with_tool_use(self):
        """Test messages with tool_use blocks get action labels."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Using the search tool."},
                    {"type": "tool_use", "name": "web_search", "id": "123"},
                ],
            },
        ]
        trace = TraceContext.from_conversation_history(messages)

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].action, "assistant:web_search")

    def test_truncation(self):
        """Test that max_text_len is respected."""
        long_text = "x" * 5000
        messages = [{"role": "user", "content": long_text}]
        trace = TraceContext.from_conversation_history(messages, max_text_len=100)

        self.assertEqual(len(trace[0].thought), 100)

    def test_raw_reasoning_built(self):
        """Test that raw_reasoning is built from messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        trace = TraceContext.from_conversation_history(messages)

        self.assertIn("[user] Hello", trace.raw_reasoning)
        self.assertIn("[assistant] Hi there", trace.raw_reasoning)

    def test_empty_messages(self):
        """Test with empty message list."""
        trace = TraceContext.from_conversation_history([])
        self.assertEqual(len(trace), 0)

    def test_raw_reasoning_limited_to_last_20(self):
        """Test that raw_reasoning only includes last 20 messages."""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(30)]
        trace = TraceContext.from_conversation_history(messages)

        # All 30 steps should be present
        self.assertEqual(len(trace), 30)
        # But raw_reasoning should only have last 20
        self.assertNotIn("Message 0", trace.raw_reasoning)
        self.assertIn("Message 29", trace.raw_reasoning)


@pytest.mark.unit
class TestFromAgentOutputConversationMarkers(unittest.TestCase):
    """Test from_agent_output with conversation marker auto-detection."""

    def _make_agent_output(self, reasoning: str, final_answer: str = "done"):
        """Create a mock AgentOutput."""
        from ace.roles import AgentOutput

        return AgentOutput(
            reasoning=reasoning,
            final_answer=final_answer,
            skill_ids=[],
        )

    def test_auto_detect_conversation_markers(self):
        """Test that [assistant]/[user] markers trigger multi-step trace."""
        reasoning = (
            "[assistant] I'll search for the answer.\n"
            "[user] Please also check the database.\n"
            "[assistant] Found it in the database."
        )
        output = self._make_agent_output(reasoning, final_answer="42")
        trace = TraceContext.from_agent_output(output)

        # Should create multiple steps from conversation
        self.assertGreater(len(trace), 1)

    def test_final_answer_appended(self):
        """Test that final answer is appended to last step's observation."""
        reasoning = (
            "[assistant] Calculating...\n"
            "[user] What's the result?\n"
            "[assistant] The result is 42."
        )
        output = self._make_agent_output(reasoning, final_answer="42")
        trace = TraceContext.from_agent_output(output)

        last_step = trace[-1]
        self.assertIn("Final Answer: 42", last_step.observation)

    def test_fallback_without_markers(self):
        """Test fallback to single-step trace without markers."""
        reasoning = "I calculated 2+2 = 4"
        output = self._make_agent_output(reasoning, final_answer="4")
        trace = TraceContext.from_agent_output(output)

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].action, "reasoning")
        self.assertIn("Answer: 4", trace[0].observation)


@pytest.mark.unit
class TestFromBrowserUse(unittest.TestCase):
    """Test from_browser_use factory method."""

    def test_basic_browser_history(self):
        """Test with mock browser-use history object."""

        class MockHistoryItem:
            def __init__(self, action, thought, result):
                self.action = action
                self.thought = thought
                self.result = result

        class MockHistory:
            def __init__(self, items):
                self.history = items

            def __str__(self):
                return "MockHistory(3 items)"

        items = [
            MockHistoryItem("navigate", "Going to page", "Page loaded"),
            MockHistoryItem("click", "Clicking button", "Button clicked"),
            MockHistoryItem("extract", "Getting text", "Text: Hello"),
        ]
        history = MockHistory(items)
        trace = TraceContext.from_browser_use(history)

        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0].action, "navigate")
        self.assertEqual(trace[0].thought, "Going to page")
        self.assertEqual(trace[0].observation, "Page loaded")
        self.assertEqual(trace[2].action, "extract")

    def test_no_history_attribute(self):
        """Test with object that has no history attribute."""

        class NoHistory:
            def __str__(self):
                return "empty"

        trace = TraceContext.from_browser_use(NoHistory())
        self.assertEqual(len(trace), 0)
        self.assertEqual(trace.raw_reasoning, "empty")

    def test_none_history(self):
        """Test with None input."""
        trace = TraceContext.from_browser_use(None)
        self.assertEqual(len(trace), 0)
        self.assertEqual(trace.raw_reasoning, "")


@pytest.mark.unit
class TestFromLangchain(unittest.TestCase):
    """Test from_langchain factory method."""

    def test_basic_intermediate_steps(self):
        """Test with mock LangChain intermediate_steps tuples."""

        class MockAction:
            def __init__(self, tool, tool_input, log=""):
                self.tool = tool
                self.tool_input = tool_input
                self.log = log

        steps = [
            (MockAction("search", "python docs", "Searching..."), "Found 10 results"),
            (
                MockAction("summarize", "results", "Summarizing..."),
                "Summary: Python is great",
            ),
        ]
        trace = TraceContext.from_langchain(steps)

        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0].action, "search")
        self.assertEqual(trace[0].thought, "Searching...")
        self.assertEqual(trace[0].observation, "Found 10 results")
        self.assertEqual(trace[1].action, "summarize")

    def test_empty_steps(self):
        """Test with empty intermediate_steps."""
        trace = TraceContext.from_langchain([])
        self.assertEqual(len(trace), 0)

    def test_non_tuple_steps_skipped(self):
        """Test that non-tuple entries are skipped."""
        steps = ["not a tuple", 42]
        trace = TraceContext.from_langchain(steps)
        self.assertEqual(len(trace), 0)


@pytest.mark.unit
class TestSearchRaw(unittest.TestCase):
    """Test search_raw and search_raw_text methods."""

    def test_search_raw_returns_indices(self):
        """search_raw() should return step indices, not matched strings."""
        messages = [
            {"role": "user", "content": "fix the error"},
            {"role": "assistant", "content": "I'll help"},
            {"role": "user", "content": "still has error"},
        ]
        trace = TraceContext.from_conversation_history(messages)
        indices = trace.search_raw(r"error")

        # Should return list of integers (indices)
        self.assertTrue(all(isinstance(i, int) for i in indices))
        # Steps 0 and 2 contain "error"
        self.assertEqual(indices, [0, 2])

    def test_search_raw_returns_empty_on_no_match(self):
        """search_raw() should return empty list when no matches."""
        trace = TraceContext(
            steps=[
                TraceStep(index=0, action="test", thought="hello", observation="world")
            ],
            raw_reasoning="hello world",
        )
        indices = trace.search_raw(r"nonexistent")
        self.assertEqual(indices, [])

    def test_search_raw_text_returns_strings(self):
        """search_raw_text() should return matched substrings."""
        trace = TraceContext(
            steps=[],
            raw_reasoning="Found error in line 5. Another error in line 10.",
        )
        matches = trace.search_raw_text(r"error")
        self.assertEqual(matches, ["error", "error"])

    def test_search_raw_indices_can_be_used_for_slicing(self):
        """Verify indices from search_raw() can be used to access steps."""
        steps = [
            TraceStep(index=0, action="start", thought="beginning", observation=""),
            TraceStep(
                index=1,
                action="process",
                thought="working",
                observation="error occurred",
            ),
            TraceStep(index=2, action="end", thought="finished", observation=""),
        ]
        trace = TraceContext(steps=steps, raw_reasoning="")
        indices = trace.search_raw(r"error")

        # Should be able to use indices to access steps
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 1)
        error_step = trace[indices[0]]
        self.assertEqual(error_step.action, "process")


@pytest.mark.unit
class TestFindStepsRegex(unittest.TestCase):
    """Test find_steps_regex method."""

    def setUp(self):
        self.steps = [
            TraceStep(
                index=0,
                action="search_api",
                thought="Querying API",
                observation="200 OK",
            ),
            TraceStep(
                index=1,
                action="parse_json",
                thought="Parsing response",
                observation="Error: invalid JSON",
            ),
            TraceStep(
                index=2, action="retry_search", thought="Retrying", observation="200 OK"
            ),
        ]
        self.trace = TraceContext(steps=self.steps, raw_reasoning="")

    def test_regex_pattern_match(self):
        """Test regex pattern matching across fields."""
        results = self.trace.find_steps_regex(r"search", re.IGNORECASE)
        self.assertEqual(len(results), 2)  # search_api and retry_search

    def test_regex_observation_match(self):
        """Test regex matching in observations."""
        results = self.trace.find_steps_regex(r"Error:.*JSON")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].action, "parse_json")

    def test_no_match(self):
        """Test regex with no matches."""
        results = self.trace.find_steps_regex(r"nonexistent_pattern")
        self.assertEqual(len(results), 0)


@pytest.mark.unit
class TestGetErrorsAllFields(unittest.TestCase):
    """Test that get_errors() searches action, thought, and observation."""

    def test_error_in_observation(self):
        """Test that errors in observation field are found."""
        steps = [
            TraceStep(
                index=0, action="run", thought="running", observation="Error: timeout"
            ),
        ]
        trace = TraceContext(steps=steps, raw_reasoning="")
        errors = trace.get_errors()
        self.assertEqual(len(errors), 1)

    def test_error_in_thought(self):
        """Test that errors in thought field are found."""
        steps = [
            TraceStep(
                index=0, action="run", thought="Got an exception here", observation=""
            ),
        ]
        trace = TraceContext(steps=steps, raw_reasoning="")
        errors = trace.get_errors()
        self.assertEqual(len(errors), 1)

    def test_error_in_action(self):
        """Test that errors in action field are found."""
        steps = [
            TraceStep(
                index=0, action="handle_failure", thought="processing", observation="ok"
            ),
        ]
        trace = TraceContext(steps=steps, raw_reasoning="")
        errors = trace.get_errors()
        self.assertEqual(len(errors), 1)

    def test_no_errors(self):
        """Test that steps without error indicators return empty."""
        steps = [
            TraceStep(
                index=0, action="search", thought="looking", observation="found it"
            ),
        ]
        trace = TraceContext(steps=steps, raw_reasoning="")
        errors = trace.get_errors()
        self.assertEqual(len(errors), 0)

    def test_error_across_multiple_fields(self):
        """Test step with errors in multiple fields only appears once."""
        steps = [
            TraceStep(
                index=0,
                action="error_handler",
                thought="exception caught",
                observation="failed",
            ),
        ]
        trace = TraceContext(steps=steps, raw_reasoning="")
        errors = trace.get_errors()
        self.assertEqual(len(errors), 1)


@pytest.mark.unit
class TestEnableSubagentFalse(unittest.TestCase):
    """Test that enable_subagent=False injects a disabled stub."""

    def test_ask_llm_disabled_stub(self):
        """Test that ask_llm returns disabled message when subagent is off."""
        from unittest.mock import MagicMock

        from ace.reflector import RecursiveReflector, RecursiveConfig
        from ace.roles import AgentOutput

        # Mock LLM that generates code calling ask_llm, then FINAL
        mock_llm = MagicMock()
        mock_llm.complete_messages.return_value = MagicMock(
            text="""```python
result = ask_llm("What happened?", "some context")
FINAL({
    "reasoning": result,
    "error_identification": "none",
    "root_cause_analysis": "N/A",
    "correct_approach": "N/A",
    "key_insight": "N/A",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
        )

        config = RecursiveConfig(enable_subagent=False, max_iterations=3)
        reflector = RecursiveReflector(mock_llm, config=config)

        agent_output = AgentOutput(
            reasoning="Simple reasoning",
            final_answer="42",
            skill_ids=[],
        )
        from ace import Skillbook

        result = reflector.reflect(
            question="Test",
            agent_output=agent_output,
            skillbook=Skillbook(),
            ground_truth="42",
            feedback="Correct",
        )

        # The stub message should appear in the reasoning
        self.assertIn("ask_llm disabled", result.reasoning)


if __name__ == "__main__":
    unittest.main()
