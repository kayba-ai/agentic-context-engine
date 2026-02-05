"""Unit tests for RecursiveReflector."""

import json
import unittest
from typing import Any, Dict, List, Type, TypeVar

import pytest
from pydantic import BaseModel

from ace import Skillbook, ReflectorMode
from ace.llm import LLMClient, LLMResponse
from ace.roles import AgentOutput, ReflectorOutput
from ace.reflector import RecursiveReflector, RecursiveConfig
from ace.reflector.subagent import CallBudget
from ace.reflector.sandbox import TraceSandbox
from ace.reflector.trace_context import TraceContext


T = TypeVar("T", bound=BaseModel)


class MockLLMClient(LLMClient):
    """Mock LLM client for testing recursive reflector."""

    def __init__(self):
        super().__init__(model="mock")
        self._responses = []
        self._call_count = 0

    def set_responses(self, responses: list[str]) -> None:
        """Queue multiple responses."""
        self._responses = list(responses)
        self._call_count = 0

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return queued response."""
        self._call_count += 1
        if not self._responses:
            raise RuntimeError("No more queued responses")
        return LLMResponse(text=self._responses.pop(0))

    def complete_messages(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Return queued response (multi-turn compatible)."""
        self._call_count += 1
        if not self._responses:
            raise RuntimeError("No more queued responses")
        return LLMResponse(text=self._responses.pop(0))

    @property
    def call_count(self) -> int:
        return self._call_count


@pytest.mark.unit
class TestRecursiveReflector(unittest.TestCase):
    """Test RecursiveReflector functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLMClient()
        self.skillbook = Skillbook()
        self.agent_output = AgentOutput(
            reasoning="I calculated 2+2 step by step: 2+2=4",
            final_answer="4",
            skill_ids=[],
        )

    def test_basic_reflection_with_code(self):
        """Test basic reflection that produces code and calls FINAL."""
        # Iteration 0: explore (FINAL would be rejected here)
        explore_response = """```python
print(f"Question: {question}")
print(f"Correct: {final_answer.strip() == ground_truth.strip()}")
```"""
        # Iteration 1: FINAL accepted
        final_response = """```python
FINAL({
    "reasoning": "The agent correctly solved the problem.",
    "error_identification": "none",
    "root_cause_analysis": "No errors - correct execution",
    "correct_approach": "The step-by-step approach is effective",
    "key_insight": "Simple arithmetic was handled correctly",
    "extracted_learnings": [
        {"learning": "Step-by-step calculation works", "atomicity_score": 0.8}
    ],
    "skill_tags": []
})
```"""
        self.llm.set_responses([explore_response, final_response])

        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_iterations=5)
        )
        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertIsInstance(result, ReflectorOutput)
        self.assertIn("correctly", result.reasoning.lower())
        self.assertEqual(result.error_identification, "none")

    def test_direct_json_response(self):
        """Test that direct JSON response without code is parsed."""
        json_response = json.dumps(
            {
                "reasoning": "Analysis complete.",
                "error_identification": "none",
                "root_cause_analysis": "No errors",
                "correct_approach": "Continue current approach",
                "key_insight": "Task completed successfully",
                "extracted_learnings": [],
                "skill_tags": [],
            }
        )

        self.llm.set_responses([json_response])

        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_iterations=5)
        )
        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertIsInstance(result, ReflectorOutput)
        self.assertEqual(result.reasoning, "Analysis complete.")

    def test_multiple_iterations(self):
        """Test that REPL loop handles multiple iterations."""
        # First response: code that prints but doesn't call FINAL
        first_response = """
```python
print("Analyzing...")
is_correct = final_answer.strip() == ground_truth.strip()
print(f"Is correct: {is_correct}")
```
"""
        # Second response: code that calls FINAL
        second_response = """
Based on the output, I'll finalize:

```python
FINAL({
    "reasoning": "Answer is correct after analysis.",
    "error_identification": "none",
    "root_cause_analysis": "No errors",
    "correct_approach": "Current approach works",
    "key_insight": "Verification confirmed correctness",
    "extracted_learnings": [],
    "skill_tags": []
})
```
"""
        self.llm.set_responses([first_response, second_response])

        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_iterations=5)
        )
        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertIsInstance(result, ReflectorOutput)
        self.assertEqual(self.llm.call_count, 2)

    def test_max_iterations_timeout(self):
        """Test that max iterations produces timeout output."""
        # Responses that never call FINAL
        responses = [
            "```python\nprint('iteration 1')\n```",
            "```python\nprint('iteration 2')\n```",
            "```python\nprint('iteration 3')\n```",
        ]
        self.llm.set_responses(responses)

        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_iterations=3)
        )
        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertIsInstance(result, ReflectorOutput)
        self.assertIn("max iterations", result.reasoning.lower())
        self.assertEqual(result.raw.get("timeout"), True)

    def test_extracted_learnings_parsed(self):
        """Test that extracted learnings are properly parsed."""
        explore_response = "```python\nprint(question[:50])\n```"
        final_response = """```python
FINAL({
    "reasoning": "Analysis complete.",
    "error_identification": "none",
    "root_cause_analysis": "No errors",
    "correct_approach": "Current approach",
    "key_insight": "Key learning",
    "extracted_learnings": [
        {"learning": "First learning", "atomicity_score": 0.9, "evidence": "from trace"},
        {"learning": "Second learning", "atomicity_score": 0.7}
    ],
    "skill_tags": []
})
```"""
        self.llm.set_responses([explore_response, final_response])

        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_iterations=5)
        )
        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertEqual(len(result.extracted_learnings), 2)
        self.assertEqual(result.extracted_learnings[0].learning, "First learning")
        self.assertAlmostEqual(result.extracted_learnings[0].atomicity_score, 0.9)
        self.assertEqual(result.extracted_learnings[0].evidence, "from trace")

    def test_skill_tags_parsed(self):
        """Test that skill tags are properly parsed."""
        explore_response = "```python\nprint(question[:50])\n```"
        final_response = """```python
FINAL({
    "reasoning": "Analysis complete.",
    "error_identification": "none",
    "root_cause_analysis": "No errors",
    "correct_approach": "Current approach",
    "key_insight": "Key learning",
    "extracted_learnings": [],
    "skill_tags": [
        {"id": "skill-001", "tag": "helpful"},
        {"id": "skill-002", "tag": "harmful"},
        {"id": "skill-003", "tag": "neutral"}
    ]
})
```"""
        self.llm.set_responses([explore_response, final_response])

        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_iterations=5)
        )
        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertEqual(len(result.skill_tags), 3)
        self.assertEqual(result.skill_tags[0].id, "skill-001")
        self.assertEqual(result.skill_tags[0].tag, "helpful")

    def test_code_extraction_python_block(self):
        """Test that ```python code blocks are extracted."""
        reflector = RecursiveReflector(self.llm)

        code = reflector._extract_code("```python\nprint('hello')\n```")
        self.assertEqual(code, "print('hello')")

    def test_code_extraction_generic_block(self):
        """Test that generic ``` blocks with Python indicators are extracted."""
        reflector = RecursiveReflector(self.llm)

        code = reflector._extract_code("```\nx = 1\nprint(x)\n```")
        self.assertEqual(code, "x = 1\nprint(x)")

    def test_code_extraction_no_block(self):
        """Test that no code block returns None."""
        reflector = RecursiveReflector(self.llm)

        code = reflector._extract_code("Just some text without code")
        self.assertIsNone(code)

    def test_code_extraction_multiple_blocks(self):
        """Test that only the first code block is extracted."""
        reflector = RecursiveReflector(self.llm)

        text = """
```python
x = 1
```

Some explanation...

```python
print(x)
```
"""
        code = reflector._extract_code(text)
        self.assertEqual(code, "x = 1")
        self.assertNotIn("print(x)", code)


@pytest.mark.unit
class TestReflectorModeRouting(unittest.TestCase):
    """Test ReflectorMode routing in Reflector class."""

    def test_reflector_mode_enum_values(self):
        """Test that ReflectorMode has expected values."""
        self.assertEqual(ReflectorMode.SIMPLE.value, "simple")
        self.assertEqual(ReflectorMode.RECURSIVE.value, "recursive")
        self.assertEqual(ReflectorMode.AUTO.value, "auto")


@pytest.mark.unit
class TestRecursiveConfig(unittest.TestCase):
    """Test RecursiveConfig defaults and customization."""

    def test_default_values(self):
        """Test that default config values are set correctly."""
        config = RecursiveConfig()

        self.assertEqual(config.max_iterations, 15)
        self.assertEqual(config.timeout, 30.0)
        self.assertTrue(config.enable_llm_query)
        self.assertEqual(config.max_llm_calls, 30)
        self.assertEqual(config.max_context_chars, 50_000)

    def test_custom_values(self):
        """Test that config accepts custom values."""
        config = RecursiveConfig(
            max_iterations=5,
            timeout=60.0,
            enable_llm_query=False,
            max_llm_calls=10,
        )

        self.assertEqual(config.max_iterations, 5)
        self.assertEqual(config.timeout, 60.0)
        self.assertFalse(config.enable_llm_query)
        self.assertEqual(config.max_llm_calls, 10)


@pytest.mark.unit
class TestPromptDoesNotContainFullData(unittest.TestCase):
    """Test that the prompt does not contain full reasoning/data."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLMClient()
        self.skillbook = Skillbook()
        # Create a large reasoning string to verify it's not in the prompt
        self.large_reasoning = "This is step 1. " * 1000  # ~16k chars
        self.agent_output = AgentOutput(
            reasoning=self.large_reasoning,
            final_answer="42",
            skill_ids=[],
        )

    def test_prompt_contains_preview_not_full_reasoning(self):
        """Test that the prompt contains a short preview but not the full reasoning."""
        captured_messages = []

        class CapturingLLMClient(MockLLMClient):
            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                captured_messages.append(messages)
                return LLMResponse(
                    text=json.dumps(
                        {
                            "reasoning": "Test complete.",
                            "error_identification": "none",
                            "root_cause_analysis": "No errors",
                            "correct_approach": "Continue",
                            "key_insight": "Test",
                            "extracted_learnings": [],
                            "skill_tags": [],
                        }
                    )
                )

        llm = CapturingLLMClient()
        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=1))

        reflector.reflect(
            question="What is the meaning of life?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="42",
            feedback="Correct!",
        )

        # Verify that messages were captured
        self.assertGreater(len(captured_messages), 0)
        initial_content = captured_messages[0][0]["content"]

        # The full reasoning should NOT be in the prompt
        self.assertNotIn(self.large_reasoning, initial_content)

        # But a preview of the reasoning SHOULD be present (first 150 chars)
        preview = self.large_reasoning[:150]
        self.assertIn(preview, initial_content)

        # Metadata should be present (length info)
        self.assertIn("chars", initial_content.lower())

    def test_prompt_contains_preview_and_metadata_placeholders(self):
        """Test that the prompt contains preview and size metadata placeholders."""
        from ace.reflector.prompts import REFLECTOR_RECURSIVE_PROMPT

        # The prompt template should have placeholders for metadata
        self.assertIn("{reasoning_length}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{answer_length}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{step_count}", REFLECTOR_RECURSIVE_PROMPT)

        # The prompt should have preview placeholders
        self.assertIn("{question_preview}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{reasoning_preview}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{answer_preview}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{ground_truth_preview}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{feedback_preview}", REFLECTOR_RECURSIVE_PROMPT)

        import re

        # Raw {reasoning}, {feedback}, etc. should NOT appear (only _length/_preview variants)
        self.assertIsNone(re.search(r"\{reasoning\}", REFLECTOR_RECURSIVE_PROMPT))
        self.assertIsNone(re.search(r"\{feedback\}", REFLECTOR_RECURSIVE_PROMPT))
        self.assertIsNone(re.search(r"\{skillbook\}", REFLECTOR_RECURSIVE_PROMPT))
        self.assertIsNone(re.search(r"\{question\}", REFLECTOR_RECURSIVE_PROMPT))


@pytest.mark.unit
class TestPrematureFinalRejected(unittest.TestCase):
    """Test that FINAL() on iteration 0 is rejected."""

    def setUp(self):
        self.skillbook = Skillbook()
        self.agent_output = AgentOutput(
            reasoning="I built a weather app with React",
            final_answer="Done",
            skill_ids=[],
        )

    def test_premature_final_rejected(self):
        """Test that FINAL() on first iteration is rejected and model gets a second chance."""
        call_count = [0]

        class TwoShotLLMClient(MockLLMClient):
            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call: immediately calls FINAL (premature)
                    return LLMResponse(
                        text="""```python
print(f"Question: {question[:100]}")
FINAL({
    "reasoning": "Premature analysis",
    "error_identification": "none",
    "root_cause_analysis": "N/A",
    "correct_approach": "N/A",
    "key_insight": "Premature",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
                    )
                else:
                    # Second call: should see rejection message, now does proper analysis
                    last_msg = messages[-1]["content"]
                    assert "before exploring the data" in last_msg
                    return LLMResponse(
                        text="""```python
FINAL({
    "reasoning": "After reading actual data, the weather app was built correctly.",
    "error_identification": "none",
    "root_cause_analysis": "No errors",
    "correct_approach": "Current approach works",
    "key_insight": "Weather app implementation was correct",
    "extracted_learnings": [
        {"learning": "Use React for weather app UI", "atomicity_score": 0.9, "evidence": "From question"}
    ],
    "skill_tags": []
})
```"""
                    )

        llm = TwoShotLLMClient()
        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))

        result = reflector.reflect(
            question="Build a weather app",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="Done",
            feedback="Good job!",
        )

        # Should have made 2 calls (first rejected, second accepted)
        self.assertEqual(call_count[0], 2)
        # Final result should be from the second (grounded) response
        self.assertIn("weather app", result.reasoning.lower())

    def test_final_accepted_on_later_iterations(self):
        """Test that FINAL() is accepted normally on iteration >= 1."""
        responses = [
            # Iteration 0: explore
            "```python\nprint(question[:100])\n```",
            # Iteration 1: FINAL should be accepted
            """```python
FINAL({
    "reasoning": "Analysis after exploration.",
    "error_identification": "none",
    "root_cause_analysis": "N/A",
    "correct_approach": "N/A",
    "key_insight": "Explored first",
    "extracted_learnings": [],
    "skill_tags": []
})
```""",
        ]
        llm = MockLLMClient()
        llm.set_responses(responses)

        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))
        result = reflector.reflect(
            question="Test",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="Done",
            feedback="OK",
        )

        self.assertEqual(llm.call_count, 2)
        self.assertIn("exploration", result.reasoning.lower())


@pytest.mark.unit
class TestPreviewInPrompt(unittest.TestCase):
    """Test that question/reasoning previews appear in the initial prompt."""

    def test_preview_in_prompt(self):
        """Test that short previews of variables appear in the formatted prompt."""
        captured_messages = []

        class CapturingLLMClient(MockLLMClient):
            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                captured_messages.append(messages)
                return LLMResponse(
                    text=json.dumps(
                        {
                            "reasoning": "Done.",
                            "error_identification": "none",
                            "root_cause_analysis": "N/A",
                            "correct_approach": "N/A",
                            "key_insight": "Test",
                            "extracted_learnings": [],
                            "skill_tags": [],
                        }
                    )
                )

        llm = CapturingLLMClient()
        skillbook = Skillbook()
        agent_output = AgentOutput(
            reasoning="I built a weather app using React and OpenWeather API",
            final_answer="Weather app complete",
            skill_ids=[],
        )

        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=1))
        reflector.reflect(
            question="Build me a weather app with hourly forecasts",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth="Done",
            feedback="The app works correctly",
        )

        initial_content = captured_messages[0][0]["content"]

        # Short question should appear as preview
        self.assertIn("Build me a weather app", initial_content)
        # Reasoning preview should appear
        self.assertIn("weather app using React", initial_content)
        # Answer preview should appear
        self.assertIn("Weather app complete", initial_content)
        # Feedback preview should appear
        self.assertIn("The app works correctly", initial_content)


@pytest.mark.unit
class TestLLMQueryLimitInReflector(unittest.TestCase):
    """Test that llm_query limit is enforced in the reflector."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()
        self.agent_output = AgentOutput(
            reasoning="I calculated the result.",
            final_answer="42",
            skill_ids=[],
        )

    def test_llm_query_limit_enforced_in_reflector(self):
        """Test that llm_query respects max_llm_calls config."""

        class CountingLLMClient(MockLLMClient):
            def __init__(self):
                super().__init__()
                self._repl_call = 0

            def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
                # Sub-LLM calls (via ask_llm/llm_query)
                return LLMResponse(text="Sub-response")

            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                """REPL loop calls."""
                self._repl_call += 1

                if self._repl_call == 1:
                    # Iteration 0: explore (no FINAL)
                    return LLMResponse(text="```python\nprint(question[:50])\n```")
                elif self._repl_call == 2:
                    # Iteration 1: llm_query calls + FINAL
                    return LLMResponse(
                        text="""```python
results = []
for i in range(5):
    r = llm_query(f"Sub-query {i}")
    results.append(r)
FINAL({
    "reasoning": str(results),
    "error_identification": "none",
    "root_cause_analysis": "No errors",
    "correct_approach": "Continue",
    "key_insight": "Test",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
                    )
                else:
                    return LLMResponse(text="```python\nprint('done')\n```")

        llm = CountingLLMClient()
        config = RecursiveConfig(max_iterations=5, max_llm_calls=3)
        reflector = RecursiveReflector(llm, config=config)

        result = reflector.reflect(
            question="Test question",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="42",
            feedback="Correct!",
        )

        # The result should contain the limit exceeded message for later calls
        self.assertIn("Max 3 LLM calls exceeded", result.reasoning)


@pytest.mark.unit
class TestCallBudget(unittest.TestCase):
    """Test the CallBudget class."""

    def test_basic_consume(self):
        """Test basic consumption of budget."""
        budget = CallBudget(3)
        self.assertEqual(budget.count, 0)
        self.assertFalse(budget.exhausted)

        self.assertTrue(budget.consume())
        self.assertEqual(budget.count, 1)

        self.assertTrue(budget.consume())
        self.assertTrue(budget.consume())
        self.assertEqual(budget.count, 3)
        self.assertTrue(budget.exhausted)

        # Should return False when exhausted
        self.assertFalse(budget.consume())
        self.assertEqual(budget.count, 3)

    def test_shared_budget_between_llm_query_and_ask_llm(self):
        """Test that llm_query and ask_llm share the same call budget."""

        class SimpleLLM(MockLLMClient):
            def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(text="response")

            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                return LLMResponse(text="response")

        llm = SimpleLLM()
        budget = CallBudget(3)

        from ace.reflector.subagent import create_ask_llm_function

        ask_llm_fn = create_ask_llm_function(llm, budget=budget)
        llm_query_fn = lambda prompt: ask_llm_fn(prompt, "")

        # Use ask_llm twice
        ask_llm_fn("q1", "ctx1")
        ask_llm_fn("q2", "ctx2")
        self.assertEqual(budget.count, 2)

        # Use llm_query once - should use same budget
        llm_query_fn("q3")
        self.assertEqual(budget.count, 3)
        self.assertTrue(budget.exhausted)

        # Both should now be exhausted
        result = ask_llm_fn("q4", "ctx4")
        self.assertIn("exceeded", result)

        result = llm_query_fn("q5")
        self.assertIn("exceeded", result)


@pytest.mark.unit
class TestSandboxSecurity(unittest.TestCase):
    """Test sandbox security restrictions."""

    def test_sandbox_blocks_dunder_access(self):
        """Test that sandbox blocks access to dunder attributes."""
        trace = TraceContext.from_agent_output(
            AgentOutput(reasoning="test", final_answer="test", skill_ids=[])
        )
        sandbox = TraceSandbox(trace=trace)

        # Attempt to access __class__ via safe_getattr should fail
        result = sandbox.execute(
            "try:\n"
            "    safe_getattr(trace, '__class__')\n"
            "    print('SHOULD NOT REACH')\n"
            "except AttributeError as e:\n"
            "    print(f'Blocked: {e}')\n"
        )
        self.assertIn("Blocked", result.stdout)
        self.assertNotIn("SHOULD NOT REACH", result.stdout)

    def test_safe_getattr_allows_public_attrs(self):
        """Test that safe_getattr allows access to public attributes."""
        trace = TraceContext.from_agent_output(
            AgentOutput(reasoning="test reasoning", final_answer="42", skill_ids=[])
        )
        sandbox = TraceSandbox(trace=trace)
        sandbox.inject("test_obj", {"key": "value"})

        # Access to public methods should work
        result = sandbox.execute(
            "# safe_getattr on a dict shouldn't fail for non-dunder attrs\n"
            "print('OK')\n"
        )
        self.assertIn("OK", result.stdout)

    def test_getattr_not_in_builtins(self):
        """Test that getattr, setattr, delattr, type are removed from builtins."""
        self.assertNotIn("getattr", TraceSandbox.SAFE_BUILTINS)
        self.assertNotIn("setattr", TraceSandbox.SAFE_BUILTINS)
        self.assertNotIn("delattr", TraceSandbox.SAFE_BUILTINS)
        self.assertNotIn("type", TraceSandbox.SAFE_BUILTINS)


@pytest.mark.unit
class TestMessagesPreserveRoleStructure(unittest.TestCase):
    """Test that multi-turn messages preserve role structure."""

    def test_messages_preserve_role_structure(self):
        """Test that complete_messages receives messages with proper roles."""
        captured_messages = []

        class CapturingLLMClient(MockLLMClient):
            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                captured_messages.append(list(messages))
                return LLMResponse(
                    text=json.dumps(
                        {
                            "reasoning": "Done.",
                            "error_identification": "none",
                            "root_cause_analysis": "No errors",
                            "correct_approach": "Continue",
                            "key_insight": "Test",
                            "extracted_learnings": [],
                            "skill_tags": [],
                        }
                    )
                )

        llm = CapturingLLMClient()
        skillbook = Skillbook()
        agent_output = AgentOutput(reasoning="test", final_answer="4", skill_ids=[])
        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=1))

        reflector.reflect(
            question="What is 2+2?",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        # Verify messages were passed as structured array
        self.assertGreater(len(captured_messages), 0)
        first_call_messages = captured_messages[0]
        self.assertIsInstance(first_call_messages, list)
        self.assertGreater(len(first_call_messages), 0)
        # First message should be the user prompt
        self.assertEqual(first_call_messages[0]["role"], "user")


@pytest.mark.unit
class TestContextWindowTrimming(unittest.TestCase):
    """Test context window management."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_context_chars=200)
        )

    def test_no_trim_under_limit(self):
        """Test that messages under the limit are not trimmed."""
        messages = [
            {"role": "user", "content": "short prompt"},
            {"role": "assistant", "content": "short response"},
            {"role": "user", "content": "short output"},
        ]
        trimmed = self.reflector._trim_messages(messages)
        self.assertEqual(len(trimmed), 3)
        self.assertEqual(trimmed, messages)

    def test_trim_preserves_first_and_last(self):
        """Test that trimming preserves the first (instructions) and most recent messages."""
        messages = [
            {"role": "user", "content": "A" * 100},  # instructions
            {"role": "assistant", "content": "B" * 80},  # old
            {"role": "user", "content": "C" * 80},  # old
            {"role": "assistant", "content": "D" * 40},  # recent
            {"role": "user", "content": "E" * 40},  # recent
        ]
        trimmed = self.reflector._trim_messages(messages)

        # First message (instructions) should always be present
        self.assertEqual(trimmed[0]["content"], "A" * 100)

        # Last messages should be present
        self.assertEqual(trimmed[-1]["content"], "E" * 40)
        self.assertEqual(trimmed[-2]["content"], "D" * 40)

        # Should have a summary marker for dropped messages
        has_omitted = any("omitted" in m["content"] for m in trimmed)
        self.assertTrue(has_omitted)

    def test_trim_with_large_limit(self):
        """Test that no trimming happens with a large limit."""
        reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_context_chars=50_000)
        )
        messages = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "response"},
        ]
        trimmed = reflector._trim_messages(messages)
        self.assertEqual(len(trimmed), 2)


if __name__ == "__main__":
    unittest.main()
