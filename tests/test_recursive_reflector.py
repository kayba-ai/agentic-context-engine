"""Unit tests for RecursiveReflector."""

import json
import unittest
from typing import Any, Type, TypeVar

import pytest
from pydantic import BaseModel

from ace import Skillbook, ReflectorMode
from ace.llm import LLMClient, LLMResponse
from ace.roles import AgentOutput, ReflectorOutput
from ace.reflector import RecursiveReflector, RecursiveConfig


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
        # LLM produces code that analyzes and calls FINAL
        code_response = """
I'll analyze this trace.

```python
is_correct = final_answer.strip() == ground_truth.strip()
print(f"Correct: {is_correct}")

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
```
"""
        self.llm.set_responses([code_response])

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
        code_response = """
```python
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
```
"""
        self.llm.set_responses([code_response])

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
        code_response = """
```python
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
```
"""
        self.llm.set_responses([code_response])

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
        """Test that multiple code blocks are concatenated."""
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
        self.assertIn("x = 1", code)
        self.assertIn("print(x)", code)


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

        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.timeout, 30.0)
        self.assertTrue(config.enable_llm_query)
        self.assertEqual(config.max_llm_calls, 20)
        self.assertTrue(config.fallback_on_error)

    def test_custom_values(self):
        """Test that config accepts custom values."""
        config = RecursiveConfig(
            max_iterations=5,
            timeout=60.0,
            enable_llm_query=False,
            max_llm_calls=10,
            fallback_on_error=False,
        )

        self.assertEqual(config.max_iterations, 5)
        self.assertEqual(config.timeout, 60.0)
        self.assertFalse(config.enable_llm_query)
        self.assertEqual(config.max_llm_calls, 10)
        self.assertFalse(config.fallback_on_error)


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

    def test_prompt_does_not_contain_full_reasoning(self):
        """Test that the prompt template only contains metadata, not full content."""
        captured_prompts = []

        class CapturingLLMClient(MockLLMClient):
            def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
                captured_prompts.append(prompt)
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

        # Verify that prompt was captured
        self.assertGreater(len(captured_prompts), 0)
        initial_prompt = captured_prompts[0]

        # The full reasoning should NOT be in the prompt
        self.assertNotIn(self.large_reasoning, initial_prompt)
        self.assertNotIn("This is step 1.", initial_prompt)

        # Instead, only metadata should be present (length info)
        # Check that "chars" is mentioned (metadata about data sizes)
        self.assertIn("chars", initial_prompt.lower())

    def test_prompt_contains_metadata_instead_of_data(self):
        """Test that the prompt contains size metadata instead of data placeholders."""
        from ace.reflector.prompts import REFLECTOR_RECURSIVE_PROMPT

        # The prompt template should have placeholders for metadata
        self.assertIn("{reasoning_length}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{answer_length}", REFLECTOR_RECURSIVE_PROMPT)
        self.assertIn("{step_count}", REFLECTOR_RECURSIVE_PROMPT)

        # The prompt template should NOT have standalone data placeholders
        # (The prompt may mention variable names like `final_answer` in docs,
        # but should not have {reasoning} or {feedback} as .format() placeholders
        # that would be filled with actual content)

        # Check that these specific format placeholders don't exist
        # (they existed in the old prompt but were removed)
        import re

        # Match {reasoning} but not {reasoning_length}
        self.assertIsNone(re.search(r"\{reasoning\}", REFLECTOR_RECURSIVE_PROMPT))
        # Match {feedback} but not {feedback_length}
        self.assertIsNone(re.search(r"\{feedback\}", REFLECTOR_RECURSIVE_PROMPT))
        # Match {skillbook} but not {skillbook_length}
        self.assertIsNone(re.search(r"\{skillbook\}", REFLECTOR_RECURSIVE_PROMPT))
        # Match {question} but not {question_length}
        self.assertIsNone(re.search(r"\{question\}", REFLECTOR_RECURSIVE_PROMPT))


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
        llm_call_count = [0]  # Use list to allow modification in nested function

        class CountingLLMClient(MockLLMClient):
            def __init__(self):
                super().__init__()
                self._iteration = 0

            def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
                llm_call_count[0] += 1
                self._iteration += 1

                if self._iteration == 1:
                    # First call: code that uses llm_query multiple times
                    return LLMResponse(
                        text="""
```python
results = []
for i in range(5):
    r = llm_query(f"Sub-query {i}")
    results.append(r)
print(f"Results: {results}")
FINAL({
    "reasoning": str(results),
    "error_identification": "none",
    "root_cause_analysis": "No errors",
    "correct_approach": "Continue",
    "key_insight": "Test",
    "extracted_learnings": [],
    "skill_tags": []
})
```
"""
                    )
                else:
                    # Sub-LLM calls return simple response
                    return LLMResponse(text=f"Sub-response {self._iteration}")

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


if __name__ == "__main__":
    unittest.main()
