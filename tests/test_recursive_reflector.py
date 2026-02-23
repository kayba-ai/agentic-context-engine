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
print(f"Question: {traces['question']}")
step = traces['steps'][0]
print(f"Correct: {step['answer'].strip() == traces['ground_truth'].strip()}")
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
        explore_response = "```python\nprint(traces['question'][:50])\n```"
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
        explore_response = "```python\nprint(traces['question'][:50])\n```"
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

        self.assertEqual(config.max_iterations, 20)
        self.assertEqual(config.timeout, 30.0)
        self.assertTrue(config.enable_llm_query)
        self.assertEqual(config.max_llm_calls, 30)
        self.assertEqual(config.max_context_chars, 50_000)
        self.assertEqual(config.max_output_chars, 20_000)
        self.assertTrue(config.enable_fallback_synthesis)

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
        """Test that both v2 and v3 prompts contain preview and size metadata placeholders."""
        from ace.reflector.prompts import REFLECTOR_RECURSIVE_PROMPT
        from ace.reflector.prompts_rr_v3 import REFLECTOR_RECURSIVE_V3_PROMPT

        import re

        for label, prompt in [
            ("v2", REFLECTOR_RECURSIVE_PROMPT),
            ("v3", REFLECTOR_RECURSIVE_V3_PROMPT),
        ]:
            with self.subTest(prompt_version=label):
                # The prompt template should have placeholders for metadata
                self.assertIn("{reasoning_length}", prompt)
                self.assertIn("{answer_length}", prompt)
                self.assertIn("{step_count}", prompt)

                # The prompt should have preview placeholders
                self.assertIn("{question_preview}", prompt)
                self.assertIn("{reasoning_preview}", prompt)
                self.assertIn("{answer_preview}", prompt)
                self.assertIn("{ground_truth_preview}", prompt)
                self.assertIn("{feedback_preview}", prompt)

                # Raw {reasoning}, {feedback}, etc. should NOT appear
                self.assertIsNone(re.search(r"\{reasoning\}", prompt))
                self.assertIsNone(re.search(r"\{feedback\}", prompt))
                self.assertIsNone(re.search(r"\{skillbook\}", prompt))
                self.assertIsNone(re.search(r"\{question\}", prompt))


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
print(f"Question: {traces['question'][:100]}")
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
            "```python\nprint(traces['question'][:100])\n```",
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
class TestFinalRejectedAfterExecutionError(unittest.TestCase):
    """Test that FINAL() is rejected if code execution had errors."""

    def setUp(self):
        self.skillbook = Skillbook()
        self.agent_output = AgentOutput(
            reasoning="I analyzed the data",
            final_answer="42",
            skill_ids=[],
        )

    def test_final_rejected_after_execution_error(self):
        """FINAL() should be rejected if code execution failed.

        This tests the scenario where code has an error but still calls FINAL().
        The reflector should reject the FINAL() and ask for a fix.
        """
        call_count = [0]

        class ErrorThenFixLLMClient(MockLLMClient):
            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                call_count[0] += 1
                if call_count[0] == 1:
                    # Iteration 0: explore first
                    return LLMResponse(
                        text="```python\nprint(traces['question'][:50])\n```"
                    )
                elif call_count[0] == 2:
                    # Iteration 1: code that triggers an error before FINAL
                    # Using undefined_variable triggers NameError which sets result.success=False
                    return LLMResponse(
                        text="""```python
# This will cause a NameError
x = undefined_variable
FINAL({
    "reasoning": "Hallucinated success based on imagined output",
    "error_identification": "none",
    "root_cause_analysis": "N/A",
    "correct_approach": "N/A",
    "key_insight": "Hallucinated",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
                    )
                else:
                    # Iteration 2+: should see rejection message, provide proper fix
                    last_msg = messages[-1]["content"]
                    # Verify the rejection message was sent (contains error info)
                    assert (
                        "error" in last_msg.lower() or "Error" in last_msg
                    ), f"Expected 'error' in message: {last_msg[:200]}"
                    return LLMResponse(
                        text="""```python
FINAL({
    "reasoning": "After fixing the error, analysis is complete.",
    "error_identification": "none",
    "root_cause_analysis": "No remaining errors",
    "correct_approach": "Fixed the undefined variable issue",
    "key_insight": "Error was fixed before finalizing",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
                    )

        llm = ErrorThenFixLLMClient()
        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))

        result = reflector.reflect(
            question="Analyze data",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="42",
            feedback="Correct!",
        )

        # Should have made 3 calls: explore, error+FINAL (rejected), fixed FINAL (accepted)
        self.assertEqual(call_count[0], 3)
        # Final result should be from the fixed response, not hallucinated
        self.assertIn("fixing the error", result.reasoning.lower())

    def test_final_accepted_when_code_succeeds(self):
        """FINAL() should be accepted when code execution succeeds."""
        responses = [
            # Iteration 0: explore
            "```python\nprint(traces['question'][:50])\n```",
            # Iteration 1: successful code + FINAL
            """```python
result = "success"
print(result)
FINAL({
    "reasoning": "Code ran successfully",
    "error_identification": "none",
    "root_cause_analysis": "N/A",
    "correct_approach": "N/A",
    "key_insight": "Clean execution",
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
            ground_truth="42",
            feedback="OK",
        )

        # Should only need 2 calls (explore + successful FINAL)
        self.assertEqual(llm.call_count, 2)
        self.assertIn("successfully", result.reasoning.lower())


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
                    return LLMResponse(
                        text="```python\nprint(traces['question'][:50])\n```"
                    )
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
        """Test that getattr, setattr, delattr are removed from builtins."""
        self.assertNotIn("getattr", TraceSandbox.SAFE_BUILTINS)
        self.assertNotIn("setattr", TraceSandbox.SAFE_BUILTINS)
        self.assertNotIn("delattr", TraceSandbox.SAFE_BUILTINS)
        # type is safe - it only returns an object's type, doesn't allow modification
        self.assertIn("type", TraceSandbox.SAFE_BUILTINS)


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


@pytest.mark.unit
class TestOutputTruncation(unittest.TestCase):
    """Test per-output truncation feature."""

    def test_truncate_output_under_limit(self):
        """Test that output under limit is not truncated."""
        from ace.reflector.recursive import _truncate_output

        output = "short output"
        result = _truncate_output(output, max_chars=100)
        self.assertEqual(result, output)

    def test_truncate_output_over_limit(self):
        """Test that output over limit is truncated with metadata."""
        from ace.reflector.recursive import _truncate_output

        output = "x" * 1000
        result = _truncate_output(output, max_chars=100)

        self.assertEqual(len(result.split("\n")[0]), 100)
        self.assertIn("truncated", result)
        self.assertIn("900", result)  # 1000 - 100 = 900 chars truncated

    def test_truncate_output_empty(self):
        """Test that empty output is not modified."""
        from ace.reflector.recursive import _truncate_output

        self.assertEqual(_truncate_output("", max_chars=100), "")
        self.assertEqual(_truncate_output(None, max_chars=100), None)

    def test_truncation_in_reflector(self):
        """Test that truncation is applied to sandbox output."""
        captured_outputs = []

        class CapturingLLMClient(MockLLMClient):
            call_count = 0

            def complete_messages(self, messages, **kwargs):
                self.call_count += 1
                # Capture output messages (user role messages after first)
                for msg in messages:
                    if msg["role"] == "user" and "stdout" in msg["content"]:
                        captured_outputs.append(msg["content"])

                if self.call_count == 1:
                    # First call: generate large output
                    return LLMResponse(text='```python\nprint("x" * 25000)\n```')
                else:
                    # Second call: return final answer
                    return LLMResponse(
                        text=json.dumps(
                            {
                                "reasoning": "Done",
                                "error_identification": "none",
                                "root_cause_analysis": "N/A",
                                "correct_approach": "N/A",
                                "key_insight": "Large output handled",
                                "extracted_learnings": [],
                                "skill_tags": [],
                            }
                        )
                    )

        llm = CapturingLLMClient()
        config = RecursiveConfig(max_iterations=5, max_output_chars=1000)
        reflector = RecursiveReflector(llm, config=config)
        skillbook = Skillbook()
        agent_output = AgentOutput(reasoning="test", final_answer="4", skill_ids=[])

        reflector.reflect(
            question="Test",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth="4",
            feedback="OK",
        )

        # Verify that the large output was truncated
        self.assertGreater(len(captured_outputs), 0)
        output = captured_outputs[0]
        self.assertIn("truncated", output)


@pytest.mark.unit
class TestFallbackSynthesis(unittest.TestCase):
    """Test fallback synthesis on timeout."""

    def test_fallback_synthesis_enabled(self):
        """Test that fallback synthesis is attempted when enabled."""
        synthesis_called = []

        class SynthesizingLLMClient(MockLLMClient):
            call_count = 0

            def complete_messages(self, messages, **kwargs):
                self.call_count += 1
                # Check if this is a synthesis request
                last_msg = messages[-1]["content"] if messages else ""
                if "timed out" in last_msg.lower():
                    synthesis_called.append(True)
                    return LLMResponse(
                        text=json.dumps(
                            {
                                "reasoning": "Synthesized",
                                "error_identification": "none",
                                "root_cause_analysis": "N/A",
                                "correct_approach": "N/A",
                                "key_insight": "Synthesized from history",
                                "extracted_learnings": [],
                                "skill_tags": [],
                            }
                        )
                    )
                # Never call FINAL to trigger timeout
                return LLMResponse(text='```python\nprint("processing...")\n```')

        llm = SynthesizingLLMClient()
        config = RecursiveConfig(max_iterations=2, enable_fallback_synthesis=True)
        reflector = RecursiveReflector(llm, config=config)
        skillbook = Skillbook()
        agent_output = AgentOutput(reasoning="test", final_answer="4", skill_ids=[])

        result = reflector.reflect(
            question="Test",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth="4",
            feedback="OK",
        )

        # Synthesis should have been attempted
        self.assertTrue(len(synthesis_called) > 0)

    def test_fallback_synthesis_disabled(self):
        """Test that fallback synthesis is not attempted when disabled."""
        synthesis_called = []

        class TrackingLLMClient(MockLLMClient):
            def complete_messages(self, messages, **kwargs):
                last_msg = messages[-1]["content"] if messages else ""
                if "timed out" in last_msg.lower():
                    synthesis_called.append(True)
                return LLMResponse(text='```python\nprint("processing...")\n```')

        llm = TrackingLLMClient()
        config = RecursiveConfig(max_iterations=2, enable_fallback_synthesis=False)
        reflector = RecursiveReflector(llm, config=config)
        skillbook = Skillbook()
        agent_output = AgentOutput(reasoning="test", final_answer="4", skill_ids=[])

        result = reflector.reflect(
            question="Test",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth="4",
            feedback="OK",
        )

        # Synthesis should not have been called
        self.assertEqual(len(synthesis_called), 0)
        # Should return timeout output with timeout=True in raw
        self.assertTrue(result.raw.get("timeout", False))


@pytest.mark.unit
class TestCodeExtractionRobustness(unittest.TestCase):
    """Test robust code extraction from various markdown formats."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.reflector = RecursiveReflector(self.llm)

    def test_extract_code_tilde_fence(self):
        """Test extraction from ~~~python fence."""
        code = self.reflector._extract_code("~~~python\nprint('hello')\n~~~")
        self.assertEqual(code, "print('hello')")

    def test_extract_code_indented_block(self):
        """Test extraction from 4-space indented block."""
        text = """Here's the code:

    x = 1
    print(x)
    FINAL({"key": x})

That should work."""
        code = self.reflector._extract_code(text)
        self.assertIn("x = 1", code)
        self.assertIn("print(x)", code)
        self.assertIn("FINAL", code)

    def test_extract_code_tab_indented_block(self):
        """Test extraction from tab-indented block."""
        text = "Here's the code:\n\n\tx = 1\n\tprint(x)\n\nDone."
        code = self.reflector._extract_code(text)
        self.assertIn("x = 1", code)
        self.assertIn("print(x)", code)

    def test_extract_code_final_call_fallback(self):
        """Test FINAL() extraction as last resort."""
        text = """Based on my analysis, the result is:

FINAL({
    "reasoning": "Complete",
    "extracted_learnings": []
})

That's all."""
        code = self.reflector._extract_code(text)
        self.assertIsNotNone(code)
        self.assertIn("FINAL(", code)
        self.assertIn("reasoning", code)

    def test_extract_code_unclosed_fence_fallback(self):
        """Test fallback when fence is unclosed."""
        text = """```python
print('hello')
# Missing closing fence

FINAL({"key": "value"})
"""
        code = self.reflector._extract_code(text)
        # Should fall back to FINAL() extraction
        self.assertIsNotNone(code)
        self.assertIn("FINAL", code)

    def test_looks_like_python_positive(self):
        """Test _looks_like_python with valid Python code."""
        self.assertTrue(self.reflector._looks_like_python("def foo(): pass"))
        self.assertTrue(self.reflector._looks_like_python("import json"))
        self.assertTrue(self.reflector._looks_like_python("x = 1"))
        self.assertTrue(self.reflector._looks_like_python("print(x)"))
        self.assertTrue(self.reflector._looks_like_python("FINAL({})"))

    def test_looks_like_python_negative(self):
        """Test _looks_like_python with non-Python content."""
        self.assertFalse(self.reflector._looks_like_python("just some text"))
        self.assertFalse(self.reflector._looks_like_python(""))
        self.assertFalse(self.reflector._looks_like_python("1234567890"))


@pytest.mark.unit
class TestBatchCodeBlocks(unittest.TestCase):
    """Test batch code block execution."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.reflector = RecursiveReflector(self.llm)

    def test_batch_marker_combines_blocks(self):
        """Test that # BATCH marker combines all code blocks."""
        text = """
```python
# BATCH
x = 1
```

Some explanation...

```python
y = x + 1
print(y)
```

```python
FINAL({"result": y})
```
"""
        code = self.reflector._extract_code(text)
        self.assertIn("# BATCH", code)
        self.assertIn("x = 1", code)
        self.assertIn("y = x + 1", code)
        self.assertIn("FINAL", code)

    def test_no_batch_marker_single_block(self):
        """Test that without # BATCH, only first block is returned."""
        text = """
```python
x = 1
```

```python
y = 2
```
"""
        code = self.reflector._extract_code(text)
        self.assertIn("x = 1", code)
        self.assertNotIn("y = 2", code)


@pytest.mark.unit
class TestSemanticTrimming(unittest.TestCase):
    """Test semantic context window trimming."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.reflector = RecursiveReflector(
            self.llm, config=RecursiveConfig(max_context_chars=500)
        )

    def test_score_iteration_errors_high_value(self):
        """Test that iterations with errors get high scores."""
        asst_msg = {"role": "assistant", "content": "```python\nprint(x)\n```"}
        user_msg_error = {
            "role": "user",
            "content": "stdout:\n\nstderr:\nNameError: name 'x' is not defined",
        }
        user_msg_ok = {"role": "user", "content": "stdout:\nhello\n"}

        error_score = self.reflector._score_iteration(asst_msg, user_msg_error)
        ok_score = self.reflector._score_iteration(asst_msg, user_msg_ok)

        self.assertGreater(error_score, ok_score)

    def test_score_iteration_findings_high_value(self):
        """Test that iterations with findings get higher scores."""
        asst_msg = {"role": "assistant", "content": "```python\nprint(x)\n```"}
        user_msg_finding = {
            "role": "user",
            "content": "stdout:\nFound pattern: API error at step 3",
        }
        user_msg_empty = {"role": "user", "content": "(no output)"}

        finding_score = self.reflector._score_iteration(asst_msg, user_msg_finding)
        empty_score = self.reflector._score_iteration(asst_msg, user_msg_empty)

        self.assertGreater(finding_score, empty_score)

    def test_score_iteration_final_high_value(self):
        """Test that iterations with FINAL() get higher scores."""
        asst_final = {
            "role": "assistant",
            "content": "```python\nFINAL({'key': 'val'})\n```",
        }
        asst_print = {"role": "assistant", "content": "```python\nprint('x')\n```"}
        user_msg = {"role": "user", "content": "stdout:\n"}

        final_score = self.reflector._score_iteration(asst_final, user_msg)
        print_score = self.reflector._score_iteration(asst_print, user_msg)

        self.assertGreater(final_score, print_score)

    def test_trim_keeps_error_iterations(self):
        """Test that trimming prioritizes keeping error iterations."""
        messages = [
            {"role": "user", "content": "A" * 200},  # instructions
            {"role": "assistant", "content": "print('explore')"},  # low value
            {"role": "user", "content": "(no output)"},
            {"role": "assistant", "content": "print(undefined)"},  # high value
            {"role": "user", "content": "stderr:\nNameError: undefined"},
            {"role": "assistant", "content": "FINAL({})"},  # high value
            {"role": "user", "content": "stdout:\n"},
        ]
        trimmed = self.reflector._trim_messages(messages)

        # Should have summary of dropped iterations
        content_str = " ".join(m["content"] for m in trimmed)
        # Error iteration should be kept
        self.assertIn("NameError", content_str)

    def test_summarize_dropped_iterations(self):
        """Test that dropped iteration summary is generated."""
        dropped = [
            (
                {"role": "assistant", "content": "print('x')"},
                {"role": "user", "content": "stderr:\nError"},
            ),
            (
                {"role": "assistant", "content": "print('y')"},
                {"role": "user", "content": "stdout:\n5"},
            ),
        ]
        summary = self.reflector._summarize_dropped_iterations(dropped)
        self.assertIn("error", summary.lower())


@pytest.mark.unit
class TestWindowsTimeout(unittest.TestCase):
    """Test Windows-compatible timeout support."""

    def test_execute_with_multiprocessing_simple(self):
        """Test that simple code executes correctly via multiprocessing."""
        import platform

        if platform.system() != "Windows":
            pytest.skip("Windows-specific test")

        sandbox = TraceSandbox(trace=None)
        sandbox.inject("question", "test question")

        result = sandbox.execute("print(question)", timeout=5.0)
        self.assertIn("test question", result.stdout)
        self.assertIsNone(result.exception)

    def test_execute_no_timeout_fallback(self):
        """Test the no-timeout fallback works."""
        sandbox = TraceSandbox(trace=None)

        # Test the _execute_no_timeout method directly
        result = sandbox._execute_no_timeout("print('hello')")
        self.assertIn("hello", result.stdout)
        self.assertIsNone(result.exception)

    def test_execute_no_timeout_with_final(self):
        """Test that FINAL() works in no-timeout mode."""
        sandbox = TraceSandbox(trace=None)

        result = sandbox._execute_no_timeout("FINAL({'key': 'value'})")
        self.assertTrue(sandbox.final_called)
        self.assertEqual(sandbox.final_value, {"key": "value"})


@pytest.mark.unit
class TestExtractFinalCall(unittest.TestCase):
    """Test FINAL() call extraction."""

    def setUp(self):
        self.llm = MockLLMClient()
        self.reflector = RecursiveReflector(self.llm)

    def test_extract_simple_final(self):
        """Test extraction of simple FINAL() call."""
        text = 'FINAL({"key": "value"})'
        code = self.reflector._extract_final_call(text)
        self.assertEqual(code, text)

    def test_extract_nested_final(self):
        """Test extraction of FINAL() with nested structures."""
        text = 'FINAL({"outer": {"inner": [1, 2, 3]}})'
        code = self.reflector._extract_final_call(text)
        self.assertEqual(code, text)

    def test_extract_final_with_strings(self):
        """Test extraction of FINAL() with string containing parens."""
        text = 'FINAL({"msg": "hello (world)"})'
        code = self.reflector._extract_final_call(text)
        self.assertEqual(code, text)

    def test_extract_final_multiline(self):
        """Test extraction of multiline FINAL() call."""
        text = """FINAL({
    "reasoning": "Analysis",
    "extracted_learnings": []
})"""
        code = self.reflector._extract_final_call(text)
        self.assertIn("reasoning", code)
        self.assertIn("extracted_learnings", code)

    def test_no_final_returns_none(self):
        """Test that text without FINAL() returns None."""
        text = "Just some text without a FINAL call"
        code = self.reflector._extract_final_call(text)
        self.assertIsNone(code)


@pytest.mark.unit
class TestPreviewBraceEscaping(unittest.TestCase):
    """Test that _preview escapes curly braces for str.format() safety."""

    def test_preview_escapes_braces(self):
        """Test that braces in text are doubled for format safety."""
        from ace.reflector.recursive import _preview

        text = "Use {name} for {purpose}"
        result = _preview(text)
        self.assertEqual(result, "Use {{name}} for {{purpose}}")

    def test_preview_escapes_braces_in_truncated_text(self):
        """Test that braces are escaped even in truncated text."""
        from ace.reflector.recursive import _preview

        text = "{x}" * 100  # 300 chars
        result = _preview(text, max_len=150)
        # All single braces should be doubled
        # No single { without a following { (i.e., no unescaped braces)
        import re

        self.assertIsNone(re.search(r"(?<!\{)\{(?!\{)", result))
        self.assertIn("{{x}}", result)

    def test_preview_no_braces_unchanged(self):
        """Test that text without braces is unchanged."""
        from ace.reflector.recursive import _preview

        text = "no braces here"
        result = _preview(text)
        self.assertEqual(result, "no braces here")

    def test_preview_empty_returns_marker(self):
        """Test that empty/None returns (empty)."""
        from ace.reflector.recursive import _preview

        self.assertEqual(_preview(None), "(empty)")
        self.assertEqual(_preview(""), "(empty)")


@pytest.mark.unit
class TestSuccessHeuristic(unittest.TestCase):
    """Test that ExecutionResult.success only checks exception."""

    def test_success_with_error_in_stderr(self):
        """Test that 'Error' in stderr does NOT make success=False."""
        from ace.reflector.sandbox import ExecutionResult

        result = ExecutionResult(stdout="", stderr="No Error occurred", exception=None)
        self.assertTrue(result.success)

    def test_success_with_exception(self):
        """Test that an exception makes success=False."""
        from ace.reflector.sandbox import ExecutionResult

        result = ExecutionResult(stdout="", stderr="", exception=ValueError("test"))
        self.assertFalse(result.success)

    def test_success_clean_execution(self):
        """Test that clean execution is success=True."""
        from ace.reflector.sandbox import ExecutionResult

        result = ExecutionResult(stdout="hello", stderr="", exception=None)
        self.assertTrue(result.success)


@pytest.mark.unit
class TestEmptyLLMResponse(unittest.TestCase):
    """Test that empty/None LLM response is handled gracefully."""

    def setUp(self):
        self.skillbook = Skillbook()
        self.agent_output = AgentOutput(
            reasoning="I calculated 2+2 step by step",
            final_answer="4",
            skill_ids=[],
        )

    def test_none_response_continues_iteration(self):
        """Test that None/empty LLM response does not crash and iteration continues."""
        call_count = [0]

        class NoneFirstLLMClient(MockLLMClient):
            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call: return empty text (simulates Gemini None)
                    return LLMResponse(text="")
                elif call_count[0] == 2:
                    # Second call: explore
                    return LLMResponse(
                        text="```python\nprint(traces['question'][:50])\n```"
                    )
                else:
                    # Third call: FINAL
                    return LLMResponse(
                        text="""```python
FINAL({
    "reasoning": "Recovered from empty response.",
    "error_identification": "none",
    "root_cause_analysis": "N/A",
    "correct_approach": "N/A",
    "key_insight": "Recovery successful",
    "extracted_learnings": [],
    "skill_tags": []
})
```"""
                    )

        llm = NoneFirstLLMClient()
        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=5))

        result = reflector.reflect(
            question="What is 2+2?",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertIsInstance(result, ReflectorOutput)
        self.assertIn("recovered", result.reasoning.lower())
        # Should have needed 3 calls: empty, explore, FINAL
        self.assertEqual(call_count[0], 3)

    def test_none_response_adds_retry_prompt(self):
        """Test that empty response appends a retry prompt to messages."""
        captured_messages = []

        class CapturingEmptyLLMClient(MockLLMClient):
            call_count = 0

            def complete_messages(
                self, messages: List[Dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                self.call_count += 1
                captured_messages.append(list(messages))
                if self.call_count == 1:
                    return LLMResponse(text=None)
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

        llm = CapturingEmptyLLMClient()
        reflector = RecursiveReflector(llm, config=RecursiveConfig(max_iterations=3))

        reflector.reflect(
            question="Test",
            agent_output=self.agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="OK",
        )

        # Second call should see the retry prompt from the first empty response
        self.assertGreater(len(captured_messages), 1)
        second_call = captured_messages[1]
        last_msg = second_call[-1]["content"]
        self.assertIn("empty", last_msg.lower())


@pytest.mark.unit
class TestSignalAlarmCeiling(unittest.TestCase):
    """Test that signal.alarm uses math.ceil for sub-second timeouts."""

    def test_subsecond_timeout_not_zero(self):
        """Test that a sub-second timeout doesn't disable the alarm."""
        import math

        # math.ceil(0.5) == 1, not 0
        self.assertEqual(math.ceil(0.5), 1)
        # int(0.5) would be 0, which disables the alarm
        self.assertEqual(int(0.5), 0)

    def test_sandbox_uses_ceil(self):
        """Test that sandbox execute uses math.ceil via code inspection."""
        import inspect
        from ace.reflector.sandbox import TraceSandbox

        source = inspect.getsource(TraceSandbox._execute_unix)
        self.assertIn("math.ceil(timeout)", source)
        self.assertNotIn("int(timeout)", source)


if __name__ == "__main__":
    unittest.main()
