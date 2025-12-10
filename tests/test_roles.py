"""Unit tests for Agent, Reflector, and SkillManager roles."""

import unittest
from pathlib import Path
import tempfile

import pytest

from ace import Agent, Reflector, SkillManager, Skillbook
from ace.roles import (
    _safe_json_loads,
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
    SkillTag,
)


@pytest.mark.unit
class TestSafeJsonLoads(unittest.TestCase):
    """Test JSON parsing utility with edge cases."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)

    def test_json_with_markdown_fences_json_lang(self):
        """Test stripping ```json markdown fences."""
        json_str = '```json\n{"key": "value"}\n```'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_json_with_generic_markdown_fences(self):
        """Test stripping generic ``` markdown fences."""
        json_str = '```\n{"key": "value"}\n```'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_json_with_only_closing_fence(self):
        """Test JSON with only closing fence."""
        json_str = '{"key": "value"}\n```'
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_json_with_whitespace(self):
        """Test JSON with leading/trailing whitespace."""
        json_str = '  \n  {"key": "value"}  \n  '
        result = _safe_json_loads(json_str)
        self.assertEqual(result["key"], "value")

    def test_invalid_json_raises_value_error(self):
        """Test that invalid JSON raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _safe_json_loads("This is not JSON")
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_non_dict_json_raises_value_error(self):
        """Test that non-dict JSON (array, string) raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _safe_json_loads('["array", "not", "object"]')
        self.assertIn("Expected a JSON object", str(ctx.exception))

    def test_truncated_json_detection_unmatched_braces(self):
        """Test detection of truncated JSON with unmatched braces."""
        truncated = '{"key": "value", "incomplete": {'
        with self.assertRaises(ValueError) as ctx:
            _safe_json_loads(truncated)
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_debug_logging_on_failure(self):
        """Test that invalid JSON is logged to debug file."""
        # Clean up any existing debug log
        debug_path = Path("logs/json_failures.log")
        if debug_path.exists():
            debug_path.unlink()

        with self.assertRaises(ValueError):
            _safe_json_loads("Invalid JSON!")

        # Verify debug log was created
        self.assertTrue(debug_path.exists())


@pytest.mark.unit
class TestAgent(unittest.TestCase):
    """Test Agent role."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()
        # Import mock from conftest
        from tests.conftest import MockLLMClient

        self.mock_llm = MockLLMClient()

    def test_generate_basic(self):
        """Test basic generation with valid JSON response."""
        self.mock_llm.set_response(
            '{"reasoning": "Test reasoning", "final_answer": "42", "skill_ids": []}'
        )

        agent = Agent(self.mock_llm)
        output = agent.generate(
            question="What is the answer?",
            context="Test context",
            skillbook=self.skillbook,
        )

        self.assertEqual(output.final_answer, "42")
        self.assertEqual(output.reasoning, "Test reasoning")
        self.assertEqual(len(output.skill_ids), 0)

    def test_generate_with_skillbook_skills(self):
        """Test generation uses skills from skillbook."""
        skill = self.skillbook.add_skill("math", "Show your work", skill_id="math-001")

        self.mock_llm.set_response(
            '{"reasoning": "Following [math-001], I will show my work to solve", "final_answer": "4"}'
        )

        agent = Agent(self.mock_llm)
        output = agent.generate(
            question="What is 2+2?",
            context="Calculate step by step",
            skillbook=self.skillbook,
        )

        self.assertEqual(output.final_answer, "4")
        self.assertIn("math-001", output.skill_ids)

    def test_generate_with_reflection(self):
        """Test generation with reflection from previous attempt."""
        self.mock_llm.set_response(
            '{"reasoning": "Improved answer", "final_answer": "Better"}'
        )

        agent = Agent(self.mock_llm)
        output = agent.generate(
            question="Test?",
            context="",
            skillbook=self.skillbook,
            reflection="Previous attempt was incorrect",
        )

        # Verify reflection was included in prompt
        prompt = self.mock_llm.call_history[0]["prompt"]
        self.assertIn("Previous attempt was incorrect", prompt)

    def test_generate_filters_invalid_skill_ids(self):
        """Test that citations are extracted from reasoning."""
        self.mock_llm.set_response(
            '{"reasoning": "Following [strategy-001] and [math-002], but also [content-123] works", "final_answer": "OK"}'
        )

        agent = Agent(self.mock_llm)
        output = agent.generate(question="Test?", context="", skillbook=self.skillbook)

        # Should extract cited IDs from reasoning
        self.assertEqual(
            len(output.skill_ids), 3
        )  # "strategy-001", "math-002", "content-123"
        self.assertIn("strategy-001", output.skill_ids)
        self.assertIn("math-002", output.skill_ids)
        self.assertIn("content-123", output.skill_ids)


@pytest.mark.unit
class TestReflector(unittest.TestCase):
    """Test Reflector role."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()
        from tests.conftest import MockLLMClient

        self.mock_llm = MockLLMClient()

    def test_reflect_basic(self):
        """Test basic reflection with valid JSON."""
        self.mock_llm.set_response(
            '{"reasoning": "Answer is correct", "error_identification": "", "root_cause_analysis": "", "correct_approach": "", "key_insight": "", "skill_tags": []}'
        )

        reflector = Reflector(self.mock_llm)
        agent_output = AgentOutput(
            reasoning="2+2 equals 4",
            final_answer="4",
            skill_ids=[],
            raw={},
        )

        reflection = reflector.reflect(
            question="What is 2+2?",
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct!",
        )

        self.assertEqual(reflection.reasoning, "Answer is correct")
        self.assertEqual(len(reflection.skill_tags), 0)

    def test_reflect_with_skill_tagging_helpful(self):
        """Test reflector tags skills as helpful."""
        skill = self.skillbook.add_skill("math", "Show your work", skill_id="b1")

        self.mock_llm.set_response(
            """
        {
            "reasoning": "Good use of step-by-step approach", "error_identification": "", "root_cause_analysis": "", "correct_approach": "", "key_insight": "",
            "skill_tags": [
                {"id": "b1", "tag": "helpful"}
            ]
        }
        """
        )

        reflector = Reflector(self.mock_llm)
        agent_output = AgentOutput(
            reasoning="Used skill b1 for step-by-step",
            final_answer="4",
            skill_ids=["b1"],
            raw={},
        )

        reflection = reflector.reflect(
            question="What is 2+2?",
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Correct",
        )

        self.assertEqual(len(reflection.skill_tags), 1)
        self.assertEqual(reflection.skill_tags[0].id, "b1")
        self.assertEqual(reflection.skill_tags[0].tag, "helpful")

    def test_reflect_with_skill_tagging_harmful(self):
        """Test reflector tags skills as harmful."""
        skill = self.skillbook.add_skill("math", "Skip showing work", skill_id="b_bad")

        self.mock_llm.set_response(
            """
        {
            "reasoning": "Skipping work led to error", "error_identification": "", "root_cause_analysis": "", "correct_approach": "", "key_insight": "",
            "skill_tags": [
                {"id": "b_bad", "tag": "harmful"}
            ]
        }
        """
        )

        reflector = Reflector(self.mock_llm)
        agent_output = AgentOutput(
            reasoning="Skipped work as suggested",
            final_answer="5",
            skill_ids=["b_bad"],
            raw={},
        )

        reflection = reflector.reflect(
            question="What is 2+2?",
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth="4",
            feedback="Incorrect!",
        )

        self.assertEqual(len(reflection.skill_tags), 1)
        self.assertEqual(reflection.skill_tags[0].id, "b_bad")
        self.assertEqual(reflection.skill_tags[0].tag, "harmful")

    def test_reflect_without_ground_truth(self):
        """Test reflection works without ground truth."""
        self.mock_llm.set_response(
            '{"reasoning": "Cannot verify without ground truth", "error_identification": "", "root_cause_analysis": "", "correct_approach": "", "key_insight": "", "skill_tags": []}'
        )

        reflector = Reflector(self.mock_llm)
        agent_output = AgentOutput(
            reasoning="Test", final_answer="Answer", skill_ids=[], raw={}
        )

        reflection = reflector.reflect(
            question="Open-ended question?",
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback="Response looks reasonable",
        )

        self.assertIsNotNone(reflection.reasoning)


@pytest.mark.unit
class TestSkillManager(unittest.TestCase):
    """Test SkillManager role."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()
        from tests.conftest import MockLLMClient

        self.mock_llm = MockLLMClient()

    def test_update_skills_basic_add_operation(self):
        """Test basic skill management with ADD operation."""
        self.mock_llm.set_response(
            """
        {
            "update": {
                "reasoning": "Need to add verification strategy",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "math",
                        "content": "Always verify calculations"
                    }
                ]
            }
        }
        """
        )

        skill_manager = SkillManager(self.mock_llm)
        reflection = ReflectorOutput(
            reasoning="Missing verification step",
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            skill_tags=[],
            raw={},
        )

        skill_manager_output = skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context="Math problem",
            progress="1/10",
        )

        self.assertEqual(len(skill_manager_output.update.operations), 1)
        self.assertEqual(skill_manager_output.update.operations[0].type, "ADD")
        self.assertEqual(skill_manager_output.update.operations[0].section, "math")
        self.assertEqual(
            skill_manager_output.update.operations[0].content,
            "Always verify calculations",
        )

    def test_update_skills_tag_operation(self):
        """Test TAG operation updates skill metadata."""
        skill = self.skillbook.add_skill("math", "Show your work", skill_id="b1")

        self.mock_llm.set_response(
            """
        {
            "update": {
                "reasoning": "Skill b1 was helpful",
                "operations": [
                    {
                        "type": "TAG",
                        "section": "math",
                        "skill_id": "b1",
                        "metadata": {"helpful": 1}
                    }
                ]
            }
        }
        """
        )

        skill_manager = SkillManager(self.mock_llm)
        reflection = ReflectorOutput(
            reasoning="Skill helped solve problem",
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            skill_tags=[],
            raw={},
        )

        skill_manager_output = skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context="Math",
            progress="1/1",
        )

        self.assertEqual(skill_manager_output.update.operations[0].type, "TAG")
        self.assertEqual(skill_manager_output.update.operations[0].skill_id, "b1")
        self.assertEqual(
            skill_manager_output.update.operations[0].metadata["helpful"], 1
        )

    def test_update_skills_multiple_operations(self):
        """Test multiple operations in one update batch."""
        self.mock_llm.set_response(
            """
        {
            "update": {
                "reasoning": "Add new strategy and tag existing one",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "math",
                        "content": "Check units"
                    },
                    {
                        "type": "TAG",
                        "section": "math",
                        "skill_id": "b1",
                        "metadata": {"helpful": 1}
                    }
                ]
            }
        }
        """
        )

        skill_manager = SkillManager(self.mock_llm)
        reflection = ReflectorOutput(
            reasoning="Multiple changes needed",
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            skill_tags=[],
            raw={},
        )

        skill_manager_output = skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context="Physics",
            progress="5/10",
        )

        self.assertEqual(len(skill_manager_output.update.operations), 2)
        self.assertEqual(skill_manager_output.update.operations[0].type, "ADD")
        self.assertEqual(skill_manager_output.update.operations[1].type, "TAG")

    def test_update_skills_empty_operations(self):
        """Test skill management with no operations needed."""
        self.mock_llm.set_response(
            """
        {
            "update": {
                "reasoning": "Skillbook is already sufficient",
                "operations": []
            }
        }
        """
        )

        skill_manager = SkillManager(self.mock_llm)
        reflection = ReflectorOutput(
            reasoning="Everything looks good",
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            skill_tags=[],
            raw={},
        )

        skill_manager_output = skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context="Test",
            progress="10/10",
        )

        self.assertEqual(len(skill_manager_output.update.operations), 0)
        self.assertIn("sufficient", skill_manager_output.update.reasoning)


class TestExtractCitedSkillIds(unittest.TestCase):
    """Test skill ID extraction utility."""

    def test_extract_single_id(self):
        """Extract single skill ID."""
        from ace.roles import extract_cited_skill_ids

        text = "Following [general-00042], I will proceed."
        result = extract_cited_skill_ids(text)
        self.assertEqual(result, ["general-00042"])

    def test_extract_multiple_ids(self):
        """Extract multiple IDs in order."""
        from ace.roles import extract_cited_skill_ids

        text = "Using [general-00042] and [geo-00003] strategies."
        result = extract_cited_skill_ids(text)
        self.assertEqual(result, ["general-00042", "geo-00003"])

    def test_deduplicate_preserving_order(self):
        """Deduplicate while preserving first occurrence."""
        from ace.roles import extract_cited_skill_ids

        text = "Start with [id-001], then [id-002], revisit [id-001]."
        result = extract_cited_skill_ids(text)
        self.assertEqual(result, ["id-001", "id-002"])

    def test_no_ids_found(self):
        """Return empty list when no IDs."""
        from ace.roles import extract_cited_skill_ids

        text = "This has no skill citations at all."
        result = extract_cited_skill_ids(text)
        self.assertEqual(result, [])

    def test_mixed_with_noise(self):
        """Extract IDs ignoring other bracketed content."""
        from ace.roles import extract_cited_skill_ids

        text = "Use [strategy-123] but not [this is not an id] or [123]."
        result = extract_cited_skill_ids(text)
        self.assertEqual(result, ["strategy-123"])

    def test_various_section_names(self):
        """Handle different section naming conventions."""
        from ace.roles import extract_cited_skill_ids

        text = "[general-001] [content_extraction-042] [API_calls-999]"
        result = extract_cited_skill_ids(text)
        self.assertEqual(
            result, ["general-001", "content_extraction-042", "API_calls-999"]
        )

    def test_empty_string(self):
        """Handle empty input."""
        from ace.roles import extract_cited_skill_ids

        self.assertEqual(extract_cited_skill_ids(""), [])

    def test_multiline_text(self):
        """Extract from multiline text."""
        from ace.roles import extract_cited_skill_ids

        text = """
        Step 1: Following [setup-001], initialize.
        Step 2: Apply [process-042] for data.
        Step 3: Using [setup-001] again.
        """
        result = extract_cited_skill_ids(text)
        self.assertEqual(result, ["setup-001", "process-042"])


if __name__ == "__main__":
    unittest.main()
