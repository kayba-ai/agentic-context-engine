"""
Unit tests for ACE v2.1 prompt enhancements.

Tests the new features introduced in v2.1:
- Enhanced validation with quality metrics
- Atomicity scoring
- Deduplication checks
- Impact scores
- Quality thresholds
"""

import unittest
import json

import pytest

from ace.prompts_v2_1 import (
    PromptManager,
    validate_prompt_output_v2_1,
    compare_prompt_versions,
    AGENT_V2_1_PROMPT,
    REFLECTOR_V2_1_PROMPT,
    SKILL_MANAGER_V2_1_PROMPT,
)


@pytest.mark.unit
class TestPromptsV21(unittest.TestCase):
    """Test suite for v2.1 prompt enhancements."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = PromptManager(default_version="2.1")

    def test_prompt_manager_v21_initialization(self):
        """Test that PromptManager correctly initializes with v2.1."""
        self.assertEqual(self.manager.default_version, "2.1")
        self.assertIsInstance(self.manager.usage_stats, dict)
        self.assertIsInstance(self.manager.quality_scores, dict)

    def test_agent_prompt_v21_retrieval(self):
        """Test retrieving v2.1 agent prompts."""
        # General agent
        prompt = self.manager.get_agent_prompt(version="2.1")
        self.assertIn("Core Mission", prompt)
        self.assertIn("Core Responsibilities", prompt)
        self.assertIn("CRITICAL", prompt)
        self.assertIn("Strategy Application", prompt)

        # Math-specific
        math_prompt = self.manager.get_agent_prompt(domain="math", version="2.1")
        self.assertIn("Mathematical Problem Solver", math_prompt)
        self.assertIn("PEMDAS/BODMAS", math_prompt)

        # Code-specific
        code_prompt = self.manager.get_agent_prompt(domain="code", version="2.1")
        self.assertIn("Software Development Specialist", code_prompt)
        self.assertIn("Type hints", code_prompt)

    def test_reflector_prompt_v21_features(self):
        """Test v2.1 reflector prompt enhancements."""
        prompt = self.manager.get_reflector_prompt(version="2.1")

        # Check for new features
        self.assertIn("EXPERIENCE-DRIVEN CONCRETE EXTRACTION", prompt)
        self.assertIn("extracted_learnings", prompt)
        self.assertIn("atomicity_score", prompt)
        self.assertIn("impact_score", prompt)
        self.assertIn("Excellent (95", prompt)  # Check without emoji

    def test_skill_manager_prompt_v21_features(self):
        """Test v2.1 skill manager prompt enhancements."""
        prompt = self.manager.get_skill_manager_prompt(version="2.1")

        # Check for atomic strategy principle
        self.assertIn("ATOMIC STRATEGY PRINCIPLE", prompt)
        self.assertIn("GOOD - Atomic Strategies", prompt)  # Without emoji
        self.assertIn("BAD - Compound Strategies", prompt)  # Without emoji
        self.assertIn("DEDUPLICATION: UPDATE > ADD", prompt)
        self.assertIn("quality_metrics", prompt)

    def test_validate_agent_output_v21(self):
        """Test v2.1 validation for agent outputs."""
        # Valid output with v2.1 fields
        valid_output = json.dumps(
            {
                "reasoning": "Step 1: Analyze. Step 2: Apply. Step 3: Solve.",
                "skill_ids": ["skill_001", "skill_002"],
                "confidence_scores": {"skill_001": 0.9, "skill_002": 0.85},
                "step_validations": ["Valid", "Verified"],
                "final_answer": "42",
                "answer_confidence": 0.95,
                "quality_check": {
                    "addresses_question": True,
                    "reasoning_complete": True,
                    "citations_provided": True,
                },
            }
        )

        is_valid, errors, metrics = validate_prompt_output_v2_1(valid_output, "agent")

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIn("completeness", metrics)
        self.assertAlmostEqual(metrics["completeness"], 1.0)
        self.assertIn("overall_confidence", metrics)
        self.assertAlmostEqual(metrics["overall_confidence"], 0.95)

    def test_validate_reflector_output_v21(self):
        """Test v2.1 validation for reflector outputs."""
        valid_output = json.dumps(
            {
                "reasoning": "Analysis of agent performance.",
                "error_identification": "Calculation error at step 3",
                "error_location": "Step 3",
                "root_cause_analysis": "Multiplication error",
                "correct_approach": "Use correct multiplication",
                "extracted_learnings": [
                    {
                        "learning": "Verify multiplication",
                        "atomicity_score": 0.92,
                        "evidence": "Error at 15×20",
                    },
                    {
                        "learning": "Check intermediate steps",
                        "atomicity_score": 0.88,
                        "evidence": "Step validation needed",
                    },
                ],
                "key_insight": "Double-check arithmetic",
                "confidence_in_analysis": 0.95,
                "skill_tags": [
                    {
                        "id": "skill_023",
                        "tag": "neutral",
                        "justification": "Strategy correct, execution failed",
                        "impact_score": 0.7,
                    }
                ],
            }
        )

        is_valid, errors, metrics = validate_prompt_output_v2_1(
            valid_output, "reflector"
        )

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIn("avg_atomicity", metrics)
        self.assertAlmostEqual(metrics["avg_atomicity"], 0.9, places=1)
        self.assertIn("impact_skill_023", metrics)

    def test_validate_skill_manager_output_v21(self):
        """Test v2.1 validation for skill manager outputs."""
        # Valid output with high atomicity
        valid_output = json.dumps(
            {
                "reasoning": "Adding atomic strategy",
                "deduplication_check": {
                    "similar_skills": ["skill_089"],
                    "similarity_scores": {"skill_089": 0.3},
                    "decision": "safe_to_add",
                },
                "operations": [
                    {
                        "type": "ADD",
                        "section": "optimization",
                        "content": "Use pandas.read_csv() for CSV",
                        "atomicity_score": 0.95,
                        "skill_id": "",
                        "metadata": {"helpful": 1, "harmful": 0},
                        "justification": "Improves performance",
                        "evidence": "3x faster in tests",
                    }
                ],
                "quality_metrics": {
                    "avg_atomicity": 0.95,
                    "operations_count": 1,
                    "estimated_impact": 0.8,
                },
            }
        )

        is_valid, errors, metrics = validate_prompt_output_v2_1(
            valid_output, "skill_manager"
        )

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertIn("avg_atomicity", metrics)
        self.assertIn("estimated_impact", metrics)

    def test_reject_low_atomicity_skill_manager_output(self):
        """Test that low atomicity strategies are rejected."""
        bad_output = json.dumps(
            {
                "reasoning": "Adding compound strategy",
                "operations": [
                    {
                        "type": "ADD",
                        "section": "general",
                        "content": "Be careful and handle errors",
                        "atomicity_score": 0.35,  # Too low!
                        "metadata": {"helpful": 1, "harmful": 0},
                    }
                ],
            }
        )

        is_valid, errors, metrics = validate_prompt_output_v2_1(
            bad_output, "skill_manager"
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("Atomicity too low" in error for error in errors))

    def test_quality_score_tracking(self):
        """Test quality score tracking in PromptManager."""
        manager = PromptManager()

        # Track some quality scores
        manager.track_quality("agent-2.1", 0.95)
        manager.track_quality("agent-2.1", 0.88)
        manager.track_quality("agent-2.1", 0.92)

        stats = manager.get_stats()

        self.assertIn("average_quality", stats)
        self.assertIn("agent-2.1", stats["average_quality"])
        avg_quality = stats["average_quality"]["agent-2.1"]
        self.assertAlmostEqual(avg_quality, 0.916, places=2)

    def test_compare_versions_functionality(self):
        """Test version comparison utility."""
        comparisons = compare_prompt_versions("agent")

        self.assertIn("length_v20", comparisons)
        self.assertIn("length_v21", comparisons)
        self.assertIn("length_increase", comparisons)
        self.assertIn("v21_enhancements", comparisons)
        self.assertIn("similarity_ratio", comparisons)

        # v2.1 should be longer due to enhancements
        self.assertGreater(comparisons["length_v21"], comparisons["length_v20"])

        # Check for v2.1 features
        enhancements = comparisons["v21_enhancements"]
        # Note: quick_reference was removed in favor of Core Mission/Responsibilities
        # Just check that we have critical markers
        self.assertGreater(enhancements["critical_markers"], 0)

    def test_version_listing(self):
        """Test listing available prompt versions."""
        versions = PromptManager.list_available_versions()

        self.assertIn("agent", versions)
        self.assertIn("reflector", versions)
        self.assertIn("skill_manager", versions)

        # Check that v2.1 is available
        self.assertIn("2.1", versions["agent"])
        self.assertIn("2.1", versions["reflector"])
        self.assertIn("2.1", versions["skill_manager"])

        # Check domain variants
        self.assertIn("2.1-math", versions["agent"])
        self.assertIn("2.1-code", versions["agent"])

    def test_backward_compatibility(self):
        """Test that v2.1 maintains backward compatibility."""
        manager = PromptManager(default_version="2.1")

        # Should still be able to get v2.0 prompts
        prompt_v20 = manager.get_agent_prompt(version="2.0")
        self.assertIsNotNone(prompt_v20)
        self.assertNotIn(
            "⚡ QUICK REFERENCE ⚡", prompt_v20
        )  # v2.0 shouldn't have this

        # Should still be able to get v1.0 prompts
        prompt_v10 = manager.get_agent_prompt(version="1.0")
        self.assertIsNotNone(prompt_v10)

    def test_usage_statistics(self):
        """Test usage tracking functionality."""
        manager = PromptManager()

        # Use different prompts
        _ = manager.get_agent_prompt(version="2.1")
        _ = manager.get_agent_prompt(version="2.1")
        _ = manager.get_agent_prompt(domain="math", version="2.1")
        _ = manager.get_reflector_prompt(version="2.1")

        stats = manager.get_stats()

        self.assertEqual(stats["usage"]["agent-2.1"], 2)
        self.assertEqual(stats["usage"]["agent-2.1-math"], 1)
        self.assertEqual(stats["usage"]["reflector-2.1"], 1)
        self.assertEqual(stats["total_calls"], 4)

    def test_date_formatting(self):
        """Test that current_date is properly formatted."""
        manager = PromptManager()
        prompt = manager.get_agent_prompt(version="2.1")

        # Should not contain the placeholder
        self.assertNotIn("{current_date}", prompt)

        # Should contain a formatted date (YYYY-MM-DD format)
        import re

        date_pattern = r"\d{4}-\d{2}-\d{2}"
        self.assertTrue(re.search(date_pattern, prompt))

    def test_a_b_testing_support(self):
        """Test A/B testing comparison functionality."""
        manager = PromptManager()

        test_input = {
            "skillbook": "test skillbook",
            "question": "test question",
            "context": "test context",
            "reflection": "test reflection",
            "current_date": "2024-01-15",
        }

        results = manager.compare_versions("agent", test_input)

        # Should have results for v2.1 variants
        self.assertIn("2.1", results)

        # Check for domain variants
        self.assertIn("2.1-math", results)

        # Results should be preview strings
        self.assertIsInstance(results["2.1"], str)
        # Check that it's a preview (trimmed)
        self.assertTrue(results["2.1"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
