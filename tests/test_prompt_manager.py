"""Tests for ace/prompt_manager.py — unified prompt management."""

import json
import re
import unittest

import pytest

from ace.prompt_manager import (
    PromptManager,
    validate_prompt_output_v2_1,
    wrap_skillbook_for_external_agent,
)
from ace.skillbook import Skillbook


@pytest.mark.unit
class TestPromptManagerResolve(unittest.TestCase):
    """Tests for _resolve_prompt() with various reference styles."""

    def setUp(self):
        self.mgr = PromptManager(default_version="2.1")

    def test_resolve_v2_1_reference(self):
        prompt = self.mgr._resolve_prompt("agent", "2.1")
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_resolve_v3_reference(self):
        prompt = self.mgr._resolve_prompt("skill_manager", "3.0")
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)

    def test_resolve_legacy_ace_module_reference(self):
        prompt = self.mgr._resolve_prompt("agent", "1.0")
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)

    def test_resolve_v2_module_reference(self):
        prompt = self.mgr._resolve_prompt("agent", "2.0")
        self.assertIsNotNone(prompt)

    def test_resolve_nonexistent_returns_none(self):
        result = self.mgr._resolve_prompt("agent", "99.9")
        self.assertIsNone(result)

    def test_resolve_nonexistent_role_returns_none(self):
        result = self.mgr._resolve_prompt("nonexistent_role", "2.1")
        self.assertIsNone(result)


@pytest.mark.unit
class TestPromptManagerGetters(unittest.TestCase):
    """Tests for get_agent_prompt / get_reflector_prompt / get_skill_manager_prompt."""

    def setUp(self):
        self.mgr = PromptManager(default_version="2.1")

    def test_get_agent_prompt_returns_nonempty(self):
        prompt = self.mgr.get_agent_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_get_reflector_prompt_returns_nonempty(self):
        prompt = self.mgr.get_reflector_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_get_skill_manager_prompt_returns_nonempty(self):
        prompt = self.mgr.get_skill_manager_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_current_date_substituted_in_agent_prompt(self):
        prompt = self.mgr.get_agent_prompt()
        self.assertNotIn("{current_date}", prompt)
        self.assertTrue(re.search(r"\d{4}-\d{2}-\d{2}", prompt))

    def test_domain_math_prompt(self):
        prompt = self.mgr.get_agent_prompt(domain="math")
        self.assertIn("Math", prompt)

    def test_domain_code_prompt(self):
        prompt = self.mgr.get_agent_prompt(domain="code")
        self.assertIn("code", prompt.lower())

    def test_invalid_version_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mgr.get_agent_prompt(version="99.9")

    def test_invalid_reflector_version_raises(self):
        with self.assertRaises(ValueError):
            self.mgr.get_reflector_prompt(version="99.9")

    def test_invalid_skill_manager_version_raises(self):
        with self.assertRaises(ValueError):
            self.mgr.get_skill_manager_prompt(version="99.9")


@pytest.mark.unit
class TestPromptManagerStats(unittest.TestCase):
    """Tests for _track_usage, track_quality, and get_stats."""

    def test_track_usage_increments(self):
        mgr = PromptManager()
        mgr._track_usage("agent-2.1")
        mgr._track_usage("agent-2.1")
        mgr._track_usage("agent-2.1")
        self.assertEqual(mgr.usage_stats["agent-2.1"], 3)

    def test_track_quality_and_average(self):
        mgr = PromptManager()
        mgr.track_quality("agent-2.1", 0.8)
        mgr.track_quality("agent-2.1", 1.0)
        stats = mgr.get_stats()
        self.assertAlmostEqual(stats["average_quality"]["agent-2.1"], 0.9)

    def test_get_stats_total_calls(self):
        mgr = PromptManager()
        mgr._track_usage("a")
        mgr._track_usage("b")
        mgr._track_usage("a")
        stats = mgr.get_stats()
        self.assertEqual(stats["total_calls"], 3)

    def test_get_stats_empty(self):
        mgr = PromptManager()
        stats = mgr.get_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["usage"], {})
        self.assertEqual(stats["average_quality"], {})


@pytest.mark.unit
class TestListAvailableVersions(unittest.TestCase):

    def test_returns_expected_keys(self):
        versions = PromptManager.list_available_versions()
        self.assertIn("agent", versions)
        self.assertIn("reflector", versions)
        self.assertIn("skill_manager", versions)

    def test_v2_1_present_in_all_roles(self):
        versions = PromptManager.list_available_versions()
        for role in ("agent", "reflector", "skill_manager"):
            self.assertIn("2.1", versions[role])


@pytest.mark.unit
class TestValidatePromptOutputV21(unittest.TestCase):
    """Tests for validate_prompt_output_v2_1."""

    def test_invalid_json_returns_false(self):
        is_valid, errors, metrics = validate_prompt_output_v2_1("not json", "agent")
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid JSON" in e for e in errors))
        self.assertEqual(metrics, {})

    def test_agent_missing_fields(self):
        is_valid, errors, _ = validate_prompt_output_v2_1(
            json.dumps({"reasoning": "ok"}), "agent"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("final_answer" in e for e in errors))

    def test_reflector_missing_fields(self):
        is_valid, errors, _ = validate_prompt_output_v2_1(
            json.dumps({"reasoning": "ok"}), "reflector"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("error_identification" in e for e in errors))

    def test_skill_manager_missing_fields(self):
        is_valid, errors, _ = validate_prompt_output_v2_1(
            json.dumps({"reasoning": "ok"}), "skill_manager"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("operations" in e for e in errors))

    def test_valid_agent_output(self):
        data = {
            "reasoning": "step by step",
            "final_answer": "42",
            "skill_ids": [],
        }
        is_valid, errors, _ = validate_prompt_output_v2_1(json.dumps(data), "agent")
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_valid_reflector_output(self):
        data = {
            "reasoning": "analysis",
            "error_identification": "none",
            "skill_tags": [{"id": "s1", "tag": "helpful"}],
        }
        is_valid, errors, _ = validate_prompt_output_v2_1(json.dumps(data), "reflector")
        self.assertTrue(is_valid)

    def test_valid_skill_manager_output(self):
        data = {
            "reasoning": "update",
            "operations": [{"type": "ADD", "section": "general", "content": "tip"}],
        }
        is_valid, errors, _ = validate_prompt_output_v2_1(
            json.dumps(data), "skill_manager"
        )
        self.assertTrue(is_valid)


@pytest.mark.unit
class TestWrapSkillbookForExternalAgent(unittest.TestCase):

    def test_empty_skillbook_returns_empty_string(self):
        sb = Skillbook()
        result = wrap_skillbook_for_external_agent(sb)
        self.assertEqual(result, "")

    def test_populated_skillbook_returns_nonempty(self):
        sb = Skillbook()
        sb.add_skill("general", "Always verify inputs")
        result = wrap_skillbook_for_external_agent(sb)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("verify inputs", result.lower())


if __name__ == "__main__":
    unittest.main()
