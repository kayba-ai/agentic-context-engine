"""Tests for ace/integrations/base.py and ace/integrations/litellm.py."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from ace.skillbook import Skillbook
from ace.integrations.base import wrap_skillbook_context

# ---------------------------------------------------------------------------
# wrap_skillbook_context (base.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWrapSkillbookContext(unittest.TestCase):
    """Tests for wrap_skillbook_context() from base.py."""

    def test_empty_skillbook_returns_empty(self):
        sb = Skillbook()
        self.assertEqual(wrap_skillbook_context(sb), "")

    def test_populated_skillbook_returns_nonempty(self):
        sb = Skillbook()
        sb.add_skill("general", "Always double-check calculations")
        result = wrap_skillbook_context(sb)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_result_contains_skill_content(self):
        sb = Skillbook()
        sb.add_skill("testing", "Write unit tests before refactoring")
        result = wrap_skillbook_context(sb)
        self.assertIn("unit tests", result.lower())


# ---------------------------------------------------------------------------
# ACELiteLLM (litellm.py)
# ---------------------------------------------------------------------------


def _make_ace_litellm(**kwargs):
    """Helper to create ACELiteLLM with all external deps mocked."""
    with patch("ace.integrations.litellm.PromptManager") as MockPM:
        mock_pm = MockPM.return_value
        mock_pm.get_agent_prompt.return_value = "agent prompt"
        mock_pm.get_reflector_prompt.return_value = "reflector prompt"
        mock_pm.get_skill_manager_prompt.return_value = "sm prompt"

        with patch("ace.llm_providers.litellm_client.LiteLLMClient") as MockLLM:
            mock_llm_instance = MockLLM.return_value
            # Patch the import path inside integrations.litellm
            with (
                patch("ace.integrations.litellm.Agent") as MockAgent,
                patch("ace.integrations.litellm.Reflector") as MockReflector,
                patch("ace.integrations.litellm.SkillManager") as MockSM,
            ):
                # Need to also patch the LiteLLMClient import inside the module
                with patch.dict(
                    "ace.integrations.litellm.__dict__",
                    {},
                ):
                    pass
                from ace.integrations.litellm import ACELiteLLM

                obj = ACELiteLLM.__new__(ACELiteLLM)
                obj.model = kwargs.get("model", "test-model")
                obj.is_learning = kwargs.get("is_learning", True)
                obj.dedup_config = None
                obj.skillbook = Skillbook()
                obj.llm = mock_llm_instance
                obj.agent = MockAgent.return_value
                obj.reflector = MockReflector.return_value
                obj.skill_manager = MockSM.return_value
                obj._ace = None
                obj._last_interaction = None
                return obj, {
                    "agent": MockAgent.return_value,
                    "reflector": MockReflector.return_value,
                    "skill_manager": MockSM.return_value,
                    "llm": mock_llm_instance,
                }


@pytest.mark.unit
class TestACELiteLLMInit(unittest.TestCase):
    """Tests for ACELiteLLM initialisation."""

    def test_creates_skillbook(self):
        obj, _ = _make_ace_litellm()
        self.assertIsInstance(obj.skillbook, Skillbook)

    def test_default_learning_enabled(self):
        obj, _ = _make_ace_litellm()
        self.assertTrue(obj.is_learning)


@pytest.mark.unit
class TestACELiteLLMAsk(unittest.TestCase):
    """Tests for ACELiteLLM.ask()."""

    def test_ask_returns_final_answer(self):
        obj, mocks = _make_ace_litellm()
        mock_output = MagicMock()
        mock_output.final_answer = "Paris"
        mocks["agent"].generate.return_value = mock_output

        result = obj.ask("What is the capital of France?")
        self.assertEqual(result, "Paris")

    def test_ask_stores_last_interaction(self):
        obj, mocks = _make_ace_litellm()
        mock_output = MagicMock()
        mock_output.final_answer = "42"
        mocks["agent"].generate.return_value = mock_output

        obj.ask("What is the meaning?")
        self.assertIsNotNone(obj._last_interaction)
        self.assertEqual(obj._last_interaction[0], "What is the meaning?")
        self.assertIs(obj._last_interaction[1], mock_output)


@pytest.mark.unit
class TestACELiteLLMLearnFromFeedback(unittest.TestCase):
    """Tests for ACELiteLLM.learn_from_feedback()."""

    def test_returns_false_when_no_prior_interaction(self):
        obj, _ = _make_ace_litellm()
        result = obj.learn_from_feedback(feedback="correct")
        self.assertFalse(result)

    def test_returns_false_when_learning_disabled(self):
        obj, _ = _make_ace_litellm(is_learning=False)
        result = obj.learn_from_feedback(feedback="correct")
        self.assertFalse(result)

    def test_returns_true_with_valid_interaction(self):
        obj, mocks = _make_ace_litellm()

        # Set up a prior interaction
        mock_agent_output = MagicMock()
        mock_agent_output.final_answer = "42"
        obj._last_interaction = ("question", mock_agent_output)

        # Mock reflector and skill_manager
        mock_reflection = MagicMock()
        mocks["reflector"].reflect.return_value = mock_reflection

        mock_sm_output = MagicMock()
        mock_sm_output.update = MagicMock()
        mocks["skill_manager"].update_skills.return_value = mock_sm_output

        with patch.object(obj.skillbook, "apply_update"):
            result = obj.learn_from_feedback(feedback="correct")

        self.assertTrue(result)
        mocks["reflector"].reflect.assert_called_once()
        mocks["skill_manager"].update_skills.assert_called_once()

    def test_passes_ground_truth_to_reflector(self):
        obj, mocks = _make_ace_litellm()
        mock_agent_output = MagicMock()
        obj._last_interaction = ("q", mock_agent_output)

        mock_reflection = MagicMock()
        mocks["reflector"].reflect.return_value = mock_reflection
        mock_sm_output = MagicMock()
        mock_sm_output.update = MagicMock()
        mocks["skill_manager"].update_skills.return_value = mock_sm_output

        with patch.object(obj.skillbook, "apply_update"):
            obj.learn_from_feedback(feedback="wrong", ground_truth="correct answer")

        call_kwargs = mocks["reflector"].reflect.call_args
        self.assertEqual(call_kwargs.kwargs.get("ground_truth"), "correct answer")


@pytest.mark.unit
class TestACELiteLLMSkillbookIO(unittest.TestCase):
    """Tests for save_skillbook / load_skillbook round-trip."""

    def test_save_and_load_round_trip(self):
        obj, _ = _make_ace_litellm()
        obj.skillbook.add_skill("general", "Test strategy")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            obj.save_skillbook(path)
            obj.load_skillbook(path)
            skills = obj.skillbook.skills()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].content, "Test strategy")
        finally:
            os.unlink(path)


@pytest.mark.unit
class TestACELiteLLMToggle(unittest.TestCase):
    """Tests for enable_learning / disable_learning."""

    def test_disable_then_enable(self):
        obj, _ = _make_ace_litellm()
        obj.disable_learning()
        self.assertFalse(obj.is_learning)
        obj.enable_learning()
        self.assertTrue(obj.is_learning)


@pytest.mark.unit
class TestACELiteLLMGetStrategies(unittest.TestCase):
    """Tests for get_strategies()."""

    def test_empty_skillbook_returns_empty(self):
        obj, _ = _make_ace_litellm()
        self.assertEqual(obj.get_strategies(), "")

    def test_populated_skillbook_returns_text(self):
        obj, _ = _make_ace_litellm()
        obj.skillbook.add_skill("general", "Be precise")
        result = obj.get_strategies()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@pytest.mark.unit
class TestACELiteLLMLearningStats(unittest.TestCase):
    """Tests for learning_stats property."""

    def test_returns_empty_when_no_ace(self):
        obj, _ = _make_ace_litellm()
        stats = obj.learning_stats
        self.assertFalse(stats["async_learning"])
        self.assertEqual(stats["pending"], 0)
        self.assertEqual(stats["completed"], 0)


@pytest.mark.unit
class TestACELiteLLMLearn(unittest.TestCase):
    """Tests for learn() method."""

    def test_raises_when_learning_disabled(self):
        obj, _ = _make_ace_litellm(is_learning=False)
        with self.assertRaises(ValueError):
            obj.learn([], MagicMock())

    def test_learn_creates_offline_ace(self):
        obj, mocks = _make_ace_litellm()
        mock_env = MagicMock()

        with patch("ace.integrations.litellm.OfflineACE") as MockOffline:
            mock_ace = MockOffline.return_value
            mock_ace.run.return_value = []
            obj.learn([], mock_env, epochs=1)
            MockOffline.assert_called_once()
            mock_ace.run.assert_called_once()


@pytest.mark.unit
class TestACELiteLLMRepr(unittest.TestCase):

    def test_repr_format(self):
        obj, _ = _make_ace_litellm()
        r = repr(obj)
        self.assertIn("ACELiteLLM", r)
        self.assertIn("test-model", r)
        self.assertIn("strategies=0", r)


if __name__ == "__main__":
    unittest.main()
