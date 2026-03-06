"""End-to-end test: ReplayAgent through the full OfflineACE pipeline.

Validates that ReplayAgent correctly feeds replayed responses into the
full ACE loop (ReplayAgent → Environment → Reflector → SkillManager → Skillbook).
Uses DummyLLMClient for Reflector/SkillManager so no API key needed.
"""

import json
import unittest

import pytest

from ace import (
    DummyLLMClient,
    EnvironmentResult,
    OfflineACE,
    ReplayAgent,
    Reflector,
    Sample,
    Skillbook,
    SkillManager,
    TaskEnvironment,
)


class ExactMatchEnvironment(TaskEnvironment):
    """Evaluates by comparing agent output to ground truth."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        ground_truth = sample.ground_truth or ""
        prediction = agent_output.final_answer.strip().lower()
        correct = prediction == ground_truth.strip().lower()
        feedback = (
            "correct"
            if correct
            else f"expected '{ground_truth}' but got '{prediction}'"
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )


def _queue_reflector_and_sm(client: DummyLLMClient, section: str, content: str):
    """Queue a valid Reflector + SkillManager response pair."""
    client.queue(
        json.dumps(
            {
                "reasoning": "Analysis complete.",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "Approach is fine.",
                "key_insight": content,
                "skill_tags": [],
            }
        )
    )
    client.queue(
        json.dumps(
            {
                "update": {
                    "reasoning": "Adding learned insight.",
                    "operations": [
                        {
                            "type": "ADD",
                            "section": section,
                            "content": content,
                            "metadata": {"helpful": 1},
                        }
                    ],
                }
            }
        )
    )


@pytest.mark.unit
class TestReplayAgentPipeline(unittest.TestCase):
    """ReplayAgent through the full OfflineACE pipeline."""

    def test_dict_mode_full_pipeline(self):
        """ReplayAgent dict-mode feeds replayed answers through the entire ACE loop."""
        responses = {
            "What is 2+2?": "4",
            "Capital of France?": "Paris",
        }
        replay_agent = ReplayAgent(responses)

        client = DummyLLMClient()
        # Queue Reflector + SkillManager for each sample
        _queue_reflector_and_sm(client, "math", "Simple arithmetic: just compute.")
        _queue_reflector_and_sm(
            client, "geography", "Capital cities: recall from memory."
        )

        skillbook = Skillbook()
        adapter = OfflineACE(
            skillbook=skillbook,
            agent=replay_agent,
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            max_refinement_rounds=1,
            enable_observability=False,
        )

        samples = [
            Sample(question="What is 2+2?", ground_truth="4"),
            Sample(question="Capital of France?", ground_truth="Paris"),
        ]
        results = adapter.run(samples, ExactMatchEnvironment(), epochs=1)

        # Verify pipeline produced results for both samples
        self.assertEqual(len(results), 2)

        # Verify ReplayAgent returned the correct replayed answers
        self.assertEqual(results[0].agent_output.final_answer, "4")
        self.assertEqual(results[1].agent_output.final_answer, "Paris")

        # Verify replay metadata is present
        self.assertEqual(
            results[0].agent_output.raw["replay_metadata"]["response_source"],
            "responses_dict",
        )

        # Verify environment evaluated correctly
        self.assertEqual(results[0].environment_result.metrics["accuracy"], 1.0)
        self.assertEqual(results[1].environment_result.metrics["accuracy"], 1.0)

        # Verify skillbook was updated by SkillManager
        skills = skillbook.skills()
        self.assertGreaterEqual(len(skills), 2)
        skill_contents = [s.content for s in skills]
        self.assertTrue(any("arithmetic" in c.lower() for c in skill_contents))
        self.assertTrue(any("capital" in c.lower() for c in skill_contents))

    def test_sample_mode_full_pipeline(self):
        """ReplayAgent sample-mode reads response from sample metadata."""
        replay_agent = ReplayAgent()  # No dict needed

        client = DummyLLMClient()
        _queue_reflector_and_sm(client, "science", "Water is H2O.")

        skillbook = Skillbook()
        adapter = OfflineACE(
            skillbook=skillbook,
            agent=replay_agent,
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            max_refinement_rounds=1,
            enable_observability=False,
        )

        # Sample carries its own response in metadata
        sample = Sample(
            question="What is water?",
            ground_truth="H2O",
            metadata={"response": "H2O"},
        )
        results = adapter.run([sample], ExactMatchEnvironment(), epochs=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].agent_output.final_answer, "H2O")
        self.assertEqual(
            results[0].agent_output.raw["replay_metadata"]["response_source"],
            "sample_metadata",
        )
        self.assertEqual(results[0].environment_result.metrics["accuracy"], 1.0)
        self.assertGreaterEqual(len(skillbook.skills()), 1)

    def test_multi_epoch_replay_pipeline(self):
        """ReplayAgent works correctly across multiple epochs."""
        responses = {"Q1": "A1"}
        replay_agent = ReplayAgent(responses)

        client = DummyLLMClient()
        # Need Reflector + SkillManager responses for each epoch pass
        _queue_reflector_and_sm(client, "general", "Epoch 1 insight.")
        _queue_reflector_and_sm(client, "general", "Epoch 2 insight.")

        skillbook = Skillbook()
        adapter = OfflineACE(
            skillbook=skillbook,
            agent=replay_agent,
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            max_refinement_rounds=1,
            enable_observability=False,
        )

        sample = Sample(question="Q1", ground_truth="A1")
        results = adapter.run([sample], ExactMatchEnvironment(), epochs=2)

        # Should get 2 results (1 sample × 2 epochs)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].agent_output.final_answer, "A1")
        self.assertEqual(results[1].agent_output.final_answer, "A1")
        self.assertGreaterEqual(len(skillbook.skills()), 1)


@pytest.mark.unit
class TestOpikObservabilityPipeline(unittest.TestCase):
    """Verify Opik observability hooks fire during the full ACE pipeline."""

    def test_observability_tracking_called_during_pipeline(self):
        """Opik integration's log_adaptation_metrics is called for each sample."""
        from unittest.mock import MagicMock, patch

        responses = {"What is 1+1?": "2"}
        replay_agent = ReplayAgent(responses)

        client = DummyLLMClient()
        _queue_reflector_and_sm(client, "math", "Basic addition.")

        skillbook = Skillbook()
        adapter = OfflineACE(
            skillbook=skillbook,
            agent=replay_agent,
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            max_refinement_rounds=1,
            enable_observability=True,
        )

        # Mock the Opik integration to verify calls
        mock_opik = MagicMock()
        mock_opik.is_available.return_value = True
        adapter.opik_integration = mock_opik

        sample = Sample(question="What is 1+1?", ground_truth="2")
        results = adapter.run([sample], ExactMatchEnvironment(), epochs=1)

        self.assertEqual(len(results), 1)
        # Verify log_adaptation_metrics was called
        mock_opik.log_adaptation_metrics.assert_called_once()

        # Verify the call included correct metadata
        call_kwargs = mock_opik.log_adaptation_metrics.call_args
        self.assertEqual(call_kwargs.kwargs["epoch"], 1)
        self.assertEqual(call_kwargs.kwargs["step"], 1)
        self.assertIn("sample_id", call_kwargs.kwargs["metadata"])

    def test_observability_graceful_when_opik_unavailable(self):
        """Pipeline works fine when Opik is not installed."""
        responses = {"Q": "A"}
        replay_agent = ReplayAgent(responses)

        client = DummyLLMClient()
        _queue_reflector_and_sm(client, "test", "Test insight.")

        skillbook = Skillbook()
        adapter = OfflineACE(
            skillbook=skillbook,
            agent=replay_agent,
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            max_refinement_rounds=1,
            enable_observability=True,
        )
        # Force Opik to be None (simulating not installed)
        adapter.opik_integration = None
        adapter.enable_observability = False

        sample = Sample(question="Q", ground_truth="A")
        results = adapter.run([sample], ExactMatchEnvironment(), epochs=1)

        # Pipeline should complete normally
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].agent_output.final_answer, "A")


if __name__ == "__main__":
    unittest.main()
