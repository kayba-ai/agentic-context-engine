import unittest

import pytest

from ace import ReplayAgent, Skillbook, Sample


@pytest.mark.unit
class ReplayAgentTest(unittest.TestCase):
    """Test ReplayAgent backward compatibility and new sample-based mode."""

    def test_dict_based_mode_backward_compatibility(self):
        """Test original dict-based mode still works."""
        responses = {"What is 2+2?": "4", "What is Python?": "A programming language"}
        agent = ReplayAgent(responses)

        # Test successful lookup
        output = agent.generate(
            question="What is 2+2?", context="", skillbook=Skillbook()
        )

        self.assertEqual(output.final_answer, "4")
        self.assertIn("[Replayed from responses dict]", output.reasoning)
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "responses_dict"
        )
        self.assertTrue(output.raw["replay_metadata"]["question_found_in_dict"])

    def test_dict_based_mode_with_default(self):
        """Test fallback to default response when question not found."""
        responses = {"Known question": "Known answer"}
        agent = ReplayAgent(responses, default_response="I don't know")

        output = agent.generate(
            question="Unknown question", context="", skillbook=Skillbook()
        )

        self.assertEqual(output.final_answer, "I don't know")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "default_response"
        )
        self.assertFalse(output.raw["replay_metadata"]["question_found_in_dict"])

    def test_sample_based_mode_dict_direct(self):
        """Test new sample-based mode with response in dict."""
        sample = {"question": "What is ACE?", "response": "Agentic Context Engineering"}
        agent = ReplayAgent()  # No dict needed

        output = agent.generate(
            question=sample["question"],
            context="",
            skillbook=Skillbook(),
            sample=sample,
        )

        self.assertEqual(output.final_answer, "Agentic Context Engineering")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_dict_direct"
        )
        self.assertTrue(output.raw["replay_metadata"]["sample_provided"])

    def test_sample_based_mode_metadata_dict(self):
        """Test sample-based mode with response in metadata dict."""
        sample = {
            "question": "What is the best framework?",
            "metadata": {"response": "ACE Framework"},
        }
        agent = ReplayAgent()

        output = agent.generate(
            question=sample["question"],
            context="",
            skillbook=Skillbook(),
            sample=sample,
        )

        self.assertEqual(output.final_answer, "ACE Framework")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_dict_metadata"
        )

    def test_sample_based_mode_sample_object(self):
        """Test sample-based mode with Sample dataclass object."""
        sample = Sample(
            question="What is 5+5?",
            ground_truth="10",
            metadata={"response": "The answer is 10"},
        )
        agent = ReplayAgent()

        output = agent.generate(
            question=sample.question, context="", skillbook=Skillbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "The answer is 10")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_metadata"
        )

    def test_priority_sample_over_dict(self):
        """Test that sample response takes priority over dict lookup."""
        responses = {"What is 2+2?": "4"}
        agent = ReplayAgent(responses)

        # Provide both dict and sample - sample should win
        sample = {"question": "What is 2+2?", "response": "5 (from sample)"}

        output = agent.generate(
            question="What is 2+2?", context="", skillbook=Skillbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "5 (from sample)")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "sample_dict_direct"
        )

    def test_fallback_to_dict_when_sample_has_no_response(self):
        """Test fallback to dict when sample exists but has no response."""
        responses = {"What is 2+2?": "4"}
        agent = ReplayAgent(responses)

        # Sample without response field
        sample = {"question": "What is 2+2?"}

        output = agent.generate(
            question="What is 2+2?", context="", skillbook=Skillbook(), sample=sample
        )

        self.assertEqual(output.final_answer, "4")
        self.assertEqual(
            output.raw["replay_metadata"]["response_source"], "responses_dict"
        )

    def test_empty_responses_dict_initialization(self):
        """Test that ReplayAgent can be initialized without responses dict."""
        agent = ReplayAgent()

        self.assertIsNotNone(agent.responses)
        self.assertEqual(len(agent.responses), 0)
        self.assertEqual(agent.default_response, "")

    def test_none_responses_dict_initialization(self):
        """Test that ReplayAgent handles None responses gracefully."""
        agent = ReplayAgent(responses=None)

        self.assertIsNotNone(agent.responses)
        self.assertEqual(len(agent.responses), 0)


if __name__ == "__main__":
    unittest.main()
