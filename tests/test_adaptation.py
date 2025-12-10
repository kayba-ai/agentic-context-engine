import json
import unittest

import pytest

from ace import (
    DummyLLMClient,
    EnvironmentResult,
    OfflineACE,
    Skillbook,
    Sample,
    TaskEnvironment,
    Agent,
    Reflector,
    SkillManager,
)


class SimpleQAEnvironment(TaskEnvironment):
    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        ground_truth = sample.ground_truth or ""
        prediction = agent_output.final_answer
        correct = prediction.strip().lower() == ground_truth.strip().lower()
        feedback = (
            "correct" if correct else f"expected {ground_truth} but got {prediction}"
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )


@pytest.mark.unit
class OfflineACETest(unittest.TestCase):
    def test_single_step_updates_skillbook(self) -> None:
        client = DummyLLMClient()
        client.queue(
            json.dumps(
                {
                    "reasoning": "The answer is given in the skillbook.",
                    "skill_ids": [],
                    "final_answer": "42",
                }
            )
        )
        client.queue(
            json.dumps(
                {
                    "reasoning": "Prediction matches ground truth.",
                    "error_identification": "",
                    "root_cause_analysis": "",
                    "correct_approach": "Keep leveraging the skillbook.",
                    "key_insight": "Store that 42 is the default answer.",
                    "skill_tags": [],
                }
            )
        )
        client.queue(
            json.dumps(
                {
                    "update": {
                        "reasoning": "Adding a reminder for future tasks.",
                        "operations": [
                            {
                                "type": "ADD",
                                "section": "default_answers",
                                "content": "If the question mentions life, universe, and everything, answer 42.",
                                "metadata": {"helpful": 1},
                            }
                        ],
                    }
                }
            )
        )

        skillbook = Skillbook()
        agent = Agent(client)
        reflector = Reflector(client)
        skill_manager = SkillManager(client)

        adapter = OfflineACE(
            skillbook=skillbook,
            agent=agent,
            reflector=reflector,
            skill_manager=skill_manager,
            max_refinement_rounds=1,
        )

        sample = Sample(
            question="What is the answer to life, the universe, and everything?",
            ground_truth="42",
        )
        environment = SimpleQAEnvironment()
        results = adapter.run([sample], environment, epochs=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].agent_output.final_answer, "42")
        self.assertGreaterEqual(skillbook.stats()["sections"], 1)
        self.assertTrue(any("life" in skill.content for skill in skillbook.skills()))


if __name__ == "__main__":
    unittest.main()
