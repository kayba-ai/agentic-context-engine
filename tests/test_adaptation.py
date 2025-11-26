import json
import unittest

import pytest

from ace import (
    DummyLLMClient,
    EnvironmentResult,
    OfflineAdapter,
    Playbook,
    Sample,
    TaskEnvironment,
    Generator,
    Reflector,
    Curator,
)


class SimpleQAEnvironment(TaskEnvironment):
    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        ground_truth = sample.ground_truth or ""
        prediction = generator_output.final_answer
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
class OfflineAdapterTest(unittest.TestCase):
    def test_single_step_updates_playbook(self) -> None:
        client = DummyLLMClient()
        client.queue(
            json.dumps(
                {
                    "reasoning": "The answer is given in the playbook.",
                    "bullet_ids": [],
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
                    "correct_approach": "Keep leveraging the playbook.",
                    "key_insight": "Store that 42 is the default answer.",
                    "bullet_tags": [],
                }
            )
        )
        client.queue(
            json.dumps(
                {
                    "delta": {
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

        playbook = Playbook()
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client)

        adapter = OfflineAdapter(
            playbook=playbook,
            generator=generator,
            reflector=reflector,
            curator=curator,
            max_refinement_rounds=1,
        )

        sample = Sample(
            question="What is the answer to life, the universe, and everything?",
            ground_truth="42",
        )
        environment = SimpleQAEnvironment()
        results = adapter.run([sample], environment, epochs=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].generator_output.final_answer, "42")
        self.assertGreaterEqual(playbook.stats()["sections"], 1)
        self.assertTrue(any("life" in bullet.content for bullet in playbook.bullets()))


if __name__ == "__main__":
    unittest.main()
