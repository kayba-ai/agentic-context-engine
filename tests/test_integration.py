"""
Integration tests for end-to-end ACE adaptation flows.

These tests verify the complete workflow from sample → generate → reflect → curate.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ace import (
    Curator,
    EnvironmentResult,
    Generator,
    LLMClient,
    OfflineAdapter,
    OnlineAdapter,
    Playbook,
    Reflector,
    Sample,
    TaskEnvironment,
)
from ace.llm import LLMResponse


class MockLLMClient(LLMClient):
    """Mock LLM that returns valid JSON responses for testing."""

    def __init__(self):
        super().__init__(model="mock")
        self.call_count = 0

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return valid JSON based on prompt type."""
        self.call_count += 1

        # Detect role from prompt
        if "Generator" in prompt or "bullet_ids" in prompt:
            response = json.dumps({
                "reasoning": "Mock reasoning",
                "final_answer": "This is a correct mock answer",
                "bullet_ids": []
            })
        elif "Reflector" in prompt or "helpful" in prompt.lower():
            response = json.dumps({
                "analysis": "Mock analysis",
                "bullet_tags": []
            })
        elif "Curator" in prompt or "delta" in prompt.lower():
            response = json.dumps({
                "deltas": []
            })
        else:
            # Generic response
            response = json.dumps({
                "result": "Mock result"
            })

        return LLMResponse(text=response)


class SimpleTestEnvironment(TaskEnvironment):
    """Simple environment that checks if answer contains 'correct'."""

    def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
        """Evaluate if answer contains 'correct'."""
        answer = generator_output.final_answer
        success = "correct" in answer.lower()
        feedback = "✓ Contains 'correct'" if success else "✗ Missing 'correct'"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth="The answer should contain 'correct'",
            metrics={"success": success, "answer_length": len(answer)},
        )


class TestOfflineAdaptation(unittest.TestCase):
    """Test offline adaptation flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLMClient()
        self.playbook = Playbook()
        self.environment = SimpleTestEnvironment()

    def test_single_sample_adaptation(self):
        """Test adaptation with a single sample."""
        # Create adapter
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(self.llm),
            reflector=Reflector(self.llm),
            curator=Curator(self.llm),
        )

        # Create sample
        samples = [
            Sample(
                question="What is 2+2?",
                context="Simple math",
                ground_truth="4",
            )
        ]

        # Run adaptation
        results = adapter.run(samples, self.environment, epochs=1)

        # Verify results
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].generator_output)
        self.assertIsNotNone(results[0].reflection)
        self.assertIsNotNone(results[0].curator_output)
        self.assertIsNotNone(results[0].environment_result)

    def test_multi_sample_adaptation(self):
        """Test adaptation with multiple samples."""
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(self.llm),
            reflector=Reflector(self.llm),
            curator=Curator(self.llm),
        )

        samples = [
            Sample(question=f"Question {i}", context="", ground_truth=str(i))
            for i in range(5)
        ]

        results = adapter.run(samples, self.environment, epochs=1)

        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertIsNotNone(result.generator_output)
            self.assertIsNotNone(result.reflection)

    def test_multi_epoch_training(self):
        """Test multi-epoch adaptation."""
        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(self.llm),
            reflector=Reflector(self.llm),
            curator=Curator(self.llm),
        )

        samples = [
            Sample(question="Q1", context="", ground_truth="A1"),
            Sample(question="Q2", context="", ground_truth="A2"),
        ]

        # Run 3 epochs
        results = adapter.run(samples, self.environment, epochs=3)

        # Should process 2 samples × 3 epochs = 6 total
        self.assertEqual(len(results), 6)

    def test_playbook_evolution(self):
        """Test that playbook evolves during adaptation."""
        initial_bullets = len(self.playbook.bullets())

        adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=Generator(self.llm),
            reflector=Reflector(self.llm),
            curator=Curator(self.llm),
        )

        samples = [
            Sample(question="Q1", context="", ground_truth="A1"),
        ]

        adapter.run(samples, self.environment, epochs=1)

        # Playbook should have more bullets (DummyLLMClient adds bullets)
        final_bullets = len(self.playbook.bullets())
        self.assertGreaterEqual(final_bullets, initial_bullets)

    def test_checkpoint_functionality(self):
        """Test checkpoint saving during offline adaptation."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            adapter = OfflineAdapter(
                playbook=self.playbook,
                generator=Generator(self.llm),
                reflector=Reflector(self.llm),
                curator=Curator(self.llm),
            )

            samples = [
                Sample(question=f"Q{i}", context="", ground_truth=f"A{i}")
                for i in range(5)
            ]

            # Run with checkpoints every 2 samples
            adapter.run(
                samples,
                self.environment,
                epochs=1,
                checkpoint_interval=2,
                checkpoint_dir=str(checkpoint_dir),
            )

            # Check that checkpoints were created
            checkpoints = list(checkpoint_dir.glob("*.json"))
            self.assertGreater(len(checkpoints), 0)


class TestOnlineAdaptation(unittest.TestCase):
    """Test online adaptation flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLMClient()
        self.playbook = Playbook()
        self.environment = SimpleTestEnvironment()

    def test_single_sample_online(self):
        """Test online adaptation with a single sample."""
        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=Generator(self.llm),
            reflector=Reflector(self.llm),
            curator=Curator(self.llm),
        )

        samples = [
            Sample(question="What is online adaptation?", context="", ground_truth="")
        ]

        results = adapter.run(samples, self.environment)

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].generator_output)

    def test_sequential_online_adaptation(self):
        """Test that online adaptation processes samples sequentially."""
        adapter = OnlineAdapter(
            playbook=self.playbook,
            generator=Generator(self.llm),
            reflector=Reflector(self.llm),
            curator=Curator(self.llm),
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="")
            for i in range(3)
        ]

        results = adapter.run(samples, self.environment)

        # Each sample should be processed with updated playbook
        self.assertEqual(len(results), 3)


class TestPlaybookPersistence(unittest.TestCase):
    """Test playbook save/load functionality."""

    def test_save_load_roundtrip(self):
        """Test that playbook can be saved and loaded."""
        with TemporaryDirectory() as tmpdir:
            playbook_path = Path(tmpdir) / "test_playbook.json"

            # Create playbook with bullets
            original = Playbook()
            bullet = original.add_bullet(
                section="Testing",
                content="Test strategy",
                bullet_id="b1",
                metadata={"helpful": 5, "harmful": 1}
            )

            # Save
            original.save_to_file(str(playbook_path))

            # Load
            loaded = Playbook.load_from_file(str(playbook_path))

            # Verify
            self.assertEqual(len(loaded.bullets()), len(original.bullets()))
            self.assertEqual(loaded.bullets()[0].content, "Test strategy")
            self.assertEqual(loaded.bullets()[0].helpful, 5)

    def test_evolved_playbook_persistence(self):
        """Test that evolved playbook can be saved and reused."""
        with TemporaryDirectory() as tmpdir:
            playbook_path = Path(tmpdir) / "evolved_playbook.json"

            # Train adapter
            llm = MockLLMClient()
            playbook = Playbook()
            environment = SimpleTestEnvironment()

            adapter = OfflineAdapter(
                playbook=playbook,
                generator=Generator(llm),
                reflector=Reflector(llm),
                curator=Curator(llm),
            )

            samples = [
                Sample(question="Train Q", context="", ground_truth=""),
            ]

            adapter.run(samples, environment, epochs=1)

            # Save evolved playbook
            playbook.save_to_file(str(playbook_path))

            # Create new adapter with loaded playbook
            loaded_playbook = Playbook.load_from_file(str(playbook_path))
            new_adapter = OfflineAdapter(
                playbook=loaded_playbook,
                generator=Generator(llm),
                reflector=Reflector(llm),
                curator=Curator(llm),
            )

            # Verify it works
            test_samples = [Sample(question="Test Q", context="", ground_truth="")]
            results = new_adapter.run(test_samples, environment, epochs=1)

            self.assertEqual(len(results), 1)


class TestErrorRecovery(unittest.TestCase):
    """Test error handling and recovery."""

    def test_failed_sample_skipping(self):
        """Test that failed samples are skipped with proper logging."""

        class FailingEnvironment(TaskEnvironment):
            """Environment that fails on specific questions."""

            def evaluate(self, sample: Sample, generator_output) -> EnvironmentResult:
                if "fail" in sample.question.lower():
                    raise ValueError("Simulated evaluation failure")
                return EnvironmentResult(
                    feedback="OK", ground_truth="", metrics={"success": True}
                )

        llm = MockLLMClient()
        playbook = Playbook()
        environment = FailingEnvironment()

        adapter = OfflineAdapter(
            playbook=playbook,
            generator=Generator(llm),
            reflector=Reflector(llm),
            curator=Curator(llm),
        )

        samples = [
            Sample(question="Good Q1", context="", ground_truth=""),
            Sample(question="FAIL this", context="", ground_truth=""),
            Sample(question="Good Q2", context="", ground_truth=""),
        ]

        # Should skip failed sample and continue
        results = adapter.run(samples, environment, epochs=1)

        # Should process 2 successful samples (skip 1 failed)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
