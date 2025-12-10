"""
Integration tests for end-to-end ACE adaptation flows.

These tests verify the complete workflow from sample → generate → reflect → update skills.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from ace import (
    SkillManager,
    EnvironmentResult,
    Agent,
    LLMClient,
    OfflineACE,
    OnlineACE,
    Skillbook,
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

        # Detect role from prompt - check more specific markers first
        # v2.1 prompts use "ACE Reflector", "ACE SkillManager", "ACE Agent"
        if "ACE Reflector" in prompt or "Reflector" in prompt:
            response = json.dumps(
                {
                    "reasoning": "Mock analysis of the outcome",
                    "error_identification": "",
                    "root_cause_analysis": "",
                    "correct_approach": "The correct approach was taken",
                    "key_insight": "Key insight from this iteration",
                    "skill_tags": [],
                }
            )
        elif (
            "ACE SkillManager" in prompt
            or "SkillManager" in prompt
            or "update" in prompt.lower()
        ):
            response = json.dumps(
                {"update": {"reasoning": "No changes needed", "operations": []}}
            )
        elif "ACE Agent" in prompt or "Agent" in prompt or "skill_ids" in prompt:
            response = json.dumps(
                {
                    "reasoning": "Mock reasoning",
                    "final_answer": "This is a correct mock answer",
                    "skill_ids": [],
                }
            )
        elif "helpful" in prompt.lower():
            response = json.dumps(
                {"update": {"reasoning": "No changes needed", "operations": []}}
            )
        else:
            # Generic response
            response = json.dumps({"result": "Mock result"})

        return LLMResponse(text=response)

    def complete_structured(self, prompt: str, response_model, **kwargs):
        """Mock structured output to prevent Instructor wrapping."""
        from ace.updates import UpdateBatch
        from ace.roles import SkillManagerOutput

        response = self.complete(prompt, **kwargs)
        data = json.loads(response.text)

        # Special handling for SkillManagerOutput (update is a dataclass, not Pydantic)
        if response_model == SkillManagerOutput:
            update_data = data.get("update", {})
            update = UpdateBatch.from_json(update_data)
            return SkillManagerOutput(update=update, raw=data)

        return response_model.model_validate(data)


class SimpleTestEnvironment(TaskEnvironment):
    """Simple environment that checks if answer contains 'correct'."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate if answer contains 'correct'."""
        answer = agent_output.final_answer
        success = "correct" in answer.lower()
        feedback = "✓ Contains 'correct'" if success else "✗ Missing 'correct'"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth="The answer should contain 'correct'",
            metrics={"success": success, "answer_length": len(answer)},
        )


@pytest.mark.integration
class TestOfflineAdaptation(unittest.TestCase):
    """Test offline adaptation flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLMClient()
        self.skillbook = Skillbook()
        self.environment = SimpleTestEnvironment()

    def test_single_sample_adaptation(self):
        """Test adaptation with a single sample."""
        # Create adapter
        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(self.llm),
            reflector=Reflector(self.llm),
            skill_manager=SkillManager(self.llm),
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
        self.assertIsNotNone(results[0].agent_output)
        self.assertIsNotNone(results[0].reflection)
        self.assertIsNotNone(results[0].skill_manager_output)
        self.assertIsNotNone(results[0].environment_result)

    def test_multi_sample_adaptation(self):
        """Test adaptation with multiple samples."""
        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(self.llm),
            reflector=Reflector(self.llm),
            skill_manager=SkillManager(self.llm),
        )

        samples = [
            Sample(question=f"Question {i}", context="", ground_truth=str(i))
            for i in range(5)
        ]

        results = adapter.run(samples, self.environment, epochs=1)

        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertIsNotNone(result.agent_output)
            self.assertIsNotNone(result.reflection)

    def test_multi_epoch_training(self):
        """Test multi-epoch adaptation."""
        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(self.llm),
            reflector=Reflector(self.llm),
            skill_manager=SkillManager(self.llm),
        )

        samples = [
            Sample(question="Q1", context="", ground_truth="A1"),
            Sample(question="Q2", context="", ground_truth="A2"),
        ]

        # Run 3 epochs
        results = adapter.run(samples, self.environment, epochs=3)

        # Should process 2 samples × 3 epochs = 6 total
        self.assertEqual(len(results), 6)

    def test_skillbook_evolution(self):
        """Test that skillbook evolves during adaptation."""
        initial_skills = len(self.skillbook.skills())

        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(self.llm),
            reflector=Reflector(self.llm),
            skill_manager=SkillManager(self.llm),
        )

        samples = [
            Sample(question="Q1", context="", ground_truth="A1"),
        ]

        adapter.run(samples, self.environment, epochs=1)

        # Skillbook should have more skills (DummyLLMClient adds skills)
        final_skills = len(self.skillbook.skills())
        self.assertGreaterEqual(final_skills, initial_skills)

    def test_checkpoint_functionality(self):
        """Test checkpoint saving during offline adaptation."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            adapter = OfflineACE(
                skillbook=self.skillbook,
                agent=Agent(self.llm),
                reflector=Reflector(self.llm),
                skill_manager=SkillManager(self.llm),
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


@pytest.mark.integration
class TestOnlineAdaptation(unittest.TestCase):
    """Test online adaptation flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm = MockLLMClient()
        self.skillbook = Skillbook()
        self.environment = SimpleTestEnvironment()

    def test_single_sample_online(self):
        """Test online adaptation with a single sample."""
        adapter = OnlineACE(
            skillbook=self.skillbook,
            agent=Agent(self.llm),
            reflector=Reflector(self.llm),
            skill_manager=SkillManager(self.llm),
        )

        samples = [
            Sample(question="What is online adaptation?", context="", ground_truth="")
        ]

        results = adapter.run(samples, self.environment)

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].agent_output)

    def test_sequential_online_adaptation(self):
        """Test that online adaptation processes samples sequentially."""
        adapter = OnlineACE(
            skillbook=self.skillbook,
            agent=Agent(self.llm),
            reflector=Reflector(self.llm),
            skill_manager=SkillManager(self.llm),
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="") for i in range(3)
        ]

        results = adapter.run(samples, self.environment)

        # Each sample should be processed with updated skillbook
        self.assertEqual(len(results), 3)


@pytest.mark.integration
class TestSkillbookPersistence(unittest.TestCase):
    """Test skillbook save/load functionality."""

    def test_save_load_roundtrip(self):
        """Test that skillbook can be saved and loaded."""
        with TemporaryDirectory() as tmpdir:
            skillbook_path = Path(tmpdir) / "test_skillbook.json"

            # Create skillbook with skills
            original = Skillbook()
            skill = original.add_skill(
                section="Testing",
                content="Test strategy",
                skill_id="b1",
                metadata={"helpful": 5, "harmful": 1},
            )

            # Save
            original.save_to_file(str(skillbook_path))

            # Load
            loaded = Skillbook.load_from_file(str(skillbook_path))

            # Verify
            self.assertEqual(len(loaded.skills()), len(original.skills()))
            self.assertEqual(loaded.skills()[0].content, "Test strategy")
            self.assertEqual(loaded.skills()[0].helpful, 5)

    def test_evolved_skillbook_persistence(self):
        """Test that evolved skillbook can be saved and reused."""
        with TemporaryDirectory() as tmpdir:
            skillbook_path = Path(tmpdir) / "evolved_skillbook.json"

            # Train adapter
            llm = MockLLMClient()
            skillbook = Skillbook()
            environment = SimpleTestEnvironment()

            adapter = OfflineACE(
                skillbook=skillbook,
                agent=Agent(llm),
                reflector=Reflector(llm),
                skill_manager=SkillManager(llm),
            )

            samples = [
                Sample(question="Train Q", context="", ground_truth=""),
            ]

            adapter.run(samples, environment, epochs=1)

            # Save evolved skillbook
            skillbook.save_to_file(str(skillbook_path))

            # Create new adapter with loaded skillbook
            loaded_skillbook = Skillbook.load_from_file(str(skillbook_path))
            new_adapter = OfflineACE(
                skillbook=loaded_skillbook,
                agent=Agent(llm),
                reflector=Reflector(llm),
                skill_manager=SkillManager(llm),
            )

            # Verify it works
            test_samples = [Sample(question="Test Q", context="", ground_truth="")]
            results = new_adapter.run(test_samples, environment, epochs=1)

            self.assertEqual(len(results), 1)


@pytest.mark.integration
class TestErrorRecovery(unittest.TestCase):
    """Test error handling and recovery."""

    def test_failed_sample_skipping(self):
        """Test that failed samples are skipped with proper logging."""

        class FailingEnvironment(TaskEnvironment):
            """Environment that fails on specific questions."""

            def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
                if "fail" in sample.question.lower():
                    raise ValueError("Simulated evaluation failure")
                return EnvironmentResult(
                    feedback="OK", ground_truth="", metrics={"success": True}
                )

        llm = MockLLMClient()
        skillbook = Skillbook()
        environment = FailingEnvironment()

        adapter = OfflineACE(
            skillbook=skillbook,
            agent=Agent(llm),
            reflector=Reflector(llm),
            skill_manager=SkillManager(llm),
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
