"""
Tests for async learning infrastructure.

Tests for ThreadSafeSkillbook, AsyncLearningPipeline, and adapter async mode.
"""

import json
import threading
import time
import unittest
from typing import List

import pytest

from ace import (
    SkillManager,
    EnvironmentResult,
    Agent,
    OfflineACE,
    OnlineACE,
    Skillbook,
    Reflector,
    Sample,
    TaskEnvironment,
)
from ace.async_learning import (
    AsyncLearningPipeline,
    LearningTask,
    ReflectionResult,
    ThreadSafeSkillbook,
)
from ace.updates import UpdateBatch
from ace.roles import SkillTag, AgentOutput, ReflectorOutput, SkillManagerOutput

# Import MockLLMClient from conftest
from tests.conftest import MockLLMClient


# ---------------------------------------------------------------------------
# Test Response Helpers
# ---------------------------------------------------------------------------


def make_agent_response(answer: str = "correct answer") -> str:
    """Create a valid Agent JSON response."""
    return json.dumps(
        {
            "reasoning": "Test reasoning",
            "final_answer": answer,
            "skill_ids": [],
        }
    )


def make_reflector_response() -> str:
    """Create a valid Reflector JSON response."""
    return json.dumps(
        {
            "reasoning": "Test reflection reasoning",
            "error_identification": "",
            "root_cause_analysis": "",
            "correct_approach": "The approach was correct",
            "key_insight": "Always verify the answer",
            "skill_tags": [],
        }
    )


def make_skill_manager_response() -> str:
    """Create a valid SkillManager JSON response with empty updates."""
    return json.dumps(
        {
            "update": {"reasoning": "No changes needed", "operations": []},
        }
    )


class SimpleTestEnvironment(TaskEnvironment):
    """Simple environment for testing."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate if answer contains 'correct'."""
        answer = agent_output.final_answer
        success = "correct" in answer.lower()
        feedback = "✓ Contains 'correct'" if success else "✗ Missing 'correct'"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth="The answer should contain 'correct'",
            metrics={"success": success},
        )


# ---------------------------------------------------------------------------
# ThreadSafeSkillbook Tests
# ---------------------------------------------------------------------------


class TestThreadSafeSkillbook(unittest.TestCase):
    """Test thread-safe skillbook wrapper."""

    def test_lock_free_reads(self):
        """Test that reads work without blocking."""
        skillbook = Skillbook()
        skillbook.add_skill("Test", "Test content", skill_id="b1")
        ts_skillbook = ThreadSafeSkillbook(skillbook)

        # Reads should work
        self.assertIn("Test content", ts_skillbook.as_prompt())
        self.assertEqual(len(ts_skillbook.skills()), 1)
        self.assertIsNotNone(ts_skillbook.get_skill("b1"))
        # Check actual stats keys
        stats = ts_skillbook.stats()
        self.assertIn("skills", stats)

    def test_locked_writes(self):
        """Test that writes are thread-safe."""
        skillbook = Skillbook()
        ts_skillbook = ThreadSafeSkillbook(skillbook)

        # Add skill through thread-safe wrapper
        ts_skillbook.add_skill("Test", "Content 1", skill_id="b1")
        self.assertEqual(len(ts_skillbook.skills()), 1)

        # Update skill
        ts_skillbook.update_skill("b1", content="Updated content")
        self.assertEqual(ts_skillbook.get_skill("b1").content, "Updated content")

        # Tag skill
        ts_skillbook.tag_skill("b1", "helpful")
        self.assertEqual(ts_skillbook.get_skill("b1").helpful, 1)

        # Remove skill
        ts_skillbook.remove_skill("b1")
        self.assertEqual(len(ts_skillbook.skills()), 0)

    def test_concurrent_writes(self):
        """Test that concurrent writes don't cause race conditions."""
        skillbook = Skillbook()
        skillbook.add_skill("Test", "Concurrent test", skill_id="b1")
        ts_skillbook = ThreadSafeSkillbook(skillbook)

        num_threads = 10
        increments_per_thread = 100
        errors: List[Exception] = []

        def increment_tags():
            try:
                for _ in range(increments_per_thread):
                    ts_skillbook.tag_skill("b1", "helpful")
            except Exception as e:
                errors.append(e)

        # Run concurrent increments
        threads = [threading.Thread(target=increment_tags) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        self.assertEqual(len(errors), 0)

        # Final count should be correct
        skill = ts_skillbook.get_skill("b1")
        expected_count = num_threads * increments_per_thread
        self.assertEqual(skill.helpful, expected_count)


# ---------------------------------------------------------------------------
# AsyncLearningPipeline Tests
# ---------------------------------------------------------------------------


class TestAsyncLearningPipeline(unittest.TestCase):
    """Test async learning pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()

    def _create_mock_llm(self, responses: List[str]) -> MockLLMClient:
        """Create MockLLMClient with queued responses."""
        llm = MockLLMClient()
        llm.set_responses(responses)
        return llm

    def test_start_stop_lifecycle(self):
        """Test pipeline start/stop lifecycle."""
        # Use separate LLMs for each role
        reflector_llm = self._create_mock_llm([])
        skill_manager_llm = self._create_mock_llm([])

        pipeline = AsyncLearningPipeline(
            skillbook=self.skillbook,
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
        )

        # Not running initially
        self.assertFalse(pipeline.is_running())

        # Start
        pipeline.start()
        self.assertTrue(pipeline.is_running())

        # Double start should be safe
        pipeline.start()
        self.assertTrue(pipeline.is_running())

        # Stop
        remaining = pipeline.stop(wait=True, timeout=5.0)
        self.assertFalse(pipeline.is_running())
        self.assertEqual(remaining, 0)

    def test_submit_before_start(self):
        """Test that submit returns None before pipeline starts."""
        reflector_llm = self._create_mock_llm([])
        skill_manager_llm = self._create_mock_llm([])

        pipeline = AsyncLearningPipeline(
            skillbook=self.skillbook,
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
        )

        task = self._create_dummy_task()
        result = pipeline.submit(task)
        self.assertIsNone(result)

    def test_submit_and_process(self):
        """Test submitting and processing a task."""
        # Create separate LLMs for each role
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        skill_manager_llm = self._create_mock_llm([make_skill_manager_response()])

        pipeline = AsyncLearningPipeline(
            skillbook=self.skillbook,
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            max_reflector_workers=2,
        )

        pipeline.start()
        try:
            task = self._create_dummy_task()
            future = pipeline.submit(task)

            self.assertIsNotNone(future)

            # Wait for completion
            completed = pipeline.wait_for_completion(timeout=10.0)
            self.assertTrue(completed)

            # Check stats
            stats = pipeline.stats
            self.assertEqual(stats["tasks_submitted"], 1)
            self.assertEqual(stats["reflections_completed"], 1)
            self.assertEqual(stats["skill_updates_completed"], 1)
            self.assertEqual(stats["tasks_failed"], 0)
        finally:
            pipeline.stop(wait=False)

    def test_multiple_tasks(self):
        """Test processing multiple tasks."""
        # Create separate LLMs with 3 responses each
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        skill_manager_llm = self._create_mock_llm(
            [make_skill_manager_response() for _ in range(3)]
        )

        pipeline = AsyncLearningPipeline(
            skillbook=self.skillbook,
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            max_reflector_workers=3,
        )

        pipeline.start()
        try:
            # Submit multiple tasks
            for i in range(3):
                task = self._create_dummy_task(i)
                pipeline.submit(task)

            pipeline.wait_for_completion(timeout=15.0)

            # All should be processed
            stats = pipeline.stats
            self.assertEqual(stats["tasks_submitted"], 3)
            self.assertEqual(stats["reflections_completed"], 3)
            self.assertEqual(stats["skill_updates_completed"], 3)
        finally:
            pipeline.stop(wait=False)

    def test_completion_callback(self):
        """Test completion callback invocation."""
        completions: List[tuple] = []

        def on_complete(task, skill_manager_output):
            completions.append((task, skill_manager_output))

        reflector_llm = self._create_mock_llm([make_reflector_response()])
        skill_manager_llm = self._create_mock_llm([make_skill_manager_response()])

        pipeline = AsyncLearningPipeline(
            skillbook=self.skillbook,
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            on_complete=on_complete,
        )

        pipeline.start()
        try:
            task = self._create_dummy_task()
            pipeline.submit(task)

            pipeline.wait_for_completion(timeout=10.0)

            # Completion callback should be invoked
            self.assertEqual(len(completions), 1)
            self.assertEqual(completions[0][0], task)
        finally:
            pipeline.stop(wait=False)

    def _create_dummy_task(self, step_index: int = 0) -> LearningTask:
        """Create a dummy learning task for testing."""
        sample = Sample(
            question=f"Test question {step_index}",
            context="Test context",
            ground_truth="correct",
        )
        agent_output = AgentOutput(
            reasoning="Test reasoning",
            final_answer="Test answer correct",
            skill_ids=[],
        )
        env_result = EnvironmentResult(
            feedback="Test feedback",
            ground_truth="correct",
            metrics={"success": True},
        )
        return LearningTask(
            sample=sample,
            agent_output=agent_output,
            environment_result=env_result,
            epoch=1,
            step_index=step_index,
            total_epochs=1,
            total_steps=1,
        )


# ---------------------------------------------------------------------------
# Adapter Async Mode Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOfflineACEAsyncMode(unittest.TestCase):
    """Test OfflineACE with async learning mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()
        self.environment = SimpleTestEnvironment()

    def _create_mock_llm(self, responses: List[str]) -> MockLLMClient:
        """Create MockLLMClient with queued responses."""
        llm = MockLLMClient()
        llm.set_responses(responses)
        return llm

    def test_sync_mode_unchanged(self):
        """Test that sync mode still works as before."""
        # Each role gets its own LLM with appropriate responses
        agent_llm = self._create_mock_llm([make_agent_response()])
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        skill_manager_llm = self._create_mock_llm([make_skill_manager_response()])

        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(agent_llm),
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            async_learning=False,  # Sync mode (default)
        )

        samples = [Sample(question="What is 2+2?", context="Math", ground_truth="4")]

        results = adapter.run(samples, self.environment, epochs=1)

        # Verify results have all fields populated (sync mode)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].agent_output)
        self.assertIsNotNone(results[0].reflection)  # Populated in sync mode
        self.assertIsNotNone(results[0].skill_manager_output)  # Populated in sync mode

    def test_async_mode_basic(self):
        """Test async mode returns results with None reflection/skill_manager."""
        # Each role gets its own LLM with appropriate responses
        agent_llm = self._create_mock_llm([make_agent_response() for _ in range(3)])
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        skill_manager_llm = self._create_mock_llm(
            [make_skill_manager_response() for _ in range(3)]
        )

        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(agent_llm),
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            async_learning=True,  # Async mode
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="correct")
            for i in range(3)
        ]

        results = adapter.run(
            samples,
            self.environment,
            epochs=1,
            wait_for_learning=False,  # Don't wait
        )

        # Results should be returned
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.agent_output)
            # In async mode, these are None (processing in background)
            self.assertIsNone(result.reflection)
            self.assertIsNone(result.skill_manager_output)

        # Clean up async pipeline
        adapter.stop_async_learning(wait=False)

    def test_async_mode_with_wait(self):
        """Test async mode that waits for learning completion."""
        agent_llm = self._create_mock_llm([make_agent_response() for _ in range(3)])
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        skill_manager_llm = self._create_mock_llm(
            [make_skill_manager_response() for _ in range(3)]
        )

        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(agent_llm),
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            async_learning=True,
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="correct")
            for i in range(3)
        ]

        results = adapter.run(
            samples,
            self.environment,
            epochs=1,
            wait_for_learning=True,  # Wait for completion
        )

        self.assertEqual(len(results), 3)

        # Check learning stats
        stats = adapter.learning_stats
        self.assertEqual(stats["tasks_submitted"], 3)
        self.assertEqual(stats["skill_updates_completed"], 3)

    def test_wait_for_learning_method(self):
        """Test explicit wait_for_learning method."""
        agent_llm = self._create_mock_llm([make_agent_response()])
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        skill_manager_llm = self._create_mock_llm([make_skill_manager_response()])

        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(agent_llm),
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            async_learning=True,
        )

        samples = [Sample(question="Q1", context="", ground_truth="correct")]

        # Run without waiting
        adapter.run(samples, self.environment, epochs=1, wait_for_learning=False)

        # Explicitly wait
        completed = adapter.wait_for_learning(timeout=10.0)
        self.assertTrue(completed)

        # Now stats should show completion
        stats = adapter.learning_stats
        self.assertEqual(stats["skill_updates_completed"], 1)

        # Clean up
        adapter.stop_async_learning(wait=False)

    def test_learning_stats_property(self):
        """Test learning_stats property."""
        agent_llm = self._create_mock_llm([make_agent_response()])
        reflector_llm = self._create_mock_llm([make_reflector_response()])
        skill_manager_llm = self._create_mock_llm([make_skill_manager_response()])

        adapter = OfflineACE(
            skillbook=self.skillbook,
            agent=Agent(agent_llm),
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            async_learning=True,
        )

        # Before starting, stats should show defaults
        stats = adapter.learning_stats
        self.assertEqual(stats["tasks_submitted"], 0)
        self.assertFalse(stats["is_running"])

        samples = [Sample(question="Q1", context="", ground_truth="correct")]
        adapter.run(samples, self.environment, epochs=1, wait_for_learning=True)

        # After running, stats should be updated
        stats = adapter.learning_stats
        self.assertGreater(stats["tasks_submitted"], 0)


@pytest.mark.integration
class TestOnlineACEAsyncMode(unittest.TestCase):
    """Test OnlineACE with async learning mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.skillbook = Skillbook()
        self.environment = SimpleTestEnvironment()

    def _create_mock_llm(self, responses: List[str]) -> MockLLMClient:
        """Create MockLLMClient with queued responses."""
        llm = MockLLMClient()
        llm.set_responses(responses)
        return llm

    def test_online_async_mode(self):
        """Test OnlineACE async mode."""
        agent_llm = self._create_mock_llm([make_agent_response() for _ in range(3)])
        reflector_llm = self._create_mock_llm(
            [make_reflector_response() for _ in range(3)]
        )
        skill_manager_llm = self._create_mock_llm(
            [make_skill_manager_response() for _ in range(3)]
        )

        adapter = OnlineACE(
            skillbook=self.skillbook,
            agent=Agent(agent_llm),
            reflector=Reflector(reflector_llm),
            skill_manager=SkillManager(skill_manager_llm),
            async_learning=True,
        )

        samples = [
            Sample(question=f"Q{i}", context="", ground_truth="correct")
            for i in range(3)
        ]

        results = adapter.run(samples, self.environment, wait_for_learning=True)

        self.assertEqual(len(results), 3)

        stats = adapter.learning_stats
        self.assertEqual(stats["tasks_submitted"], 3)


# ---------------------------------------------------------------------------
# Data Classes Tests
# ---------------------------------------------------------------------------


class TestLearningTask(unittest.TestCase):
    """Test LearningTask dataclass."""

    def test_creation_with_defaults(self):
        """Test LearningTask creation with default values."""
        sample = Sample(question="Q", context="C", ground_truth="A")
        agent_out = AgentOutput(reasoning="R", final_answer="A", skill_ids=[])
        env_result = EnvironmentResult(feedback="F", ground_truth="A", metrics={})

        task = LearningTask(
            sample=sample,
            agent_output=agent_out,
            environment_result=env_result,
            epoch=1,
            step_index=0,
        )

        self.assertEqual(task.epoch, 1)
        self.assertEqual(task.step_index, 0)
        self.assertEqual(task.total_epochs, 1)  # Default
        self.assertEqual(task.total_steps, 1)  # Default
        self.assertIsNotNone(task.timestamp)
        self.assertEqual(task.metadata, {})


class TestReflectionResult(unittest.TestCase):
    """Test ReflectionResult dataclass."""

    def test_creation(self):
        """Test ReflectionResult creation."""
        sample = Sample(question="Q", context="C", ground_truth="A")
        agent_out = AgentOutput(reasoning="R", final_answer="A", skill_ids=[])
        env_result = EnvironmentResult(feedback="F", ground_truth="A", metrics={})
        task = LearningTask(
            sample=sample,
            agent_output=agent_out,
            environment_result=env_result,
            epoch=1,
            step_index=0,
        )
        # Create proper ReflectorOutput with all required fields
        reflection = ReflectorOutput(
            reasoning="Analysis reasoning",
            correct_approach="The correct approach",
            key_insight="Key insight learned",
            skill_tags=[],
        )

        result = ReflectionResult(task=task, reflection=reflection)

        self.assertEqual(result.task, task)
        self.assertEqual(result.reflection, reflection)
        self.assertIsNotNone(result.timestamp)


if __name__ == "__main__":
    unittest.main()
