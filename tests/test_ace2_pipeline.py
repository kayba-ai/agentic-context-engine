"""Tests for the ace2 pipeline — steps, runners, and end-to-end."""

import json

import pytest

from ace.adaptation import EnvironmentResult, Sample, TaskEnvironment
from ace.llm import DummyLLMClient
from ace.roles import Agent, Reflector, SkillManager
from ace.skillbook import Skillbook
from pipeline import Pipeline, StepContext
from pipeline.protocol import SampleResult

from ace2.steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep
from ace2.pipelines import ace_pipeline, OfflineACE, OnlineACE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_RESPONSE = json.dumps(
    {
        "reasoning": "The answer is clearly 42.",
        "skill_ids": [],
        "final_answer": "42",
    }
)

REFLECTOR_RESPONSE = json.dumps(
    {
        "reasoning": "Prediction matches ground truth.",
        "error_identification": "",
        "root_cause_analysis": "",
        "correct_approach": "Keep doing this.",
        "key_insight": "42 is always the answer.",
        "skill_tags": [],
    }
)

SKILL_MANAGER_RESPONSE = json.dumps(
    {
        "update": {
            "reasoning": "Adding a new strategy.",
            "operations": [
                {
                    "type": "ADD",
                    "section": "default_answers",
                    "content": "When asked about life, answer 42.",
                    "metadata": {"helpful": 1},
                }
            ],
        }
    }
)


def _make_client() -> DummyLLMClient:
    """Create a DummyLLMClient pre-loaded with one full cycle of responses."""
    client = DummyLLMClient()
    client.queue(AGENT_RESPONSE)
    client.queue(REFLECTOR_RESPONSE)
    client.queue(SKILL_MANAGER_RESPONSE)
    return client


class SimpleQAEnvironment(TaskEnvironment):
    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        gt = sample.ground_truth or ""
        correct = agent_output.final_answer.strip().lower() == gt.strip().lower()
        return EnvironmentResult(
            feedback="correct" if correct else f"expected {gt}",
            ground_truth=gt,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )


SAMPLE = Sample(
    question="What is the answer to life, the universe, and everything?",
    ground_truth="42",
)
ENV = SimpleQAEnvironment()


# ---------------------------------------------------------------------------
# Step-level unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentStep:
    def test_produces_agent_output(self):
        client = DummyLLMClient()
        client.queue(AGENT_RESPONSE)
        agent = Agent(client)
        step = AgentStep(agent)

        ctx = StepContext(
            sample=SAMPLE,
            skillbook=Skillbook(),
            recent_reflections=(),
        )
        out = step(ctx)

        assert out.agent_output is not None
        assert out.agent_output.final_answer == "42"

    def test_passes_recent_reflections(self):
        client = DummyLLMClient()
        client.queue(AGENT_RESPONSE)
        agent = Agent(client)
        step = AgentStep(agent)

        ctx = StepContext(
            sample=SAMPLE,
            skillbook=Skillbook(),
            recent_reflections=("reflection 1", "reflection 2"),
        )
        out = step(ctx)
        assert out.agent_output is not None


@pytest.mark.unit
class TestEvaluateStep:
    def test_produces_environment_result(self):
        from ace.roles import AgentOutput

        step = EvaluateStep()
        agent_output = AgentOutput(
            reasoning="test", final_answer="42", skill_ids=[], raw={}
        )
        ctx = StepContext(
            sample=SAMPLE,
            agent_output=agent_output,
            environment=ENV,
        )
        out = step(ctx)

        assert out.environment_result is not None
        assert out.environment_result.feedback == "correct"


@pytest.mark.unit
class TestReflectStep:
    def test_produces_reflection(self):
        client = DummyLLMClient()
        client.queue(REFLECTOR_RESPONSE)
        reflector = Reflector(client)
        step = ReflectStep(reflector, reflection_window=3)

        from ace.roles import AgentOutput

        ctx = StepContext(
            sample=SAMPLE,
            agent_output=AgentOutput(
                reasoning="test", final_answer="42", skill_ids=[], raw={}
            ),
            environment_result=EnvironmentResult(feedback="correct", ground_truth="42"),
            skillbook=Skillbook(),
            recent_reflections=(),
        )
        out = step(ctx)

        assert out.reflection is not None
        assert out.reflection.key_insight == "42 is always the answer."
        # recent_reflections should have grown by 1
        assert len(out.recent_reflections) == 1

    def test_reflection_window_trims(self):
        client = DummyLLMClient()
        client.queue(REFLECTOR_RESPONSE)
        reflector = Reflector(client)
        step = ReflectStep(reflector, reflection_window=2)

        from ace.roles import AgentOutput

        ctx = StepContext(
            sample=SAMPLE,
            agent_output=AgentOutput(
                reasoning="test", final_answer="42", skill_ids=[], raw={}
            ),
            environment_result=EnvironmentResult(feedback="correct", ground_truth="42"),
            skillbook=Skillbook(),
            recent_reflections=("old1", "old2"),  # already at capacity
        )
        out = step(ctx)

        # Window should still be 2 (oldest trimmed)
        assert len(out.recent_reflections) == 2


@pytest.mark.unit
class TestUpdateStep:
    def test_applies_update_to_skillbook(self):
        client = DummyLLMClient()
        client.queue(SKILL_MANAGER_RESPONSE)
        sm = SkillManager(client)
        step = UpdateStep(sm)

        from ace.roles import AgentOutput, ReflectorOutput

        skillbook = Skillbook()
        ctx = StepContext(
            sample=SAMPLE,
            agent_output=AgentOutput(
                reasoning="test", final_answer="42", skill_ids=[], raw={}
            ),
            environment_result=EnvironmentResult(feedback="correct", ground_truth="42"),
            reflection=ReflectorOutput(
                reasoning="ok",
                error_identification="",
                correct_approach="keep going",
                key_insight="42",
                raw={},
            ),
            skillbook=skillbook,
            epoch=1,
            total_epochs=1,
            step_index=1,
            total_steps=1,
        )
        out = step(ctx)

        assert out.skill_manager_output is not None
        assert skillbook.stats()["skills"] >= 1


# ---------------------------------------------------------------------------
# Pipeline wiring test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAcePipeline:
    def test_ace_pipeline_builds(self):
        client = DummyLLMClient()
        pipe = ace_pipeline(Agent(client), Reflector(client), SkillManager(client))

        assert isinstance(pipe, Pipeline)
        assert "agent_output" in pipe.provides
        assert "skill_manager_output" in pipe.provides


# ---------------------------------------------------------------------------
# End-to-end runner tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOfflineACE:
    def test_single_sample_single_epoch(self):
        client = _make_client()
        skillbook = Skillbook()
        ace = OfflineACE.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            skillbook=skillbook,
        )

        results = ace.run([SAMPLE], ENV, epochs=1)

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].output is not None
        assert results[0].output.agent_output.final_answer == "42"
        assert skillbook.stats()["skills"] >= 1

    def test_multi_epoch(self):
        client = DummyLLMClient()
        # Queue 2 full cycles (2 epochs × 1 sample)
        for _ in range(2):
            client.queue(AGENT_RESPONSE)
            client.queue(REFLECTOR_RESPONSE)
            client.queue(SKILL_MANAGER_RESPONSE)

        skillbook = Skillbook()
        ace = OfflineACE.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            skillbook=skillbook,
        )

        results = ace.run([SAMPLE], ENV, epochs=2)
        assert len(results) == 2

    def test_checkpoint_raises_without_dir(self):
        client = _make_client()
        ace = OfflineACE.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
        )
        with pytest.raises(ValueError, match="checkpoint_dir"):
            ace.run([SAMPLE], ENV, checkpoint_interval=1)

    def test_failed_sample_captured(self):
        """A client with no queued responses should produce an error result."""
        client = DummyLLMClient()  # empty — will raise on first call
        ace = OfflineACE.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
        )

        results = ace.run([SAMPLE], ENV)
        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].output is None


@pytest.mark.unit
class TestOnlineACE:
    def test_single_sample(self):
        client = _make_client()
        skillbook = Skillbook()
        ace = OnlineACE.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
            skillbook=skillbook,
        )

        results = ace.run([SAMPLE], ENV)

        assert len(results) == 1
        assert results[0].error is None
        assert skillbook.stats()["skills"] >= 1

    def test_stream(self):
        client = DummyLLMClient()
        for _ in range(3):
            client.queue(AGENT_RESPONSE)
            client.queue(REFLECTOR_RESPONSE)
            client.queue(SKILL_MANAGER_RESPONSE)

        ace = OnlineACE.from_roles(
            agent=Agent(client),
            reflector=Reflector(client),
            skill_manager=SkillManager(client),
        )

        def sample_gen():
            for _ in range(3):
                yield SAMPLE

        results = ace.run(sample_gen(), ENV)
        assert len(results) == 3
        assert all(r.error is None for r in results)


# ---------------------------------------------------------------------------
# from_client shorthand tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromClient:
    def test_offline_from_client(self):
        client = _make_client()
        skillbook = Skillbook()
        ace = OfflineACE.from_client(client, skillbook=skillbook)

        results = ace.run([SAMPLE], ENV)
        assert len(results) == 1
        assert results[0].error is None
        assert skillbook.stats()["skills"] >= 1

    def test_online_from_client(self):
        client = _make_client()
        skillbook = Skillbook()
        ace = OnlineACE.from_client(client, skillbook=skillbook)

        results = ace.run([SAMPLE], ENV)
        assert len(results) == 1
        assert results[0].error is None
        assert skillbook.stats()["skills"] >= 1

    def test_from_client_creates_default_skillbook(self):
        client = _make_client()
        ace = OfflineACE.from_client(client)

        results = ace.run([SAMPLE], ENV)
        assert results[0].error is None
        assert ace.skillbook.stats()["skills"] >= 1
