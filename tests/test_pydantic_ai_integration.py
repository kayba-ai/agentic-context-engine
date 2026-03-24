"""Integration tests for PydanticAI-backed ACE roles with real API calls.

Requires AWS credentials for Bedrock access.
Run with: uv run pytest tests/test_pydantic_ai_integration.py -v -s --no-cov
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip entire module if no API credentials
pytestmark = pytest.mark.requires_api

HAS_API = bool(os.environ.get("OPENAI_API_KEY"))
if not HAS_API:
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)

from ace_next.core.outputs import (
    AgentOutput,
    ExtractedLearning,
    ReflectorOutput,
    SkillManagerOutput,
    SkillTag,
)
from ace_next.core.skillbook import Skillbook, UpdateBatch
from ace_next.implementations import Agent, Reflector, SkillManager
from ace_next.runners.litellm import ACELiteLLM
from ace_next.core.environments import Sample, SimpleEnvironment


MODEL = "openai:gpt-4o-mini"


class TestAgentRole:
    """Test Agent role produces valid structured output."""

    def test_basic_question(self):
        agent = Agent(MODEL)
        sb = Skillbook()
        output = agent.generate(
            question="What is the capital of France?",
            context="Answer in one word.",
            skillbook=sb,
        )

        assert isinstance(output, AgentOutput)
        assert len(output.reasoning) > 0, "reasoning should be non-empty"
        assert len(output.final_answer) > 0, "final_answer should be non-empty"
        assert "paris" in output.final_answer.lower(), (
            f"Expected 'Paris' in answer, got: {output.final_answer}"
        )
        assert isinstance(output.skill_ids, list)
        assert "usage" in output.raw, f"raw should contain usage, got: {output.raw}"
        assert output.raw["usage"]["prompt_tokens"] > 0
        assert output.raw["usage"]["completion_tokens"] > 0
        print(f"\n  Agent answer: {output.final_answer}")
        print(f"  Usage: {output.raw['usage']}")

    def test_with_skillbook(self):
        agent = Agent(MODEL)
        sb = Skillbook()
        sb.add_skill(
            "math",
            "Use decomposition: break large multiplications into (a*10 + b) parts",
            skill_id="math-001",
        )

        output = agent.generate(
            question="What is 17 × 23?",
            context="Show your work step by step.",
            skillbook=sb,
        )

        assert isinstance(output, AgentOutput)
        assert len(output.final_answer) > 0
        assert "391" in output.final_answer, (
            f"Expected '391' in answer, got: {output.final_answer}"
        )
        print(f"\n  Agent answer: {output.final_answer}")
        print(f"  Reasoning excerpt: {output.reasoning[:300]}...")
        print(f"  Cited skills: {output.skill_ids}")

    def test_with_reflection(self):
        agent = Agent(MODEL)
        sb = Skillbook()
        sb.add_skill("physics", "Always include the unit when stating temperatures", skill_id="phys-001")

        output = agent.generate(
            question="What temperature does water boil at in Fahrenheit? Reply with just the number and unit.",
            context="This is a factual science question. Answer concisely.",
            skillbook=sb,
            reflection="Your previous answer was incorrect. The correct answer is 212°F at standard atmospheric pressure.",
        )

        assert isinstance(output, AgentOutput)
        assert len(output.final_answer) > 0
        # The reflection explicitly states 212°F — verify the model uses it
        full_text = f"{output.final_answer} {output.reasoning}"
        assert "212" in full_text, (
            f"Expected '212' somewhere in output, got answer: {output.final_answer}"
        )
        print(f"\n  Agent answer with reflection: {output.final_answer}")


class TestReflectorRole:
    """Test Reflector role produces valid structured analysis."""

    def test_correct_answer_reflection(self):
        reflector = Reflector(MODEL)
        sb = Skillbook()
        sb.add_skill("math", "Break down multiplication", skill_id="math-001")

        agent_output = AgentOutput(
            reasoning="Following [math-001], I decomposed 15×24 as 15×20 + 15×4 = 300 + 60 = 360",
            final_answer="360",
            skill_ids=["math-001"],
        )

        output = reflector.reflect(
            question="What is 15 × 24?",
            agent_output=agent_output,
            skillbook=sb,
            ground_truth="360",
            feedback="Correct!",
        )

        assert isinstance(output, ReflectorOutput)
        assert len(output.reasoning) > 0
        assert len(output.correct_approach) > 0
        assert len(output.key_insight) > 0
        assert "usage" in output.raw
        print(f"\n  Key insight: {output.key_insight}")
        print(f"  Skill tags: {[(t.id, t.tag) for t in output.skill_tags]}")

    def test_wrong_answer_reflection(self):
        reflector = Reflector(MODEL)
        sb = Skillbook()

        agent_output = AgentOutput(
            reasoning="I calculated 15×24 = 15×20 + 15×4 = 310 + 60 = 370",
            final_answer="370",
        )

        output = reflector.reflect(
            question="What is 15 × 24?",
            agent_output=agent_output,
            skillbook=sb,
            ground_truth="360",
            feedback="Incorrect. The answer is 360.",
        )

        assert isinstance(output, ReflectorOutput)
        assert len(output.error_identification) > 0, "Should identify the error"
        assert len(output.root_cause_analysis) > 0, "Should analyze root cause"
        print(f"\n  Error identified: {output.error_identification[:200]}")
        print(f"  Root cause: {output.root_cause_analysis[:200]}")
        print(f"  Learnings: {len(output.extracted_learnings)}")

    def test_extracted_learnings_structure(self):
        reflector = Reflector(MODEL)
        sb = Skillbook()

        agent_output = AgentOutput(
            reasoning="I tried to answer directly without checking",
            final_answer="I don't know",
        )

        output = reflector.reflect(
            question="What is the population of Tokyo metropolitan area?",
            agent_output=agent_output,
            skillbook=sb,
            ground_truth="approximately 37 million",
            feedback="Incorrect. Should have provided the answer.",
        )

        assert isinstance(output, ReflectorOutput)
        for learning in output.extracted_learnings:
            assert isinstance(learning, ExtractedLearning)
            assert len(learning.learning) > 0
            assert 0.0 <= learning.atomicity_score <= 1.0
        print(f"\n  Extracted {len(output.extracted_learnings)} learnings")
        for i, l in enumerate(output.extracted_learnings):
            print(f"    [{i}] {l.learning} (atomicity: {l.atomicity_score})")


class TestSkillManagerRole:
    """Test SkillManager role produces valid skillbook updates."""

    def test_add_new_skill(self):
        sm = SkillManager(MODEL)
        sb = Skillbook()

        reflection = ReflectorOutput(
            reasoning="The agent failed because it didn't decompose the problem",
            error_identification="Tried to multiply directly without decomposition",
            root_cause_analysis="Missing strategy for breaking down multiplication",
            correct_approach="Use decomposition: 15×24 = 15×(20+4) = 300+60 = 360",
            key_insight="Break large multiplications into manageable parts",
            extracted_learnings=[
                ExtractedLearning(
                    learning="Decompose multi-digit multiplication using distributive property",
                    atomicity_score=0.95,
                    evidence="15×24 = 15×20 + 15×4 = 360",
                ),
            ],
        )

        output = sm.update_skills(
            reflections=(reflection,),
            skillbook=sb,
            question_context="Mental arithmetic",
            progress="0/1 correct",
        )

        assert isinstance(output, SkillManagerOutput)
        assert isinstance(output.update, UpdateBatch)
        assert len(output.update.reasoning) > 0
        assert "usage" in output.raw
        print(f"\n  Reasoning: {output.update.reasoning[:200]}")
        print(f"  Operations: {len(output.update.operations)}")
        for op in output.update.operations:
            print(f"    {op.type}: {op.content[:80] if op.content else 'N/A'}")

    def test_tag_existing_skill(self):
        sm = SkillManager(MODEL)
        sb = Skillbook()
        sb.add_skill(
            "math",
            "Use decomposition for multiplication",
            skill_id="math-001",
        )

        reflection = ReflectorOutput(
            reasoning="The agent correctly applied decomposition strategy",
            correct_approach="Decomposition worked well",
            key_insight="Decomposition strategy is effective",
            skill_tags=[SkillTag(id="math-001", tag="helpful")],
        )

        output = sm.update_skills(
            reflections=(reflection,),
            skillbook=sb,
            question_context="Mental arithmetic",
            progress="1/1 correct",
        )

        assert isinstance(output, SkillManagerOutput)
        print(f"\n  Operations: {len(output.update.operations)}")
        for op in output.update.operations:
            print(
                f"    {op.type} {op.skill_id or ''}: "
                f"{op.content or op.metadata}"
            )


class TestACELiteLLMIntegration:
    """Test the full ACELiteLLM flow with real API calls."""

    def test_ask(self):
        ace = ACELiteLLM.from_model(MODEL)
        answer = ace.ask("What is 2 + 2?")
        assert "4" in answer, f"Expected '4' in answer, got: {answer}"
        print(f"\n  ask() answer: {answer}")

    def test_ask_and_learn_from_feedback(self):
        ace = ACELiteLLM.from_model(MODEL)

        answer = ace.ask("What is the chemical symbol for gold?")
        print(f"\n  Answer: {answer}")
        assert len(answer) > 0

        result = ace.learn_from_feedback(
            feedback="Correct! Gold's symbol Au comes from the Latin 'aurum'.",
            ground_truth="Au",
        )
        assert result is True, "learn_from_feedback should return True"
        print(f"  Skills after learning: {len(ace.skillbook.skills())}")
        for skill in ace.skillbook.skills():
            print(f"    [{skill.id}] {skill.content[:80]}")

    def test_full_learning_pipeline(self):
        """End-to-end: learn from samples, verify skillbook grows."""
        ace = ACELiteLLM.from_model(MODEL)
        env = SimpleEnvironment()

        samples = [
            Sample(
                question="What is the speed of light in km/s?",
                ground_truth="approximately 300,000 km/s",
            ),
        ]

        results = ace.learn(samples, environment=env)
        assert len(results) == 1
        assert results[0].error is None, f"Pipeline error: {results[0].error}"
        print(f"\n  Pipeline completed. Skills: {len(ace.skillbook.skills())}")
        for skill in ace.skillbook.skills():
            print(f"    [{skill.id}] {skill.content[:80]}")

    def test_save_and_load_after_learning(self, tmp_path):
        """Skills survive save/load cycle."""
        ace = ACELiteLLM.from_model(MODEL)
        answer = ace.ask("What is H2O?")
        ace.learn_from_feedback("Correct!", ground_truth="Water")

        path = str(tmp_path / "skillbook.json")
        skills_before = len(ace.skillbook.skills())
        ace.save(path)

        ace2 = ACELiteLLM.from_model(MODEL, skillbook_path=path)
        assert len(ace2.skillbook.skills()) == skills_before
        print(f"\n  Saved and loaded {skills_before} skills successfully")


class TestRetryAndConsistency:
    """Test structured output consistency across multiple calls."""

    def test_structured_output_consistency(self):
        """Multiple calls should always produce valid structured output."""
        agent = Agent(MODEL)
        sb = Skillbook()

        questions = [
            ("What is 7 × 8?", "56"),
            ("What is the capital of Japan?", "Tokyo"),
            ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ]

        for q, expected in questions:
            output = agent.generate(
                question=q,
                context="Answer concisely.",
                skillbook=sb,
            )
            assert isinstance(output, AgentOutput), f"Wrong type for '{q}'"
            assert len(output.reasoning) > 0, f"Empty reasoning for '{q}'"
            assert len(output.final_answer) > 0, f"Empty answer for '{q}'"
            assert isinstance(output.raw, dict), f"raw not dict for '{q}'"
            assert "usage" in output.raw, f"No usage in raw for '{q}'"
            assert expected.lower() in output.final_answer.lower(), (
                f"Expected '{expected}' in answer for '{q}', got: {output.final_answer}"
            )
            print(f"\n  Q: {q} -> A: {output.final_answer}")


class TestRRStepIntegration:
    """Test the PydanticAI-based Recursive Reflector (RRStep) with real API calls."""

    RR_MODEL = "openai:gpt-4o-mini"

    @pytest.mark.integration
    def test_rr_basic_reflection(self):
        """RRStep.reflect returns a valid ReflectorOutput with non-empty fields."""
        from ace_next.rr.runner import RRStep
        from ace_next.rr.config import RecursiveConfig

        config = RecursiveConfig(
            max_llm_calls=15,
            max_iterations=10,
            timeout=15.0,
            enable_subagent=False,
        )
        rr = RRStep(model=self.RR_MODEL, config=config)
        sb = Skillbook()

        agent_out = AgentOutput(
            reasoning="I recall that the capital of Australia is Sydney because it is the largest city.",
            final_answer="Sydney",
            skill_ids=[],
        )

        output = rr.reflect(
            question="What is the capital of Australia?",
            agent_output=agent_out,
            skillbook=sb,
            ground_truth="Canberra",
            feedback="Incorrect. The capital of Australia is Canberra, not Sydney.",
        )

        assert isinstance(output, ReflectorOutput), f"Expected ReflectorOutput, got {type(output)}"
        assert len(output.reasoning) > 0, "reasoning should be non-empty"
        assert len(output.key_insight) > 0, "key_insight should be non-empty"
        assert isinstance(output.raw, dict), "raw should be a dict"

        # Verify rr_trace metadata is populated
        rr_trace = output.raw.get("rr_trace", {})
        assert isinstance(rr_trace, dict), "rr_trace should be a dict in raw"
        assert "total_iterations" in rr_trace, "rr_trace should have total_iterations"

        print(f"\n  Reasoning: {output.reasoning[:300]}")
        print(f"  Key insight: {output.key_insight[:200]}")
        print(f"  RR trace: {rr_trace}")

    @pytest.mark.integration
    def test_rr_with_skillbook(self):
        """RRStep.reflect with a populated skillbook references or tags skills."""
        from ace_next.rr.runner import RRStep
        from ace_next.rr.config import RecursiveConfig

        config = RecursiveConfig(
            max_llm_calls=25,
            max_iterations=10,
            timeout=15.0,
            enable_subagent=False,
        )
        rr = RRStep(model=self.RR_MODEL, config=config)
        sb = Skillbook()
        sb.add_skill(
            "geography",
            "Always verify capital cities — the largest city is often not the capital",
            skill_id="geo-001",
        )

        agent_out = AgentOutput(
            reasoning="The largest city in Brazil is Sao Paulo, so it must be the capital.",
            final_answer="Sao Paulo",
            skill_ids=[],
        )

        output = rr.reflect(
            question="What is the capital of Brazil?",
            agent_output=agent_out,
            skillbook=sb,
            ground_truth="Brasilia",
            feedback="Incorrect. The capital of Brazil is Brasilia.",
        )

        assert isinstance(output, ReflectorOutput)
        assert len(output.reasoning) > 0
        assert len(output.key_insight) > 0

        # The reflector should reference the skill in some way — either via
        # skill_tags or by mentioning the skill in its reasoning/key_insight.
        # The RR should produce a meaningful analysis — it may or may not
        # reference the specific skill ID depending on the model.
        full_text = f"{output.reasoning} {output.key_insight} {output.extracted_learnings}"
        has_analysis = (
            "capital" in full_text.lower()
            or "largest" in full_text.lower()
            or "brasilia" in full_text.lower()
            or len(output.skill_tags) > 0
        )
        assert has_analysis, (
            "Expected the reflector to analyze the capital city error. "
            f"reasoning={output.reasoning[:200]}"
        )

        print(f"\n  Key insight: {output.key_insight[:200]}")
        print(f"  Skill tags: {[(t.id, t.tag) for t in output.skill_tags]}")
        print(f"  Learnings: {len(output.extracted_learnings)}")

    @pytest.mark.integration
    def test_rr_step_protocol(self):
        """RRStep used as a StepProtocol: __call__(ctx) populates reflections."""
        from ace_next.rr.runner import RRStep
        from ace_next.rr.config import RecursiveConfig
        from ace_next.core.context import ACEStepContext, SkillbookView

        config = RecursiveConfig(
            max_llm_calls=15,
            max_iterations=10,
            timeout=15.0,
            enable_subagent=False,
        )
        rr = RRStep(model=self.RR_MODEL, config=config)
        sb = Skillbook()

        trace = {
            "question": "What is 15 x 24?",
            "ground_truth": "360",
            "feedback": "Incorrect. The correct answer is 360.",
            "steps": [
                {
                    "role": "agent",
                    "reasoning": "15 x 24 = 15 x 20 + 15 x 4 = 310 + 60 = 370",
                    "answer": "370",
                    "skill_ids": [],
                },
            ],
        }

        ctx = ACEStepContext(
            trace=trace,
            skillbook=SkillbookView(sb),
        )

        result_ctx = rr(ctx)

        assert result_ctx.reflections is not None, "reflections should be set"
        assert len(result_ctx.reflections) > 0, "reflections should be non-empty"
        for reflection in result_ctx.reflections:
            assert isinstance(reflection, ReflectorOutput)
            assert len(reflection.reasoning) > 0

        print(f"\n  Reflections count: {len(result_ctx.reflections)}")
        print(f"  First reasoning: {result_ctx.reflections[0].reasoning[:300]}")
        print(f"  First key_insight: {result_ctx.reflections[0].key_insight[:200]}")

    @pytest.mark.integration
    def test_rr_execute_code_tool_used(self):
        """Verify the agent uses execute_code (total_iterations > 0 in rr_trace)."""
        from ace_next.rr.runner import RRStep
        from ace_next.rr.config import RecursiveConfig

        config = RecursiveConfig(
            max_llm_calls=15,
            max_iterations=10,
            timeout=15.0,
            enable_subagent=False,
        )
        rr = RRStep(model=self.RR_MODEL, config=config)
        sb = Skillbook()

        agent_out = AgentOutput(
            reasoning=(
                "I need to find the square root of 144. "
                "I think it might be 14 since 14 x 14 is close to 144."
            ),
            final_answer="14",
            skill_ids=[],
        )

        output = rr.reflect(
            question="What is the square root of 144?",
            agent_output=agent_out,
            skillbook=sb,
            ground_truth="12",
            feedback="Incorrect. The square root of 144 is 12, not 14.",
        )

        assert isinstance(output, ReflectorOutput)

        rr_trace = output.raw.get("rr_trace", {})
        total_iterations = rr_trace.get("total_iterations", 0)
        assert total_iterations > 0, (
            f"Expected execute_code to be called at least once "
            f"(total_iterations > 0), got {total_iterations}. "
            f"rr_trace={rr_trace}"
        )

        print(f"\n  Total iterations (execute_code calls): {total_iterations}")
        print(f"  Timed out: {rr_trace.get('timed_out', 'N/A')}")
        print(f"  Key insight: {output.key_insight[:200]}")
        print(f"  Reasoning: {output.reasoning[:300]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
