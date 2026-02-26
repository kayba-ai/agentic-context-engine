"""Pre-refactor smoke test — exercises the ACE public API with DummyLLMClient.

Run:  uv run python scripts/pre_refactor_smoke_test.py
No API keys needed.
"""

import json
import tempfile
from pathlib import Path

from ace import (
    # Core data
    Skill,
    Skillbook,
    UpdateOperation,
    UpdateBatch,
    Sample,
    EnvironmentResult,
    SimpleEnvironment,
    ACEStepResult,
    # Roles
    Agent,
    ReplayAgent,
    Reflector,
    SkillManager,
    AgentOutput,
    ReflectorOutput,
    SkillManagerOutput,
    # LLM
    DummyLLMClient,
    LLMClient,
    # Orchestration
    OfflineACE,
    OnlineACE,
    # Async
    LearningTask,
    ReflectionResult,
    ThreadSafeSkillbook,
    AsyncLearningPipeline,
    # Dedup
    DeduplicationConfig,
    DeduplicationManager,
    # Prompts
    PromptManager,
    # Features
    LITELLM_AVAILABLE,
    OBSERVABILITY_AVAILABLE,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Skillbook & Skill ─────────────────────────────────────

section("Skillbook & Skill")

sb = Skillbook()
check("empty skillbook", len(sb.skills()) == 0)

sb.add_skill(section="testing", content="Always write tests first", skill_id="test-001")
check("add skill", len(sb.skills()) == 1)
check("skill lookup", sb.get_skill("test-001") is not None)
check("skill content", sb.get_skill("test-001").content == "Always write tests first")

sb.add_skill(
    section="testing", content="Use mocks for external calls", skill_id="test-002"
)
check("two skills", len(sb.skills()) == 2)

sb.tag_skill("test-001", "helpful")
check("tag helpful", sb.get_skill("test-001").helpful == 1)

sb.tag_skill("test-001", "harmful")
check("tag harmful", sb.get_skill("test-001").harmful == 1)

sb.remove_skill("test-002")
check("remove skill", len(sb.skills()) == 1)
check("removed not found", sb.get_skill("test-002") is None)

# Prompt rendering
prompt_text = sb.as_prompt()
check("as_prompt returns string", isinstance(prompt_text, str) and len(prompt_text) > 0)

# Serialization round-trip
with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    tmp_path = f.name
sb.save_to_file(tmp_path)
loaded = Skillbook.load_from_file(tmp_path)
check("save/load round-trip", len(loaded.skills()) == 1)
check(
    "loaded content matches",
    loaded.get_skill("test-001").content == "Always write tests first",
)
Path(tmp_path).unlink()


# ── 2. UpdateOperation & UpdateBatch ─────────────────────────

section("UpdateOperation & UpdateBatch")

op_add = UpdateOperation(
    type="ADD", section="general", content="Be concise", skill_id="new-001"
)
check("create ADD op", op_add.type == "ADD")

op_update = UpdateOperation(
    type="UPDATE", section="testing", skill_id="test-001", content="Updated"
)
check("create UPDATE op", op_update.type == "UPDATE")

op_tag = UpdateOperation(
    type="TAG", section="testing", skill_id="test-001", metadata={"helpful": 1}
)
check("create TAG op", op_tag.type == "TAG")

op_remove = UpdateOperation(type="REMOVE", section="testing", skill_id="test-001")
check("create REMOVE op", op_remove.type == "REMOVE")

batch = UpdateBatch(reasoning="test batch", operations=[op_add, op_tag])
check("batch creation", len(batch.operations) == 2)

# from_json
op_from_json = UpdateOperation.from_json(
    {"type": "ADD", "section": "s", "content": "c"}
)
check("UpdateOperation.from_json", op_from_json.type == "ADD")

batch_from_json = UpdateBatch.from_json(
    {
        "reasoning": "r",
        "operations": [{"type": "ADD", "section": "s", "content": "c"}],
    }
)
check("UpdateBatch.from_json", len(batch_from_json.operations) == 1)

# Apply batch to skillbook
sb2 = Skillbook()
sb2.add_skill(section="testing", content="original", skill_id="test-001")
sb2.apply_update(batch)
check("apply_update adds skill", sb2.get_skill("new-001") is not None)
check("apply_update tags skill", sb2.get_skill("test-001").helpful >= 1)


# ── 3. Pydantic Output Models ────────────────────────────────

section("Pydantic Output Models")

agent_out = AgentOutput(
    reasoning="Step 1: think",
    final_answer="42",
    skill_ids=["test-001"],
)
check("AgentOutput", agent_out.final_answer == "42")

reflector_out = ReflectorOutput(
    reasoning="The answer was correct",
    correct_approach="Use arithmetic",
    key_insight="Addition is commutative",
    skill_tags=[],
)
check("ReflectorOutput", reflector_out.key_insight == "Addition is commutative")

sm_out = SkillManagerOutput(
    update=UpdateBatch(reasoning="add skill", operations=[op_add]),
)
check("SkillManagerOutput", len(sm_out.update.operations) == 1)


# ── 4. DummyLLMClient ────────────────────────────────────────

section("DummyLLMClient")

llm = DummyLLMClient()
llm.queue("hello")
resp = llm.complete("say hi")
check("complete returns text", resp.text == "hello")

llm.queue(json.dumps({"reasoning": "r", "final_answer": "42", "skill_ids": []}))
structured = llm.complete_structured("q", AgentOutput)
check("complete_structured", structured.final_answer == "42")


# ── 5. Roles (Agent, Reflector, SkillManager) ────────────────

section("Roles — Agent")

llm = DummyLLMClient()
agent = Agent(llm)
check("Agent created", agent is not None)
check("Agent has llm", agent.llm is not None)

llm.queue(
    json.dumps(
        {
            "reasoning": "2+2 is basic arithmetic",
            "final_answer": "4",
            "skill_ids": [],
        }
    )
)
sb_for_roles = Skillbook()
out = agent.generate(question="What is 2+2?", context="", skillbook=sb_for_roles)
check("Agent.generate returns AgentOutput", isinstance(out, AgentOutput))
check("Agent answer correct", out.final_answer == "4")

section("Roles — ReplayAgent")

replay = ReplayAgent(responses={"What is 2+2?": "4", "Capital of France?": "Paris"})
check("ReplayAgent created", replay is not None)
out2 = replay.generate(question="What is 2+2?", context="", skillbook=sb_for_roles)
check("ReplayAgent returns AgentOutput", isinstance(out2, AgentOutput))
check("ReplayAgent answer", out2.final_answer == "4")

section("Roles — Reflector")

llm2 = DummyLLMClient()
reflector = Reflector(llm2)
check("Reflector created", reflector is not None)

llm2.queue(
    json.dumps(
        {
            "reasoning": "The agent got the right answer",
            "error_identification": "",
            "root_cause_analysis": "",
            "correct_approach": "Basic arithmetic",
            "key_insight": "Simple addition works",
            "extracted_learnings": [
                {
                    "learning": "Always verify arithmetic",
                    "atomicity_score": 0.9,
                    "evidence": "2+2=4",
                    "justification": "fundamental",
                }
            ],
            "skill_tags": [],
        }
    )
)
ref_out = reflector.reflect(
    question="What is 2+2?",
    agent_output=out,
    feedback="Correct!",
    ground_truth="4",
    skillbook=sb_for_roles,
)
check("Reflector.reflect returns ReflectorOutput", isinstance(ref_out, ReflectorOutput))
check("Reflector key_insight", ref_out.key_insight == "Simple addition works")

section("Roles — SkillManager")

llm3 = DummyLLMClient()
skill_mgr = SkillManager(llm3)
check("SkillManager created", skill_mgr is not None)

llm3.queue(
    json.dumps(
        {
            "update": {
                "reasoning": "Add arithmetic verification skill",
                "operations": [
                    {
                        "type": "ADD",
                        "skill_id": "arith-001",
                        "section": "math",
                        "content": "Verify arithmetic",
                    }
                ],
            }
        }
    )
)
sm_result = skill_mgr.update_skills(
    reflection=ref_out,
    skillbook=sb_for_roles,
    question_context="Math problems",
    progress="1/1 correct",
)
check(
    "SkillManager.update_skills returns SkillManagerOutput",
    isinstance(sm_result, SkillManagerOutput),
)
check("SkillManager produced ops", len(sm_result.update.operations) >= 1)


# ── 6. Sample & SimpleEnvironment ─────────────────────────────

section("Sample & SimpleEnvironment")

sample = Sample(question="What is 2+2?", ground_truth="4")
check("Sample created", sample.question == "What is 2+2?")

env = SimpleEnvironment()
env_result = env.evaluate(sample, out)
check("SimpleEnvironment evaluate", isinstance(env_result, EnvironmentResult))
check("correct answer detected", env_result.metrics.get("correct") == 1.0)

wrong_out = AgentOutput(reasoning="guess", final_answer="5", skill_ids=[])
env_result2 = env.evaluate(sample, wrong_out)
check("wrong answer detected", env_result2.metrics.get("correct") == 0.0)


# ── 7. OfflineACE end-to-end ──────────────────────────────────

section("OfflineACE — full pipeline")

llm_full = DummyLLMClient()
sb_full = Skillbook()
agent_full = Agent(llm_full)
reflector_full = Reflector(llm_full)
sm_full = SkillManager(llm_full)

adapter = OfflineACE(
    skillbook=sb_full,
    agent=agent_full,
    reflector=reflector_full,
    skill_manager=sm_full,
    enable_observability=False,
)
check("OfflineACE created", adapter is not None)

samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="Capital of France?", ground_truth="Paris"),
]

# Queue responses for 2 samples × 1 epoch: agent + reflector + skill_manager each
for q, a in [("2+2", "4"), ("France", "Paris")]:
    llm_full.queue(
        json.dumps(
            {
                "reasoning": f"Thinking about {q}",
                "final_answer": a,
                "skill_ids": [],
            }
        )
    )
    llm_full.queue(
        json.dumps(
            {
                "reasoning": "Good answer",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "Direct recall",
                "key_insight": f"Know the answer to {q}",
                "extracted_learnings": [],
                "skill_tags": [],
            }
        )
    )
    llm_full.queue(
        json.dumps({"update": {"reasoning": "no changes needed", "operations": []}})
    )

results = adapter.run(samples, env, epochs=1)
check("OfflineACE.run returned results", len(results) == 2)
check("results are ACEStepResult", all(isinstance(r, ACEStepResult) for r in results))


# ── 8. OnlineACE ──────────────────────────────────────────────

section("OnlineACE — full pipeline")

llm_online = DummyLLMClient()
sb_online = Skillbook()
agent_online = Agent(llm_online)
reflector_online = Reflector(llm_online)
sm_online = SkillManager(llm_online)

online = OnlineACE(
    skillbook=sb_online,
    agent=agent_online,
    reflector=reflector_online,
    skill_manager=sm_online,
    enable_observability=False,
)
check("OnlineACE created", online is not None)

llm_online.queue(
    json.dumps(
        {
            "reasoning": "thinking",
            "final_answer": "4",
            "skill_ids": [],
        }
    )
)
llm_online.queue(
    json.dumps(
        {
            "reasoning": "correct",
            "error_identification": "",
            "root_cause_analysis": "",
            "correct_approach": "arithmetic",
            "key_insight": "addition",
            "extracted_learnings": [],
            "skill_tags": [],
        }
    )
)
llm_online.queue(json.dumps({"update": {"reasoning": "no changes", "operations": []}}))

online_results = online.run([Sample(question="2+2?", ground_truth="4")], env)
check("OnlineACE.run returned results", len(online_results) == 1)


# ── 9. PromptManager ─────────────────────────────────────────

section("PromptManager")

pm = PromptManager()
check("PromptManager default version", pm.default_version is not None)

agent_prompt = pm.get_agent_prompt()
check("agent prompt is string", isinstance(agent_prompt, str) and len(agent_prompt) > 0)

reflector_prompt = pm.get_reflector_prompt()
check(
    "reflector prompt is string",
    isinstance(reflector_prompt, str) and len(reflector_prompt) > 0,
)

sm_prompt = pm.get_skill_manager_prompt()
check(
    "skill_manager prompt is string", isinstance(sm_prompt, str) and len(sm_prompt) > 0
)

versions = pm.list_available_versions()
check("list_available_versions non-empty", len(versions) >= 1)


# ── 10. DeduplicationConfig ──────────────────────────────────

section("DeduplicationConfig")

dedup_cfg = DeduplicationConfig()
check("default config created", dedup_cfg.enabled is True)
check("default threshold", dedup_cfg.similarity_threshold == 0.85)

custom_cfg = DeduplicationConfig(similarity_threshold=0.7, enabled=False)
check(
    "custom config", custom_cfg.similarity_threshold == 0.7 and not custom_cfg.enabled
)


# ── 11. Feature Flags ────────────────────────────────────────

section("Feature Flags")

check("LITELLM_AVAILABLE is bool", isinstance(LITELLM_AVAILABLE, bool))
check("OBSERVABILITY_AVAILABLE is bool", isinstance(OBSERVABILITY_AVAILABLE, bool))


# ── 12. Async Learning types ─────────────────────────────────

section("Async Learning Types")

check("LearningTask importable", LearningTask is not None)
check("ReflectionResult importable", ReflectionResult is not None)
check("ThreadSafeSkillbook importable", ThreadSafeSkillbook is not None)
check("AsyncLearningPipeline importable", AsyncLearningPipeline is not None)

ts_sb = ThreadSafeSkillbook(Skillbook())
ts_sb.add_skill(section="test", content="thread safe", skill_id="ts-001")
check("ThreadSafeSkillbook add_skill", ts_sb.get_skill("ts-001") is not None)


# ── Summary ──────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*60}")

exit(0 if failed == 0 else 1)
