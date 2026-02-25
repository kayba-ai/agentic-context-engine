#!/usr/bin/env python3
"""Focused e2e smoke test for ace_next — exercises core runners with a real LLM.

Verifies that each runner actually generates insights (skills with non-empty
content), not just that the pipeline runs without errors.

Usage:
    ACE_MODEL=anthropic/claude-haiku-4-5-20251001 uv run python examples/ace_next/smoke_test.py
"""

import os
import sys
import tempfile
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

# Ensure project root is importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

from ace_next import (
    ACE,
    ACELiteLLM,
    Agent,
    LiteLLMClient,
    Reflector,
    Sample,
    SimpleEnvironment,
    Skillbook,
    SkillManager,
    TraceAnalyser,
)

MODEL = os.getenv("ACE_MODEL", "anthropic/claude-haiku-4-5-20251001")
passed = 0
total = 5


def section(name: str) -> None:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


def assert_skills_have_content(skillbook: Skillbook, label: str) -> None:
    """Verify every skill has a non-empty content field."""
    for skill in skillbook.skills():
        assert (
            skill.content and skill.content.strip()
        ), f"{label}: skill {skill.id} has empty content"


# ── Shared setup ────────────────────────────────────────────
client = LiteLLMClient(model=MODEL)
agent = Agent(client)
reflector = Reflector(client)
skill_manager = SkillManager(client)

# ── 1. ACE runner (full pipeline) ───────────────────────────
section("1. ACE runner — 3 samples, 1 epoch")
skillbook = Skillbook()
ace = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=SimpleEnvironment(),
    skillbook=skillbook,
)
results = ace.run(
    [
        Sample(question="What is the capital of France?", ground_truth="Paris"),
        Sample(question="What is the capital of Japan?", ground_truth="Tokyo"),
        Sample(question="What is the capital of Brazil?", ground_truth="Brasilia"),
    ],
    epochs=1,
)
assert len(results) == 3, f"Expected 3 results, got {len(results)}"
errors = [r for r in results if r.error]
assert not errors, f"Pipeline errors: {errors}"

# Verify agent produced answers
for r in results:
    assert r.output is not None, f"No output for {r.sample.question}"
    ao = getattr(r.output, "agent_output", None)
    assert ao is not None, f"No agent_output for {r.sample.question}"
    assert ao.final_answer.strip(), f"Empty answer for {r.sample.question}"

# Verify insights were generated
ace_skill_count = len(skillbook.skills())
assert ace_skill_count > 0, "ACE runner produced zero skills"
assert_skills_have_content(skillbook, "ACE runner")
print(f"  OK — {len(results)} results, {ace_skill_count} skills learned")
for s in skillbook.skills()[:3]:
    print(f"    [{s.id}] {s.content[:70]}")
passed += 1

# ── 2. TraceAnalyser ────────────────────────────────────────
section("2. TraceAnalyser — 2 pre-recorded traces")
skills_before = len(skillbook.skills())
analyser = TraceAnalyser.from_roles(
    reflector=reflector,
    skill_manager=skill_manager,
    skillbook=skillbook,  # continues from ACE run
)
traces = [
    {
        "question": "Translate 'hello' to Spanish",
        "answer": "hola",
        "feedback": "Correct! Simple and accurate.",
    },
    {
        "question": "What is 12 * 15?",
        "answer": "170",
        "feedback": "Incorrect. The correct answer is 180.",
    },
]
trace_results = analyser.run(traces, epochs=1)
assert len(trace_results) == 2, f"Expected 2 results, got {len(trace_results)}"
trace_errors = [r for r in trace_results if r.error]
assert not trace_errors, f"TraceAnalyser errors: {trace_errors}"

# Verify new insights were added
skills_after = len(skillbook.skills())
new_skills = skills_after - skills_before
assert (
    new_skills > 0
), f"TraceAnalyser added zero new skills (before={skills_before}, after={skills_after})"
assert_skills_have_content(skillbook, "TraceAnalyser")
print(
    f"  OK — {len(trace_results)} traces, {new_skills} new skills, {skills_after} total"
)
passed += 1

# ── 3. ACELiteLLM ask + learn_from_feedback ─────────────────
section("3. ACELiteLLM — ask + learn_from_feedback")
llm_skillbook = Skillbook()
ace_llm = ACELiteLLM(client, skillbook=llm_skillbook)
answer = ace_llm.ask("What colour is the sky on a clear day?")
assert isinstance(answer, str) and len(answer.strip()) > 0, f"Bad answer: {answer!r}"
print(f"  ask() → {answer[:80]}")

learned = ace_llm.learn_from_feedback(
    feedback="Good answer but could mention why (Rayleigh scattering).",
    ground_truth="Blue",
)
assert learned, "learn_from_feedback returned False"

# Verify insights were generated
llm_skill_count = len(ace_llm.skillbook.skills())
assert llm_skill_count > 0, "learn_from_feedback produced zero skills"
assert_skills_have_content(ace_llm.skillbook, "ACELiteLLM")

# Verify as_prompt() returns something useful
prompt = ace_llm.skillbook.as_prompt()
assert (
    prompt and len(prompt.strip()) > 0
), "Skillbook as_prompt() is empty after learning"
print(f"  learn_from_feedback() → OK, {llm_skill_count} skills")
print(f"  as_prompt() → {len(prompt)} chars")
passed += 1

# ── 4. Skillbook persistence ────────────────────────────────
section("4. Skillbook persistence — save, reload, verify")
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    tmp_path = f.name
try:
    skillbook.save_to_file(tmp_path)
    reloaded = Skillbook.load_from_file(tmp_path)

    # Verify counts match
    orig_stats = skillbook.stats()
    new_stats = reloaded.stats()
    assert (
        orig_stats["skills"] == new_stats["skills"]
    ), f"Skill count mismatch: {orig_stats} vs {new_stats}"

    # Verify content survives round-trip
    orig_prompt = skillbook.as_prompt()
    reloaded_prompt = reloaded.as_prompt()
    assert (
        orig_prompt == reloaded_prompt
    ), f"as_prompt() differs after reload:\n  original: {orig_prompt[:100]}...\n  reloaded: {reloaded_prompt[:100]}..."

    # Verify individual skill content preserved
    orig_ids = {s.id for s in skillbook.skills()}
    reloaded_ids = {s.id for s in reloaded.skills()}
    assert orig_ids == reloaded_ids, f"Skill IDs differ: {orig_ids} vs {reloaded_ids}"

    print(
        f"  OK — saved/loaded {orig_stats['skills']} skills, content round-trip verified"
    )
    passed += 1
finally:
    Path(tmp_path).unlink(missing_ok=True)

# ── 5. max_retries wiring ───────────────────────────────────
section("5. max_retries wiring")
a = Agent(client, max_retries=5)
r = Reflector(client, max_retries=7)
s = SkillManager(client, max_retries=9)
assert a.max_retries == 5, f"Agent max_retries={a.max_retries}"
assert r.max_retries == 7, f"Reflector max_retries={r.max_retries}"
assert s.max_retries == 9, f"SkillManager max_retries={s.max_retries}"
# Verify defaults
a_default = Agent(client)
r_default = Reflector(client)
s_default = SkillManager(client)
assert a_default.max_retries == 3, f"Agent default max_retries={a_default.max_retries}"
assert (
    r_default.max_retries == 3
), f"Reflector default max_retries={r_default.max_retries}"
assert (
    s_default.max_retries == 3
), f"SkillManager default max_retries={s_default.max_retries}"
print("  OK — custom: Agent=5, Reflector=7, SkillManager=9; defaults=3")
passed += 1

# ── Summary ─────────────────────────────────────────────────
section(f"RESULT: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
