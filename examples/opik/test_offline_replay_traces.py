#!/usr/bin/env python3
"""E2E: OfflineACE + ReplayAgent + pre-recorded multi-step traces.

No real agent LLM calls — ReplayAgent returns canned answers while the
RecursiveReflector analyzes the rich traces in its sandbox.  Validates
that pre-recorded traces flow correctly through the full pipeline.

Usage:
    uv run python examples/opik/test_offline_replay_traces.py

Check traces at: https://www.comet.com/opik → project "ace-offline-replay-traces"
"""

import os
import logging

from ace import (
    OfflineACE,
    ReplayAgent,
    SkillManager,
    Skillbook,
    Sample,
    SimpleEnvironment,
)
from ace.llm_providers.litellm_client import LiteLLMClient
from ace.observability import configure_opik
from ace.prompts_v2_1 import PromptManager
from ace.reflector import RecursiveReflector, RecursiveConfig

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger(__name__)

# ── Opik ──────────────────────────────────────────────────────────────
integration = configure_opik(
    project_name="ace-offline-replay-traces",
    tags=["e2e", "replay-agent", "pre-recorded-traces"],
)
log.info("Opik: %s", "enabled" if integration.is_available() else "disabled")

# ── LLM (only used by reflector + skill_manager, NOT the agent) ──────
model = os.getenv("ACE_MODEL", "claude-sonnet-4-5-20250929")
llm = LiteLLMClient(model=model, temperature=0.3, max_tokens=2000)
log.info("Model: %s", model)

# ── Components ────────────────────────────────────────────────────────
prompt_mgr = PromptManager()
skillbook = Skillbook()

# ReplayAgent reads responses from sample.metadata["response"]
agent = ReplayAgent()

reflector = RecursiveReflector(
    llm,
    config=RecursiveConfig(
        max_iterations=8,
        enable_subagent=True,
        max_llm_calls=10,
    ),
)

skill_manager = SkillManager(llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

# ── Samples with pre-recorded multi-step traces ──────────────────────
samples = [
    # Sample 1 — correct answer, multi-step reasoning with tool use
    Sample(
        question="What causes tides on Earth?",
        ground_truth="The Moon's gravitational pull",
        metadata={
            "response": "The Moon's gravitational pull causes tides on Earth.",
            "traces": {
                "question": "What causes tides on Earth?",
                "ground_truth": "The Moon's gravitational pull",
                "feedback": "Correct",
                "steps": [
                    {
                        "role": "agent",
                        "reasoning": "Tides are caused by gravitational forces. Let me verify.",
                        "answer": "Gravitational forces from the Moon and Sun",
                        "skill_ids": [],
                    },
                    {
                        "role": "tool",
                        "tool_name": "web_search",
                        "input": "what causes ocean tides earth",
                        "output": "Tides are caused primarily by the gravitational pull of the Moon on Earth's oceans. The Sun also contributes but to a lesser degree.",
                    },
                    {
                        "role": "agent",
                        "reasoning": "Confirmed — primarily the Moon's gravity, with some Sun contribution.",
                        "answer": "The Moon's gravitational pull causes tides on Earth.",
                        "skill_ids": [],
                    },
                ],
            },
        },
    ),
    # Sample 2 — incorrect answer, agent got confused by a common misconception
    Sample(
        question="What is the largest desert in the world?",
        ground_truth="Antarctic Desert",
        metadata={
            "response": "The Sahara Desert is the largest desert in the world.",
            "traces": {
                "question": "What is the largest desert in the world?",
                "ground_truth": "Antarctic Desert",
                "feedback": "Incorrect. The Antarctic Desert is the largest desert by area.",
                "steps": [
                    {
                        "role": "agent",
                        "reasoning": "The Sahara is commonly known as the largest desert.",
                        "answer": "The Sahara Desert",
                        "skill_ids": [],
                    },
                    {
                        "role": "tool",
                        "tool_name": "knowledge_lookup",
                        "input": "largest desert world by area",
                        "output": "Antarctica is the largest desert (14.2M km²), followed by the Arctic (13.9M km²), then the Sahara (9.2M km²).",
                    },
                    {
                        "role": "agent",
                        "reasoning": "The tool says Antarctica is bigger, but I'll go with Sahara since people usually mean hot deserts.",
                        "answer": "The Sahara Desert is the largest desert in the world.",
                        "skill_ids": [],
                    },
                ],
            },
        },
    ),
    # Sample 3 — correct answer, single-step reasoning (no tool use)
    Sample(
        question="What programming language is CPython written in?",
        ground_truth="C",
        metadata={
            "response": "C",
            "traces": {
                "question": "What programming language is CPython written in?",
                "ground_truth": "C",
                "feedback": "Correct",
                "steps": [
                    {
                        "role": "agent",
                        "reasoning": "CPython is the reference implementation of Python, written in C. The 'C' prefix literally indicates this.",
                        "answer": "C",
                        "skill_ids": [],
                    },
                ],
            },
        },
    ),
    # Sample 4 — incorrect answer, reasoning chain went wrong
    Sample(
        question="How many bones does an adult human body have?",
        ground_truth="206",
        metadata={
            "response": "The adult human body has 213 bones.",
            "traces": {
                "question": "How many bones does an adult human body have?",
                "ground_truth": "206",
                "feedback": "Incorrect. An adult human has 206 bones.",
                "steps": [
                    {
                        "role": "agent",
                        "reasoning": "I recall that babies have about 270 bones which fuse as they grow. Adults have fewer.",
                        "answer": "Around 210 bones",
                        "skill_ids": [],
                    },
                    {
                        "role": "tool",
                        "tool_name": "calculator",
                        "input": "270 - 57",
                        "output": "213",
                    },
                    {
                        "role": "agent",
                        "reasoning": "Subtracted the commonly cited fusion count. 270 - 57 = 213.",
                        "answer": "The adult human body has 213 bones.",
                        "skill_ids": [],
                    },
                ],
            },
        },
    ),
    # Sample 5 — correct answer after self-correction
    Sample(
        question="What is the chemical symbol for gold?",
        ground_truth="Au",
        metadata={
            "response": "Au",
            "traces": {
                "question": "What is the chemical symbol for gold?",
                "ground_truth": "Au",
                "feedback": "Correct",
                "steps": [
                    {
                        "role": "agent",
                        "reasoning": "Gold... the symbol might be Go or Gd? Let me think — it's from Latin.",
                        "answer": "Go",
                        "skill_ids": [],
                    },
                    {
                        "role": "tool",
                        "tool_name": "periodic_table",
                        "input": "gold symbol",
                        "output": "Gold (Au) — from Latin 'aurum'. Atomic number 79.",
                    },
                    {
                        "role": "agent",
                        "reasoning": "Right, Au from Latin 'aurum'. My initial guess was wrong.",
                        "answer": "Au",
                        "skill_ids": [],
                    },
                ],
            },
        },
    ),
]

# ── Run ───────────────────────────────────────────────────────────────
environment = SimpleEnvironment()

ace = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
)

EPOCHS = 2
log.info(
    "Running OfflineACE with %d samples × %d epochs (ReplayAgent) ...",
    len(samples),
    EPOCHS,
)
results = ace.run(samples, environment, epochs=EPOCHS)

# ── Report ────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESULTS (%d total)", len(results))
log.info("=" * 60)

for i, r in enumerate(results):
    is_correct = (
        r.environment_result
        and r.environment_result.feedback
        and "correct" in r.environment_result.feedback.lower()
    )
    status = "PASS" if is_correct else "FAIL"
    epoch = r.epoch if hasattr(r, "epoch") else "?"
    log.info(
        "  [Epoch %s] Sample %d [%s]: Q=%s  A=%s",
        epoch,
        i,
        status,
        r.sample.question[:45],
        (r.agent_output.final_answer or "")[:50] if r.agent_output else "N/A",
    )
    if r.reflection:
        log.info(
            "    Reflection: insight=%s",
            (r.reflection.key_insight or "")[:80],
        )

log.info("-" * 60)
log.info("Skills learned: %d", len(skillbook.skills()))
log.info("Skillbook:\n%s", skillbook.as_prompt() or "(empty)")
log.info("-" * 60)
log.info("Done. Check Opik dashboard → project 'ace-offline-replay-traces'")
