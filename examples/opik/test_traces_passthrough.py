#!/usr/bin/env python3
"""End-to-end test: OfflineACE + RecursiveReflector + Opik tracing.

Runs the full offline pipeline with real LLM calls so you can inspect
the Opik trace to verify that `traces` are injected into the RR sandbox.

Usage:
    uv run python examples/opik/test_traces_passthrough.py

Check traces at: https://www.comet.com/opik (or your local Opik instance)
"""

import os
import logging

from ace import (
    OfflineACE,
    Agent,
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
    project_name="ace-traces-passthrough-test",
    tags=["test", "recursive-reflector", "traces"],
)
log.info("Opik: %s", "enabled" if integration.is_available() else "disabled")

# ── LLM ───────────────────────────────────────────────────────────────
model = os.getenv("ACE_MODEL", "claude-sonnet-4-5-20250929")
llm = LiteLLMClient(model=model, temperature=0.3, max_tokens=2000)
log.info("Model: %s", model)

# ── Components ────────────────────────────────────────────────────────
prompt_mgr = PromptManager()
skillbook = Skillbook()

agent = Agent(llm, prompt_template=prompt_mgr.get_agent_prompt())

reflector = RecursiveReflector(
    llm,
    config=RecursiveConfig(
        max_iterations=8,
        enable_subagent=True,
        max_llm_calls=10,
    ),
)

skill_manager = SkillManager(llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

# ── Samples ───────────────────────────────────────────────────────────
# Sample 1: basic, auto-builds traces from agent output
samples = [
    Sample(
        question="What is the capital of France?",
        ground_truth="Paris",
    ),
    # Sample 2: pre-recorded traces as full dict (new format)
    # question/ground_truth inside traces dict — self-contained
    Sample(
        question="Explain why the sky is blue in one sentence.",
        ground_truth="Rayleigh scattering",
        metadata={
            "traces": {
                "question": "Explain why the sky is blue in one sentence.",
                "ground_truth": "Rayleigh scattering",
                "feedback": "Correct!",
                "steps": [
                    {
                        "role": "agent",
                        "reasoning": "Light scatters - shorter wavelengths scatter more.",
                        "answer": "Rayleigh scattering of sunlight by the atmosphere.",
                        "skill_ids": [],
                    },
                    {
                        "role": "tool",
                        "tool_name": "web_search",
                        "input": "why is the sky blue physics",
                        "output": "Rayleigh scattering causes shorter (blue) wavelengths...",
                    },
                    {
                        "role": "agent",
                        "reasoning": "Confirmed via search. Rayleigh scattering is the answer.",
                        "answer": "Rayleigh scattering of sunlight by atmospheric molecules.",
                        "skill_ids": [],
                    },
                ],
            }
        },
    ),
    # Sample 3: plain List[Dict] format (backward compat — auto-wrapped)
    Sample(
        question="What is 2+2?",
        ground_truth="4",
        metadata={
            "traces": [
                {
                    "role": "agent",
                    "reasoning": "Simple arithmetic: 2+2=4.",
                    "answer": "4",
                    "skill_ids": [],
                },
            ]
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

log.info("Running OfflineACE with %d samples, 1 epoch ...", len(samples))
results = ace.run(samples, environment, epochs=1)

# ── Report ────────────────────────────────────────────────────────────
for i, r in enumerate(results):
    status = (
        "PASS"
        if r.environment_result
        and r.environment_result.feedback
        and "correct" in r.environment_result.feedback.lower()
        else "FAIL"
    )
    log.info(
        "Sample %d [%s]: Q=%s  A=%s",
        i,
        status,
        r.sample.question[:50],
        (r.agent_output.final_answer or "")[:60] if r.agent_output else "N/A",
    )
    if r.reflection:
        log.info(
            "  Reflection: insight=%s, learnings=%d, tags=%d",
            (r.reflection.key_insight or "")[:80],
            len(r.reflection.extracted_learnings),
            len(r.reflection.skill_tags),
        )

log.info("Skillbook after training:\n%s", skillbook.as_prompt() or "(empty)")
log.info(
    "Done. Check Opik dashboard for traces → project 'ace-traces-passthrough-test'"
)
