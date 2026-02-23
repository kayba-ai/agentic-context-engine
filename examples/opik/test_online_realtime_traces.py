#!/usr/bin/env python3
"""E2E: OnlineACE + real Agent + streaming samples + auto-built traces.

Traces are auto-built from agent output (no pre-recorded traces).
Validates the online/streaming learning path and that auto-built
traces flow to the RecursiveReflector correctly.

Usage:
    uv run python examples/opik/test_online_realtime_traces.py

Check traces at: https://www.comet.com/opik → project "ace-online-realtime-traces"
"""

import os
import logging
from typing import Iterator

from ace import (
    OnlineACE,
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
    project_name="ace-online-realtime-traces",
    tags=["e2e", "online", "auto-built-traces", "streaming"],
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


# ── Streaming sample generator ───────────────────────────────────────
def sample_stream() -> Iterator[Sample]:
    """Simulates a stream of incoming QA samples."""
    samples = [
        Sample(
            question="What is the boiling point of water in Celsius?",
            ground_truth="100",
        ),
        Sample(
            question="Who wrote the play Romeo and Juliet?",
            ground_truth="William Shakespeare",
        ),
        Sample(
            question="What is the speed of light in meters per second?",
            ground_truth="299792458",
        ),
        Sample(
            question="What element has the atomic number 1?",
            ground_truth="Hydrogen",
        ),
    ]
    for s in samples:
        log.info(">> Streaming sample: %s", s.question[:50])
        yield s


# ── Run ───────────────────────────────────────────────────────────────
environment = SimpleEnvironment()

ace = OnlineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
)

log.info("Running OnlineACE with streaming samples ...")
results = ace.run(sample_stream(), environment)

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
    log.info(
        "  Sample %d [%s]: Q=%s",
        i,
        status,
        r.sample.question[:50],
    )
    log.info(
        "    Agent answer: %s",
        (r.agent_output.final_answer or "")[:60] if r.agent_output else "N/A",
    )
    if r.reflection:
        log.info(
            "    Reflection: insight=%s",
            (r.reflection.key_insight or "")[:80],
        )
    log.info(
        "    Skillbook size: %d skills",
        len(skillbook.skills()),
    )

log.info("-" * 60)
log.info("Final skillbook (%d skills):", len(skillbook.skills()))
log.info("%s", skillbook.as_prompt() or "(empty)")
log.info("-" * 60)
log.info("Done. Check Opik dashboard → project 'ace-online-realtime-traces'")
