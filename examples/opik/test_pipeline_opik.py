#!/usr/bin/env python3
"""Test: Opik tracing via OpikStep in a manual ACE pipeline.

Uses ACE.from_roles() with extra_steps=[OpikStep(...)] instead of
ACELiteLLM's opik=True flag. This gives full control over the pipeline
while still getting per-sample Opik traces + optional LLM cost tracking.

Prerequisites:
    1. .env with AWS_BEARER_TOKEN_BEDROCK, OPIK_API_KEY, OPIK_WORKSPACE
    2. Opik installed: pip install ace-framework[observability]

Usage:
    uv run python examples/opik/test_pipeline_opik.py

View traces at: https://www.comet.com/opik → project "ace-pipeline-opik-test"
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from ace_next import (
    ACE,
    Agent,
    LiteLLMClient,
    OpikStep,
    Reflector,
    Sample,
    SimpleEnvironment,
    SkillManager,
    register_opik_litellm_callback,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger(__name__)

# ── LLM client ────────────────────────────────────────────────────
llm = LiteLLMClient(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")

# ── Roles ─────────────────────────────────────────────────────────
agent = Agent(llm)
reflector = Reflector(llm)
skill_manager = SkillManager(llm)

# ── Pipeline with OpikStep ────────────────────────────────────────
PROJECT = "ace-pipeline-opik-test"

opik_step = OpikStep(project_name=PROJECT, tags=["pipeline", "manual"])

runner = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=SimpleEnvironment(),
    extra_steps=[opik_step],
)

# Optional: also track per-LLM-call tokens and costs
register_opik_litellm_callback(project_name=PROJECT)

# ── Training samples ──────────────────────────────────────────────
samples = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
    Sample(question="Capital of France?", context="", ground_truth="Paris"),
    Sample(question="Who wrote Hamlet?", context="", ground_truth="Shakespeare"),
]

# ── Run ───────────────────────────────────────────────────────────
log.info("Running ACE pipeline with OpikStep ...")
results = runner.run(samples, epochs=1)

# ── Report ────────────────────────────────────────────────────────
log.info("=" * 60)
for i, r in enumerate(results):
    status = "FAIL" if r.error else "OK"
    log.info("Sample %d: %s", i, status)
log.info("Learned %d skills", len(runner.skillbook.skills()))
log.info("=" * 60)

log.info("Done. Check https://www.comet.com/opik → project '%s'", PROJECT)
