#!/usr/bin/env python3
"""Quick test: verify Opik tracing works with ACELiteLLM + Bedrock.

Prerequisites:
    1. .env with AWS_BEARER_TOKEN_BEDROCK, OPIK_API_KEY, OPIK_WORKSPACE
    2. Opik installed: pip install ace-framework[observability]

Usage:
    uv run python examples/opik/test_litellm_opik.py

View traces at: https://www.comet.com/opik → project "ace-litellm-opik-test"
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from ace_next import ACELiteLLM, Sample, SimpleEnvironment

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger(__name__)

# ── Create agent with Opik enabled ────────────────────────────────
ace = ACELiteLLM.from_model(
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    opik=True,
    opik_project="ace-litellm-opik-test",
    opik_tags=["e2e", "litellm", "bedrock", "opik-test"],
)

# ── Training samples ──────────────────────────────────────────────
samples = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
    Sample(question="Capital of France?", context="", ground_truth="Paris"),
    Sample(question="Who wrote Hamlet?", context="", ground_truth="Shakespeare"),
]

# ── Run learning (traces go to Opik) ─────────────────────────────
log.info("Running learning with Opik tracing ...")
results = ace.learn(samples, environment=SimpleEnvironment(), epochs=1)

# ── Report ────────────────────────────────────────────────────────
log.info("=" * 60)
for i, r in enumerate(results):
    status = "FAIL" if r.error else "OK"
    log.info("Sample %d: %s", i, status)
log.info("Learned %d skills", len(ace.skillbook.skills()))
log.info("=" * 60)

log.info("Done. Check https://www.comet.com/opik → project 'ace-litellm-opik-test'")
