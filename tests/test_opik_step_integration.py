"""Integration tests for OpikStep with real LLM endpoints and Opik server.

Runs the full ACE pipeline against real Bedrock/Anthropic LLMs and verifies
that traces with correct metadata land in a real Opik project.

Prerequisites:
    - AWS Bedrock credentials configured
    - Either OPIK_API_KEY env var (cloud) or a local Opik server on localhost:5173

Usage:
    uv run pytest tests/test_opik_step_integration.py -v -m integration
"""

import os
import time
import uuid

import pytest
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
)
from ace_next.rr import ClaudeRRConfig, ClaudeRRStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

SAMPLES = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
]


def _unique_project() -> str:
    return f"ace-integration-test-{uuid.uuid4().hex[:8]}"


def _query_traces(project_name: str, max_results: int = 10):
    """Query Opik for traces in the given project."""
    import opik

    client = opik.Opik(
        project_name=project_name,
        api_key=os.environ.get("OPIK_API_KEY") or None,
        workspace=os.environ.get("OPIK_WORKSPACE") or None,
        host=os.environ.get("OPIK_URL_OVERRIDE") or None,
    )
    return client.search_traces(project_name=project_name, max_results=max_results)


def _skip_if_no_opik_server():
    """Skip if neither Opik cloud credentials nor a reachable local server."""
    if os.environ.get("OPIK_API_KEY"):
        return
    import urllib.request

    host = os.environ.get("OPIK_URL_OVERRIDE", "http://localhost:5173/api")
    try:
        urllib.request.urlopen(f"{host}/is-alive/ping", timeout=3)
    except Exception:
        pytest.skip(
            "No Opik server reachable (no OPIK_API_KEY, local server not responding)"
        )
    # Ensure OpikStep and _query_traces pick up the local host/workspace
    if not os.environ.get("OPIK_URL_OVERRIDE"):
        os.environ["OPIK_URL_OVERRIDE"] = host
    if not os.environ.get("OPIK_WORKSPACE"):
        os.environ["OPIK_WORKSPACE"] = "default"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_opik(monkeypatch):
    """Override conftest's autouse _suppress_opik by clearing the disable flag.

    Also unset CLAUDECODE so ClaudeRRStep can launch a nested Claude Code
    session (it refuses to start inside an existing session otherwise).
    """
    monkeypatch.setenv("OPIK_DISABLED", "")
    monkeypatch.delenv("CLAUDECODE", raising=False)


@pytest.mark.integration
def test_standard_reflector_traces_to_opik():
    """Full ACE pipeline with Reflector(llm) — verify standard metadata in Opik."""
    _skip_if_no_opik_server()

    project = _unique_project()
    llm = LiteLLMClient(model=MODEL)

    runner = ACE.from_roles(
        agent=Agent(llm),
        reflector=Reflector(llm),
        skill_manager=SkillManager(llm),
        environment=SimpleEnvironment(),
        extra_steps=[OpikStep(project_name=project, tags=["integration", "standard"])],
    )

    runner.run(SAMPLES, epochs=1)

    # Allow Opik ingestion
    time.sleep(5)

    traces = _query_traces(project)
    assert len(traces) >= 1, f"Expected >=1 trace, got {len(traces)}"

    trace = traces[0]
    meta = trace.metadata or {}

    # Standard metadata fields
    assert "epoch" in meta, f"Missing 'epoch' in metadata: {meta}"
    assert "skill_count" in meta, f"Missing 'skill_count' in metadata: {meta}"
    assert "key_insight" in meta, f"Missing 'key_insight' in metadata: {meta}"
    assert "learnings_count" in meta, f"Missing 'learnings_count' in metadata: {meta}"

    # Should NOT have ClaudeRR-specific metadata
    assert meta.get("reflector_type") != "claude_rr"


@pytest.mark.integration
def test_claude_rr_reflector_traces_to_opik():
    """Full ACE pipeline with ClaudeRRStep as reflector — verify RR metadata in Opik."""
    _skip_if_no_opik_server()

    project = _unique_project()
    llm = LiteLLMClient(model=MODEL)

    rr = ClaudeRRStep(ClaudeRRConfig(max_turns=5, max_budget_usd=0.25))

    runner = ACE.from_roles(
        agent=Agent(llm),
        reflector=rr,
        skill_manager=SkillManager(llm),
        environment=SimpleEnvironment(),
        extra_steps=[OpikStep(project_name=project, tags=["integration", "claude-rr"])],
    )

    runner.run(SAMPLES, epochs=1)

    # Allow Opik ingestion
    time.sleep(5)

    traces = _query_traces(project)
    assert len(traces) >= 1, f"Expected >=1 trace, got {len(traces)}"

    trace = traces[0]
    meta = trace.metadata or {}

    # Standard metadata
    assert "epoch" in meta, f"Missing 'epoch' in metadata: {meta}"
    assert "skill_count" in meta, f"Missing 'skill_count' in metadata: {meta}"
    assert "key_insight" in meta, f"Missing 'key_insight' in metadata: {meta}"
    assert "learnings_count" in meta, f"Missing 'learnings_count' in metadata: {meta}"

    # ClaudeRR-specific metadata
    assert (
        meta.get("reflector_type") == "claude_rr"
    ), f"Expected reflector_type='claude_rr', got: {meta}"
    assert "rr_cost_usd" in meta, f"Missing 'rr_cost_usd' in metadata: {meta}"
    assert "rr_turns" in meta, f"Missing 'rr_turns' in metadata: {meta}"
    assert "rr_duration_ms" in meta, f"Missing 'rr_duration_ms' in metadata: {meta}"
