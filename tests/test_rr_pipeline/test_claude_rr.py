"""Unit tests for ClaudeRRStep — the Claude Agent SDK-based Recursive Reflector."""

import json
import os
from unittest.mock import patch

import pytest

from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.outputs import ReflectorOutput
from ace_next.core.skillbook import Skillbook
from ace_next.rr.claude_rr import ClaudeRRConfig, ClaudeRRStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JSON_RESPONSE = json.dumps(
    {
        "reasoning": "The agent answered correctly using step-by-step calculation",
        "error_identification": "none",
        "root_cause_analysis": "No errors",
        "correct_approach": "Step-by-step arithmetic",
        "key_insight": "Simple arithmetic is handled well",
        "extracted_learnings": [
            {
                "learning": "Step-by-step works for arithmetic",
                "atomicity_score": 0.8,
                "evidence": "Agent correctly computed 2+2=4",
            }
        ],
        "skill_tags": [{"id": "arithmetic-1", "tag": "helpful"}],
    }
)

SAMPLE_TRACES = {
    "question": "What is 2+2?",
    "ground_truth": "4",
    "feedback": "Correct!",
    "steps": [
        {
            "role": "agent",
            "reasoning": "2+2=4",
            "answer": "4",
            "skill_ids": ["arithmetic-1"],
        }
    ],
}

COST_INFO = {
    "total_cost_usd": 0.05,
    "num_turns": 3,
    "duration_ms": 1000,
    "is_error": False,
    "session_id": "test-session",
}


def _mock_query_sdk(final_text: str, cost_info: dict | None = None):
    """Return an AsyncMock for ClaudeRRStep._query_sdk."""
    info = cost_info or COST_INFO

    async def mock(tmp_dir):
        return final_text, info

    return mock


# ---------------------------------------------------------------------------
# Tests: SDK response parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRStepParsing:
    """Test that ClaudeRRStep correctly parses various response formats."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        """Valid JSON response is parsed into ReflectorOutput."""
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(VALID_JSON_RESPONSE)
        result = await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "Simple arithmetic is handled well"
        assert len(result.extracted_learnings) == 1
        assert result.extracted_learnings[0].atomicity_score == 0.8
        assert len(result.skill_tags) == 1
        assert result.skill_tags[0].id == "arithmetic-1"
        assert result.raw.get("claude_rr") is True

    @pytest.mark.asyncio
    async def test_json_in_code_fence(self):
        """JSON wrapped in ```json fences is extracted and parsed."""
        fenced = f"Here is my analysis:\n\n```json\n{VALID_JSON_RESPONSE}\n```\n\nDone."
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(fenced)
        result = await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "Simple arithmetic is handled well"

    @pytest.mark.asyncio
    async def test_empty_response_produces_error_output(self):
        """Empty SDK response produces a fallback error ReflectorOutput."""
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk("")
        result = await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert isinstance(result, ReflectorOutput)
        assert (
            "failed" in result.reasoning.lower() or "empty" in result.reasoning.lower()
        )
        assert result.raw.get("claude_rr") is True

    @pytest.mark.asyncio
    async def test_malformed_json_produces_error_output(self):
        """Malformed JSON produces a fallback error ReflectorOutput."""
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk("This is not JSON at all, just text.")
        result = await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert isinstance(result, ReflectorOutput)
        assert result.raw.get("error") is not None

    @pytest.mark.asyncio
    async def test_cost_info_propagated(self):
        """Cost info from SDK is included in the raw output."""
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(VALID_JSON_RESPONSE)
        result = await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert result.raw["cost_info"]["total_cost_usd"] == 0.05
        assert result.raw["cost_info"]["num_turns"] == 3


# ---------------------------------------------------------------------------
# Tests: Sandbox file setup and cleanup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRStepSandbox:
    """Test temp directory creation, file contents, and cleanup."""

    def test_write_sandbox_files(self):
        """Verify trace.json and skillbook.md are written correctly."""
        step = ClaudeRRStep(ClaudeRRConfig())
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="claude-rr-test-")
        try:
            step._write_sandbox_files(tmp_dir, SAMPLE_TRACES, "my skillbook text")

            # trace.json
            trace_path = os.path.join(tmp_dir, "trace.json")
            assert os.path.exists(trace_path)
            with open(trace_path) as f:
                data = json.load(f)
            assert data["question"] == "What is 2+2?"
            assert data["ground_truth"] == "4"
            assert len(data["steps"]) == 1

            # skillbook.md
            sb_path = os.path.join(tmp_dir, "skillbook.md")
            assert os.path.exists(sb_path)
            with open(sb_path) as f:
                assert f.read() == "my skillbook text"

            # workspace/
            assert os.path.isdir(os.path.join(tmp_dir, "workspace"))
        finally:
            import shutil

            shutil.rmtree(tmp_dir)

    @pytest.mark.asyncio
    async def test_cleanup_after_run(self):
        """Temp directory is cleaned up after _run completes."""
        created_dirs: list[str] = []
        original_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(VALID_JSON_RESPONSE)

        with patch("ace_next.rr.claude_rr.tempfile.mkdtemp", tracking_mkdtemp):
            await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert len(created_dirs) == 1
        assert not os.path.exists(created_dirs[0])


# ---------------------------------------------------------------------------
# Tests: StepProtocol (__call__)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRStepProtocol:
    """Test StepProtocol compliance."""

    def test_step_protocol_attributes(self):
        step = ClaudeRRStep(ClaudeRRConfig())
        assert "trace" in step.requires
        assert "skillbook" in step.requires
        assert "reflection" in step.provides

    @pytest.mark.asyncio
    async def test_call_produces_reflection(self):
        """__call__ populates ctx.reflection."""
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(VALID_JSON_RESPONSE)

        sb = Skillbook()
        ctx = ACEStepContext(
            trace=SAMPLE_TRACES,
            skillbook=SkillbookView(sb),
        )
        result_ctx = await step(ctx)

        assert result_ctx.reflection is not None
        assert isinstance(result_ctx.reflection, ReflectorOutput)
        assert result_ctx.reflection.key_insight == "Simple arithmetic is handled well"


# ---------------------------------------------------------------------------
# Tests: ReflectorLike (reflect)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRStepReflect:
    """Test the synchronous reflect() entry point."""

    def test_reflect_with_agent_output(self):
        """reflect() with explicit parameters works."""
        from ace_next.core.outputs import AgentOutput

        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(VALID_JSON_RESPONSE)

        result = step.reflect(
            question="What is 2+2?",
            agent_output=AgentOutput(
                reasoning="2+2=4", final_answer="4", skill_ids=["arithmetic-1"]
            ),
            skillbook=Skillbook(),
            ground_truth="4",
            feedback="Correct!",
        )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "Simple arithmetic is handled well"

    def test_reflect_with_trace_kwarg(self):
        """reflect() with trace dict via kwargs works."""
        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = _mock_query_sdk(VALID_JSON_RESPONSE)

        result = step.reflect(
            question="",
            agent_output=None,
            skillbook=Skillbook(),
            trace=SAMPLE_TRACES,
        )

        assert isinstance(result, ReflectorOutput)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRStepErrors:
    """Test error handling and fallback outputs."""

    @pytest.mark.asyncio
    async def test_sdk_exception_produces_fallback(self):
        """SDK exception produces a fallback ReflectorOutput."""

        async def exploding_query_sdk(tmp_dir):
            raise RuntimeError("SDK connection failed")

        step = ClaudeRRStep(ClaudeRRConfig())
        step._query_sdk = exploding_query_sdk
        result = await step._run(SAMPLE_TRACES, "(empty skillbook)")

        assert isinstance(result, ReflectorOutput)
        assert "SDK connection failed" in result.raw.get("error", "")

    def test_build_error_output(self):
        """_build_error_output produces valid ReflectorOutput."""
        result = ClaudeRRStep._build_error_output("test error")
        assert isinstance(result, ReflectorOutput)
        assert "test error" in result.reasoning
        assert result.raw["claude_rr"] is True
        assert result.raw["error"] == "test error"


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRConfig:
    """Test config defaults and custom values."""

    def test_defaults(self):
        cfg = ClaudeRRConfig()
        assert cfg.model is None
        assert cfg.max_turns == 10
        assert cfg.max_budget_usd == 0.50
        assert cfg.permission_mode == "bypassPermissions"
        assert cfg.allowed_tools is None
        assert cfg.extra_instructions == ""

    def test_custom_values(self):
        cfg = ClaudeRRConfig(
            model="claude-sonnet-4-5-20250929",
            max_turns=5,
            max_budget_usd=1.0,
            extra_instructions="Be brief.",
        )
        assert cfg.model == "claude-sonnet-4-5-20250929"
        assert cfg.max_turns == 5
        assert cfg.max_budget_usd == 1.0
        assert cfg.extra_instructions == "Be brief."


# ---------------------------------------------------------------------------
# Tests: Trace building
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClaudeRRTraceBuilding:
    """Test trace dict construction from various inputs."""

    def test_build_traces_dict_from_agent_output(self):
        from ace_next.core.outputs import AgentOutput

        ao = AgentOutput(reasoning="r", final_answer="a", skill_ids=["s1"])
        result = ClaudeRRStep._build_traces_dict("q", ao, "gt", "fb")
        assert result["question"] == "q"
        assert result["ground_truth"] == "gt"
        assert result["feedback"] == "fb"
        assert result["steps"][0]["reasoning"] == "r"
        assert result["steps"][0]["answer"] == "a"

    def test_build_traces_dict_none_agent_output(self):
        result = ClaudeRRStep._build_traces_dict("q", None, None, None)
        assert result["question"] == "q"
        assert result["steps"][0]["reasoning"] == ""

    def test_extract_skillbook_text_from_view(self):
        sb = Skillbook()
        view = SkillbookView(sb)
        text = ClaudeRRStep._extract_skillbook_text(view)
        assert isinstance(text, str)

    def test_extract_skillbook_text_none(self):
        assert ClaudeRRStep._extract_skillbook_text(None) == "(empty skillbook)"

    def test_extract_skillbook_text_string(self):
        assert ClaudeRRStep._extract_skillbook_text("my skills") == "my skills"

    def test_build_traces_dict_from_ctx_with_dict(self):
        """When ctx.trace is already a dict, use it directly."""
        step = ClaudeRRStep(ClaudeRRConfig())
        ctx = ACEStepContext(trace=SAMPLE_TRACES)
        result = step._build_traces_dict_from_ctx(ctx)
        assert result is SAMPLE_TRACES

    def test_build_traces_dict_from_ctx_with_object(self):
        """When ctx.trace is an object, extract attrs."""

        class FakeTrace:
            question = "q"
            ground_truth = "gt"
            feedback = "fb"
            steps = []

        step = ClaudeRRStep(ClaudeRRConfig())
        ctx = ACEStepContext(trace=FakeTrace())
        result = step._build_traces_dict_from_ctx(ctx)
        assert result["question"] == "q"
        assert result["ground_truth"] == "gt"
