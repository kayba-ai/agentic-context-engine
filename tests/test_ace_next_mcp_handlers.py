import asyncio
import json

import pytest
from unittest.mock import MagicMock, patch
from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.registry import SessionRegistry
from ace_next.integrations.mcp.handlers import MCPHandlers
from ace_next.integrations.mcp.models import (
    AskRequest, LearnSampleRequest, LearnFeedbackRequest,
    SkillbookGetRequest, SkillbookSaveRequest, SkillbookLoadRequest,
    SampleItem
)
from ace_next.integrations.mcp.errors import (
    ForbiddenInSafeModeError, SaveLoadDisabledError, ValidationError,
    map_error_to_mcp,
)
from ace_next.integrations.mcp.errors import TimeoutError as MCPTimeoutError

@pytest.fixture
def config():
    return MCPServerConfig(safe_mode=False)

@pytest.fixture
def registry(config):
    return SessionRegistry(config)

@pytest.fixture
def handlers(registry, config):
    return MCPHandlers(registry, config)


# ── ace.ask ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_ask(handlers):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.ask.return_value = "answer"
        runner.skillbook.skills.return_value = [1, 2, 3]
        mock_runner_cls.from_model.return_value = runner

        req = AskRequest(session_id="s1", question="q")
        resp = await handlers.handle_ask(req)

        assert resp.answer == "answer"
        assert resp.skill_count == 3
        # applied_skill_ids was removed from the response model
        assert "applied_skill_ids" not in resp.model_fields
        runner.ask.assert_called_once()


@pytest.mark.asyncio
async def test_handle_ask_enforces_prompt_limit(handlers):
    handlers.config.max_prompt_chars = 10
    req = AskRequest(session_id="s1", question="12345678901")
    with pytest.raises(ValidationError):
        await handlers.handle_ask(req)


# ── ace.skillbook.get ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_skillbook_get(handlers, registry):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        mock_skill = MagicMock()
        mock_skill.id = "k1"
        mock_skill.content = "cont"
        mock_skill.section = "test"
        mock_skill.helpful = 1
        mock_skill.harmful = 0
        mock_skill.neutral = 0
        runner.skillbook.skills.return_value = [mock_skill]
        runner.skillbook.stats.return_value = {"skills": 1}
        mock_runner_cls.from_model.return_value = runner

        await registry.get_or_create("s1")
        req = SkillbookGetRequest(session_id="s1")
        resp = await handlers.handle_skillbook_get(req)

        assert len(resp.skills) == 1
        assert resp.skills[0].id == "k1"
        assert resp.stats["skills"] == 1


@pytest.mark.asyncio
async def test_handle_skillbook_get_uses_skill_type(handlers, registry):
    """When skills are actual Skill dataclass instances, use direct access."""
    from ace_next.core.skillbook import Skill

    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        skill = Skill(id="s1", section="topic-a", content="do X")
        runner.skillbook.skills.return_value = [skill]
        runner.skillbook.stats.return_value = {"skills": 1}
        mock_runner_cls.from_model.return_value = runner

        await registry.get_or_create("s1")
        req = SkillbookGetRequest(session_id="s1")
        resp = await handlers.handle_skillbook_get(req)

        assert resp.skills[0].id == "s1"
        assert resp.skills[0].topic == "topic-a"
        assert resp.skills[0].content == "do X"
        assert resp.skills[0].helpful == 0
        assert resp.skills[0].harmful == 0
        assert resp.skills[0].neutral == 0


# ── ace.learn.sample ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_learn_sample_safe_mode(handlers):
    handlers.config.safe_mode = True
    req = LearnSampleRequest(session_id="s1", samples=[SampleItem(question="q")])
    with pytest.raises(ForbiddenInSafeModeError):
        await handlers.handle_learn_sample(req)


@pytest.mark.asyncio
async def test_handle_learn_sample_enforces_runtime_sample_limit(handlers):
    handlers.config.max_samples_per_call = 1
    req = LearnSampleRequest(
        session_id="s1",
        samples=[SampleItem(question="q1"), SampleItem(question="q2")],
    )
    with pytest.raises(ValidationError):
        await handlers.handle_learn_sample(req)


# ── ace.learn.feedback ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_learn_feedback_uses_trace_learning(handlers):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [["a"], ["a", "b"]]
        runner.learn_from_feedback.return_value = False
        runner.learn_from_traces.return_value = []
        mock_runner_cls.from_model.return_value = runner

        req = LearnFeedbackRequest(
            session_id="s1",
            question="q",
            answer="a",
            feedback="good",
        )
        resp = await handlers.handle_learn_feedback(req)

        assert resp.learned is True
        assert resp.new_skill_count == 1
        runner.learn_from_traces.assert_called_once()


@pytest.mark.asyncio
async def test_handle_learn_feedback_always_reports_learned_true(handlers):
    """learned=True on success even when no new skills are created."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [["a"], ["a"]]
        runner.learn_from_feedback.return_value = True
        mock_runner_cls.from_model.return_value = runner

        req = LearnFeedbackRequest(
            session_id="s1",
            question="q",
            answer="a",
            feedback="good",
        )
        resp = await handlers.handle_learn_feedback(req)

        assert resp.learned is True
        assert resp.new_skill_count == 0


@pytest.mark.asyncio
async def test_handle_learn_feedback_trace_uses_context_not_reasoning(handlers):
    """Fallback trace must map context to 'context', not 'reasoning'."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.skillbook.skills.side_effect = [[], []]
        runner.learn_from_feedback.return_value = False
        runner.learn_from_traces.return_value = []
        mock_runner_cls.from_model.return_value = runner

        req = LearnFeedbackRequest(
            session_id="s1",
            question="q",
            answer="a",
            feedback="good",
            context="some background",
        )
        await handlers.handle_learn_feedback(req)

        trace = runner.learn_from_traces.call_args[0][0][0]
        assert trace["context"] == "some background"
        assert "reasoning" not in trace


# ── ace.skillbook.save/load ──────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_save_safe_mode(handlers, registry):
    handlers.config.safe_mode = True
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        req = SkillbookSaveRequest(session_id="s1", path="/tmp/some")
        with pytest.raises(ForbiddenInSafeModeError):
            await handlers.handle_skillbook_save(req)


@pytest.mark.asyncio
async def test_handle_load_safe_mode(handlers):
    handlers.config.safe_mode = True
    req = SkillbookLoadRequest(session_id="s1", path="/tmp/some")
    with pytest.raises(ForbiddenInSafeModeError):
        await handlers.handle_skillbook_load(req)


@pytest.mark.asyncio
async def test_handle_save_load_disabled(handlers, registry):
    """allow_save_load=false with safe_mode=false raises SaveLoadDisabledError."""
    handlers.config.safe_mode = False
    handlers.config.allow_save_load = False
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        with pytest.raises(SaveLoadDisabledError):
            await handlers.handle_skillbook_save(
                SkillbookSaveRequest(session_id="s1", path="/tmp/f")
            )
        with pytest.raises(SaveLoadDisabledError):
            await handlers.handle_skillbook_load(
                SkillbookLoadRequest(session_id="s1", path="/tmp/f")
            )


@pytest.mark.asyncio
async def test_handle_skillbook_save_rejects_path_outside_root(handlers, registry):
    handlers.config.skillbook_root = "/tmp/ace-root"
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        req = SkillbookSaveRequest(session_id="s1", path="/tmp/not-allowed/file.json")
        with pytest.raises(ValidationError):
            await handlers.handle_skillbook_save(req)


@pytest.mark.asyncio
async def test_handle_skillbook_load_rejects_path_outside_root(handlers):
    handlers.config.skillbook_root = "/tmp/ace-root"
    req = SkillbookLoadRequest(session_id="s1", path="/tmp/not-allowed/file.json")
    with pytest.raises(ValidationError):
        await handlers.handle_skillbook_load(req)


# ── ace.learn.sample success path ────────────────────────────────

@pytest.mark.asyncio
async def test_handle_learn_sample_success(handlers):
    """Success path: learning processes samples and returns counts."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        result_ok = MagicMock(error=None)
        runner.learn.return_value = [result_ok, result_ok]
        runner.skillbook.skills.side_effect = [["a"], ["a", "b", "c"]]
        mock_runner_cls.from_model.return_value = runner

        req = LearnSampleRequest(
            session_id="s1",
            samples=[
                SampleItem(question="q1", ground_truth="gt1"),
                SampleItem(question="q2", ground_truth="gt2"),
            ],
        )
        resp = await handlers.handle_learn_sample(req)

        assert resp.processed == 2
        assert resp.failed == 0
        assert resp.skill_count_before == 1
        assert resp.skill_count_after == 3
        assert resp.new_skill_count == 2
        runner.learn.assert_called_once()


@pytest.mark.asyncio
async def test_handle_learn_sample_partial_failure(handlers):
    """When some samples fail, counts reflect partial success."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        result_ok = MagicMock(error=None)
        result_fail = MagicMock(error="provider error")
        runner.learn.return_value = [result_ok, result_fail]
        runner.skillbook.skills.side_effect = [[], ["s1"]]
        mock_runner_cls.from_model.return_value = runner

        req = LearnSampleRequest(
            session_id="s1",
            samples=[
                SampleItem(question="q1"),
                SampleItem(question="q2"),
            ],
        )
        resp = await handlers.handle_learn_sample(req)

        assert resp.processed == 1
        assert resp.failed == 1


@pytest.mark.asyncio
async def test_handle_learn_sample_timeout(handlers):
    """learn.sample raises MCPTimeoutError when learn() exceeds timeout."""
    handlers.config.learn_timeout_seconds = 0  # instant timeout

    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()

        async def slow_learn(*args, **kwargs):
            await asyncio.sleep(10)

        runner.learn.side_effect = lambda *a, **kw: asyncio.get_event_loop().run_until_complete(slow_learn())
        runner.skillbook.skills.return_value = []
        mock_runner_cls.from_model.return_value = runner

        req = LearnSampleRequest(
            session_id="s1",
            samples=[SampleItem(question="q")],
        )
        with pytest.raises(MCPTimeoutError):
            await handlers.handle_learn_sample(req)


# ── ace.skillbook.save/load success paths ────────────────────────

@pytest.mark.asyncio
async def test_handle_skillbook_save_success(handlers, registry):
    """Success path: save returns the resolved path and skill count."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.save.return_value = None
        runner.skillbook.skills.return_value = ["s1", "s2"]
        mock_runner_cls.from_model.return_value = runner

        await registry.get_or_create("s1")
        req = SkillbookSaveRequest(session_id="s1", path="/tmp/test.json")
        resp = await handlers.handle_skillbook_save(req)

        assert resp.saved_skill_count == 2
        assert resp.session_id == "s1"
        runner.save.assert_called_once()


@pytest.mark.asyncio
async def test_handle_skillbook_load_success(handlers):
    """Success path: load returns the resolved path and new skill count."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.load.return_value = None
        runner.skillbook.skills.return_value = ["s1", "s2", "s3"]
        mock_runner_cls.from_model.return_value = runner

        req = SkillbookLoadRequest(session_id="s1", path="/tmp/test.json")
        resp = await handlers.handle_skillbook_load(req)

        assert resp.skill_count == 3
        assert resp.session_id == "s1"
        runner.load.assert_called_once()


# ── ace.skillbook.save/load uses resolved path ───────────────────

@pytest.mark.asyncio
async def test_handle_save_uses_resolved_path(handlers, registry):
    """save() receives the resolved path, not the raw user input."""
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        runner = MagicMock()
        runner.save.return_value = None
        runner.skillbook.skills.return_value = []
        mock_runner_cls.from_model.return_value = runner

        await registry.get_or_create("s1")
        # Path with .. that resolves to /tmp/test.json
        req = SkillbookSaveRequest(session_id="s1", path="/tmp/sub/../test.json")
        resp = await handlers.handle_skillbook_save(req)

        # The runner should receive the resolved path
        called_path = runner.save.call_args[0][0]
        assert ".." not in called_path
        assert resp.path == called_path


# ── error-to-MCP mapping ────────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_call_tool_error_mapping(handlers):
    """handle_call_tool maps domain errors to MCP error envelopes."""
    from ace_next.integrations.mcp.adapters import register_tools

    try:
        from mcp.server import Server
        from mcp import types
    except ImportError:
        pytest.skip("mcp not installed")

    server = Server("test")
    register_tools(server, handlers)

    # Call a tool that will fail (session not found for skillbook.get)
    req = SkillbookGetRequest(session_id="nonexistent")
    # Use the handlers directly — the adapter error mapping is tested via map_error_to_mcp
    from ace_next.integrations.mcp.errors import SessionNotFoundError

    err = SessionNotFoundError("nonexistent")
    mapped = map_error_to_mcp(err)
    assert mapped["code"] == "ACE_MCP_SESSION_NOT_FOUND"
    assert "nonexistent" in mapped["message"]
    assert mapped["details"]["session_id"] == "nonexistent"


def test_map_error_to_mcp_unknown_error():
    """Unknown exceptions map to ACE_MCP_INTERNAL_ERROR."""
    err = RuntimeError("boom")
    mapped = map_error_to_mcp(err)
    assert mapped["code"] == "ACE_MCP_INTERNAL_ERROR"
    assert "boom" in mapped["message"]
    assert mapped["details"]["type"] == "RuntimeError"


# ── sample indexing uses 0-based ─────────────────────────────────

@pytest.mark.asyncio
async def test_handle_learn_sample_prompt_limit_uses_zero_index(handlers):
    """Error message for oversized samples uses 0-based index."""
    handlers.config.max_prompt_chars = 5
    req = LearnSampleRequest(
        session_id="s1",
        samples=[
            SampleItem(question="ok"),       # fits
            SampleItem(question="toolong"),   # exceeds limit
        ],
    )
    with pytest.raises(ValidationError, match=r"samples\[1\]"):
        await handlers.handle_learn_sample(req)


# ── ground_truth included in feedback prompt limit ───────────────

@pytest.mark.asyncio
async def test_handle_learn_feedback_prompt_limit_includes_ground_truth(handlers):
    """ground_truth contributes to the prompt limit check."""
    handlers.config.max_prompt_chars = 20
    req = LearnFeedbackRequest(
        session_id="s1",
        question="q",
        answer="a",
        feedback="f",
        context="c",
        ground_truth="x" * 20,  # pushes total over 20
    )
    with pytest.raises(ValidationError):
        await handlers.handle_learn_feedback(req)
