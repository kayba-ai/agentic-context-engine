import pytest
from pydantic import ValidationError
from ace_next.integrations.mcp.models import (
    AskRequest,
    AskResponse,
    LearnSampleRequest,
    LearnSampleResponse,
    LearnFeedbackRequest,
    LearnFeedbackResponse,
    SkillbookGetRequest,
    SkillbookGetResponse,
    SkillbookSaveRequest,
    SkillbookSaveResponse,
    SkillbookLoadRequest,
    SkillbookLoadResponse,
    SessionConfig,
    SampleItem,
    SkillItem,
    ErrorEnvelope,
)


def test_session_config_validation():
    # Valid with all fields
    config = SessionConfig(model="gpt-4o", temperature=0.7, max_tokens=100)
    assert config.model == "gpt-4o"

    # Valid without model (optional per contract)
    config2 = SessionConfig(temperature=0.5)
    assert config2.model is None

    # Invalid: empty string for model
    with pytest.raises(ValidationError):
        SessionConfig(model="")

    # Invalid temp
    with pytest.raises(ValidationError):
        SessionConfig(model="gpt-4o", temperature=2.5)


def test_ask_request_validation():
    # Valid
    req = AskRequest(session_id="s1", question="hello")
    assert req.context == ""
    assert req.metadata is None

    # Max length question
    with pytest.raises(ValidationError):
        AskRequest(session_id="s1", question="a" * 100001)

    # Removed compatibility flags must remain invalid
    with pytest.raises(ValidationError):
        AskRequest(session_id="s1", question="hello", learn=True)


def test_learn_sample_request_limits():
    # Min items
    with pytest.raises(ValidationError):
        LearnSampleRequest(session_id="s1", samples=[])

    # Max items
    samples = [{"question": f"q{i}"} for i in range(26)]
    with pytest.raises(ValidationError):
        LearnSampleRequest(session_id="s1", samples=samples)


def test_skillbook_get_limits():
    # Valid
    req = SkillbookGetRequest(session_id="s", limit=50)
    assert req.limit == 50

    # Max limit
    with pytest.raises(ValidationError):
        SkillbookGetRequest(session_id="s", limit=201)


def test_error_envelope():
    env = ErrorEnvelope(code="ERR1", message="Error message")
    assert env.code == "ERR1"

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        ErrorEnvelope(code="ERR1", message="m", extra="not allowed")


def test_skillbook_load_request_disallows_unsupported_flags():
    req = SkillbookLoadRequest(session_id="s1", path="/tmp/skillbook.json")
    assert req.path == "/tmp/skillbook.json"

    with pytest.raises(ValidationError):
        SkillbookLoadRequest(
            session_id="s1",
            path="/tmp/skillbook.json",
            replace=False,
        )
