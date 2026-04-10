"""Tests for the judge model routing normalization in trace_judge.

Regression guard for the 401 screening bug where bare OpenAI model
names (e.g. ``gpt-4.1-mini``) were routed through PydanticAI's litellm
fallback, which hard-codes a placeholder api_key and yields a 401 at
call time even when ``OPENAI_API_KEY`` is present in the environment.
"""

from __future__ import annotations

import pytest

from ace_eval.meta_agent.trace_judge import _resolve_judge_model

pytestmark = pytest.mark.unit


class TestResolveJudgeModel:
    def test_bare_gpt_is_routed_native_openai(self):
        """gpt-4.1-mini should resolve to the openai: native prefix."""
        resolved = _resolve_judge_model("gpt-4.1-mini")
        # resolve_model() passes explicit openai: prefixes through unchanged
        assert resolved == "openai:gpt-4.1-mini"

    def test_bare_o3_is_routed_native_openai(self):
        resolved = _resolve_judge_model("o3-mini")
        assert resolved == "openai:o3-mini"

    def test_bare_chatgpt_is_routed_native_openai(self):
        resolved = _resolve_judge_model("chatgpt-4o-latest")
        assert resolved == "openai:chatgpt-4o-latest"

    def test_explicit_openai_prefix_passthrough(self):
        """openai: prefix should be preserved as-is."""
        resolved = _resolve_judge_model("openai:gpt-4o")
        assert resolved == "openai:gpt-4o"

    def test_bedrock_slash_prefix_preserved(self):
        """bedrock/ models should continue to use the bedrock routing path."""
        # The Bedrock resolver uses AWS_BEARER_TOKEN_BEDROCK when present.
        # Without it, resolve_model rewrites the slash to a colon.
        resolved = _resolve_judge_model(
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        # either a BedrockConverseModel instance (api-key path) or the
        # native pydantic-ai "bedrock:..." string — both are acceptable;
        # what matters is that we did NOT downgrade it to litellm:
        assert not (
            isinstance(resolved, str) and resolved.startswith("litellm:")
        )

    def test_anthropic_prefix_passthrough(self):
        """Anthropic native prefix should be untouched."""
        resolved = _resolve_judge_model("anthropic:claude-3-5-sonnet-latest")
        assert resolved == "anthropic:claude-3-5-sonnet-latest"

    def test_non_openai_bare_model_keeps_default(self):
        """Non-OpenAI bare names fall back to the default resolve_model path."""
        resolved = _resolve_judge_model("llama3-70b")
        # default resolve_model falls back to litellm: for unknown bare names
        assert resolved == "litellm:llama3-70b"
