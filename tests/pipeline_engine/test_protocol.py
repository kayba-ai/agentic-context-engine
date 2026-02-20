"""Unit tests for StepProtocol and SampleResult."""

from __future__ import annotations

import pytest

from pipeline import SampleResult, StepContext, StepProtocol


# ---------------------------------------------------------------------------
# Helper objects
# ---------------------------------------------------------------------------

class ValidStep:
    requires = frozenset({"a"})
    provides = frozenset({"b"})
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class ValidStepWithPlainSets:
    requires = {"a"}   # plain set, not frozenset
    provides = {"b"}
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class MissingRequires:
    provides = frozenset({"b"})
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class MissingProvides:
    requires = frozenset({"a"})
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class MissingCall:
    requires = frozenset({"a"})
    provides = frozenset({"b"})


class EmptyStep:
    """Valid step with empty requires/provides (e.g. a pure side-effect step)."""
    requires = frozenset()
    provides = frozenset()
    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


# ---------------------------------------------------------------------------
# StepProtocol
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStepProtocol:
    def test_valid_step_passes_isinstance(self):
        assert isinstance(ValidStep(), StepProtocol)

    def test_valid_step_with_plain_sets_passes_isinstance(self):
        # AbstractSet[str] accepts both set and frozenset
        assert isinstance(ValidStepWithPlainSets(), StepProtocol)

    def test_empty_requires_provides_is_valid(self):
        assert isinstance(EmptyStep(), StepProtocol)

    def test_missing_requires_fails_isinstance(self):
        assert not isinstance(MissingRequires(), StepProtocol)

    def test_missing_provides_fails_isinstance(self):
        assert not isinstance(MissingProvides(), StepProtocol)

    def test_missing_call_fails_isinstance(self):
        assert not isinstance(MissingCall(), StepProtocol)

    def test_plain_object_fails_isinstance(self):
        assert not isinstance(object(), StepProtocol)

    def test_none_fails_isinstance(self):
        assert not isinstance(None, StepProtocol)


# ---------------------------------------------------------------------------
# SampleResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSampleResult:
    def test_basic_construction(self):
        ctx = StepContext(sample="s")
        r = SampleResult(sample="s", output=ctx, error=None, failed_at=None)
        assert r.sample == "s"
        assert r.output is ctx
        assert r.error is None
        assert r.failed_at is None

    def test_cause_defaults_to_none(self):
        r = SampleResult(sample="s", output=None, error=None, failed_at=None)
        assert r.cause is None

    def test_is_mutable(self):
        r = SampleResult(sample="s", output=None, error=None, failed_at=None)
        exc = RuntimeError("oops")
        r.error = exc
        r.failed_at = "SomeStep"
        assert r.error is exc
        assert r.failed_at == "SomeStep"

    def test_failure_result(self):
        exc = RuntimeError("boom")
        r = SampleResult(sample="x", output=None, error=exc, failed_at="BoomStep")
        assert r.output is None
        assert r.error is exc
        assert r.failed_at == "BoomStep"

    def test_branch_failure_has_cause(self):
        inner = RuntimeError("inner")
        from pipeline import BranchError
        outer = BranchError([inner])
        r = SampleResult(
            sample="x", output=None, error=outer, failed_at="Branch", cause=inner
        )
        assert r.cause is inner
        assert r.failed_at == "Branch"
