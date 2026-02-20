"""Unit tests for pipeline error types."""

from __future__ import annotations

import pytest

from pipeline import BranchError, PipelineConfigError, PipelineOrderError


@pytest.mark.unit
class TestErrorHierarchy:
    def test_pipeline_order_error_is_exception(self):
        err = PipelineOrderError("bad order")
        assert isinstance(err, Exception)

    def test_pipeline_config_error_is_exception(self):
        err = PipelineConfigError("bad config")
        assert isinstance(err, Exception)

    def test_branch_error_is_exception(self):
        err = BranchError([RuntimeError("x")])
        assert isinstance(err, Exception)


@pytest.mark.unit
class TestBranchError:
    def test_stores_failures_list(self):
        failures = [RuntimeError("a"), ValueError("b")]
        err = BranchError(failures)
        assert err.failures is failures

    def test_message_includes_failure_count(self):
        err = BranchError([RuntimeError("x"), OSError("y")])
        assert "2" in str(err)

    def test_message_includes_type_names(self):
        err = BranchError([RuntimeError("x"), ValueError("y")])
        msg = str(err)
        assert "RuntimeError" in msg
        assert "ValueError" in msg

    def test_single_failure(self):
        inner = RuntimeError("boom")
        err = BranchError([inner])
        assert err.failures == [inner]
        assert "1" in str(err)

    def test_empty_failures_list(self):
        err = BranchError([])
        assert err.failures == []
        assert "0" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(BranchError) as exc_info:
            raise BranchError([RuntimeError("boom")])
        assert len(exc_info.value.failures) == 1
