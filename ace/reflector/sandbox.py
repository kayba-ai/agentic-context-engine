"""Backward-compat shim — canonical location is ace_next.rr.sandbox."""

from ace_next.rr.sandbox import ExecutionResult, ExecutionTimeoutError, TraceSandbox

__all__ = ["ExecutionResult", "ExecutionTimeoutError", "TraceSandbox"]
