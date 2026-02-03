"""Lightweight sandbox for executing LLM-generated Python code."""

from __future__ import annotations

import collections
import io
import json
import platform
import re
import signal
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time, timezone
from typing import Any, Callable, Dict, Optional

from .trace_context import TraceContext


class ExecutionTimeoutError(Exception):
    """Raised when code execution exceeds the timeout."""

    pass


@dataclass
class ExecutionResult:
    """Result of executing code in the sandbox.

    Attributes:
        stdout: Captured standard output
        stderr: Captured standard error
        final_value: Value passed to FINAL() if called, otherwise None
        exception: Exception that occurred during execution, if any
    """

    stdout: str = ""
    stderr: str = ""
    final_value: Any = None
    exception: Optional[Exception] = None

    @property
    def success(self) -> bool:
        """Return True if execution completed without errors."""
        return self.exception is None and "Error" not in self.stderr


class TraceSandbox:
    """Lightweight sandbox using exec() with restricted builtins.

    This sandbox restricts builtins but is NOT secure against determined escape
    attempts. Security relies on trusting the LLM not to generate malicious code.
    Do not use this sandbox to execute untrusted or user-provided code.

    Restrictions (defense-in-depth, not security guarantees):
        - No file access: `open` and `__import__` are blocked
        - No code injection: `eval`, `exec`, `compile` are blocked
        - Read-only trace: TraceContext is immutable
        - Timeout protection: Configurable per-execution timeout (Unix only)
        - Worst case: bad code fails -> fallback to simple reflector

    Example:
        >>> sandbox = TraceSandbox(trace=trace, llm_query_fn=llm_query)
        >>> result = sandbox.execute("print(len(trace.steps))", timeout=30.0)
        >>> print(result.stdout)
        5
    """

    # Safe builtins that don't allow file/network access or code injection
    SAFE_BUILTINS: Dict[str, Any] = {
        # Core types
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "bool": bool,
        "type": type,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "range": range,
        "bytes": bytes,
        "bytearray": bytearray,
        # Iteration
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "iter": iter,
        "next": next,
        # Math
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        # Logic
        "any": any,
        "all": all,
        "not": lambda x: not x,
        # String/Formatting
        "chr": chr,
        "ord": ord,
        "repr": repr,
        "format": format,
        "ascii": ascii,
        "bin": bin,
        "hex": hex,
        "oct": oct,
        # Object inspection
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "delattr": delattr,
        "dir": dir,
        "vars": lambda obj=None: {} if obj is None else vars(obj),
        "id": id,
        "hash": hash,
        "callable": callable,
        # Exceptions (for try/except in generated code)
        "Exception": Exception,
        "BaseException": BaseException,
        "ValueError": ValueError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "TypeError": TypeError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "AssertionError": AssertionError,
        "LookupError": LookupError,
        "ZeroDivisionError": ZeroDivisionError,
        "NameError": NameError,
        "OverflowError": OverflowError,
        "FloatingPointError": FloatingPointError,
        "ArithmeticError": ArithmeticError,
        "SyntaxError": SyntaxError,
        "IndentationError": IndentationError,
        "TabError": TabError,
        "UnicodeError": UnicodeError,
        "UnicodeDecodeError": UnicodeDecodeError,
        "UnicodeEncodeError": UnicodeEncodeError,
        "NotImplementedError": NotImplementedError,
        "RecursionError": RecursionError,
        # Constants
        "True": True,
        "False": False,
        "None": None,
        # BLOCKED - security sensitive
        "open": None,
        "__import__": None,
        "eval": None,
        "exec": None,
        "compile": None,
        "input": None,
        "globals": None,
        "locals": None,
        "breakpoint": None,
        "memoryview": None,
        "__build_class__": None,
    }

    def __init__(
        self,
        trace: Optional[TraceContext],
        llm_query_fn: Optional[Callable[[str], str]] = None,
        additional_globals: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the sandbox with trace and optional LLM query function.

        Args:
            trace: TraceContext for trace exploration (can be None)
            llm_query_fn: Function to call for sub-LLM queries
            additional_globals: Extra variables to inject into the namespace
        """
        self._final_value: Any = None
        self._final_called = False

        # Build the namespace
        self.namespace: Dict[str, Any] = {
            "__builtins__": self.SAFE_BUILTINS.copy(),
            # Core analysis objects
            "trace": trace,
            "FINAL": self._final,
            # Safe stdlib modules
            "json": json,
            "re": re,
            "collections": collections,
            # datetime module and commonly used classes
            "datetime": datetime,
            "timedelta": timedelta,
            "date": date,
            "time": time,
            "timezone": timezone,
        }

        # Add llm_query if provided
        if llm_query_fn is not None:
            self.namespace["llm_query"] = llm_query_fn
        else:
            # Provide a stub that explains the feature is disabled
            self.namespace["llm_query"] = lambda _prompt: (
                "(llm_query disabled - analyze with available data)"
            )

        # Add any additional globals
        if additional_globals:
            self.namespace.update(additional_globals)

    def _final(self, value: Any) -> None:
        """Called by LLM code to output the final result.

        Args:
            value: The final analysis result (should be a dict matching ReflectorOutput)

        Raises:
            StopIteration: Always raised to signal completion
        """
        self._final_value = value
        self._final_called = True
        raise StopIteration("FINAL called - analysis complete")

    @property
    def final_value(self) -> Any:
        """Return the value passed to FINAL(), or None if not called."""
        return self._final_value

    @property
    def final_called(self) -> bool:
        """Return True if FINAL() was called."""
        return self._final_called

    def inject(self, name: str, value: Any) -> None:
        """Inject a variable into the sandbox namespace.

        Args:
            name: Variable name
            value: Variable value
        """
        self.namespace[name] = value

    def execute(self, code: str, timeout: float = 30.0) -> ExecutionResult:
        """Execute code in the sandbox and capture output.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds (default: 30.0).
                     Only enforced on Unix systems (macOS, Linux).
                     On Windows, timeout is not enforced.

        Returns:
            ExecutionResult with stdout, stderr, final_value, and exception
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Set up timeout handler (Unix only)
        use_timeout = platform.system() != "Windows" and timeout > 0
        old_handler = None

        def timeout_handler(_signum: int, _frame: Any) -> None:
            raise ExecutionTimeoutError(f"Execution exceeded {timeout}s timeout")

        if use_timeout:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace, self.namespace)
        except StopIteration:
            # FINAL() was called - this is expected
            pass
        except ExecutionTimeoutError as e:
            stderr_buf.write(f"\nExecutionTimeoutError: {e}")
            return ExecutionResult(
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                final_value=self._final_value,
                exception=e,
            )
        except Exception as e:
            # Capture the exception info
            stderr_buf.write(f"\n{type(e).__name__}: {e}")
            return ExecutionResult(
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                final_value=self._final_value,
                exception=e,
            )
        finally:
            if use_timeout:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            final_value=self._final_value,
            exception=None,
        )

    def reset(self) -> None:
        """Reset the sandbox state for a new execution."""
        self._final_value = None
        self._final_called = False
