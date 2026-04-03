"""Lightweight sandbox for executing LLM-generated Python code."""

from __future__ import annotations

import collections
import copy
import io
import json
import logging
import platform
import re
import math
import signal
import threading
import time as _time_mod
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time, timezone
from typing import Any, Callable, Dict, List, Optional

from .trace_context import TraceContext

logger = logging.getLogger(__name__)


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
        return self.exception is None


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
        # Object inspection (getattr blocks dunder access — see __init__)
        "hasattr": hasattr,
        "getattr": None,  # Replaced with safe_getattr in __init__
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
        # BLOCKED - security sensitive (raise clear errors, not NoneType)
        "open": None,
        "__import__": None,  # replaced with _safe_import in __init__
        "eval": None,
        "exec": None,
        "compile": None,
        "input": None,
        "globals": None,
        "locals": None,
        "breakpoint": None,
        "memoryview": None,
    }

    def __init__(
        self,
        trace: Optional[TraceContext],
        llm_query_fn: Optional[Callable[[str], str]] = None,
        additional_globals: Optional[Dict[str, Any]] = None,
        *,
        parallel_max_concurrency: int = 10,
        parallel_max_retries: int = 3,
        parallel_retry_delay: float = 1.0,
        parallel_timeout: Optional[float] = None,
    ) -> None:
        """Initialize the sandbox with trace and optional LLM query function.

        Args:
            trace: TraceContext for trace exploration (can be None)
            llm_query_fn: Function to call for sub-LLM queries
            additional_globals: Extra variables to inject into the namespace
            parallel_max_concurrency: Max concurrent workers for parallel_map
            parallel_max_retries: Max retries per item in parallel_map
            parallel_retry_delay: Base delay (seconds) for exponential backoff
            parallel_timeout: Per-item timeout in seconds (None = no timeout)
        """
        self._final_value: Any = None
        self._final_called = False

        # parallel_map configuration (infrastructure-side only)
        self._parallel_max_concurrency = parallel_max_concurrency
        self._parallel_max_retries = parallel_max_retries
        self._parallel_retry_delay = parallel_retry_delay
        self._parallel_timeout = parallel_timeout

        # Build the namespace
        self.namespace: Dict[str, Any] = {
            "__builtins__": self.SAFE_BUILTINS.copy(),
            # Core analysis objects
            "trace": trace,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
            "helper_registry": {},
            "register_helper": self._register_helper,
            "list_helpers": self._list_helpers,
            "run_helper": self._run_helper,
            "get_batch_item": self._get_batch_item,
            "get_item_payload": self._get_item_payload,
            "get_item_messages": self._get_item_messages,
            "get_item_question": self._get_item_question,
            "get_item_feedback": self._get_item_feedback,
            "get_item_id": self._get_item_id,
            "get_message_text": self._get_message_text,
            "preview_item": self._preview_item,
            "parallel_map": self._parallel_map,
            # Safe stdlib modules
            "json": json,
            "re": re,
            "math": math,
            "collections": collections,
            # datetime module and commonly used classes
            "datetime": datetime,
            "timedelta": timedelta,
            "date": date,
            "time": time,
            "timezone": timezone,
        }

        # Safe getattr that blocks dunder access — override in both
        # builtins (so bare getattr() works) and namespace (for direct ref)
        def safe_getattr(obj, name, *default):
            if name.startswith("_"):
                raise AttributeError(f"Access to '{name}' blocked")
            return getattr(obj, name, *default) if default else getattr(obj, name)

        self.namespace["__builtins__"]["getattr"] = safe_getattr
        self.namespace["safe_getattr"] = safe_getattr

        # Safe import — allows pre-loaded modules, blocks everything else.
        # LLMs often write `import json` even when json is already available.
        _allowed_modules = {
            "json": json,
            "re": re,
            "math": math,
            "collections": collections,
            "datetime": __import__("datetime"),
        }

        def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in _allowed_modules:
                return _allowed_modules[name]
            raise ImportError(
                f"import {name!r} is blocked in sandbox. "
                f"Pre-loaded modules ({', '.join(sorted(_allowed_modules))}) "
                f"are already available — use them directly."
            )

        self.namespace["__builtins__"]["__import__"] = _safe_import

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

    def _final_var(self, var_name: str) -> None:
        """Called by LLM code to output a variable as the final result.

        Convenience function to finalize with a pre-built result stored in a variable.
        Useful when the analysis result is built up across multiple code blocks.

        Args:
            var_name: Name of the variable in the namespace to use as the result

        Raises:
            ValueError: If the variable doesn't exist
            StopIteration: Always raised to signal completion
        """
        if var_name not in self.namespace:
            available = [k for k in self.namespace.keys() if not k.startswith("_")]
            raise ValueError(
                f"Variable '{var_name}' not found. Available: {available[:20]}"
            )
        self._final(self.namespace[var_name])

    def _show_vars(self) -> None:
        """Print available variables in the namespace for debugging.

        Prints a list of user-accessible variables (excludes internal/dunder names).
        """
        user_vars = [k for k in self.namespace.keys() if not k.startswith("_")]
        # Exclude builtins and modules for cleaner output
        excluded = {
            "__builtins__",
            "json",
            "re",
            "collections",
            "datetime",
            "timedelta",
            "date",
            "time",
            "timezone",
            "safe_getattr",
        }
        user_vars = [k for k in user_vars if k not in excluded]
        logger.debug("Available variables: %s", sorted(user_vars))

    def _register_helper(
        self,
        name: str,
        source: str,
        description: str = "",
    ) -> str:
        """Register reusable helper code in the sandbox.

        The helper source is executed immediately and stored so that future
        sandbox snapshots can recreate the same helper definitions for
        sub-agents.
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid helper name: {name!r}")
        if not source.strip():
            raise ValueError("Helper source cannot be empty")

        exec(source, self.namespace, self.namespace)
        helper = self.namespace.get(name)
        if not callable(helper):
            raise ValueError(f"Helper source must define a callable named {name!r}")

        registry = self.namespace.setdefault("helper_registry", {})
        registry[name] = {
            "description": description,
            "source": source,
        }
        return f"Registered helper {name}"

    def _list_helpers(self) -> list[dict[str, str]]:
        """Return metadata for registered helpers."""
        registry = self.namespace.get("helper_registry", {})
        if not isinstance(registry, dict):
            return []

        helpers: list[dict[str, str]] = []
        for name, meta in registry.items():
            if not isinstance(meta, dict):
                continue
            helpers.append(
                {
                    "name": str(name),
                    "description": str(meta.get("description", "")),
                }
            )
        return helpers

    def _run_helper(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a registered helper by name."""
        helper = self.namespace.get(name)
        if not callable(helper):
            raise KeyError(f"Helper {name!r} is not registered")
        return helper(*args, **kwargs)

    def _get_batch_item(self, index: int) -> Any:
        """Return a batch item by index when batch helpers are available."""
        batch_items = self.namespace.get("batch_items")
        if not isinstance(batch_items, list):
            raise RuntimeError("batch_items is not available in this sandbox")
        return batch_items[index]

    def _resolve_batch_item(self, item_or_index: Any) -> Any:
        """Resolve a batch helper argument to the underlying item."""
        if isinstance(item_or_index, int):
            return self._get_batch_item(item_or_index)
        return item_or_index

    def _get_item_payload(self, item_or_index: Any) -> Any:
        """Return the payload for a batch item or index without rewriting it."""
        item = self._resolve_batch_item(item_or_index)
        if (
            isinstance(item, dict)
            and item.get("role") == "conversation"
            and isinstance(item.get("content"), dict)
        ):
            return item["content"]
        return item

    def _get_item_messages(self, item_or_index: Any) -> list[Any]:
        """Return a best-effort message list for a batch item or index."""
        payload = self._get_item_payload(item_or_index)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            trace_value = payload.get("trace")
            if isinstance(trace_value, list):
                return trace_value
            if isinstance(trace_value, dict):
                for key in ("messages", "steps", "trace"):
                    nested = trace_value.get(key)
                    if isinstance(nested, list):
                        return nested
            for key in ("messages", "steps"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _get_item_field(self, item_or_index: Any, field: str) -> str:
        """Extract a string field from a batch item payload when present."""
        payload = self._get_item_payload(item_or_index)
        if isinstance(payload, dict):
            value = payload.get(field)
            if value is not None:
                return str(value)
        return ""

    def _get_item_question(self, item_or_index: Any) -> str:
        """Return the question field for a batch item or index."""
        return self._get_item_field(item_or_index, "question")

    def _get_item_feedback(self, item_or_index: Any) -> str:
        """Return the feedback field for a batch item or index."""
        return self._get_item_field(item_or_index, "feedback")

    def _get_item_id(self, item_or_index: Any) -> str:
        """Return a stable identifier for a batch item or index."""
        item = self._resolve_batch_item(item_or_index)
        payload = self._get_item_payload(item)

        if isinstance(item_or_index, int):
            item_ids = self.namespace.get("item_ids")
            if isinstance(item_ids, list) and 0 <= item_or_index < len(item_ids):
                return str(item_ids[item_or_index])

        if isinstance(item, dict):
            for key in ("item_id", "task_id", "id"):
                value = item.get(key)
                if value is not None:
                    return str(value)
        if isinstance(payload, dict):
            for key in ("item_id", "task_id", "id"):
                value = payload.get(key)
                if value is not None:
                    return str(value)

        return "unknown_item"

    def _get_message_text(self, message: Any) -> str:
        """Return a readable text summary for a message-like object."""
        if isinstance(message, dict):
            content = message.get("content")
            if content not in (None, ""):
                if isinstance(content, str):
                    return content
                try:
                    return json.dumps(content, default=str)
                except Exception:
                    return str(content)

            tool_calls = message.get("tool_calls")
            if tool_calls:
                return f"tool_calls={json.dumps(tool_calls, default=str)}"

            tool_results = message.get("tool_results")
            if tool_results:
                return f"tool_results={json.dumps(tool_results, default=str)}"

            for key in ("reasoning", "answer", "text"):
                value = message.get(key)
                if value not in (None, ""):
                    return str(value)

            try:
                return json.dumps(message, default=str)
            except Exception:
                return str(message)

        return str(message)

    def _preview_item(self, item_or_index: Any) -> dict[str, Any]:
        """Return a compact preview for a batch item or index."""
        messages = self._get_item_messages(item_or_index)
        first_message = self._get_message_text(messages[0]) if messages else ""
        payload = self._get_item_payload(item_or_index)
        return {
            "item_id": self._get_item_id(item_or_index),
            "question_preview": self._get_item_question(item_or_index)[:120],
            "feedback_preview": self._get_item_feedback(item_or_index)[:120],
            "message_count": len(messages),
            "first_message_preview": first_message[:120],
            "payload_type": type(payload).__name__,
        }

    def _parallel_map(
        self, fn: Callable[[Any], Any], inputs: list, *, return_exceptions: bool = False
    ) -> List[Any]:
        """Execute fn over inputs in parallel using a thread pool.

        Concurrency, retries, backoff, and timeout are controlled by the
        sandbox configuration — the agent cannot override them.

        Args:
            fn: A callable to apply to each input
            inputs: Ordered list of inputs
            return_exceptions: If True, failed items appear as exceptions in
                the results list instead of raising immediately

        Returns:
            Ordered list of results (same length/order as inputs)

        Raises:
            Exception: Re-raises the first worker exception when
                return_exceptions is False
        """
        if not inputs:
            return []

        max_concurrency = self._parallel_max_concurrency
        max_retries = self._parallel_max_retries
        retry_delay = self._parallel_retry_delay
        timeout = self._parallel_timeout

        def _worker(item: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(item)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        backoff = retry_delay * (2**attempt)
                        _time_mod.sleep(backoff)
            raise last_exc  # type: ignore[misc]

        pool_size = min(len(inputs), max_concurrency)
        results: List[Any] = [None] * len(inputs)
        first_exc: Optional[Exception] = None
        first_exc_idx: Optional[int] = None

        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            futures = {
                pool.submit(_worker, item): idx for idx, item in enumerate(inputs)
            }
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=timeout)
                except Exception as exc:
                    if return_exceptions:
                        results[idx] = exc
                    else:
                        if first_exc_idx is None or idx < first_exc_idx:
                            first_exc = exc
                            first_exc_idx = idx

        if first_exc is not None and not return_exceptions:
            raise first_exc

        return results

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
                     - Unix: uses signal.SIGALRM
                     - Windows: not enforced (in-process execution)

        Returns:
            ExecutionResult with stdout, stderr, final_value, and exception
        """
        if platform.system() == "Windows":
            return self._execute_no_timeout(code)
        elif threading.current_thread() is not threading.main_thread():
            return self._execute_no_timeout(code)
        else:
            return self._execute_unix(code, timeout)

    def _execute_unix(self, code: str, timeout: float) -> ExecutionResult:
        """Execute code using signal-based timeout (Unix only).

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult with stdout, stderr, final_value, and exception
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Set up timeout handler (Unix only)
        use_timeout = timeout > 0
        old_handler = None

        def timeout_handler(_signum: int, _frame: Any) -> None:
            raise ExecutionTimeoutError(f"Execution exceeded {timeout}s timeout")

        if use_timeout:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(math.ceil(timeout))

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

    def _execute_windows(self, code: str, timeout: float) -> ExecutionResult:
        """Execute code on Windows without timeout enforcement.

        Windows multiprocessing uses 'spawn' which cannot pass functions,
        trace objects, or injected variables to subprocesses. Instead,
        execute in-process for full feature support (no timeout enforcement).

        Args:
            code: Python code to execute
            timeout: Ignored on Windows (logged as warning)

        Returns:
            ExecutionResult with stdout, stderr, final_value, and exception
        """
        logger.debug("Windows: executing in-process (timeout not enforced)")
        return self._execute_no_timeout(code)

    def _execute_no_timeout(self, code: str) -> ExecutionResult:
        """Execute code without timeout enforcement.

        Fallback when multiprocessing is unavailable or fails.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with stdout, stderr, final_value, and exception
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace, self.namespace)
        except StopIteration:
            # FINAL() was called - this is expected
            pass
        except Exception as e:
            stderr_buf.write(f"\n{type(e).__name__}: {e}")
            return ExecutionResult(
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                final_value=self._final_value,
                exception=e,
            )

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


def create_readonly_sandbox(parent: TraceSandbox) -> TraceSandbox:
    """Create an isolated sandbox snapshot for sub-agent use.

    Deep-copies data variables from the parent sandbox so the sub-agent
    can explore trace data via ``execute_code`` without affecting the
    parent's state.  Safe for parallel use — each snapshot is independent.

    Args:
        parent: The parent sandbox to snapshot.

    Returns:
        A new TraceSandbox with deep-copied data variables.
    """
    sandbox = TraceSandbox(
        trace=None,
        llm_query_fn=None,
        parallel_max_concurrency=parent._parallel_max_concurrency,
        parallel_max_retries=parent._parallel_max_retries,
        parallel_retry_delay=parent._parallel_retry_delay,
        parallel_timeout=parent._parallel_timeout,
    )

    # Keys already set up by TraceSandbox.__init__ — skip them
    infrastructure = {
        "__builtins__",
        "FINAL",
        "FINAL_VAR",
        "SHOW_VARS",
        "parallel_map",
        "llm_query",
        "safe_getattr",
        "trace",
        "register_helper",
        "list_helpers",
        "run_helper",
        "get_batch_item",
        "get_item_payload",
        "get_item_messages",
        "get_item_question",
        "get_item_feedback",
        "get_item_id",
        "get_message_text",
        "preview_item",
        "json",
        "re",
        "math",
        "collections",
        "datetime",
        "timedelta",
        "date",
        "time",
        "timezone",
    }

    for key, value in parent.namespace.items():
        if key in infrastructure or key.startswith("_"):
            continue
        try:
            sandbox.namespace[key] = copy.deepcopy(value)
        except (TypeError, copy.Error):
            # Modules, functions, etc. — share by reference
            sandbox.namespace[key] = value

    registry = sandbox.namespace.get("helper_registry", {})
    if isinstance(registry, dict):
        for name, meta in registry.items():
            if not isinstance(meta, dict):
                continue
            source = meta.get("source")
            if not isinstance(source, str) or not source.strip():
                continue
            try:
                exec(source, sandbox.namespace, sandbox.namespace)
            except Exception as exc:
                logger.warning("Failed to restore helper %s in snapshot: %s", name, exc)

    return sandbox
