"""Thin Kayba-branded wrapper around MLflow tracing.

All public symbols re-export MLflow functionality so that users never
need to ``import mlflow`` directly.  The :func:`configure` helper sets
the MLflow tracking URI and auth to point at the Kayba backend.
"""

from __future__ import annotations

import functools
import os
import re
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar, overload

_TRACING_INSTALL_HINT = (
    "Tracing requires the 'tracing' extra: pip install ace-framework[tracing]"
)

try:
    import mlflow
    import mlflow.tracing  # noqa: F401 — ensure tracing sub-module is loaded
except ImportError as exc:
    raise ImportError(_TRACING_INSTALL_HINT) from exc

DEFAULT_BASE_URL = "https://use.kayba.ai"

# Module-level state set by configure() / set_folder().
_folder: str | None = None

_MAX_FOLDER_LENGTH = 256
_SAFE_FOLDER_RE = re.compile(r"[^a-zA-Z0-9 _\-/.]")


def _sanitize_folder(name: str) -> str:
    """Sanitize a folder name to prevent injection attacks.

    Strips control characters, HTML tags, and characters outside an
    allowlist.  Truncates to ``_MAX_FOLDER_LENGTH``.
    """
    # Strip HTML tags.
    clean = re.sub(r"<[^>]*>", "", name)
    # Remove anything outside the safe set.
    clean = _SAFE_FOLDER_RE.sub("", clean)
    return clean.strip()[:_MAX_FOLDER_LENGTH]


_P = TypeVar("_P")
_R = TypeVar("_R")


def configure(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    experiment: str | None = None,
    folder: str | None = None,
) -> None:
    """Configure Kayba tracing.

    Sets the MLflow tracking URI and authentication so that all
    subsequent ``@trace`` / ``start_span`` calls export to Kayba.

    Args:
        api_key: Kayba API key. Falls back to ``KAYBA_API_KEY`` env var.
        base_url: Kayba API base URL. Falls back to ``KAYBA_API_URL`` env
                  var, then to ``https://use.kayba.ai``.
        experiment: Alias for ``folder``. If both are provided, ``folder``
                    takes precedence.
        folder: Optional folder name. Traces will be filed into this
                folder in the Kayba dashboard.
    """
    global _folder

    resolved_key = api_key or os.environ.get("KAYBA_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "No API key provided. Pass api_key= or set the KAYBA_API_KEY "
            "environment variable."
        )

    resolved_url = base_url or os.environ.get("KAYBA_API_URL") or DEFAULT_BASE_URL
    # Strip trailing slash, then append the MLflow-compatible mount path.
    tracking_uri = resolved_url.rstrip("/") + "/api/mlflow"

    # Configure MLflow under the hood.
    os.environ["MLFLOW_TRACKING_TOKEN"] = resolved_key
    mlflow.set_tracking_uri(tracking_uri)

    resolved_folder = folder or experiment
    _folder = _sanitize_folder(resolved_folder) or None if resolved_folder else None


def set_folder(folder: str | None) -> None:
    """Change the target folder for subsequent traces.

    Args:
        folder: Folder name, or ``None`` to clear (traces go to Unfiled).
    """
    global _folder
    _folder = _sanitize_folder(folder) or None if folder else None


def get_folder() -> str | None:
    """Return the currently configured folder, or ``None``."""
    return _folder


# ---------------------------------------------------------------------------
# Wrapped MLflow tracing primitives that inject the folder tag
# ---------------------------------------------------------------------------



def _inject_folder_tag() -> None:
    """Inject ``kayba.folder`` tag into the active trace if a folder is set."""
    if _folder is not None:
        mlflow.update_current_trace(tags={"kayba.folder": _folder})


@overload
def trace(func: Callable[..., _R]) -> Callable[..., _R]: ...


@overload
def trace(
    func: None = None,
    *,
    name: str | None = None,
    span_type: str = ...,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]: ...


def trace(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    span_type: str = "UNKNOWN",
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Decorator that creates a trace span for the decorated function.

    Works identically to ``mlflow.trace`` but automatically tags
    the trace with the configured Kayba folder.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Wrap the original function so the folder tag is injected
        # *inside* the trace context (before MLflow closes it).
        @functools.wraps(fn)
        def fn_with_tag(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            _inject_folder_tag()
            return result

        # Let MLflow handle the actual tracing.
        mlflow_kwargs: dict[str, Any] = {}
        if name is not None:
            mlflow_kwargs["name"] = name
        if span_type != "UNKNOWN":
            mlflow_kwargs["span_type"] = span_type
        if attributes is not None:
            mlflow_kwargs["attributes"] = attributes

        if mlflow_kwargs:
            traced = mlflow.trace(**mlflow_kwargs)(fn_with_tag)
        else:
            traced = mlflow.trace(fn_with_tag)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return traced(*args, **kwargs)

        return wrapper

    if func is not None:
        # Called as @trace without parentheses.
        return decorator(func)
    return decorator


@contextmanager
def start_span(
    name: str = "span",
    span_type: str | None = "UNKNOWN",
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager that creates a child span.

    Works identically to ``mlflow.start_span`` but automatically tags
    the trace with the configured Kayba folder when used as a root span.
    """
    with mlflow.start_span(
        name=name, span_type=span_type, attributes=attributes
    ) as span:
        yield span
        # Inject folder tag while the trace context is still open.
        _inject_folder_tag()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def enable() -> None:
    """Enable Kayba tracing (enabled by default after :func:`configure`)."""
    mlflow.tracing.enable()


def disable() -> None:
    """Disable Kayba tracing without removing the configuration."""
    mlflow.tracing.disable()


def get_trace(trace_id: str) -> Any:
    """Retrieve a trace by ID."""
    return mlflow.get_trace(trace_id)


def search_traces(
    experiment_names: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Search for traces, optionally filtered by experiment names."""
    if experiment_names is None:
        experiment_names = ["Default"]
    return mlflow.search_traces(experiment_names=experiment_names, **kwargs)
