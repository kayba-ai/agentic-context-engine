"""Kayba tracing — instrument your agents and send traces to Kayba.

This module re-exports from the standalone ``kayba-tracing`` package.
Both import paths are supported::

    # Standalone package
    from kayba_tracing import configure, trace, start_span

    # Via ace-framework
    from ace.tracing import configure, trace, start_span

Requires the ``tracing`` extra::

    pip install ace-framework[tracing]
"""

from kayba_tracing import (
    configure,
    disable,
    enable,
    get_folder,
    get_trace,
    search_traces,
    set_folder,
    start_span,
    trace,
)

__all__ = [
    "configure",
    "disable",
    "enable",
    "get_folder",
    "get_trace",
    "search_traces",
    "set_folder",
    "start_span",
    "trace",
]
