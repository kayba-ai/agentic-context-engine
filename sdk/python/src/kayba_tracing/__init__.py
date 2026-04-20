"""Kayba tracing — instrument your agents and send traces to Kayba.

Usage::

    from kayba_tracing import configure, trace, start_span

    configure(api_key="kb-...")

    @trace
    def my_agent(query: str) -> str:
        with start_span("retrieval") as span:
            span.set_inputs({"query": query})
            results = search(query)
            span.set_outputs(results)
        return synthesize(results)

Install::

    pip install kayba-tracing
"""

from kayba_tracing._wrapper import (
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
