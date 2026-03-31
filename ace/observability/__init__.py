"""Observability utilities for ACE.

Provides opt-in Logfire integration that auto-instruments all PydanticAI
agents (Agent, Reflector, SkillManager, RR).

Usage::

    from ace.observability import configure_logfire

    if configure_logfire():
        print("Logfire active")

Or via runner::

    ace = ACELiteLLM.from_model("gpt-4o-mini", logfire=True)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_logfire_configured = False


def configure_logfire() -> bool:
    """Configure Logfire and instrument PydanticAI agents.

    Reads ``LOGFIRE_TOKEN`` from the environment.  Set
    ``LOGFIRE_SEND_TO_LOGFIRE=false`` to disable sending in CI/local dev.

    Returns:
        ``True`` if Logfire was configured successfully, ``False`` if the
        ``logfire`` package is not installed.

    Raises:
        No exceptions — returns False on ImportError.
    """
    global _logfire_configured
    if _logfire_configured:
        return True

    try:
        import logfire

        def scrubbing_callback(m: logfire.ScrubMatch):
            if m.path == ("attributes", "trace", "reasoning"):
                return m.value
            if m.path == ("attributes", "trace", "answer"):
                return m.value
            if "messages" in m.path and "content" in m.path:
                return m.value
            if "payment_id" in m.path:
                return m.value

        logfire.configure(
            scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback)
        )
        logfire.instrument_pydantic_ai()
        _logfire_configured = True
        logger.info("Logfire configured — PydanticAI agents instrumented")
        return True
    except ImportError:
        logger.debug("logfire not installed — skipping instrumentation")
        return False


def is_configured() -> bool:
    """Return ``True`` if :func:`configure_logfire` has been called successfully."""
    return _logfire_configured
