"""
ACE Observability Module

Provides production-grade observability for ACE framework using Opik.
Replaces custom explainability implementation with industry-standard tracing.
"""

from .opik_integration import OpikIntegration, configure_opik, get_integration
from .tracers import ace_track, track_role, maybe_track

__all__ = [
    "OpikIntegration",
    "configure_opik",
    "get_integration",
    "ace_track",
    "track_role",
    "maybe_track",
]
