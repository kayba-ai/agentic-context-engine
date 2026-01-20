"""ACE Daemon - Background queue processing for Claude Code hooks.

This package provides a daemon-based architecture that allows hooks to exit
instantly (< 50ms) while learning happens asynchronously in the background.

Architecture:
    Hook (fast) → Queue file → Daemon → AsyncLearningPipeline → Skillbook

Usage:
    # Start daemon
    ace-daemon start

    # Check status
    ace-daemon status

    # Install for auto-start on login (macOS)
    ace-daemon install
"""

from .watcher import QueueWatcher
from .queue_consumer import QueueConsumer, ProcessingStats
from .service import DaemonService, DaemonConfig, is_daemon_running, stop_daemon

__all__ = [
    "QueueWatcher",
    "QueueConsumer",
    "ProcessingStats",
    "DaemonService",
    "DaemonConfig",
    "is_daemon_running",
    "stop_daemon",
]
