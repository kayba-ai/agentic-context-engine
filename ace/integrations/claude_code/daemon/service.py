"""Daemon service for ACE queue processing with lifecycle management.

Handles signal handling, PID file management, and graceful shutdown.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .queue_consumer import QueueConsumer, ProcessingStats

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Daemon configuration.

    ACE uses subscription-only mode via Claude CLI. No API keys required.
    """

    queue_dir: Path = field(default_factory=lambda: Path.home() / ".ace" / "queue")
    log_dir: Path = field(default_factory=lambda: Path.home() / ".ace" / "logs")
    pid_file: Path = field(
        default_factory=lambda: Path.home() / ".ace" / "logs" / "daemon.pid"
    )
    ace_model: str = "anthropic/claude-sonnet-4-5-20250929"  # Deprecated - ignored
    max_workers: int = 2
    log_level: str = "INFO"
    stats_interval: int = 300  # Log stats every 5 minutes


class DaemonService:
    """
    Long-running daemon service for ACE queue processing.

    Handles:
    - PID file management
    - Signal handling (SIGTERM, SIGINT)
    - Graceful shutdown
    - Periodic stats logging

    Example:
        >>> config = DaemonConfig(use_cli=True)
        >>> daemon = DaemonService(config)
        >>> daemon.run()  # Blocks until signal
    """

    def __init__(self, config: DaemonConfig):
        self.config = config
        self._consumer: Optional[QueueConsumer] = None
        self._running = False
        self._last_stats_time = 0.0

    def run(self) -> int:
        """
        Run the daemon (blocking).

        Returns:
            Exit code (0 for success, 1 for error)
        """
        self._setup_logging()
        self._setup_signals()

        # Check if already running
        existing_pid = is_daemon_running(self.config.pid_file)
        if existing_pid:
            logger.error(f"Daemon already running (PID: {existing_pid})")
            return 1

        # Write PID file
        self._write_pid_file()

        try:
            logger.info("ACE Daemon starting...")
            logger.info(f"  Queue directory: {self.config.queue_dir}")
            logger.info(f"  Mode: CLI subscription (no API keys required)")
            logger.info(f"  Max workers: {self.config.max_workers}")

            # Create and start consumer (CLI subscription mode)
            self._consumer = QueueConsumer(
                queue_dir=self.config.queue_dir,
                max_workers=self.config.max_workers,
            )
            self._consumer.start()
            self._running = True

            logger.info(f"ACE Daemon running (PID: {os.getpid()})")

            # Main loop - wait for signals, log periodic stats
            self._last_stats_time = time.time()
            while self._running:
                time.sleep(1)
                self._maybe_log_stats()

            return 0

        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)
            return 1

        finally:
            self._cleanup()

    def stop(self) -> None:
        """Signal the daemon to stop."""
        logger.info("Stopping daemon...")
        self._running = False

    def _setup_logging(self) -> None:
        """Configure logging."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.config.log_dir / "daemon.log"

        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _setup_signals(self) -> None:
        """Set up signal handlers."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle SIGTERM/SIGINT."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down...")
        self.stop()

    def _write_pid_file(self) -> None:
        """Write PID file."""
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.write_text(str(os.getpid()))
        logger.debug(f"Wrote PID file: {self.config.pid_file}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._consumer:
            stats = self._consumer.stop(wait=True, timeout=30.0)
            logger.info(f"Final stats: {stats}")

        # Remove PID file
        self.config.pid_file.unlink(missing_ok=True)
        logger.info("Daemon stopped")

    def _maybe_log_stats(self) -> None:
        """Log stats periodically."""
        now = time.time()
        if now - self._last_stats_time >= self.config.stats_interval:
            self._last_stats_time = now
            if self._consumer:
                stats = self._consumer.stats
                logger.info(f"Stats: {stats}")


def is_daemon_running(
    pid_file: Path = Path.home() / ".ace" / "logs" / "daemon.pid",
) -> Optional[int]:
    """
    Check if daemon is running.

    Args:
        pid_file: Path to PID file

    Returns:
        PID if daemon is running, None otherwise
    """
    pid_file = Path(pid_file).expanduser()

    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file is stale
        pid_file.unlink(missing_ok=True)
        return None


def stop_daemon(
    pid_file: Path = Path.home() / ".ace" / "logs" / "daemon.pid",
    timeout: float = 30.0,
) -> bool:
    """
    Stop running daemon.

    Args:
        pid_file: Path to PID file
        timeout: Max time to wait for graceful shutdown

    Returns:
        True if daemon was stopped, False if not running
    """
    pid_file = Path(pid_file).expanduser()
    pid = is_daemon_running(pid_file)

    if pid is None:
        return False

    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            if not is_daemon_running(pid_file):
                return True

        # Force kill if still running
        logger.warning(f"Daemon did not stop gracefully, sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
        return True

    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return True
