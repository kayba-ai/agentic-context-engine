"""File system watcher for ACE queue directory.

Monitors ~/.ace/queue/ for new hook files and triggers processing callbacks.
Uses polling by default, with optional FSEvents support on macOS.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)


class QueueWatcher:
    """
    Watch queue directory for new hook files.

    Uses polling-based watching (portable across platforms).
    Can be extended with FSEvents on macOS for efficiency.

    Example:
        >>> def on_file(path: Path):
        ...     print(f"New file: {path}")
        >>> watcher = QueueWatcher(Path("~/.ace/queue"), on_file)
        >>> watcher.start()
        >>> # ... later
        >>> watcher.stop()
    """

    def __init__(
        self,
        queue_dir: Path,
        on_new_file: Callable[[Path], None],
        poll_interval: float = 0.5,  # seconds
    ):
        """
        Initialize the queue watcher.

        Args:
            queue_dir: Directory to watch for new files
            on_new_file: Callback function called with path when new file detected
            poll_interval: How often to check for new files (seconds)
        """
        self.queue_dir = queue_dir.expanduser()
        self.on_new_file = on_new_file
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._watcher_thread: Optional[threading.Thread] = None

        # Track processed files to avoid duplicates
        self._processed: Set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start watching for new files."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()

        # Process any existing files first
        self._scan_existing_files()

        # Start watcher thread
        self._watcher_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="ace-queue-watcher",
        )
        self._watcher_thread.start()
        logger.info(f"QueueWatcher started: {self.queue_dir}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching."""
        self._stop_event.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=timeout)
        logger.info("QueueWatcher stopped")

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return (
            self._watcher_thread is not None
            and self._watcher_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def _polling_loop(self) -> None:
        """Polling-based file watching."""
        while not self._stop_event.is_set():
            try:
                self._scan_for_new_files()
            except Exception as e:
                logger.warning(f"Polling scan error: {e}")

            self._stop_event.wait(self.poll_interval)

    def _scan_existing_files(self) -> None:
        """Process any files that exist before watcher started."""
        try:
            for path in sorted(self.queue_dir.glob("hook-*.json")):
                self._handle_file(path)
        except Exception as e:
            logger.error(f"Error scanning existing files: {e}")

    def _scan_for_new_files(self) -> None:
        """Scan for new files (polling mode)."""
        try:
            for path in self.queue_dir.glob("hook-*.json"):
                self._handle_file(path)
        except Exception as e:
            logger.warning(f"Error scanning for new files: {e}")

    def _handle_file(self, path: Path) -> None:
        """Handle a detected file, avoiding duplicates."""
        with self._lock:
            if path.name in self._processed:
                return
            self._processed.add(path.name)

        logger.debug(f"New queue file detected: {path.name}")

        try:
            self.on_new_file(path)
        except Exception as e:
            logger.error(f"Error in on_new_file callback for {path}: {e}")

    def mark_processed(self, filename: str) -> None:
        """Mark a file as processed (for external tracking)."""
        with self._lock:
            self._processed.add(filename)

    def clear_processed(self) -> None:
        """Clear the processed files set."""
        with self._lock:
            self._processed.clear()
