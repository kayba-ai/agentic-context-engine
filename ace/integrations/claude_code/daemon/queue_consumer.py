"""Queue consumer for processing ACE hook files.

Processes queue files through the existing ACEHookLearner infrastructure,
running learning in a thread pool for parallel processing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

from .watcher import QueueWatcher

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for queue processing."""

    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def __str__(self) -> str:
        return (
            f"ProcessingStats(processed={self.files_processed}, "
            f"failed={self.files_failed}, skipped={self.files_skipped}, "
            f"uptime={self.uptime_seconds:.0f}s)"
        )


class QueueConsumer:
    """
    Consumes hook queue files and processes them through ACE learning.

    Uses the existing ACEHookLearner infrastructure for actual learning,
    running multiple learners in parallel via ThreadPoolExecutor.

    ACE uses subscription-only mode via Claude CLI (no API keys required).

    Example:
        >>> consumer = QueueConsumer()
        >>> consumer.start()
        >>> # ... runs until stopped
        >>> consumer.stop()
    """

    def __init__(
        self,
        queue_dir: Optional[Path] = None,
        ace_model: str = "anthropic/claude-sonnet-4-5-20250929",
        use_cli: bool = True,  # Deprecated - always uses CLI
        max_workers: int = 2,
        on_error: Optional[Callable[[Exception, Path], None]] = None,
        on_complete: Optional[Callable[[Path, bool], None]] = None,
    ):
        """
        Initialize the queue consumer.

        ACE now uses subscription-only mode via Claude CLI. No API keys required.

        Args:
            queue_dir: Directory to watch for queue files (default: ~/.ace/queue)
            ace_model: Deprecated - ignored (CLI only)
            use_cli: Deprecated - always uses CLI subscription
            max_workers: Max concurrent learning tasks
            on_error: Callback for errors (exception, queue_file_path)
            on_complete: Callback on completion (queue_file_path, success)
        """
        self.queue_dir = (queue_dir or Path.home() / ".ace" / "queue").expanduser()
        self.ace_model = ace_model  # Kept for compatibility, but ignored
        self.max_workers = max_workers
        self.on_error = on_error
        self.on_complete = on_complete

        self._watcher: Optional[QueueWatcher] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._stats = ProcessingStats()
        self._futures: List[Future] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start consuming queue."""
        if self._running:
            logger.warning("QueueConsumer already running")
            return

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._stats = ProcessingStats()

        # Create thread pool for parallel processing
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ace-learner",
        )

        # Start file watcher
        self._watcher = QueueWatcher(
            queue_dir=self.queue_dir,
            on_new_file=self._on_queue_file,
        )
        self._watcher.start()

        logger.info(
            f"QueueConsumer started: {self.queue_dir} "
            f"(workers={self.max_workers}, mode=CLI subscription)"
        )

    def stop(self, wait: bool = True, timeout: float = 60.0) -> ProcessingStats:
        """
        Stop consuming queue.

        Args:
            wait: Wait for pending tasks to complete
            timeout: Max time to wait for pending tasks

        Returns:
            Final processing statistics
        """
        if not self._running:
            return self._stats

        self._running = False

        # Stop file watcher first
        if self._watcher:
            self._watcher.stop()

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)

        logger.info(f"QueueConsumer stopped: {self._stats}")
        return self._stats

    @property
    def stats(self) -> ProcessingStats:
        """Get processing statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running

    def _on_queue_file(self, queue_file: Path) -> None:
        """Handle a new queue file - submit to thread pool."""
        if not self._running:
            return

        if not self._executor:
            logger.error("Executor not initialized")
            return

        logger.debug(f"Submitting queue file for processing: {queue_file}")

        # Submit to thread pool
        future = self._executor.submit(self._process_queue_file, queue_file)

        with self._lock:
            self._futures.append(future)

    def _process_queue_file(self, queue_file: Path) -> None:
        """
        Process a single queue file through ACE learning.

        This runs in a worker thread.
        """
        logger.info(f"Processing: {queue_file.name}")

        try:
            # Parse hook input
            try:
                with queue_file.open("r", encoding="utf-8") as f:
                    hook_input = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in queue file: {e}")

            # Validate required fields
            cwd = hook_input.get("cwd")
            transcript_path = hook_input.get("transcript_path")

            if not cwd:
                raise ValueError("Missing 'cwd' in hook input")
            if not transcript_path:
                raise ValueError("Missing 'transcript_path' in hook input")

            # Run learning using existing infrastructure
            success = self._run_learning(hook_input)

            if success:
                self._stats.files_processed += 1
                logger.info(f"Processed successfully: {queue_file.name}")
            else:
                self._stats.files_skipped += 1
                logger.info(f"Skipped (trivial): {queue_file.name}")

            # Clean up queue file
            queue_file.unlink(missing_ok=True)

            if self.on_complete:
                self.on_complete(queue_file, success)

        except Exception as e:
            self._stats.files_failed += 1
            logger.error(f"Failed to process {queue_file}: {e}", exc_info=True)

            # Move to failed directory
            self._move_to_failed(queue_file)

            if self.on_error:
                self.on_error(e, queue_file)

    def _run_learning(self, hook_input: Dict[str, Any]) -> bool:
        """
        Run ACE learning using the existing ACEHookLearner.

        Uses subscription-only mode via Claude CLI.

        Returns True if learning was performed, False if skipped.
        """
        # Import here to avoid circular imports
        from ..hook import ACEHookLearner

        try:
            return ACEHookLearner.learn_from_hook_input(hook_input)
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            raise

    def _move_to_failed(self, queue_file: Path) -> None:
        """Move a failed queue file to the failed directory."""
        try:
            failed_dir = self.queue_dir / "failed"
            failed_dir.mkdir(exist_ok=True)

            dest = failed_dir / queue_file.name
            if queue_file.exists():
                queue_file.rename(dest)
                logger.debug(f"Moved to failed: {dest}")
        except Exception as e:
            logger.warning(f"Could not move to failed directory: {e}")
