"""Async learning infrastructure for ACE.

This module provides parallel Reflector execution with serialized SkillManager
for efficient learning without skillbook conflicts.

Architecture:
    Sample 1 ──► Agent ──► Env ──► Reflector ─┐
    Sample 2 ──► Agent ──► Env ──► Reflector ─┼──► [Queue] ──► SkillManager ──► Skillbook
    Sample 3 ──► Agent ──► Env ──► Reflector ─┘                (serialized)
               (parallel)        (parallel)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .adaptation import EnvironmentResult, Sample
    from .skillbook import Skillbook
    from .roles import SkillManager, AgentOutput, Reflector, ReflectorOutput

from .updates import UpdateBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LearningTask:
    """Input to Reflector (from main thread).

    Contains all data needed to run reflection on a sample's results.
    """

    sample: "Sample"
    agent_output: "AgentOutput"
    environment_result: "EnvironmentResult"
    epoch: int
    step_index: int
    total_epochs: int = 1
    total_steps: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """Output from Reflector (goes to SkillManager queue).

    Contains the original task plus the reflection analysis.
    """

    task: LearningTask
    reflection: "ReflectorOutput"
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Thread-Safe Skillbook Wrapper
# ---------------------------------------------------------------------------


class ThreadSafeSkillbook:
    """Thread-safe wrapper for Skillbook with RWLock semantics.

    Provides:
    - Lock-free reads for eventual consistency (Agent can read anytime)
    - Locked writes to ensure atomic skillbook updates (SkillManager serialized)

    Example:
        >>> ts_skillbook = ThreadSafeSkillbook(skillbook)
        >>> # Reads are lock-free
        >>> prompt = ts_skillbook.as_prompt()
        >>> # Writes are locked
        >>> ts_skillbook.apply_update(update_batch)
    """

    def __init__(self, skillbook: "Skillbook") -> None:
        self._skillbook = skillbook
        self._lock = threading.RLock()

    @property
    def skillbook(self) -> "Skillbook":
        """Direct access to underlying skillbook (for read operations)."""
        return self._skillbook

    # -----------------------------------------------------------------------
    # Lock-free reads (eventual consistency)
    # -----------------------------------------------------------------------

    def as_prompt(self) -> str:
        """Get TOON-encoded skillbook for LLM prompts (lock-free)."""
        return self._skillbook.as_prompt()

    def skills(self) -> List[Any]:
        """Get all skills (lock-free)."""
        return self._skillbook.skills()

    def get_skill(self, skill_id: str) -> Optional[Any]:
        """Get a skill by ID (lock-free)."""
        return self._skillbook.get_skill(skill_id)

    def stats(self) -> Dict[str, object]:
        """Get skillbook statistics (lock-free)."""
        return self._skillbook.stats()

    # -----------------------------------------------------------------------
    # Locked writes (serialized for thread safety)
    # -----------------------------------------------------------------------

    def apply_update(self, update: UpdateBatch) -> None:
        """Apply update operations to skillbook (thread-safe)."""
        with self._lock:
            self._skillbook.apply_update(update)

    def tag_skill(self, skill_id: str, tag: str, increment: int = 1) -> Optional[Any]:
        """Tag a skill (thread-safe)."""
        with self._lock:
            return self._skillbook.tag_skill(skill_id, tag, increment)

    def add_skill(
        self,
        section: str,
        content: str,
        skill_id: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Any:
        """Add a skill (thread-safe)."""
        with self._lock:
            return self._skillbook.add_skill(section, content, skill_id, metadata)

    def update_skill(
        self,
        skill_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, int]] = None,
    ) -> Optional[Any]:
        """Update a skill (thread-safe)."""
        with self._lock:
            return self._skillbook.update_skill(
                skill_id, content=content, metadata=metadata
            )

    def remove_skill(self, skill_id: str) -> None:
        """Remove a skill (thread-safe)."""
        with self._lock:
            self._skillbook.remove_skill(skill_id)


# ---------------------------------------------------------------------------
# Async Learning Pipeline
# ---------------------------------------------------------------------------


class AsyncLearningPipeline:
    """Parallel Reflectors + Serialized SkillManager pipeline.

    This class orchestrates async learning with:
    1. ThreadPoolExecutor for parallel Reflector.reflect() calls
    2. Single SkillManager thread processing a queue sequentially
    3. Thread-safe skillbook wrapper for safe concurrent access

    Flow:
        1. Main thread submits LearningTask via submit()
        2. ThreadPoolExecutor runs Reflector.reflect() in parallel
        3. ReflectionResult queued to SkillManager
        4. Single SkillManager thread processes queue sequentially

    Example:
        >>> pipeline = AsyncLearningPipeline(
        ...     skillbook=skillbook,
        ...     reflector=reflector,
        ...     skill_manager=skill_manager,
        ...     max_reflector_workers=3,
        ... )
        >>> pipeline.start()
        >>> pipeline.submit(task)  # Non-blocking
        >>> pipeline.wait_for_completion()
        >>> pipeline.stop()
    """

    def __init__(
        self,
        skillbook: "Skillbook",
        reflector: "Reflector",
        skill_manager: "SkillManager",
        *,
        max_reflector_workers: int = 3,
        skill_manager_queue_size: int = 100,
        max_refinement_rounds: int = 1,
        on_error: Optional[Callable[[Exception, LearningTask], None]] = None,
        on_complete: Optional[Callable[[LearningTask, Any], None]] = None,
    ) -> None:
        """Initialize the async learning pipeline.

        Args:
            skillbook: Skillbook instance to update
            reflector: Reflector instance for analysis
            skill_manager: SkillManager instance for skillbook updates
            max_reflector_workers: Max concurrent Reflector threads (default: 3)
            skill_manager_queue_size: Max pending SkillManager tasks (default: 100)
            max_refinement_rounds: Reflector refinement rounds (default: 1)
            on_error: Callback for task errors (task, exception)
            on_complete: Callback for task completion (task, skill_manager_output)
        """
        self._skillbook = ThreadSafeSkillbook(skillbook)
        self._reflector = reflector
        self._skill_manager = skill_manager
        self._max_reflector_workers = max_reflector_workers
        self._max_refinement_rounds = max_refinement_rounds
        self._on_error = on_error
        self._on_complete = on_complete

        # Thread pool for parallel Reflectors
        self._reflector_pool: Optional[ThreadPoolExecutor] = None

        # Queue for SkillManager (serialized processing)
        self._skill_manager_queue: Queue[ReflectionResult] = Queue(
            maxsize=skill_manager_queue_size
        )

        # SkillManager thread
        self._skill_manager_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Stats
        self._tasks_submitted = 0
        self._reflections_completed = 0
        self._skill_updates_completed = 0
        self._tasks_failed = 0
        self._stats_lock = threading.Lock()

        # Track pending futures for wait_for_completion
        self._pending_futures: List[Future] = []
        self._futures_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the async learning pipeline."""
        if self._reflector_pool is not None:
            logger.warning("AsyncLearningPipeline already started")
            return

        self._stop_event.clear()

        # Start thread pool for Reflectors
        self._reflector_pool = ThreadPoolExecutor(
            max_workers=self._max_reflector_workers,
            thread_name_prefix="ace-reflector",
        )

        # Start SkillManager thread
        self._skill_manager_thread = threading.Thread(
            target=self._skill_manager_loop,
            daemon=True,
            name="ace-skill-manager",
        )
        self._skill_manager_thread.start()

        logger.info(
            f"AsyncLearningPipeline started with {self._max_reflector_workers} "
            f"Reflector workers"
        )

    def stop(self, wait: bool = True, timeout: float = 30.0) -> int:
        """Stop the async learning pipeline.

        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Max seconds to wait for completion

        Returns:
            Number of tasks remaining in queues
        """
        if self._reflector_pool is None:
            return 0

        if wait:
            self.wait_for_completion(timeout=timeout)

        # Signal stop
        self._stop_event.set()

        # Shutdown thread pool
        self._reflector_pool.shutdown(wait=False)
        self._reflector_pool = None

        # Wait for SkillManager thread
        if self._skill_manager_thread and self._skill_manager_thread.is_alive():
            self._skill_manager_thread.join(timeout=min(timeout, 5.0))

        remaining = self._skill_manager_queue.qsize()
        logger.info(f"AsyncLearningPipeline stopped, {remaining} tasks remaining")
        return remaining

    def is_running(self) -> bool:
        """Check if the pipeline is running."""
        return self._reflector_pool is not None and not self._stop_event.is_set()

    # -----------------------------------------------------------------------
    # Task Submission
    # -----------------------------------------------------------------------

    def submit(self, task: LearningTask) -> Optional[Future]:
        """Submit a learning task (non-blocking).

        Args:
            task: LearningTask containing sample results to learn from

        Returns:
            Future for tracking completion, or None if pipeline not running
        """
        if self._reflector_pool is None:
            logger.warning("Cannot submit task: pipeline not started")
            return None

        with self._stats_lock:
            self._tasks_submitted += 1

        # Submit to thread pool for Reflector processing
        future = self._reflector_pool.submit(self._reflector_worker, task)

        # Track future for wait_for_completion
        with self._futures_lock:
            self._pending_futures.append(future)
            # Clean up completed futures
            self._pending_futures = [f for f in self._pending_futures if not f.done()]

        return future

    # -----------------------------------------------------------------------
    # Synchronization
    # -----------------------------------------------------------------------

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all pending learning tasks to complete.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        start_time = time.time()

        # Wait for all Reflector futures to complete
        with self._futures_lock:
            pending = list(self._pending_futures)

        for future in pending:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                if remaining_timeout <= 0:
                    return False
            try:
                future.result(timeout=remaining_timeout)
            except Exception:
                pass  # Errors already handled in worker

        # Wait for SkillManager queue to drain
        try:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)

            # Poll queue until empty or timeout
            poll_start = time.time()
            while not self._skill_manager_queue.empty():
                if remaining_timeout is not None:
                    if time.time() - poll_start > remaining_timeout:
                        return False
                time.sleep(0.1)

            return True
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        with self._stats_lock:
            return {
                "tasks_submitted": self._tasks_submitted,
                "reflections_completed": self._reflections_completed,
                "skill_updates_completed": self._skill_updates_completed,
                "tasks_failed": self._tasks_failed,
                "skill_manager_queue_size": self._skill_manager_queue.qsize(),
                "is_running": self.is_running(),
            }

    # -----------------------------------------------------------------------
    # Internal Workers
    # -----------------------------------------------------------------------

    def _reflector_worker(self, task: LearningTask) -> Optional[ReflectionResult]:
        """Run Reflector.reflect() - executes in thread pool.

        Can have multiple instances running concurrently.
        """
        try:
            # Run reflection (safe to parallelize - reads only)
            reflection = self._reflector.reflect(
                question=task.sample.question,
                agent_output=task.agent_output,
                skillbook=self._skillbook.skillbook,  # Read-only access
                ground_truth=task.environment_result.ground_truth,
                feedback=task.environment_result.feedback,
                max_refinement_rounds=self._max_refinement_rounds,
            )

            # Create result
            result = ReflectionResult(task=task, reflection=reflection)

            # Queue for SkillManager
            try:
                self._skill_manager_queue.put(result, timeout=10.0)
            except Full:
                logger.warning(
                    f"SkillManager queue full, dropping reflection for sample "
                    f"{task.step_index}"
                )
                with self._stats_lock:
                    self._tasks_failed += 1
                return None

            with self._stats_lock:
                self._reflections_completed += 1

            return result

        except Exception as e:
            logger.warning(f"Reflector failed for sample {task.step_index}: {e}")
            with self._stats_lock:
                self._tasks_failed += 1

            if self._on_error:
                try:
                    self._on_error(e, task)
                except Exception:
                    pass  # Don't let callback errors propagate

            return None

    def _skill_manager_loop(self) -> None:
        """Single SkillManager thread - processes ReflectionResults sequentially.

        Only one instance runs at a time to serialize skillbook updates.
        """
        while not self._stop_event.is_set():
            try:
                # Block with timeout to check stop_event periodically
                result = self._skill_manager_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                self._process_skill_update(result)
            except Exception as e:
                logger.warning(
                    f"SkillManager failed for sample {result.task.step_index}: {e}"
                )
                with self._stats_lock:
                    self._tasks_failed += 1

                if self._on_error:
                    try:
                        self._on_error(e, result.task)
                    except Exception:
                        pass
            finally:
                self._skill_manager_queue.task_done()

    def _process_skill_update(self, result: ReflectionResult) -> None:
        """Process a single reflection result through SkillManager.

        Runs in the single SkillManager thread - serialized execution.
        """
        task = result.task
        reflection = result.reflection

        # Apply skill tags (thread-safe)
        for tag in reflection.skill_tags:
            try:
                self._skillbook.tag_skill(tag.id, tag.tag)
            except ValueError:
                continue  # Skill not found, skip

        # Build question context
        question_context = self._build_question_context(task)
        progress = self._build_progress_string(task)

        # Run SkillManager (sees latest skillbook state)
        skill_manager_output = self._skill_manager.update_skills(
            reflection=reflection,
            skillbook=self._skillbook.skillbook,  # Pass underlying skillbook
            question_context=question_context,
            progress=progress,
        )

        # Apply update (thread-safe)
        self._skillbook.apply_update(skill_manager_output.update)

        with self._stats_lock:
            self._skill_updates_completed += 1

        # Completion callback
        if self._on_complete:
            try:
                self._on_complete(task, skill_manager_output)
            except Exception:
                pass  # Don't let callback errors propagate

    def _build_question_context(self, task: LearningTask) -> str:
        """Build question context string for SkillManager."""
        parts = [
            f"question: {task.sample.question}",
            f"context: {task.sample.context}",
            f"metadata: {json.dumps(task.sample.metadata)}",
            f"feedback: {task.environment_result.feedback}",
            f"ground_truth: {task.environment_result.ground_truth}",
        ]
        return "\n".join(parts)

    def _build_progress_string(self, task: LearningTask) -> str:
        """Build progress string for SkillManager."""
        return (
            f"epoch {task.epoch}/{task.total_epochs} · "
            f"sample {task.step_index}/{task.total_steps}"
        )
