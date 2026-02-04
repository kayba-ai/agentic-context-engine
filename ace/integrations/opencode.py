"""
OpenCode integration for ACE framework.

This module provides ACEOpenCode, a wrapper for OpenCode CLI
that automatically learns from execution feedback via the OpenCode server API.

Example:
    from ace.integrations import ACEOpenCode

    agent = ACEOpenCode(working_dir="./my_project")
    result = agent.run(task="Refactor the auth module")
    agent.save_skillbook("learned.json")

    # With async learning
    agent = ACEOpenCode(working_dir="./project", async_learning=True)
    result = agent.run(task="Task 1")  # Returns immediately
    agent.wait_for_learning()  # Wait for learning to complete
"""

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import requests

from ..llm_providers import LiteLLMClient
from ..prompts_v2_1 import PromptManager
from ..roles import AgentOutput, Reflector, SkillManager
from ..skillbook import Skillbook
from .base import wrap_skillbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig, DeduplicationManager


# Availability flag - requests is required for OpenCode integration
OPENCODE_AVAILABLE = True


@dataclass
class OpenCodeResult:
    """Result from OpenCode execution."""

    success: bool
    output: str
    execution_trace: str
    session_id: str
    error: Optional[str] = None


class OpenCodeClient:
    """
    HTTP client for OpenCode server API.

    Handles communication with OpenCode's headless server including:
    - Session management (create, list, get)
    - Prompt execution (sync and async)
    - Message retrieval
    - Event streaming (SSE)
    """

    DEFAULT_SERVER_URL = "http://127.0.0.1:4096"

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout: int = 300):
        """
        Initialize OpenCode client.

        Args:
            server_url: URL of OpenCode server (default: http://127.0.0.1:4096)
            timeout: Request timeout in seconds (default: 300)
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def check_health(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Health info dict with 'healthy' and 'version' keys

        Raises:
            requests.ConnectionError: If server is unreachable
        """
        resp = self._session.get(
            f"{self.server_url}/global/health", timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def create_session(
        self,
        working_directory: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new OpenCode session.

        Args:
            working_directory: Working directory for the session
            title: Optional session title

        Returns:
            Session info dict with 'id', 'slug', 'directory', etc.

        Raises:
            requests.HTTPError: If session creation fails
        """
        payload: Dict[str, Any] = {"workingDirectory": working_directory}
        if title:
            payload["title"] = title

        resp = self._session.post(
            f"{self.server_url}/session", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Session info dict

        Raises:
            requests.HTTPError: If session not found
        """
        resp = self._session.get(
            f"{self.server_url}/session/{session_id}", timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts

        Raises:
            requests.HTTPError: If listing fails
        """
        resp = self._session.get(f"{self.server_url}/session", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_session_status(self) -> Dict[str, Any]:
        """
        Get status of all sessions.

        Returns:
            Dict mapping session IDs to status info

        Raises:
            requests.HTTPError: If request fails
        """
        resp = self._session.get(
            f"{self.server_url}/session/status", timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def send_prompt(
        self,
        session_id: str,
        parts: List[Dict[str, str]],
        async_mode: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a prompt to a session.

        Args:
            session_id: Session identifier
            parts: List of message parts (e.g., [{"type": "text", "text": "..."}])
            async_mode: If True, use prompt_async (returns 204). If False, wait for response.

        Returns:
            Response dict if async_mode=False, None if async_mode=True

        Raises:
            requests.HTTPError: If prompt fails
        """
        endpoint = (
            f"{self.server_url}/session/{session_id}/prompt_async"
            if async_mode
            else f"{self.server_url}/session/{session_id}/prompt"
        )

        payload = {"parts": parts}
        resp = self._session.post(endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()

        if async_mode:
            return None
        return resp.json()

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a session.

        Args:
            session_id: Session identifier
            limit: Optional maximum number of messages to return

        Returns:
            List of message dict with info and parts

        Raises:
            requests.HTTPError: If retrieval fails
        """
        params = {}
        if limit:
            params["limit"] = limit

        resp = self._session.get(
            f"{self.server_url}/session/{session_id}/message",
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_message(self, session_id: str, message_id: str) -> Dict[str, Any]:
        """
        Get a specific message.

        Args:
            session_id: Session identifier
            message_id: Message identifier

        Returns:
            Message dict with info and parts

        Raises:
            requests.HTTPError: If message not found
        """
        resp = self._session.get(
            f"{self.server_url}/session/{session_id}/message/{message_id}",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def stream_events(
        self, timeout_seconds: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream SSE events from the global event endpoint.

        Args:
            timeout_seconds: Optional timeout for streaming. If None, streams indefinitely.

        Yields:
            Event dicts as they arrive

        Raises:
            requests.ConnectionError: If server disconnects
        """
        try:
            response = self._session.get(
                f"{self.server_url}/event",
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=10,  # Connection timeout only
            )
            response.raise_for_status()

            start_time = time.time()

            while True:
                # Check timeout if specified
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    break

                try:
                    for line in response.iter_lines(chunk_size=1, decode_unicode=False):
                        if (
                            timeout_seconds
                            and (time.time() - start_time) > timeout_seconds
                        ):
                            break

                        if line:
                            line_text = line.decode("utf-8")
                            if line_text.startswith("data: "):
                                data = line_text[6:]
                                try:
                                    event = json.loads(data)
                                    yield event
                                except json.JSONDecodeError:
                                    continue
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                ):
                    # Timeout or disconnection - check if we should continue
                    if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                        break
                    # Otherwise retry
                    response.close()
                    response = self._session.get(
                        f"{self.server_url}/event",
                        stream=True,
                        headers={"Accept": "text/event-stream"},
                        timeout=10,
                    )
                    response.raise_for_status()
        finally:
            if "response" in locals():
                response.close()

    def close(self):
        """Close the HTTP session."""
        self._session.close()


class ACEOpenCode:
    """
    OpenCode with ACE learning capabilities.

    Executes tasks via OpenCode server and learns from execution.
    Drop-in wrapper that automatically:
    - Injects learned strategies into prompts
    - Reflects on execution results
    - Updates skillbook with new learnings

    Usage:
        # Simple usage
        agent = ACEOpenCode(working_dir="./project")
        result = agent.run(task="Add unit tests for utils.py")

        # Reuse across tasks (learns from each)
        agent = ACEOpenCode(working_dir="./project")
        agent.run(task="Task 1")
        agent.run(task="Task 2")  # Uses Task 1 learnings
        agent.save_skillbook("expert.json")

        # Start with existing knowledge
        agent = ACEOpenCode(
            working_dir="./project",
            skillbook_path="expert.json"
        )
        agent.run(task="New task")
    """

    def __init__(
        self,
        working_dir: str,
        server_url: str = OpenCodeClient.DEFAULT_SERVER_URL,
        ace_model: str = "gpt-4o-mini",
        ace_llm: Optional[LiteLLMClient] = None,
        ace_max_tokens: int = 2048,
        skillbook: Optional[Skillbook] = None,
        skillbook_path: Optional[str] = None,
        is_learning: bool = True,
        timeout: int = 600,
        async_learning: bool = False,
        max_reflector_workers: int = 3,
        dedup_config: Optional["DeduplicationConfig"] = None,
    ):
        """
        Initialize ACEOpenCode.

        Args:
            working_dir: Directory where OpenCode will execute
            server_url: URL of OpenCode server (default: http://127.0.0.1:4096)
            ace_model: Model for ACE learning (Reflector/SkillManager)
            ace_llm: Custom LLM client for ACE (overrides ace_model)
            ace_max_tokens: Max tokens for ACE learning LLM
            skillbook: Existing Skillbook instance
            skillbook_path: Path to load skillbook from
            is_learning: Enable/disable ACE learning
            timeout: Execution timeout in seconds (default: 600)
            async_learning: Run learning in background (default: False)
            max_reflector_workers: Parallel Reflector threads (default: 3)
            dedup_config: Optional DeduplicationConfig for skill deduplication
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.is_learning = is_learning
        self.timeout = timeout
        self.async_learning = async_learning
        self.max_reflector_workers = max_reflector_workers
        self.dedup_config = dedup_config

        # Create OpenCode client
        self.client = OpenCodeClient(server_url=server_url, timeout=timeout)

        # Load or create skillbook
        if skillbook_path:
            self.skillbook = Skillbook.load_from_file(skillbook_path)
        elif skillbook:
            self.skillbook = skillbook
        else:
            self.skillbook = Skillbook()

        # Create ACE LLM (for Reflector/SkillManager)
        self.ace_llm = ace_llm or LiteLLMClient(
            model=ace_model, max_tokens=ace_max_tokens
        )

        # Create ACE learning components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.skill_manager = SkillManager(
            self.ace_llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

        # Initialize deduplication manager if config provided
        self._dedup_manager: Optional["DeduplicationManager"] = None
        if dedup_config:
            from ..deduplication import DeduplicationManager

            self._dedup_manager = DeduplicationManager(dedup_config)

        # Async learning state
        self._learning_queue: queue.Queue = queue.Queue()
        self._learning_thread: Optional[threading.Thread] = None
        self._stop_learning = threading.Event()
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._lock = threading.Lock()

        # Session management
        self._session_id: Optional[str] = None

        # Start async learning thread if enabled
        if async_learning:
            self._start_async_learning()

    def run(self, task: str, context: str = "") -> OpenCodeResult:
        """
        Execute task via OpenCode with ACE learning.

        Args:
            task: Task description for OpenCode
            context: Additional context (optional)

        Returns:
            OpenCodeResult with execution details
        """
        # 1. INJECT: Add skillbook context if learning enabled and has skills
        if self.is_learning and self.skillbook.skills():
            skillbook_context = wrap_skillbook_context(self.skillbook)
            prompt = (
                f"{task}\n\n{context}\n\n{skillbook_context}"
                if context
                else f"{task}\n\n{skillbook_context}"
            )
        else:
            prompt = f"{task}\n\n{context}" if context else task

        # 2. EXECUTE: Run OpenCode
        result = self._execute_opencode(prompt)

        # 3. LEARN: Run ACE learning if enabled
        if self.is_learning:
            if self.async_learning:
                # Queue learning task for background processing
                with self._lock:
                    self._tasks_submitted += 1
                self._learning_queue.put((task, result))
            else:
                # Synchronous learning
                self._learn_from_execution(task, result)

        return result

    def _execute_opencode(self, prompt: str) -> OpenCodeResult:
        """Execute OpenCode via server API and capture events."""
        try:
            # Create or reuse session
            if not self._session_id:
                session_info = self.client.create_session(
                    working_directory=str(self.working_dir)
                )
                self._session_id = session_info.get("id")

            if not self._session_id:
                raise RuntimeError("Failed to create session")

            # Collect events during execution
            execution_trace = self._capture_execution_trace(prompt)

            # Get final messages for summary
            messages = self.client.get_messages(self._session_id, limit=10)
            output = self._extract_summary(messages)

            return OpenCodeResult(
                success=True,
                output=output,
                execution_trace=execution_trace,
                session_id=self._session_id,
            )

        except Exception as e:
            return OpenCodeResult(
                success=False,
                output="",
                execution_trace="",
                session_id=self._session_id or "",
                error=str(e),
            )

    def _capture_execution_trace(self, prompt: str) -> str:
        """Capture and parse execution trace from SSE events."""
        trace_parts = []
        step_num = 0
        seen_assistant_start = False

        # Track cumulative reasoning
        current_reasoning = ""
        last_reasoning_length = 0

        if not self._session_id:
            return "(No session available)"

        # Send async prompt
        self.client.send_prompt(
            session_id=self._session_id,
            parts=[{"type": "text", "text": prompt}],
            async_mode=True,
        )

        # Listen to events for execution trace
        start_time = time.time()
        for event in self.client.stream_events(timeout_seconds=self.timeout - 10):
            event_type = event.get("type")
            properties = event.get("properties", {})

            # Less restrictive session filtering
            event_session = properties.get("info", {}).get("sessionID")
            if event_session and event_session != self._session_id:
                continue

            if event_type == "message.updated":
                role = properties.get("info", {}).get("role")
                if role == "assistant":
                    seen_assistant_start = True

            elif event_type == "message.part.updated":
                part = properties.get("part", {})
                part_type = part.get("type")
                part_session = part.get("sessionID")

                # Only skip if DEFINITELY from another session
                if part_session and part_session != self._session_id:
                    continue

                # Handle reasoning (incremental updates)
                if part_type == "reasoning" and part.get("text"):
                    text = part.get("text", "")
                    if len(text) > last_reasoning_length:
                        new_text = text[last_reasoning_length:]
                        current_reasoning += new_text
                        last_reasoning_length = len(text)

                # NEW: Handle OpenCode's "tool" events
                elif part_type == "tool":
                    state = part.get("state", {})
                    status = state.get("status")
                    tool_name = part.get("tool", "unknown")

                    # Only capture when tool completes
                    if status == "completed":
                        step_num += 1
                        tool_input = state.get("input", {})

                        # Format based on tool type
                        if tool_name == "read":
                            file_path = tool_input.get("filePath", "")
                            trace_parts.append(f"[Step {step_num}] Read: {file_path}")
                        elif tool_name == "write":
                            file_path = tool_input.get("filePath", "")
                            trace_parts.append(f"[Step {step_num}] Write: {file_path}")
                        elif tool_name == "bash":
                            cmd = tool_input.get("command", "")[:80]
                            trace_parts.append(f"[Step {step_num}] Bash: {cmd}")
                        elif tool_name == "glob":
                            pattern = tool_input.get("pattern", "")
                            trace_parts.append(f"[Step {step_num}] Glob: {pattern}")
                        else:
                            trace_parts.append(f"[Step {step_num}] {tool_name}")

                        # Optionally capture output
                        output = state.get("output", "")
                        if output and len(output) < 200:
                            trace_parts.append(f"[Result] {output[:200]}")

                    # Capture errors
                    elif status == "error":
                        error = state.get("error", "")
                        if error:
                            trace_parts.append(f"[Error] {error[:200]}")

            elif event_type == "session.status":
                status = properties.get("status", {}).get("type")
                if status == "idle" and seen_assistant_start:
                    # Session completed - add final reasoning
                    if current_reasoning.strip():
                        sentences = current_reasoning.split(". ")
                        if len(sentences) > 3:
                            summary = ". ".join(sentences[:2] + [sentences[-1]])
                            trace_parts.insert(0, f"[Reasoning] {summary}")
                        else:
                            trace_parts.insert(0, f"[Reasoning] {current_reasoning}")
                    break

            # Safety timeout
            if time.time() - start_time > self.timeout - 5:
                break

        return "\n".join(trace_parts) if trace_parts else "(No trace captured)"

    def _extract_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Extract final summary from messages."""
        if not messages:
            return "(No messages)"

        # Get the last assistant message
        assistant_messages = [
            m for m in messages if m.get("info", {}).get("role") == "assistant"
        ]

        if not assistant_messages:
            return f"Completed {len(messages)} messages"

        # Extract text parts
        parts = assistant_messages[-1].get("parts", [])
        text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]

        if text_parts:
            return text_parts[-1][:500]

        # Fall back to session title
        last_message = messages[-1]
        summary = last_message.get("info", {}).get("summary", {})
        title = summary.get("title", "")
        return title if title else f"Completed {len(messages)} messages"

    def _learn_from_execution(self, task: str, result: OpenCodeResult):
        """Run ACE learning pipeline after execution."""
        # Create AgentOutput for Reflector
        agent_output = AgentOutput(
            reasoning=result.execution_trace,
            final_answer=result.output,
            skill_ids=[],  # External agents don't pre-select skills
            raw={
                "success": result.success,
                "session_id": result.session_id,
            },
        )

        # Build feedback
        status = "succeeded" if result.success else "failed"
        feedback = f"OpenCode task {status}"
        if result.error:
            feedback += f"\nError: {result.error}"

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Run SkillManager
        skill_manager_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"task: {task}",
            progress=f"OpenCode: {task}",
        )

        # Update skillbook
        self.skillbook.apply_update(skill_manager_output.update)

        # Apply consolidation operations if deduplication enabled
        if self._dedup_manager and skill_manager_output.raw:
            self._dedup_manager.apply_operations_from_response(
                skill_manager_output.raw, self.skillbook
            )

    def save_skillbook(self, path: str):
        """Save learned skillbook to file."""
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str):
        """Load skillbook from file."""
        self.skillbook = Skillbook.load_from_file(path)

    def get_strategies(self) -> str:
        """Get current skillbook strategies as formatted text."""
        if not self.skillbook.skills():
            return ""
        return wrap_skillbook_context(self.skillbook)

    def enable_learning(self):
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable ACE learning (execution only)."""
        self.is_learning = False

    def _start_async_learning(self):
        """Start the background learning thread."""
        if self._learning_thread is not None and self._learning_thread.is_alive():
            return

        self._stop_learning.clear()
        self._learning_thread = threading.Thread(
            target=self._learning_worker, daemon=True
        )
        self._learning_thread.start()

    def _learning_worker(self):
        """Background worker that processes learning tasks."""
        while not self._stop_learning.is_set():
            try:
                # Wait for a task with timeout to allow checking stop flag
                task, result = self._learning_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._learn_from_execution(task, result)
            finally:
                with self._lock:
                    self._tasks_completed += 1
                self._learning_queue.task_done()

    def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for async learning to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if all learning completed, False if timeout reached
        """
        if not self.async_learning:
            return True

        try:
            if timeout is not None:
                start = time.time()
                while not self._learning_queue.empty():
                    elapsed = time.time() - start
                    if elapsed >= timeout:
                        return False
                    time.sleep(0.1)
                return True
            else:
                self._learning_queue.join()
                return True
        except Exception:
            return False

    def stop_async_learning(self, wait: bool = True):
        """
        Stop async learning pipeline.

        Args:
            wait: If True, wait for current tasks to complete (default: True)
        """
        if not self.async_learning:
            return

        if wait:
            self.wait_for_learning()

        self._stop_learning.set()
        if self._learning_thread and self._learning_thread.is_alive():
            self._learning_thread.join(timeout=5.0)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """
        Get async learning statistics.

        Returns:
            Dictionary with learning progress info
        """
        with self._lock:
            submitted = self._tasks_submitted
            completed = self._tasks_completed

        return {
            "async_learning": self.async_learning,
            "tasks_submitted": submitted,
            "tasks_completed": completed,
            "pending": submitted - completed,
            "queue_size": self._learning_queue.qsize(),
        }

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_async_learning(wait=False)
            self.client.close()
        except Exception:
            pass


__all__ = ["OpenCodeClient", "OpenCodeResult", "ACEOpenCode"]
