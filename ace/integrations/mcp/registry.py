import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any

from ace.runners import ACELiteLLM
from ace.integrations.mcp.config import MCPServerConfig
from ace.integrations.mcp.errors import SessionNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    runner: ACELiteLLM
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionRegistry:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._sessions: Dict[str, Session] = {}
        self._registry_lock = asyncio.Lock()

    async def get_or_create(
        self, session_id: str, model: str | None = None, **runner_kwargs: Any
    ) -> Session:
        """Get an existing session or create a new one, sweeping expired sessions first."""
        async with self._registry_lock:
            expired = self._collect_expired()

            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = time.time()
                result = session
            else:
                # Create new runner
                target_model = model or self.config.default_model
                runner = ACELiteLLM.from_model(target_model, **runner_kwargs)
                session = Session(session_id=session_id, runner=runner)
                self._sessions[session_id] = session
                result = session

        # Drain outside the lock so other callers aren't blocked
        self._drain_sessions(expired)
        return result

    async def get(self, session_id: str) -> Session:
        """Get an existing session. Raises SessionNotFoundError if not found."""
        async with self._registry_lock:
            expired = self._collect_expired()

            if session_id not in self._sessions:
                # Drain before raising so we don't leak
                self._drain_sessions(expired)
                raise SessionNotFoundError(session_id)

            session = self._sessions[session_id]
            session.last_accessed = time.time()

        self._drain_sessions(expired)
        return session

    async def delete(self, session_id: str) -> None:
        """Delete a session if it exists."""
        async with self._registry_lock:
            session = self._sessions.pop(session_id, None)

        if session is not None:
            self._drain_sessions([session])

    def _collect_expired(self) -> list[Session]:
        """Remove and return sessions that have exceeded the TTL.

        Must be called while holding ``_registry_lock``.
        """
        now = time.time()
        ttl = self.config.session_ttl_seconds

        expired_ids = [
            sid
            for sid, session in self._sessions.items()
            if now - session.last_accessed > ttl
        ]
        return [self._sessions.pop(sid) for sid in expired_ids]

    @staticmethod
    def _drain_sessions(sessions: list[Session]) -> None:
        """Best-effort wait for any in-progress background learning."""
        for session in sessions:
            try:
                session.runner.wait_for_background(timeout=2.0)
            except Exception:
                logger.debug(
                    "Failed to drain session %s", session.session_id, exc_info=True
                )
