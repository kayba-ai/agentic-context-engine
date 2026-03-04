from typing import Any
import asyncio
from pathlib import Path

from ace_next.integrations.mcp.registry import SessionRegistry
from ace_next.integrations.mcp.models import (
    AskRequest,
    AskResponse,
    LearnSampleRequest,
    LearnSampleResponse,
    LearnFeedbackRequest,
    LearnFeedbackResponse,
    SkillbookGetRequest,
    SkillbookGetResponse,
    SkillbookSaveRequest,
    SkillbookSaveResponse,
    SkillbookLoadRequest,
    SkillbookLoadResponse,
    SkillItem,
)
from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.errors import (
    ACEMCPError,
    ForbiddenInSafeModeError,
    InternalError,
    SaveLoadDisabledError,
    TimeoutError as MCPTimeoutError,
    ValidationError,
)
from ace_next.core.environments import Sample
from ace_next.core.skillbook import Skill


class MCPHandlers:
    def __init__(self, registry: SessionRegistry, config: MCPServerConfig):
        self.registry = registry
        self.config = config

    def _get_session_kwargs(self, config_model) -> tuple[str | None, dict[str, Any]]:
        target_model = None
        kwargs: dict[str, Any] = {}
        if config_model:
            target_model = config_model.model  # may be None per contract
            if config_model.temperature is not None:
                kwargs["temperature"] = config_model.temperature
            if config_model.max_tokens is not None:
                kwargs["max_tokens"] = config_model.max_tokens
        return target_model, kwargs

    def _enforce_prompt_limit(self, char_count: int, field_name: str) -> None:
        if char_count > self.config.max_prompt_chars:
            raise ValidationError(
                f"{field_name} exceeds max_prompt_chars ({self.config.max_prompt_chars})",
                details={
                    "field": field_name,
                    "char_count": char_count,
                    "max_prompt_chars": self.config.max_prompt_chars,
                },
            )

    def _resolve_skillbook_path(self, path: str) -> str:
        """Resolve a user-provided path and validate it against skillbook_root.

        Returns the resolved absolute path string so callers use the
        validated path — not the raw user input — for file operations,
        eliminating TOCTOU races with symlinks or ``..`` components.
        """
        resolved = str(Path(path).expanduser().resolve())

        if not self.config.skillbook_root:
            return resolved

        root = Path(self.config.skillbook_root).expanduser().resolve()
        try:
            Path(resolved).relative_to(root)
        except ValueError as exc:
            raise ValidationError(
                "Path is outside configured skillbook_root",
                details={
                    "path": resolved,
                    "skillbook_root": str(root),
                },
            ) from exc

        return resolved

    async def handle_ask(self, request: AskRequest) -> AskResponse:
        self._enforce_prompt_limit(len(request.question) + len(request.context), "ask")

        target_model, kwargs = self._get_session_kwargs(request.session_config)
        session = await self.registry.get_or_create(
            request.session_id, model=target_model, **kwargs
        )

        async with session.lock:
            try:
                answer = await asyncio.to_thread(
                    session.runner.ask, request.question, request.context
                )
                skill_count = len(session.runner.skillbook.skills())

                return AskResponse(
                    session_id=request.session_id,
                    answer=str(answer),
                    skill_count=skill_count,
                )
            except ACEMCPError:
                raise
            except Exception as e:
                raise InternalError(str(e))

    async def handle_skillbook_get(
        self, request: SkillbookGetRequest
    ) -> SkillbookGetResponse:
        session = await self.registry.get(request.session_id)

        async with session.lock:
            try:
                skillbook = session.runner.skillbook
                skills = skillbook.skills(include_invalid=request.include_invalid)

                limited_skills = []
                for s in skills:
                    if isinstance(s, Skill):
                        limited_skills.append(
                            SkillItem(
                                id=s.id,
                                content=s.content,
                                topic=s.section,
                                helpful=s.helpful,
                                harmful=s.harmful,
                                neutral=s.neutral,
                            )
                        )
                    else:
                        # Defensive fallback for non-standard skill objects
                        limited_skills.append(
                            SkillItem(
                                id=getattr(s, "id", str(len(limited_skills))),
                                content=getattr(s, "content", str(s)),
                                topic=getattr(s, "section", None),
                                helpful=getattr(s, "helpful", None),
                                harmful=getattr(s, "harmful", None),
                                neutral=getattr(s, "neutral", None),
                            )
                        )

                limited_skills = limited_skills[: request.limit]

                stats = skillbook.stats()

                return SkillbookGetResponse(
                    session_id=request.session_id, stats=stats, skills=limited_skills
                )
            except ACEMCPError:
                raise
            except Exception as e:
                raise InternalError(str(e))

    async def handle_learn_sample(
        self, request: LearnSampleRequest
    ) -> LearnSampleResponse:
        if self.config.safe_mode:
            raise ForbiddenInSafeModeError("ace.learn.sample")

        if len(request.samples) > self.config.max_samples_per_call:
            raise ValidationError(
                f"samples exceeds max_samples_per_call ({self.config.max_samples_per_call})",
                details={
                    "sample_count": len(request.samples),
                    "max_samples_per_call": self.config.max_samples_per_call,
                },
            )

        for idx, s in enumerate(request.samples):
            self._enforce_prompt_limit(
                len(s.question) + len(s.context),
                f"samples[{idx}]",
            )

        target_model, kwargs = self._get_session_kwargs(request.session_config)
        session = await self.registry.get_or_create(
            request.session_id, model=target_model, **kwargs
        )

        async with session.lock:
            try:
                samples = []
                for s in request.samples:
                    samples.append(
                        Sample(
                            question=s.question,
                            context=s.context,
                            ground_truth=s.ground_truth,
                            metadata=s.metadata or {},
                        )
                    )

                count_before = len(session.runner.skillbook.skills())

                results = await asyncio.wait_for(
                    asyncio.to_thread(
                        session.runner.learn,
                        samples,
                        None,
                        request.epochs,
                    ),
                    timeout=self.config.learn_timeout_seconds,
                )

                failed = sum(1 for r in results if r.error is not None)
                count_after = len(session.runner.skillbook.skills())

                return LearnSampleResponse(
                    session_id=request.session_id,
                    processed=len(samples) - failed,
                    failed=failed,
                    skill_count_before=count_before,
                    skill_count_after=count_after,
                    new_skill_count=max(0, count_after - count_before),
                )
            except ACEMCPError:
                raise
            except asyncio.TimeoutError:
                raise MCPTimeoutError(
                    f"learn.sample timed out after {self.config.learn_timeout_seconds}s"
                )
            except Exception as e:
                raise InternalError(str(e))

    async def handle_learn_feedback(
        self, request: LearnFeedbackRequest
    ) -> LearnFeedbackResponse:
        if self.config.safe_mode:
            raise ForbiddenInSafeModeError("ace.learn.feedback")

        self._enforce_prompt_limit(
            len(request.question)
            + len(request.context)
            + len(request.answer)
            + len(request.feedback)
            + len(request.ground_truth or ""),
            "learn.feedback",
        )

        target_model, kwargs = self._get_session_kwargs(request.session_config)
        session = await self.registry.get_or_create(
            request.session_id, model=target_model, **kwargs
        )

        async with session.lock:
            try:
                count_before = len(session.runner.skillbook.skills())

                # Prefer the direct feedback path when a prior ask exists;
                # fall back to learn_from_traces for standalone feedback.
                timeout = self.config.learn_timeout_seconds
                learned = await asyncio.wait_for(
                    asyncio.to_thread(
                        session.runner.learn_from_feedback,
                        request.feedback,
                        request.ground_truth or None,
                    ),
                    timeout=timeout,
                )

                if not learned:
                    # No prior ask interaction — build a trace and learn
                    trace: dict[str, object] = {
                        "question": request.question,
                        "context": request.context,
                        "answer": request.answer,
                        "skill_ids": [],
                        "feedback": request.feedback,
                        "ground_truth": request.ground_truth,
                    }
                    await asyncio.wait_for(
                        asyncio.to_thread(session.runner.learn_from_traces, [trace]),
                        timeout=timeout,
                    )

                count_after = len(session.runner.skillbook.skills())

                return LearnFeedbackResponse(
                    session_id=request.session_id,
                    learned=True,
                    skill_count_before=count_before,
                    skill_count_after=count_after,
                    new_skill_count=max(0, count_after - count_before),
                )
            except ACEMCPError:
                raise
            except asyncio.TimeoutError:
                raise MCPTimeoutError(
                    f"learn.feedback timed out after {self.config.learn_timeout_seconds}s"
                )
            except Exception as e:
                raise InternalError(str(e))

    async def handle_skillbook_save(
        self, request: SkillbookSaveRequest
    ) -> SkillbookSaveResponse:
        if self.config.safe_mode:
            raise ForbiddenInSafeModeError("ace.skillbook.save")
        if not self.config.allow_save_load:
            raise SaveLoadDisabledError("ace.skillbook.save")

        resolved = self._resolve_skillbook_path(request.path)

        session = await self.registry.get(request.session_id)

        async with session.lock:
            try:
                await asyncio.to_thread(session.runner.save, resolved)
                skill_count = len(session.runner.skillbook.skills())

                return SkillbookSaveResponse(
                    session_id=request.session_id,
                    path=resolved,
                    saved_skill_count=skill_count,
                )
            except ACEMCPError:
                raise
            except Exception as e:
                raise InternalError(str(e))

    async def handle_skillbook_load(
        self, request: SkillbookLoadRequest
    ) -> SkillbookLoadResponse:
        if self.config.safe_mode:
            raise ForbiddenInSafeModeError("ace.skillbook.load")
        if not self.config.allow_save_load:
            raise SaveLoadDisabledError("ace.skillbook.load")

        resolved = self._resolve_skillbook_path(request.path)

        session = await self.registry.get_or_create(request.session_id)

        async with session.lock:
            try:
                await asyncio.to_thread(session.runner.load, resolved)
                skill_count = len(session.runner.skillbook.skills())

                return SkillbookLoadResponse(
                    session_id=request.session_id,
                    path=resolved,
                    skill_count=skill_count,
                )
            except ACEMCPError:
                raise
            except Exception as e:
                raise InternalError(str(e))
