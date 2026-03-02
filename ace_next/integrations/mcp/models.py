from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class SessionConfig(BaseModel):
    model: str | None = Field(default=None, min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    
    model_config = ConfigDict(extra="forbid")

class ErrorEnvelope(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    
    model_config = ConfigDict(extra="forbid")

class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=100000)
    context: str = Field(default="")
    session_config: SessionConfig | None = None
    metadata: dict[str, Any] | None = None
    
    model_config = ConfigDict(extra="forbid")

class AskResponse(BaseModel):
    session_id: str
    answer: str
    skill_count: int = Field(ge=0)
    applied_skill_ids: list[str] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

class SampleItem(BaseModel):
    question: str = Field(min_length=1)
    context: str = Field(default="")
    ground_truth: str | None = Field(default=None)
    metadata: dict[str, Any] | None = None
    
    model_config = ConfigDict(extra="forbid")

class LearnSampleRequest(BaseModel):
    session_id: str = Field(min_length=1)
    samples: list[SampleItem] = Field(min_length=1, max_length=25)
    epochs: int = Field(default=1, ge=1, le=20)
    session_config: SessionConfig | None = None
    
    model_config = ConfigDict(extra="forbid")

class LearnSampleResponse(BaseModel):
    session_id: str
    processed: int = Field(ge=0)
    failed: int = Field(default=0, ge=0)
    skill_count_before: int = Field(ge=0)
    skill_count_after: int = Field(ge=0)
    new_skill_count: int = Field(ge=0)
    
    model_config = ConfigDict(extra="forbid")

class LearnFeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    feedback: str = Field(min_length=1)
    context: str = Field(default="")
    ground_truth: str | None = Field(default=None)
    session_config: SessionConfig | None = None
    
    model_config = ConfigDict(extra="forbid")

class LearnFeedbackResponse(BaseModel):
    session_id: str
    learned: bool
    skill_count_before: int = Field(ge=0)
    skill_count_after: int = Field(ge=0)
    new_skill_count: int = Field(ge=0)
    
    model_config = ConfigDict(extra="forbid")

class SkillbookGetRequest(BaseModel):
    session_id: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=200)
    include_invalid: bool = Field(default=False)
    
    model_config = ConfigDict(extra="forbid")

class SkillItem(BaseModel):
    id: str
    content: str
    topic: str | None = None
    helpful: int | None = None
    harmful: int | None = None
    neutral: int | None = None
    
    model_config = ConfigDict(extra="allow")

class SkillbookGetResponse(BaseModel):
    session_id: str
    stats: dict[str, Any]
    skills: list[SkillItem]
    
    model_config = ConfigDict(extra="forbid")

class SkillbookSaveRequest(BaseModel):
    session_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    
    model_config = ConfigDict(extra="forbid")

class SkillbookSaveResponse(BaseModel):
    session_id: str
    path: str
    saved_skill_count: int = Field(ge=0)
    
    model_config = ConfigDict(extra="forbid")

class SkillbookLoadRequest(BaseModel):
    session_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    
    model_config = ConfigDict(extra="forbid")

class SkillbookLoadResponse(BaseModel):
    session_id: str
    path: str
    skill_count: int = Field(ge=0)
    
    model_config = ConfigDict(extra="forbid")
