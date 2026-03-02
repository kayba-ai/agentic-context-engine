from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class MCPServerConfig(BaseSettings):
    """Configuration for the ACE MCP Server."""
    
    default_model: str = Field(default="gpt-4o-mini")
    safe_mode: bool = Field(default=False)
    max_samples_per_call: int = Field(default=25)
    max_prompt_chars: int = Field(default=100_000)
    session_ttl_seconds: int = Field(default=3600)
    allow_save_load: bool = Field(default=True)
    learn_timeout_seconds: int = Field(default=300)
    skillbook_root: str | None = Field(default=None)
    log_level: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_prefix="ACE_MCP_",
        case_sensitive=False,
    )
