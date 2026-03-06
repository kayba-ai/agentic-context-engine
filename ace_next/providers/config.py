"""Model configuration — secrets-free config for ACE roles.

``ModelConfig`` describes which model to use and how. No API keys —
those come from the environment (via ``.env`` or exported variables).

``ACEModelConfig`` maps ACE roles (agent, reflector, skill_manager)
to individual ``ModelConfig`` instances, enabling per-role model selection.

Config is persisted in ``ace.toml`` (committable, no secrets).
Keys are persisted in ``.env`` (gitignored).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "ace.toml"
ENV_FILENAME = ".env"


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a single LLM role. No secrets."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    extra_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict, omitting None/default values."""
        d: dict[str, Any] = {"model": self.model}
        if self.temperature != 0.0:
            d["temperature"] = self.temperature
        if self.max_tokens != 2048:
            d["max_tokens"] = self.max_tokens
        if self.extra_params:
            d["extra_params"] = self.extra_params
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# ACEModelConfig
# ---------------------------------------------------------------------------


@dataclass
class ACEModelConfig:
    """Model selection per ACE role. No secrets — keys come from env."""

    default: ModelConfig
    agent: ModelConfig | None = None
    reflector: ModelConfig | None = None
    skill_manager: ModelConfig | None = None

    def for_role(self, role: str) -> ModelConfig:
        """Return the ModelConfig for *role*, falling back to default."""
        explicit = getattr(self, role, None)
        if explicit is not None:
            return explicit
        return self.default

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"default": self.default.to_dict()}
        for role in ("agent", "reflector", "skill_manager"):
            cfg = getattr(self, role)
            if cfg is not None:
                d[role] = cfg.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ACEModelConfig:
        default = ModelConfig.from_dict(d["default"])
        agent = ModelConfig.from_dict(d["agent"]) if "agent" in d else None
        reflector = ModelConfig.from_dict(d["reflector"]) if "reflector" in d else None
        skill_manager = (
            ModelConfig.from_dict(d["skill_manager"]) if "skill_manager" in d else None
        )
        return cls(
            default=default,
            agent=agent,
            reflector=reflector,
            skill_manager=skill_manager,
        )


# ---------------------------------------------------------------------------
# TOML persistence
# ---------------------------------------------------------------------------


def _to_toml(config: ACEModelConfig) -> str:
    """Serialise ACEModelConfig to TOML string."""
    lines: list[str] = []
    for section_name in ("default", "agent", "reflector", "skill_manager"):
        cfg = getattr(config, section_name)
        if cfg is None:
            continue
        lines.append(f"[{section_name}]")
        d = cfg.to_dict()
        for key, value in d.items():
            if key == "extra_params":
                # Inline table for extra_params
                inner = ", ".join(f'{k} = {_toml_value(v)}' for k, v in value.items())
                lines.append(f"extra_params = {{ {inner} }}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(v: Any) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return repr(v)


def save_config(config: ACEModelConfig, directory: str | Path = ".") -> Path:
    """Write ace.toml to *directory*."""
    path = Path(directory) / CONFIG_FILENAME
    path.write_text(_to_toml(config), encoding="utf-8")
    logger.info("Saved config to %s", path)
    return path


def load_config(directory: str | Path = ".") -> ACEModelConfig:
    """Load ace.toml from *directory*.

    Raises:
        FileNotFoundError: If ace.toml does not exist.
    """
    path = Path(directory) / CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME} found in {Path(directory).resolve()}. "
            "Run `ace setup` to create one."
        )
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ACEModelConfig.from_dict(data)


def find_config(start: str | Path = ".") -> Path | None:
    """Walk up from *start* looking for ace.toml. Return path or None."""
    current = Path(start).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------


def load_dotenv() -> bool:
    """Load .env if python-dotenv is installed. Return True if loaded."""
    try:
        from dotenv import load_dotenv as _load

        return _load()
    except ImportError:
        return False


def save_env_var(key: str, value: str, directory: str | Path = ".") -> None:
    """Append or update a key in .env file."""
    path = Path(directory) / ENV_FILENAME
    lines: list[str] = []
    found = False

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                lines.append(f'{key}="{value}"')
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f'{key}="{value}"')

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
