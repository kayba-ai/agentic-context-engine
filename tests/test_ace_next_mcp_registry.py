import pytest
import asyncio
from unittest.mock import MagicMock, patch
from ace_next.integrations.mcp.config import MCPServerConfig
from ace_next.integrations.mcp.registry import SessionRegistry
from ace_next.integrations.mcp.errors import SessionNotFoundError

@pytest.fixture
def config():
    return MCPServerConfig(session_ttl_seconds=1)

@pytest.fixture
def registry(config):
    return SessionRegistry(config)

@pytest.mark.asyncio
async def test_get_or_create(registry):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM") as mock_runner_cls:
        mock_runner_cls.from_model.return_value = MagicMock()
        
        # Create
        s1 = await registry.get_or_create("s1")
        assert s1.session_id == "s1"
        assert s1.runner is not None
        mock_runner_cls.from_model.assert_called_once_with("gpt-4o-mini")
        
        # Get existing
        s1_again = await registry.get_or_create("s1")
        assert s1 is s1_again
        assert mock_runner_cls.from_model.call_count == 1

@pytest.mark.asyncio
async def test_get_existing(registry):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        s1 = await registry.get_or_create("s1")
        s1_get = await registry.get("s1")
        assert s1 is s1_get

@pytest.mark.asyncio
async def test_get_not_found(registry):
    with pytest.raises(SessionNotFoundError):
        await registry.get("nonexistent")

@pytest.mark.asyncio
async def test_sweep_expired(registry):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        s1 = await registry.get_or_create("s1")
        
        # Should not expire immediately
        await registry.get("s1")
        
        # Wait for TTL to pass (config TTL is 1 sec)
        await asyncio.sleep(1.1)
        
        with pytest.raises(SessionNotFoundError):
            await registry.get("s1")

@pytest.mark.asyncio
async def test_delete(registry):
    with patch("ace_next.integrations.mcp.registry.ACELiteLLM"):
        await registry.get_or_create("s1")
        await registry.delete("s1")
        
        with pytest.raises(SessionNotFoundError):
            await registry.get("s1")
