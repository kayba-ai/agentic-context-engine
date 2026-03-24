"""Shared pytest fixtures for test suite."""

import pytest


@pytest.fixture(autouse=True)
def _suppress_opik(monkeypatch):
    """Disable Opik connections during tests to avoid connection noise."""
    monkeypatch.setenv("OPIK_DISABLED", "true")


# Test markers configuration
pytest_configure_done = False


def pytest_configure(config):
    """Register custom markers."""
    global pytest_configure_done
    if not pytest_configure_done:
        config.addinivalue_line(
            "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
        )
        config.addinivalue_line(
            "markers", "integration: marks tests as integration tests"
        )
        config.addinivalue_line("markers", "unit: marks tests as unit tests")
        config.addinivalue_line(
            "markers",
            "requires_api: marks tests requiring external API keys (skipped in CI)",
        )
        pytest_configure_done = True
