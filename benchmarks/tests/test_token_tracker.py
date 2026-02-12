"""Tests for token tracking."""

import pytest
from unittest.mock import MagicMock

# Import after setting up path
import sys
from pathlib import Path

# Add benchmarks to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import TokenTracker, TokenTrackingContext


class TestTokenTracker:
    """Tests for TokenTracker dataclass."""

    def test_initial_state(self):
        """Test tracker starts with zero values."""
        tracker = TokenTracker()
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0
        assert tracker.total_cost == 0.0
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0

    def test_reset(self):
        """Test reset clears all counters."""
        tracker = TokenTracker()
        tracker.prompt_tokens = 100
        tracker.completion_tokens = 50
        tracker.total_tokens = 150
        tracker.call_count = 5
        tracker.total_cost = 0.01

        tracker.reset()

        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0
        assert tracker.total_cost == 0.0

    def test_update_with_usage(self):
        """Test update with response containing usage data."""
        tracker = TokenTracker()

        # Create mock response
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response._hidden_params = {"response_cost": 0.001}

        tracker.update(mock_response)

        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 150
        assert tracker.call_count == 1
        assert tracker.total_cost == 0.001

    def test_update_without_usage(self):
        """Test update with response missing usage data."""
        tracker = TokenTracker()

        mock_response = MagicMock()
        mock_response.usage = None
        del mock_response._hidden_params  # Remove attribute

        tracker.update(mock_response)

        assert tracker.call_count == 1
        assert tracker.total_tokens == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tracker = TokenTracker()
        tracker.prompt_tokens = 100
        tracker.completion_tokens = 50
        tracker.total_tokens = 150
        tracker.call_count = 2
        tracker.total_cost = 0.0012345

        result = tracker.to_dict()

        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["call_count"] == 2
        assert result["total_cost"] == 0.001234  # Rounded to 6 decimal places

    def test_multiple_updates(self):
        """Test accumulation across multiple updates."""
        tracker = TokenTracker()

        for i in range(3):
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response._hidden_params = {"response_cost": 0.001}
            tracker.update(mock_response)

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 150
        assert tracker.total_tokens == 450
        assert tracker.call_count == 3
        assert tracker.total_cost == 0.003


class TestTokenTrackingContext:
    """Tests for TokenTrackingContext context manager."""

    def test_context_manager_returns_tracker(self):
        """Test context manager returns a tracker instance."""
        import litellm

        with TokenTrackingContext() as tracker:
            assert isinstance(tracker, TokenTracker)

    def test_context_manager_registers_callback(self):
        """Test callback is registered on entry."""
        import litellm

        initial_count = len(litellm.success_callback)

        with TokenTrackingContext() as _tracker:
            assert len(litellm.success_callback) == initial_count + 1

    def test_context_manager_cleanup(self):
        """Test callback is removed on exit."""
        import litellm

        initial_count = len(litellm.success_callback)

        with TokenTrackingContext() as _tracker:
            pass  # Exit context

        assert len(litellm.success_callback) == initial_count

    def test_context_manager_cleanup_on_exception(self):
        """Test callback is removed even when exception occurs."""
        import litellm

        initial_count = len(litellm.success_callback)

        try:
            with TokenTrackingContext() as _tracker:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert len(litellm.success_callback) == initial_count
