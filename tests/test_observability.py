"""Tests for ace/observability/ — Opik integration and tracers."""

import os
import unittest
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestShouldSkipOpik(unittest.TestCase):
    """Tests for _should_skip_opik() env-var logic."""

    def _import_fresh(self):
        from ace.observability.opik_integration import _should_skip_opik

        return _should_skip_opik

    def test_disabled_true(self):
        fn = self._import_fresh()
        with patch.dict(os.environ, {"OPIK_DISABLED": "true"}, clear=False):
            self.assertTrue(fn())

    def test_disabled_1(self):
        fn = self._import_fresh()
        with patch.dict(os.environ, {"OPIK_DISABLED": "1"}, clear=False):
            self.assertTrue(fn())

    def test_disabled_yes(self):
        fn = self._import_fresh()
        with patch.dict(os.environ, {"OPIK_DISABLED": "yes"}, clear=False):
            self.assertTrue(fn())

    def test_enabled_false(self):
        fn = self._import_fresh()
        with patch.dict(os.environ, {"OPIK_ENABLED": "false"}, clear=False):
            self.assertTrue(fn())

    def test_enabled_0(self):
        fn = self._import_fresh()
        with patch.dict(os.environ, {"OPIK_ENABLED": "0"}, clear=False):
            self.assertTrue(fn())

    def test_no_env_vars_returns_false(self):
        fn = self._import_fresh()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPIK_DISABLED", "OPIK_ENABLED")
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(fn())


@pytest.mark.unit
class TestOpikIntegration(unittest.TestCase):
    """Tests for OpikIntegration class."""

    @patch("ace.observability.opik_integration.OPIK_AVAILABLE", False)
    def test_init_when_opik_unavailable(self):
        from ace.observability.opik_integration import OpikIntegration

        integration = OpikIntegration(enable_auto_config=False)
        self.assertFalse(integration.enabled)

    @patch("ace.observability.opik_integration.OPIK_AVAILABLE", False)
    def test_is_available_false_when_opik_missing(self):
        from ace.observability.opik_integration import OpikIntegration

        integration = OpikIntegration(enable_auto_config=False)
        self.assertFalse(integration.is_available())

    @patch("ace.observability.opik_integration.OPIK_AVAILABLE", False)
    def test_log_methods_noop_when_disabled(self):
        from ace.observability.opik_integration import OpikIntegration

        integration = OpikIntegration(enable_auto_config=False)
        # Should not raise even when Opik unavailable
        integration.log_skill_evolution("s1", "content", 1, 0, 0, "general")
        integration.log_skillbook_update("ADD")
        integration.log_role_performance("Agent", 1.0, True)
        integration.log_adaptation_metrics(1, 1, 0.5, 5, 3, 5)
        integration.create_experiment("test")


@pytest.mark.unit
class TestGetIntegration(unittest.TestCase):
    """Tests for get_integration() singleton."""

    def test_returns_opik_integration_instance(self):
        import ace.observability.opik_integration as mod

        # Reset global
        mod._global_integration = None
        with patch.object(mod, "OPIK_AVAILABLE", False):
            result = mod.get_integration()
            self.assertIsInstance(result, mod.OpikIntegration)

    def test_singleton_returns_same_instance(self):
        import ace.observability.opik_integration as mod

        mod._global_integration = None
        with patch.object(mod, "OPIK_AVAILABLE", False):
            a = mod.get_integration()
            b = mod.get_integration()
            self.assertIs(a, b)

    def test_get_integration_disabled_via_env(self):
        import ace.observability.opik_integration as mod

        mod._global_integration = None
        with patch.dict(os.environ, {"OPIK_DISABLED": "true"}, clear=False):
            result = mod.get_integration()
            self.assertFalse(result.enabled)
        mod._global_integration = None


@pytest.mark.unit
class TestConfigureOpik(unittest.TestCase):
    """Tests for configure_opik() global config."""

    def test_configure_sets_global(self):
        import ace.observability.opik_integration as mod

        mod._global_integration = None
        with patch.object(mod, "OPIK_AVAILABLE", False):
            result = mod.configure_opik(project_name="test-proj")
            self.assertIsInstance(result, mod.OpikIntegration)
        mod._global_integration = None

    def test_configure_when_disabled(self):
        import ace.observability.opik_integration as mod

        mod._global_integration = None
        with patch.dict(os.environ, {"OPIK_DISABLED": "true"}, clear=False):
            result = mod.configure_opik()
            self.assertFalse(result.enabled)
        mod._global_integration = None


@pytest.mark.unit
class TestSetupLitellmCallback(unittest.TestCase):
    """Tests for setup_litellm_callback() deduplication."""

    @patch("ace.observability.opik_integration.OPIK_AVAILABLE", False)
    def test_returns_false_when_opik_unavailable(self):
        from ace.observability.opik_integration import OpikIntegration

        integration = OpikIntegration(enable_auto_config=False)
        self.assertFalse(integration.setup_litellm_callback())


@pytest.mark.unit
class TestTracers(unittest.TestCase):
    """Tests for ace/observability/tracers.py — maybe_track and aliases."""

    @patch("ace.observability.tracers._OPIK_INSTALLED", False)
    def test_maybe_track_returns_undecorated_when_not_installed(self):
        from ace.observability.tracers import maybe_track

        def my_func():
            return 42

        decorated = maybe_track(name="test")(my_func)
        self.assertIs(decorated, my_func)

    @patch("ace.observability.tracers._OPIK_INSTALLED", True)
    @patch("ace.observability.opik_integration._should_skip_opik", return_value=True)
    def test_maybe_track_returns_undecorated_when_disabled(self, _mock_skip):
        from ace.observability.tracers import maybe_track

        def my_func():
            return 42

        decorated = maybe_track(name="test")(my_func)
        self.assertIs(decorated, my_func)

    def test_track_role_alias(self):
        from ace.observability.tracers import track_role, maybe_track

        # track_role should be a callable (alias of maybe_track)
        self.assertTrue(callable(track_role))

    def test_ace_track_alias(self):
        from ace.observability.tracers import ace_track, maybe_track

        self.assertTrue(callable(ace_track))


if __name__ == "__main__":
    unittest.main()
