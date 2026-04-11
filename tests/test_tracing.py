"""Tests for the ace.tracing wrapper."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestConfigure:
    """Tests for ace.tracing.configure()."""

    def test_configure_sets_tracking_uri_and_token(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            from ace.tracing._wrapper import configure

            with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
                configure(api_key="kb-test-key")

            mock_mlflow.set_tracking_uri.assert_called_once_with(
                "https://use.kayba.ai/api/mlflow"
            )
            assert os.environ["MLFLOW_TRACKING_TOKEN"] == "kb-test-key"

    def test_configure_custom_base_url(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            from ace.tracing._wrapper import configure

            with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
                configure(
                    api_key="kb-test-key",
                    base_url="https://custom.example.com",
                )

            mock_mlflow.set_tracking_uri.assert_called_once_with(
                "https://custom.example.com/api/mlflow"
            )

    def test_configure_strips_trailing_slash(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            from ace.tracing._wrapper import configure

            with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
                configure(
                    api_key="kb-test-key",
                    base_url="https://custom.example.com/",
                )

            mock_mlflow.set_tracking_uri.assert_called_once_with(
                "https://custom.example.com/api/mlflow"
            )

    def test_configure_reads_api_key_from_env(self) -> None:
        with patch.dict(
            os.environ, {"KAYBA_API_KEY": "kb-env-key"}, clear=False
        ):
            from ace.tracing._wrapper import configure

            with patch("ace.tracing._wrapper.mlflow"):
                configure()

            assert os.environ["MLFLOW_TRACKING_TOKEN"] == "kb-env-key"

    def test_configure_reads_base_url_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KAYBA_API_KEY": "kb-key",
                "KAYBA_API_URL": "https://env.example.com",
            },
            clear=False,
        ):
            from ace.tracing._wrapper import configure

            with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
                configure()

            mock_mlflow.set_tracking_uri.assert_called_once_with(
                "https://env.example.com/api/mlflow"
            )

    def test_configure_raises_without_api_key(self) -> None:
        with patch.dict(os.environ, {"KAYBA_API_KEY": ""}, clear=False):
            from ace.tracing._wrapper import configure

            with pytest.raises(ValueError, match="No API key provided"):
                configure()

    def test_experiment_is_alias_for_folder(self) -> None:
        import ace.tracing._wrapper as w

        with patch.dict(os.environ, {}, clear=False):
            with patch("ace.tracing._wrapper.mlflow"):
                w.configure(api_key="kb-key", experiment="my-project")

            assert w._folder == "my-project"

    def test_folder_takes_precedence_over_experiment(self) -> None:
        import ace.tracing._wrapper as w

        with patch.dict(os.environ, {}, clear=False):
            with patch("ace.tracing._wrapper.mlflow"):
                w.configure(
                    api_key="kb-key",
                    experiment="from-experiment",
                    folder="from-folder",
                )

            assert w._folder == "from-folder"

    def test_configure_sets_folder(self) -> None:
        import ace.tracing._wrapper as w

        with patch.dict(os.environ, {}, clear=False):
            with patch("ace.tracing._wrapper.mlflow"):
                w.configure(api_key="kb-key", folder="my-folder")

            assert w._folder == "my-folder"

    def test_configure_clears_folder_when_none(self) -> None:
        import ace.tracing._wrapper as w

        with patch.dict(os.environ, {}, clear=False):
            with patch("ace.tracing._wrapper.mlflow"):
                w.configure(api_key="kb-key", folder="old")
                w.configure(api_key="kb-key")

            assert w._folder is None


@pytest.mark.unit
class TestSanitizeFolder:
    """Tests for folder name sanitization."""

    def test_strips_html_tags(self) -> None:
        from ace.tracing._wrapper import _sanitize_folder

        assert _sanitize_folder('<script>alert("xss")</script>') == "alertxss"

    def test_strips_control_characters(self) -> None:
        from ace.tracing._wrapper import _sanitize_folder

        assert _sanitize_folder("folder\x00\x1f\nname") == "foldername"

    def test_allows_safe_characters(self) -> None:
        from ace.tracing._wrapper import _sanitize_folder

        assert _sanitize_folder("my-folder/sub_dir 2.0") == "my-folder/sub_dir 2.0"

    def test_truncates_long_names(self) -> None:
        from ace.tracing._wrapper import _sanitize_folder

        assert len(_sanitize_folder("a" * 500)) == 256

    def test_strips_sql_injection_chars(self) -> None:
        from ace.tracing._wrapper import _sanitize_folder

        assert _sanitize_folder("folder'; DROP TABLE--") == "folder DROP TABLE--"

    def test_configure_sanitizes_folder(self) -> None:
        import ace.tracing._wrapper as w

        with patch.dict(os.environ, {}, clear=False):
            with patch("ace.tracing._wrapper.mlflow"):
                w.configure(api_key="kb-key", folder='<img onerror="xss">')

            # Entire input is an HTML tag, stripped to empty string
            assert w._folder is None

    def test_set_folder_sanitizes(self) -> None:
        import ace.tracing._wrapper as w

        w.set_folder("<b>bold</b>")
        assert w.get_folder() == "bold"


@pytest.mark.unit
class TestFolder:
    """Tests for set_folder / get_folder."""

    def test_set_and_get_folder(self) -> None:
        import ace.tracing._wrapper as w

        w.set_folder("production")
        assert w.get_folder() == "production"

    def test_clear_folder(self) -> None:
        import ace.tracing._wrapper as w

        w.set_folder("production")
        w.set_folder(None)
        assert w.get_folder() is None

    def test_inject_folder_tag(self) -> None:
        import ace.tracing._wrapper as w

        w._folder = "my-folder"
        with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
            w._inject_folder_tag()
            mock_mlflow.update_current_trace.assert_called_once_with(
                tags={"kayba.folder": "my-folder"}
            )

    def test_inject_folder_tag_noop_when_none(self) -> None:
        import ace.tracing._wrapper as w

        w._folder = None
        with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
            w._inject_folder_tag()
            mock_mlflow.update_current_trace.assert_not_called()


@pytest.mark.unit
class TestTraceDecorator:
    """Tests for the trace decorator wrapper."""

    def test_trace_wraps_function(self) -> None:
        import ace.tracing._wrapper as w

        w._folder = "test-folder"

        with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
            # Make mlflow.trace return a passthrough decorator
            mock_mlflow.trace.side_effect = lambda fn=None, **kw: (
                fn if fn is not None else (lambda f: f)
            )

            @w.trace
            def my_func(x: int) -> int:
                return x + 1

            result = my_func(5)
            assert result == 6
            mock_mlflow.update_current_trace.assert_called_with(
                tags={"kayba.folder": "test-folder"}
            )

    def test_trace_with_params(self) -> None:
        import ace.tracing._wrapper as w

        w._folder = None

        with patch("ace.tracing._wrapper.mlflow") as mock_mlflow:
            mock_mlflow.trace.return_value = lambda fn: fn

            @w.trace(name="custom", span_type="LLM")
            def my_func() -> str:
                return "ok"

            result = my_func()
            assert result == "ok"
            mock_mlflow.trace.assert_called_once_with(
                name="custom", span_type="LLM"
            )
            # No folder set, so no tag injection
            mock_mlflow.update_current_trace.assert_not_called()


@pytest.mark.unit
class TestReExports:
    """Verify utility re-exports."""

    def test_enable_calls_mlflow(self) -> None:
        from ace.tracing._wrapper import enable

        with patch("ace.tracing._wrapper.mlflow.tracing.enable") as mock:
            enable()
            mock.assert_called_once()

    def test_disable_calls_mlflow(self) -> None:
        from ace.tracing._wrapper import disable

        with patch("ace.tracing._wrapper.mlflow.tracing.disable") as mock:
            disable()
            mock.assert_called_once()


@pytest.mark.unit
class TestPackageInit:
    """Verify the public __init__ exports."""

    def test_all_exports(self) -> None:
        import ace.tracing

        expected = {
            "configure",
            "disable",
            "enable",
            "get_folder",
            "get_trace",
            "search_traces",
            "set_folder",
            "start_span",
            "trace",
        }
        assert set(ace.tracing.__all__) == expected
