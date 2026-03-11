"""Tests for Kayba CLI commands and KaybaClient."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from ace.cli import cli
from ace.cli.client import KaybaClient, KaybaAPIError

# ---------------------------------------------------------------------------
# KaybaClient
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKaybaClient(unittest.TestCase):
    """Test KaybaClient initialization and request handling."""

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KAYBA_API_KEY", None)
            with self.assertRaises(KaybaAPIError) as ctx:
                KaybaClient()
            self.assertEqual(ctx.exception.code, "AUTH_MISSING")

    def test_init_with_explicit_key(self):
        client = KaybaClient(api_key="test-key")
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.base_url, "https://use.kayba.ai/api")
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-key")

    def test_init_with_env_key(self):
        with patch.dict(os.environ, {"KAYBA_API_KEY": "env-key"}):
            client = KaybaClient()
            self.assertEqual(client.api_key, "env-key")

    def test_custom_base_url(self):
        client = KaybaClient(api_key="k", base_url="http://localhost:3000/api/")
        self.assertEqual(client.base_url, "http://localhost:3000/api")

    def test_env_base_url(self):
        with patch.dict(
            os.environ,
            {"KAYBA_API_KEY": "k", "KAYBA_API_URL": "http://local/api"},
        ):
            client = KaybaClient()
            self.assertEqual(client.base_url, "http://local/api")

    @patch("ace.cli.client.requests.Session")
    def test_request_parses_structured_error(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {
            "error": {"code": "VALIDATION_ERROR", "message": "Bad input"}
        }
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        with self.assertRaises(KaybaAPIError) as ctx:
            client.upload_traces([])
        self.assertEqual(ctx.exception.code, "VALIDATION_ERROR")
        self.assertEqual(ctx.exception.message, "Bad input")
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("ace.cli.client.requests.Session")
    def test_request_success(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"traces": [], "count": 0}
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        result = client.upload_traces([])
        self.assertEqual(result, {"traces": [], "count": 0})

    @patch("ace.cli.client.requests.Session")
    def test_upload_traces_sends_correct_payload(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"traces": [], "count": 1}
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        traces = [{"filename": "a.md", "content": "hello", "fileType": "md"}]
        client.upload_traces(traces)

        session.request.assert_called_once_with(
            "POST",
            "https://use.kayba.ai/api/traces",
            json={"traces": traces},
            params=None,
        )

    @patch("ace.cli.client.requests.Session")
    def test_generate_insights_params(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 202
        resp.json.return_value = {"jobId": "job-1", "status": "pending"}
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        client.generate_insights(
            trace_ids=["t1"],
            model="claude-sonnet-4-6",
            epochs=2,
            reflector_mode="recursive",
            anthropic_key="ak",
        )

        session.request.assert_called_once_with(
            "POST",
            "https://use.kayba.ai/api/insights/generate",
            json={
                "traceIds": ["t1"],
                "model": "claude-sonnet-4-6",
                "epochs": 2,
                "reflectorMode": "recursive",
                "anthropicApiKey": "ak",
            },
            params=None,
        )

    @patch("ace.cli.client.requests.Session")
    def test_triage_insight(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "ins-1", "status": "accepted"}
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        client.triage_insight("ins-1", "accepted", note="looks good")

        session.request.assert_called_once_with(
            "PATCH",
            "https://use.kayba.ai/api/insights/ins-1",
            json={"status": "accepted", "note": "looks good"},
            params=None,
        )

    @patch("ace.cli.client.requests.Session")
    def test_list_insights_with_filters(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"insights": []}
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        client.list_insights(status="pending", section="errors")

        session.request.assert_called_once_with(
            "GET",
            "https://use.kayba.ai/api/insights",
            json=None,
            params={"status": "pending", "section": "errors"},
        )

    @patch("ace.cli.client.requests.Session")
    def test_materialize_job(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "jobId": "job-1",
            "message": "ok",
            "skillsGenerated": 5,
        }
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        result = client.materialize_job("job-1")
        self.assertEqual(result["skillsGenerated"], 5)

    @patch("ace.cli.client.requests.Session")
    def test_generate_prompt(self, MockSession):
        session = MockSession.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "promptId": "prompt-1",
            "version": 1,
            "content": {"text": "prompt text", "sectionMapping": {}},
            "label": "my-label",
        }
        session.request.return_value = resp

        client = KaybaClient(api_key="k")
        client.session = session

        client.generate_prompt(insight_ids=["i1", "i2"], label="my-label")

        session.request.assert_called_once_with(
            "POST",
            "https://use.kayba.ai/api/prompts/generate",
            json={"insightIds": ["i1", "i2"], "label": "my-label"},
            params=None,
        )


# ---------------------------------------------------------------------------
# Upload command
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUploadCommand(unittest.TestCase):
    """Test kayba upload command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("ace.cli.cloud.KaybaClient")
    def test_upload_file(self, MockClient):
        mock = MockClient.return_value
        mock.upload_traces.return_value = {
            "traces": [{"id": "conv-1", "filename": "test.md"}],
            "count": 1,
        }

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# trace content")
            f.flush()
            result = self.runner.invoke(cli, ["upload", f.name, "--api-key", "k"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Uploaded 1 trace(s)", result.output)
        os.unlink(f.name)

    @patch("ace.cli.cloud.KaybaClient")
    def test_upload_detects_json_type(self, MockClient):
        mock = MockClient.return_value
        mock.upload_traces.return_value = {"traces": [], "count": 1}

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            f.flush()
            self.runner.invoke(cli, ["upload", f.name, "--api-key", "k"])

        call_args = mock.upload_traces.call_args[0][0]
        self.assertEqual(call_args[0]["fileType"], "json")
        os.unlink(f.name)

    @patch("ace.cli.cloud.KaybaClient")
    def test_upload_forced_type(self, MockClient):
        mock = MockClient.return_value
        mock.upload_traces.return_value = {"traces": [], "count": 1}

        with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as f:
            f.write("log data")
            f.flush()
            self.runner.invoke(
                cli,
                ["upload", f.name, "--type", "md", "--api-key", "k"],
            )

        call_args = mock.upload_traces.call_args[0][0]
        self.assertEqual(call_args[0]["fileType"], "md")
        os.unlink(f.name)

    @patch("ace.cli.cloud.KaybaClient")
    def test_upload_directory(self, MockClient):
        mock = MockClient.return_value
        mock.upload_traces.return_value = {"traces": [], "count": 2}

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("alpha")
            (Path(d) / "b.txt").write_text("beta")
            result = self.runner.invoke(cli, ["upload", d, "--api-key", "k"])

        self.assertEqual(result.exit_code, 0, result.output)
        call_args = mock.upload_traces.call_args[0][0]
        self.assertEqual(len(call_args), 2)

    @patch("ace.cli.cloud.KaybaClient")
    def test_upload_stdin(self, MockClient):
        mock = MockClient.return_value
        mock.upload_traces.return_value = {"traces": [], "count": 1}

        result = self.runner.invoke(
            cli, ["upload", "-", "--api-key", "k"], input="stdin data"
        )

        self.assertEqual(result.exit_code, 0, result.output)
        call_args = mock.upload_traces.call_args[0][0]
        self.assertEqual(call_args[0]["filename"], "stdin.txt")
        self.assertEqual(call_args[0]["content"], "stdin data")

    def test_upload_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KAYBA_API_KEY", None)
            result = self.runner.invoke(cli, ["upload", "foo.md"])
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Insights commands
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInsightsCommands(unittest.TestCase):
    """Test kayba insights subcommands."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("ace.cli.cloud.KaybaClient")
    def test_generate_basic(self, MockClient):
        mock = MockClient.return_value
        mock.generate_insights.return_value = {
            "jobId": "job-1",
            "status": "pending",
        }

        result = self.runner.invoke(
            cli,
            ["insights", "generate", "--api-key", "k"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("job-1", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_generate_with_all_options(self, MockClient):
        mock = MockClient.return_value
        mock.generate_insights.return_value = {
            "jobId": "job-2",
            "status": "pending",
        }

        result = self.runner.invoke(
            cli,
            [
                "insights",
                "generate",
                "--traces",
                "t1",
                "--traces",
                "t2",
                "--model",
                "claude-opus-4-6",
                "--epochs",
                "3",
                "--reflector-mode",
                "recursive",
                "--anthropic-key",
                "ak",
                "--api-key",
                "k",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock.generate_insights.assert_called_once_with(
            trace_ids=["t1", "t2"],
            model="claude-opus-4-6",
            epochs=3,
            reflector_mode="recursive",
            anthropic_key="ak",
        )

    @patch("ace.cli.cloud.KaybaClient")
    def test_list_insights(self, MockClient):
        mock = MockClient.return_value
        mock.list_insights.return_value = {
            "insights": [
                {
                    "id": "ins-1",
                    "section": "errors",
                    "content": "insight text",
                    "status": "pending",
                }
            ]
        }

        result = self.runner.invoke(
            cli,
            ["insights", "list", "--status", "pending", "--api-key", "k"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("ins-1", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_list_insights_json(self, MockClient):
        mock = MockClient.return_value
        mock.list_insights.return_value = {
            "insights": [{"id": "ins-1", "status": "pending"}]
        }

        result = self.runner.invoke(
            cli,
            ["insights", "list", "--json", "--api-key", "k"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.output)
        self.assertEqual(parsed[0]["id"], "ins-1")

    @patch("ace.cli.cloud.KaybaClient")
    def test_triage_accept(self, MockClient):
        mock = MockClient.return_value
        mock.triage_insight.return_value = {"id": "ins-1", "status": "accepted"}

        result = self.runner.invoke(
            cli,
            [
                "insights",
                "triage",
                "--accept",
                "ins-1",
                "--note",
                "lgtm",
                "--api-key",
                "k",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock.triage_insight.assert_called_once_with("ins-1", "accepted", note="lgtm")

    @patch("ace.cli.cloud.KaybaClient")
    def test_triage_accept_all(self, MockClient):
        mock = MockClient.return_value
        mock.list_insights.return_value = {
            "insights": [{"id": "ins-1"}, {"id": "ins-2"}]
        }
        mock.triage_insight.return_value = {"id": "x", "status": "accepted"}

        result = self.runner.invoke(
            cli,
            ["insights", "triage", "--accept-all", "--api-key", "k"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock.triage_insight.call_count, 2)

    @patch("ace.cli.cloud.KaybaClient")
    def test_triage_no_args_errors(self, MockClient):
        MockClient.return_value
        result = self.runner.invoke(
            cli,
            ["insights", "triage", "--api-key", "k"],
        )
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# Job commands
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJobCommands(unittest.TestCase):
    """Test kayba status and materialize commands."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("ace.cli.cloud.KaybaClient")
    def test_status(self, MockClient):
        mock = MockClient.return_value
        mock.get_job.return_value = {
            "jobId": "job-1",
            "status": "completed",
            "startedAt": "2025-01-01T00:00:00Z",
            "completedAt": "2025-01-01T00:05:00Z",
            "result": {
                "skillsGenerated": 3,
                "materialized": False,
            },
        }

        result = self.runner.invoke(cli, ["status", "job-1", "--api-key", "k"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("completed", result.output)
        self.assertIn("3", result.output)

    @patch("ace.cli.cloud.time.sleep", return_value=None)
    @patch("ace.cli.cloud.KaybaClient")
    def test_status_wait_polls(self, MockClient, mock_sleep):
        mock = MockClient.return_value
        mock.get_job.side_effect = [
            {"jobId": "job-1", "status": "running"},
            {
                "jobId": "job-1",
                "status": "completed",
                "result": {"skillsGenerated": 2, "materialized": False},
            },
        ]

        result = self.runner.invoke(
            cli, ["status", "job-1", "--wait", "--api-key", "k"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock.get_job.call_count, 2)
        self.assertIn("materialize", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_materialize(self, MockClient):
        mock = MockClient.return_value
        mock.materialize_job.return_value = {
            "jobId": "job-1",
            "message": "ok",
            "skillsGenerated": 5,
        }

        result = self.runner.invoke(cli, ["materialize", "job-1", "--api-key", "k"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("5 skill(s)", result.output)


# ---------------------------------------------------------------------------
# Prompt commands
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptCommands(unittest.TestCase):
    """Test kayba prompts subcommands."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("ace.cli.cloud.KaybaClient")
    def test_generate_prompt(self, MockClient):
        mock = MockClient.return_value
        mock.generate_prompt.return_value = {
            "promptId": "prompt-1",
            "version": 1,
            "content": {"text": "the prompt", "sectionMapping": {}},
            "label": "v1",
        }

        result = self.runner.invoke(
            cli,
            [
                "prompts",
                "generate",
                "--insights",
                "i1",
                "--label",
                "v1",
                "--api-key",
                "k",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("prompt-1", result.output)
        self.assertIn("the prompt", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_generate_prompt_to_file(self, MockClient):
        mock = MockClient.return_value
        mock.generate_prompt.return_value = {
            "promptId": "prompt-1",
            "version": 1,
            "content": {"text": "saved prompt", "sectionMapping": {}},
            "label": "v1",
        }

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            out_path = f.name

        result = self.runner.invoke(
            cli,
            [
                "prompts",
                "generate",
                "--label",
                "v1",
                "-o",
                out_path,
                "--api-key",
                "k",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(Path(out_path).read_text(), "saved prompt")
        os.unlink(out_path)

    @patch("ace.cli.cloud.KaybaClient")
    def test_list_prompts(self, MockClient):
        mock = MockClient.return_value
        mock.list_prompts.return_value = {
            "prompts": [
                {"id": "p1", "label": "first"},
                {"id": "p2", "label": "second"},
            ]
        }

        result = self.runner.invoke(cli, ["prompts", "list", "--api-key", "k"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("p1", result.output)
        self.assertIn("p2", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_pull_by_id(self, MockClient):
        mock = MockClient.return_value
        mock.get_prompt.return_value = {
            "id": "p1",
            "content": {"text": "pulled text", "sectionMapping": {}},
        }

        result = self.runner.invoke(
            cli,
            ["prompts", "pull", "--id", "p1", "--api-key", "k"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("pulled text", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_pull_latest(self, MockClient):
        mock = MockClient.return_value
        mock.list_prompts.return_value = {
            "prompts": [{"id": "p-latest", "label": "latest"}]
        }
        mock.get_prompt.return_value = {
            "id": "p-latest",
            "content": {"text": "latest prompt", "sectionMapping": {}},
        }

        result = self.runner.invoke(cli, ["prompts", "pull", "--api-key", "k"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("latest prompt", result.output)

    @patch("ace.cli.cloud.KaybaClient")
    def test_pull_pretty_json(self, MockClient):
        mock = MockClient.return_value
        mock.get_prompt.return_value = {
            "id": "p1",
            "content": {"text": "txt", "sectionMapping": {"a": ["s1"]}},
        }

        result = self.runner.invoke(
            cli,
            ["prompts", "pull", "--id", "p1", "--pretty", "--api-key", "k"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["id"], "p1")


# ---------------------------------------------------------------------------
# Batch command
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBatchCommand(unittest.TestCase):
    """Test kayba batch command (prepare/apply modes) and helpers."""

    def setUp(self):
        self.runner = CliRunner()

    # -- helper: _extract_trace_metadata --

    def test_batch_metadata_extraction_json(self):
        from ace.cli.cloud import _extract_trace_metadata

        content = json.dumps(
            {
                "task_id": "t-1",
                "user_request": "Do something important",
                "tools": ["search", "write"],
                "steps": [1, 2, 3],
            }
        )
        meta = _extract_trace_metadata("trace.json", content, "json")
        self.assertEqual(meta["filename"], "trace.json")
        self.assertEqual(meta["type"], "json")
        self.assertEqual(meta["task_id"], "t-1")
        self.assertEqual(meta["user_request"], "Do something important")
        self.assertEqual(meta["tools"], ["search", "write"])
        self.assertEqual(meta["step_count"], 3)

    def test_batch_metadata_extraction_markdown(self):
        from ace.cli.cloud import _extract_trace_metadata

        content = "# Title\n## Section 1\nSome text\n## Section 2\nMore text"
        meta = _extract_trace_metadata("trace.md", content, "md")
        self.assertIn("summary", meta)
        self.assertIn("headings", meta)
        self.assertIn("# Title", meta["headings"])
        self.assertIn("## Section 1", meta["headings"])

    # -- helper: _build_classification_prompt --

    def test_batch_prompt_default(self):
        from ace.cli.cloud import _build_classification_prompt

        traces = [{"filename": "a.json", "type": "json", "size": 100}]
        prompt = _build_classification_prompt(
            traces, "min_batch_size=10, max_batch_size=30"
        )
        self.assertIn("min_batch_size=10", prompt)
        self.assertIn("max_batch_size=30", prompt)
        self.assertIn("a.json", prompt)
        self.assertIn("trace classification system", prompt)

    def test_batch_prompt_custom_file(self):
        from ace.cli.cloud import _build_classification_prompt

        custom = "Custom prompt: {constraints}\nTraces: {traces_json}"
        traces = [{"filename": "b.md"}]
        prompt = _build_classification_prompt(
            traces, "min=5, max=20", custom_prompt=custom
        )
        self.assertIn("Custom prompt:", prompt)
        self.assertIn("min=5, max=20", prompt)
        self.assertIn("b.md", prompt)

    # -- helper: _validate_batch_plan --

    def test_batch_validation_all_assigned(self):
        from ace.cli.cloud import _validate_batch_plan

        plan = {
            "batches": {
                "group-a": {"description": "A", "trace_files": ["a.json"]},
            }
        }
        errors = _validate_batch_plan(plan, ["a.json", "b.json"], 1, 30)
        self.assertTrue(any("not assigned" in e.lower() for e in errors))

    def test_batch_validation_size_constraints(self):
        from ace.cli.cloud import _validate_batch_plan

        plan = {
            "batches": {
                "big": {
                    "description": "Big batch",
                    "trace_files": ["a.json", "b.json", "c.json"],
                },
            }
        }
        errors = _validate_batch_plan(
            plan, ["a.json", "b.json", "c.json"], min_size=1, max_size=2
        )
        self.assertTrue(any("max 2" in e for e in errors))

    # -- CLI: no paths errors --

    def test_batch_no_paths_errors(self):
        result = self.runner.invoke(cli, ["batch", "--api-key", "k"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Provide at least one path", result.output)

    # -- CLI: prepare mode outputs prompt to stdout --

    def test_batch_prepare_outputs_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("# Trace A\nSome content")
            out = os.path.join(d, "out.json")
            result = self.runner.invoke(
                cli,
                ["batch", d, "-o", out, "--min-batch-size", "1"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            # Prompt printed to stdout
            self.assertIn("trace classification system", result.output)
            self.assertIn("a.md", result.output)
            # Starter file written
            written = json.loads(Path(out).read_text())
            self.assertIn("batches", written)
            self.assertEqual(written["summary"]["total_traces"], 1)

    # -- CLI: prepare mode with custom prompt file --

    def test_batch_prepare_custom_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("trace data")
            prompt_path = os.path.join(d, "prompt.txt")
            Path(prompt_path).write_text("Custom: {constraints}\nData: {traces_json}")
            out = os.path.join(d, "out.json")
            result = self.runner.invoke(
                cli,
                [
                    "batch",
                    d,
                    "--prompt",
                    prompt_path,
                    "-o",
                    out,
                    "--min-batch-size",
                    "1",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Custom:", result.output)

    # -- CLI: apply mode validates --

    def test_batch_apply_validates(self):
        with (
            tempfile.TemporaryDirectory() as traces_dir,
            tempfile.TemporaryDirectory() as plan_dir,
        ):
            (Path(traces_dir) / "a.md").write_text("data")
            plan_path = os.path.join(plan_dir, "plan.json")
            # Plan references b.md which doesn't exist → validation error
            plan = {
                "batches": {
                    "group": {"description": "g", "trace_files": ["a.md", "b.md"]}
                }
            }
            Path(plan_path).write_text(json.dumps(plan))
            result = self.runner.invoke(
                cli,
                [
                    "batch",
                    traces_dir,
                    "--apply",
                    plan_path,
                    "--min-batch-size",
                    "1",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("validation failed", result.output.lower())

    # -- CLI: apply + upload --

    @patch("ace.cli.cloud._upload_batches")
    @patch("ace.cli.cloud.KaybaClient")
    def test_batch_apply_upload(self, MockKayba, mock_upload):
        """--apply + --upload triggers per-batch uploads."""
        with (
            tempfile.TemporaryDirectory() as traces_dir,
            tempfile.TemporaryDirectory() as plan_dir,
        ):
            (Path(traces_dir) / "a.md").write_text("data")
            plan_path = os.path.join(plan_dir, "plan.json")
            plan = {"batches": {"group": {"description": "g", "trace_files": ["a.md"]}}}
            Path(plan_path).write_text(json.dumps(plan))
            result = self.runner.invoke(
                cli,
                [
                    "batch",
                    traces_dir,
                    "--apply",
                    plan_path,
                    "--upload",
                    "--min-batch-size",
                    "1",
                    "--api-key",
                    "k",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Upload complete", result.output)
            mock_upload.assert_called_once()

    # -- CLI: --upload without --apply errors --

    def test_batch_upload_without_apply_errors(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("data")
            result = self.runner.invoke(
                cli,
                ["batch", d, "--upload", "--min-batch-size", "1"],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--upload requires --apply", result.output)


if __name__ == "__main__":
    unittest.main()
