"""End-to-end integration tests for Kayba CLI against the live Kayba API.

Required env vars:
    KAYBA_API_KEY      – Kayba platform key (kayba_ak_...)
    ANTHROPIC_API_KEY  – Anthropic key for insight generation

Optional:
    KAYBA_API_URL      – Override API base URL (default: https://use.kayba.ai/api)

Run:
    KAYBA_API_KEY=... ANTHROPIC_API_KEY=... uv run pytest tests/test_cloud_e2e.py -v --no-cov
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ace.cli import cli

_HAS_KAYBA_KEY = bool(os.environ.get("KAYBA_API_KEY"))
_HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

SAMPLE_TRACE = """\
User: How do I reset my password?
Assistant: Go to Settings > Security > Reset Password.
User: Thanks, that worked!
Assistant: You're welcome! Let me know if you need anything else."""


def _base_args() -> list[str]:
    """Build shared CLI args from env vars."""
    args = ["--api-key", os.environ["KAYBA_API_KEY"]]
    base_url = os.environ.get("KAYBA_API_URL")
    if base_url:
        args += ["--base-url", base_url]
    return args


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.skipif(
    not (_HAS_KAYBA_KEY and _HAS_ANTHROPIC_KEY),
    reason="KAYBA_API_KEY and ANTHROPIC_API_KEY required",
)
class TestCloudE2E(unittest.TestCase):
    """Full E2E: upload → generate → poll → materialize → triage → prompt."""

    runner: CliRunner
    base_args: list[str]
    anthropic_key: str

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()
        cls.base_args = _base_args()
        cls.anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    # ------------------------------------------------------------------
    # Full workflow
    # ------------------------------------------------------------------

    def test_full_e2e_workflow(self):
        """Complete flow: upload → generate → list → triage → prompt → pull."""
        runner = self.runner
        base = self.base_args

        # 1. Upload a trace file
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(SAMPLE_TRACE)
            f.flush()
            trace_path = f.name

        try:
            result = runner.invoke(cli, ["upload", trace_path] + base)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Uploaded 1 trace(s)", result.output)
        finally:
            os.unlink(trace_path)

        # 2. Generate insights (with --wait to poll until complete)
        result = runner.invoke(
            cli,
            [
                "insights",
                "generate",
                "--anthropic-key",
                self.anthropic_key,
                "--wait",
            ]
            + base,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Job started:", result.output)

        # 3. List insights
        result = runner.invoke(
            cli,
            ["insights", "list", "--json"] + base,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        insights = json.loads(result.output)
        self.assertGreater(len(insights), 0, "Expected at least one insight")

        # 4. Triage — accept all pending
        result = runner.invoke(
            cli,
            ["insights", "triage", "--accept-all"] + base,
        )
        self.assertEqual(result.exit_code, 0, result.output)

        # 5. Generate prompt
        result = runner.invoke(
            cli,
            ["prompts", "generate", "--label", "E2E Test"] + base,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("generated", result.output.lower())

        # 6. List prompts
        result = runner.invoke(cli, ["prompts", "list"] + base)
        self.assertEqual(result.exit_code, 0, result.output)

        # 7. Pull latest prompt
        result = runner.invoke(cli, ["prompts", "pull"] + base)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(
            len(result.output.strip()) > 0, "Prompt content should be non-empty"
        )

    # ------------------------------------------------------------------
    # Individual command tests
    # ------------------------------------------------------------------

    def test_upload_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(SAMPLE_TRACE)
            f.flush()
            path = f.name
        try:
            result = self.runner.invoke(cli, ["upload", path] + self.base_args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Uploaded 1 trace(s)", result.output)
        finally:
            os.unlink(path)

    def test_upload_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "trace_a.md").write_text("User: Hi\nAssistant: Hello!")
            (Path(d) / "trace_b.md").write_text("User: Bye\nAssistant: Goodbye!")
            result = self.runner.invoke(cli, ["upload", d] + self.base_args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Uploaded 2 trace(s)", result.output)

    def test_insights_list_filters(self):
        # List with --status filter and --json
        result = self.runner.invoke(
            cli,
            ["insights", "list", "--status", "pending", "--json"] + self.base_args,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.output)
        self.assertIsInstance(parsed, list)

    def test_status_nonexistent_job(self):
        result = self.runner.invoke(
            cli,
            ["status", "nonexistent-job-id-000"] + self.base_args,
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_materialize_nonexistent_job(self):
        result = self.runner.invoke(
            cli,
            ["materialize", "nonexistent-job-id-000"] + self.base_args,
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_prompts_pull_nonexistent(self):
        result = self.runner.invoke(
            cli,
            ["prompts", "pull", "--id", "nonexistent-prompt-000"] + self.base_args,
        )
        self.assertNotEqual(result.exit_code, 0)


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.skipif(not _HAS_KAYBA_KEY, reason="KAYBA_API_KEY required")
class TestCloudE2ENoAnthropic(unittest.TestCase):
    """Tests that only need the Kayba API key (no Anthropic key)."""

    runner: CliRunner
    base_args: list[str]

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()
        cls.base_args = _base_args()

    def test_upload_no_auth(self):
        """Upload without API key should fail with auth error."""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(SAMPLE_TRACE)
            f.flush()
            path = f.name
        try:
            env = os.environ.copy()
            env.pop("KAYBA_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                result = self.runner.invoke(cli, ["upload", path])
            self.assertNotEqual(result.exit_code, 0)
        finally:
            os.unlink(path)

    def test_batch_prepare_mode(self):
        """Batch prepare mode should print classification prompt to stdout."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("User: Test\nAssistant: Response")
            out = os.path.join(d, "out.json")
            result = self.runner.invoke(
                cli,
                ["batch", d, "-o", out, "--min-batch-size", "1"],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("trace classification system", result.output)

    def test_batch_apply_and_upload(self):
        """Create a batch plan JSON, validate, and upload to real API."""
        with (
            tempfile.TemporaryDirectory() as traces_dir,
            tempfile.TemporaryDirectory() as plan_dir,
        ):
            (Path(traces_dir) / "x.md").write_text("User: Ping\nAssistant: Pong")
            plan_path = os.path.join(plan_dir, "plan.json")
            plan = {
                "batches": {
                    "group-a": {
                        "description": "single trace batch",
                        "trace_files": ["x.md"],
                    }
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
                    "--upload",
                    "--min-batch-size",
                    "1",
                ]
                + self.base_args,
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Upload complete", result.output)


if __name__ == "__main__":
    unittest.main()
