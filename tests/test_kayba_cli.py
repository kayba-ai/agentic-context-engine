from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ace.cli import cloud
from ace.cli.client import KaybaAPIError, KaybaClient


class _FakeTraceClient:
    def __init__(self, traces=None):
        self._traces = traces or []
        self.upload_calls: list[list[dict[str, str]]] = []

    def list_traces(self):
        return {"traces": self._traces}

    def upload_traces(self, traces):
        self.upload_calls.append(traces)
        return {
            "count": len(traces),
            "traces": [
                {"id": f"trace-{idx}", "filename": trace["filename"]}
                for idx, trace in enumerate(traces, start=1)
            ],
        }


class _FakeResponse:
    def __init__(self, status_code: int, text: str, json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.headers = {}

    def request(self, method: str, url: str, json=None, params=None):
        return self.response


def test_detect_file_type_handles_jsonl_and_markdown():
    assert cloud._detect_file_type("trace.jsonl") == "json"
    assert cloud._detect_file_type("trace.markdown") == "md"


def test_traces_list_empty_state_explains_manual_upload(monkeypatch):
    runner = CliRunner()
    fake_client = _FakeTraceClient()
    monkeypatch.setattr(cloud, "_client", lambda api_key, base_url: fake_client)

    result = runner.invoke(cloud.traces_list, [])

    assert result.exit_code == 0
    assert "Kayba does not auto-import local agent transcripts yet." in result.output
    assert "~/.claude/projects/<project>/*.jsonl" in result.output


def test_traces_upload_skips_oversized_files(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    fake_client = _FakeTraceClient()
    monkeypatch.setattr(cloud, "_client", lambda api_key, base_url: fake_client)

    small = tmp_path / "small.jsonl"
    small.write_text('{"ok": true}\n', encoding="utf-8")
    large = tmp_path / "large.txt"
    large.write_text("x" * (cloud.MAX_TRACE_CHARS + 1), encoding="utf-8")

    result = runner.invoke(cloud.traces_upload, [str(small), str(large)])

    assert result.exit_code == 0
    assert "Uploaded 1 trace(s)." in result.output
    assert "Skipping large.txt" in result.output
    assert len(fake_client.upload_calls) == 1
    assert fake_client.upload_calls[0][0]["filename"] == "small.jsonl"
    assert fake_client.upload_calls[0][0]["fileType"] == "json"


def test_prompts_install_replaces_managed_block(tmp_path: Path):
    runner = CliRunner()
    prompt_file = tmp_path / "prompt.md"
    target_file = tmp_path / "CLAUDE.md"
    prompt_file.write_text("First prompt", encoding="utf-8")

    first = runner.invoke(
        cloud.prompts_install,
        ["--input", str(prompt_file), "--target", "claude-code", "--file", str(target_file)],
    )
    assert first.exit_code == 0
    first_text = target_file.read_text(encoding="utf-8")
    assert "First prompt" in first_text
    assert first_text.count(cloud.PROMPT_BLOCK_START) == 1

    prompt_file.write_text("Second prompt", encoding="utf-8")
    second = runner.invoke(
        cloud.prompts_install,
        ["--input", str(prompt_file), "--target", "claude-code", "--file", str(target_file)],
    )
    assert second.exit_code == 0
    second_text = target_file.read_text(encoding="utf-8")
    assert "Second prompt" in second_text
    assert "First prompt" not in second_text
    assert second_text.count(cloud.PROMPT_BLOCK_START) == 1


def test_client_formats_non_json_http_errors():
    client = KaybaClient(api_key="test-key", base_url="https://example.com")
    client.session = _FakeSession(
        _FakeResponse(502, "<html><body>gateway exploded in a surprisingly long way</body></html>")
    )

    with pytest.raises(KaybaAPIError) as exc:
        client._request("GET", "/traces")

    assert exc.value.code == "HTTP_ERROR"
    assert "HTTP 502 from Kayba API:" in exc.value.message
    assert "<html><body>gateway exploded" in exc.value.message
