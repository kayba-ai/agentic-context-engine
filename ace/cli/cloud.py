"""Kayba CLI — commands for the Kayba hosted API."""

from __future__ import annotations

import importlib.resources
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from ace.cli.client import KaybaClient, KaybaAPIError

# Shared options applied to every command.
_api_key_option = click.option(
    "--api-key",
    envvar="KAYBA_API_KEY",
    help="Kayba API key (or set KAYBA_API_KEY).",
)
_base_url_option = click.option(
    "--base-url",
    envvar="KAYBA_API_URL",
    help="API base URL (default: https://use.kayba.ai/api).",
)

MAX_TRACE_CHARS = 350_000
PROMPT_BLOCK_START = "<!-- KAYBA:PROMPT:START -->"
PROMPT_BLOCK_END = "<!-- KAYBA:PROMPT:END -->"
PROMPT_INSTALL_TARGETS = {
    "universal": ("AGENTS.md", "most coding agents"),
    "codex": ("AGENTS.md", "Codex"),
    "windsurf": ("AGENTS.md", "Windsurf"),
    "claude-code": ("CLAUDE.md", "Claude Code"),
    "cursor": (".cursorrules", "Cursor"),
}


def _client(api_key: Optional[str], base_url: Optional[str]) -> KaybaClient:
    """Build a KaybaClient, surfacing auth errors as click failures."""
    try:
        return KaybaClient(api_key=api_key, base_url=base_url)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))


def _detect_file_type(filename: str) -> str:
    """Infer fileType from extension."""
    ext = Path(filename).suffix.lower()
    return {"md": "md", "markdown": "md", "json": "json", "jsonl": "json", "toon": "json"}.get(
        ext.lstrip("."),
        "txt",
    )


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


@click.command()
@click.argument("paths", nargs=-1)
@click.option(
    "--type",
    "file_type",
    type=click.Choice(["md", "json", "txt"]),
    default=None,
    help="Force file type (auto-detected from extension by default).",
)
@_api_key_option
@_base_url_option
def upload(paths, file_type, api_key, base_url):
    """Upload trace files to Kayba.

    PATHS can be files, directories, or '-' for stdin.
    Directories are walked recursively.
    """
    client = _client(api_key, base_url)
    traces = _collect_upload_traces(paths, file_type)

    if not traces:
        raise click.ClickException("No traces to upload.")

    _warn_large_trace_batch(traces)

    try:
        result = client.upload_traces(traces)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    count = result.get("count", len(result.get("traces", [])))
    click.echo(f"Uploaded {count} trace(s).")
    for t in result.get("traces", []):
        click.echo(f"  {t['id']}  {t['filename']}")


def _add_file(traces: list, path: Path, forced_type: Optional[str]):
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_TRACE_CHARS:
        click.echo(
            f"Skipping {path.name}: {len(content)} chars exceeds the Kayba API "
            f"limit of {MAX_TRACE_CHARS}. Split or trim the trace and re-upload.",
            err=True,
        )
        return False
    ft = forced_type or _detect_file_type(path.name)
    traces.append({"filename": path.name, "content": content, "fileType": ft})
    return True


def _collect_upload_traces(
    paths: tuple[str, ...],
    forced_type: Optional[str],
) -> list[dict[str, str]]:
    """Collect uploadable traces from files, directories, or stdin."""
    traces: list[dict[str, str]] = []
    items = list(paths) if paths else ["-"]

    for item in items:
        if item == "-":
            content = sys.stdin.read()
            if len(content) > MAX_TRACE_CHARS:
                click.echo(
                    f"Skipping stdin.txt: {len(content)} chars exceeds the Kayba API "
                    f"limit of {MAX_TRACE_CHARS}. Split or trim the trace and re-upload.",
                    err=True,
                )
                continue
            ft = forced_type or "txt"
            traces.append({"filename": "stdin.txt", "content": content, "fileType": ft})
            continue

        p = Path(item)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    _add_file(traces, child, forced_type)
        elif p.is_file():
            _add_file(traces, p, forced_type)
        else:
            click.echo(f"Warning: skipping {item} (not found)", err=True)

    return traces


def _warn_large_trace_batch(traces: list[dict[str, str]]) -> None:
    """Warn once when a batch contains very large trace files."""
    oversized = sum(1 for trace in traces if len(trace["content"]) > MAX_TRACE_CHARS)
    if oversized:
        click.echo(
            "Warning: "
            f"{oversized} trace(s) exceed {MAX_TRACE_CHARS} chars; the CLI will chunk uploads "
            "into smaller requests, but very large individual files may still be "
            "rejected.",
            err=True,
        )


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------


@click.group()
def traces():
    """List, view, upload, and delete traces."""
    pass


@traces.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@_api_key_option
@_base_url_option
def traces_list(as_json, api_key, base_url):
    """List uploaded traces."""
    client = _client(api_key, base_url)
    try:
        result = client.list_traces()
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    items = result.get("traces", [])

    if as_json:
        click.echo(json.dumps(items, indent=2))
        return

    if not items:
        click.echo(_no_traces_message())
        return

    # Table header
    click.echo(f"  {'ID':<36}  {'Filename':<40}  {'Type':<6}  {'Size':>8}  {'Uploaded'}")
    click.echo(f"  {'-' * 36}  {'-' * 40}  {'-' * 6}  {'-' * 8}  {'-' * 10}")
    for t in items:
        tid = t.get("id", "?")
        fname = t.get("filename", "?")
        ftype = t.get("fileType", t.get("type", "?"))
        size = _format_size(t.get("size", 0))
        age = _format_age(t.get("uploadedAt", ""))
        click.echo(f"  {tid:<36}  {fname:<40}  {ftype:<6}  {size:>8}  {age}")

    click.echo(f"\n  {len(items)} trace(s)")


@traces.command("show")
@click.argument("trace_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@click.option("--meta", is_flag=True, help="Show only metadata, no content.")
@_api_key_option
@_base_url_option
def traces_show(trace_id, as_json, meta, api_key, base_url):
    """View a trace."""
    client = _client(api_key, base_url)
    try:
        result = client.get_trace(trace_id)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    # Metadata header
    click.echo(f"ID:       {result.get('id', '?')}")
    click.echo(f"Filename: {result.get('filename', '?')}")
    click.echo(f"Type:     {result.get('fileType', result.get('type', '?'))}")
    click.echo(f"Size:     {_format_size(result.get('size', 0))}")
    click.echo(f"Uploaded: {result.get('uploadedAt', '?')}")

    if not meta:
        content = result.get("content", "")
        if content:
            click.echo(f"\n{'─' * 60}")
            click.echo(content)


@traces.command("delete")
@click.argument("trace_ids", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
@_api_key_option
@_base_url_option
def traces_delete(trace_ids, force, api_key, base_url):
    """Delete one or more traces."""
    if not force and sys.stdin.isatty():
        count = len(trace_ids)
        if not click.confirm(f"Delete {count} trace(s)?"):
            click.echo("Aborted.")
            return

    client = _client(api_key, base_url)
    try:
        result = client.delete_traces(list(trace_ids))
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    deleted = result.get("deleted", [])
    errors = result.get("errors", [])

    for tid in deleted:
        click.echo(f"  Deleted {tid}")
    for err in errors:
        click.echo(f"  Error deleting {err['id']}: {err['error']}", err=True)

    if errors:
        raise click.ClickException(f"{len(errors)} deletion(s) failed.")


@traces.command("upload")
@click.argument("paths", nargs=-1)
@click.option(
    "--type",
    "file_type",
    type=click.Choice(["md", "json", "txt"]),
    default=None,
    help="Force file type (auto-detected from extension by default).",
)
@_api_key_option
@_base_url_option
def traces_upload(paths, file_type, api_key, base_url):
    """Upload trace files to Kayba.

    PATHS can be files, directories, or '-' for stdin.
    Directories are walked recursively.
    """
    client = _client(api_key, base_url)
    trace_list = _collect_upload_traces(paths, file_type)

    if not trace_list:
        raise click.ClickException("No traces to upload.")

    _warn_large_trace_batch(trace_list)

    try:
        result = client.upload_traces(trace_list)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    count = result.get("count", len(result.get("traces", [])))
    click.echo(f"Uploaded {count} trace(s).")
    for t in result.get("traces", []):
        click.echo(f"  {t['id']}  {t['filename']}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@click.command()
@click.option("--traces", "trace_ids", multiple=True, help="Trace IDs to analyze.")
@click.option("--all", "select_all", is_flag=True, help="Select all traces.")
@click.option(
    "--model",
    type=click.Choice(["claude-sonnet-4-6", "claude-opus-4-6"]),
    default=None,
    help="Model to use for analysis.",
)
@click.option("--epochs", type=int, default=None, help="Analysis epochs (default 1).")
@click.option(
    "--reflector-mode",
    type=click.Choice(["recursive", "standard"]),
    default=None,
    help="Reflector mode.",
)
@click.option(
    "--anthropic-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (or set ANTHROPIC_API_KEY).",
)
@click.option("--wait", is_flag=True, help="Poll until the job completes.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@_api_key_option
@_base_url_option
def run(
    trace_ids, select_all, model, epochs, reflector_mode, anthropic_key, wait, as_json,
    api_key, base_url,
):
    """Run the analysis pipeline on selected traces.

    Interactive mode: shows a visual trace selector.
    Programmatic mode: use --traces ID or --all.
    """
    client = _client(api_key, base_url)

    # Fetch available traces
    try:
        result = client.list_traces()
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    available = result.get("traces", [])

    if not available:
        raise click.ClickException(_no_traces_message())

    if select_all:
        selected_ids = [t["id"] for t in available]
    elif trace_ids:
        selected_ids = list(trace_ids)
    elif sys.stdin.isatty() and not as_json:
        # Interactive mode: visual checkbox selector
        selected_ids = _interactive_trace_select(available)
        if not selected_ids:
            click.echo("No traces selected.")
            return
    else:
        raise click.ClickException(
            "Provide --traces ID, --all, or run interactively (TTY)."
        )

    # Start the pipeline
    try:
        result = client.generate_insights(
            trace_ids=selected_ids,
            model=model,
            epochs=epochs,
            reflector_mode=reflector_mode,
            anthropic_key=anthropic_key,
        )
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    job_id = result["jobId"]

    if as_json:
        click.echo(json.dumps({"jobId": job_id, "traces": len(selected_ids)}))
    else:
        click.echo(f"Job started: {job_id} ({len(selected_ids)} traces)")

    if wait:
        _poll_job(client, job_id)


# ---------------------------------------------------------------------------
# insights
# ---------------------------------------------------------------------------


@click.group()
def insights():
    """Generate, list, and triage insights."""
    pass


@insights.command("generate")
@click.option("--traces", "trace_ids", multiple=True, help="Trace IDs to analyze.")
@click.option(
    "--model",
    type=click.Choice(["claude-sonnet-4-6", "claude-opus-4-6"]),
    default=None,
    help="Model to use for analysis.",
)
@click.option("--epochs", type=int, default=None, help="Analysis epochs (default 1).")
@click.option(
    "--reflector-mode",
    type=click.Choice(["recursive", "standard"]),
    default=None,
    help="Reflector mode.",
)
@click.option(
    "--anthropic-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (or set ANTHROPIC_API_KEY).",
)
@click.option("--wait", is_flag=True, help="Poll until the job completes.")
@_api_key_option
@_base_url_option
def insights_generate(
    trace_ids, model, epochs, reflector_mode, anthropic_key, wait, api_key, base_url
):
    """Trigger insight generation from uploaded traces."""
    client = _client(api_key, base_url)
    try:
        result = client.generate_insights(
            trace_ids=list(trace_ids) or None,
            model=model,
            epochs=epochs,
            reflector_mode=reflector_mode,
            anthropic_key=anthropic_key,
        )
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    job_id = result["jobId"]
    click.echo(f"Job started: {job_id}")

    if wait:
        _poll_job(client, job_id)


@insights.command("list")
@click.option(
    "--status",
    type=click.Choice(["pending", "new", "accepted", "rejected"]),
    default=None,
    help="Filter by review status.",
)
@click.option("--section", default=None, help="Filter by skillbook section.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@_api_key_option
@_base_url_option
def insights_list(status, section, as_json, api_key, base_url):
    """List insights."""
    client = _client(api_key, base_url)
    try:
        result = client.list_insights(status=status, section=section)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    items = result.get("insights", [])
    if as_json:
        click.echo(json.dumps(items, indent=2))
        return

    if not items:
        click.echo("No insights found.")
        return

    for ins in items:
        status_str = ins.get("status", "?")
        click.echo(f"  [{status_str:>8}]  {ins['id']}  {ins.get('section', '')}")
        click.echo(f"            {ins.get('content', '')[:120]}")


@insights.command("triage")
@click.option("--accept", "accept_ids", multiple=True, help="Insight IDs to accept.")
@click.option("--reject", "reject_ids", multiple=True, help="Insight IDs to reject.")
@click.option("--accept-all", is_flag=True, help="Accept all pending insights.")
@click.option("--note", default=None, help="Optional triage note.")
@_api_key_option
@_base_url_option
def insights_triage(accept_ids, reject_ids, accept_all, note, api_key, base_url):
    """Accept or reject insights."""
    client = _client(api_key, base_url)

    if accept_all:
        try:
            result = client.list_insights(status="pending")
        except KaybaAPIError as exc:
            raise click.ClickException(str(exc))
        accept_ids = tuple(ins["id"] for ins in result.get("insights", []))
        if not accept_ids:
            click.echo("No pending insights to accept.")
            return

    if not accept_ids and not reject_ids:
        raise click.ClickException("Provide --accept, --reject, or --accept-all.")

    errors = []
    for iid in accept_ids:
        try:
            client.triage_insight(iid, "accepted", note=note)
            click.echo(f"  Accepted {iid}")
        except KaybaAPIError as exc:
            errors.append(str(exc))
            click.echo(f"  Error accepting {iid}: {exc}", err=True)

    for iid in reject_ids:
        try:
            client.triage_insight(iid, "rejected", note=note)
            click.echo(f"  Rejected {iid}")
        except KaybaAPIError as exc:
            errors.append(str(exc))
            click.echo(f"  Error rejecting {iid}: {exc}", err=True)

    if errors:
        raise click.ClickException(f"{len(errors)} triage operation(s) failed.")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


@click.group()
def prompts():
    """Generate, list, and pull prompts."""
    pass


@prompts.command("generate")
@click.option(
    "--insights", "insight_ids", multiple=True, help="Insight IDs to include."
)
@click.option("--label", default=None, help="Label for the generated prompt.")
@click.option("-o", "--output", "output_path", default=None, help="Save to file.")
@_api_key_option
@_base_url_option
def prompts_generate(insight_ids, label, output_path, api_key, base_url):
    """Generate a prompt from accepted insights."""
    client = _client(api_key, base_url)
    try:
        result = client.generate_prompt(
            insight_ids=list(insight_ids) or None,
            label=label,
        )
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    prompt_id = result.get("promptId", "?")
    version = result.get("version", "?")
    text = result.get("content", {}).get("text", "")

    click.echo(f"Prompt {prompt_id} (v{version}) generated.")

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        click.echo(f"Saved to {output_path}")
    else:
        click.echo(text)


@prompts.command("list")
@_api_key_option
@_base_url_option
def prompts_list(api_key, base_url):
    """List prompt versions."""
    client = _client(api_key, base_url)
    try:
        result = client.list_prompts()
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    items = result if isinstance(result, list) else result.get("prompts", [])
    if not items:
        click.echo("No prompts found.")
        return

    for p in items:
        pid = p.get("id", p.get("promptId", "?"))
        label = p.get("label", "")
        click.echo(f"  {pid}  {label}")


@prompts.command("pull")
@click.option("--id", "prompt_id", default=None, help="Prompt ID (default: latest).")
@click.option("-o", "--output", "output_path", default=None, help="Save to file.")
@click.option("--pretty", is_flag=True, help="Pretty-print JSON output.")
@_api_key_option
@_base_url_option
def prompts_pull(prompt_id, output_path, pretty, api_key, base_url):
    """Download a prompt."""
    client = _client(api_key, base_url)

    if prompt_id:
        try:
            result = client.get_prompt(prompt_id)
        except KaybaAPIError as exc:
            raise click.ClickException(str(exc))
    else:
        # Get latest by listing and picking first
        try:
            listing = client.list_prompts()
        except KaybaAPIError as exc:
            raise click.ClickException(str(exc))
        items = listing if isinstance(listing, list) else listing.get("prompts", [])
        if not items:
            raise click.ClickException("No prompts available.")
        first = items[0]
        pid = first.get("id", first.get("promptId"))
        try:
            result = client.get_prompt(pid)
        except KaybaAPIError as exc:
            raise click.ClickException(str(exc))

    text = result.get("content", {}).get("text", "")

    if pretty:
        output = json.dumps(result, indent=2)
    else:
        output = text

    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
        click.echo(f"Saved to {output_path}")
    else:
        click.echo(output)


@prompts.command("install")
@click.option("--id", "prompt_id", default=None, help="Prompt ID (default: latest).")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read prompt text from a local file instead of the API.",
)
@click.option(
    "--target",
    type=click.Choice(sorted(PROMPT_INSTALL_TARGETS)),
    default="universal",
    show_default=True,
    help="Agent to install the prompt for.",
)
@click.option(
    "--file",
    "target_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Override the destination file path.",
)
@_api_key_option
@_base_url_option
def prompts_install(input_path, prompt_id, target, target_path, api_key, base_url):
    """Install a generated prompt into an agent instruction file."""
    if input_path and prompt_id:
        raise click.ClickException("Use either --input or --id, not both.")

    if input_path:
        prompt_ref = input_path
        text = Path(input_path).read_text(encoding="utf-8")
    else:
        client = _client(api_key, base_url)
        prompt_ref, text = _fetch_prompt_text(client, prompt_id)

    if not text.strip():
        raise click.ClickException("Prompt content is empty.")

    default_filename, target_label = PROMPT_INSTALL_TARGETS[target]
    destination = Path(target_path) if target_path else Path(default_filename)
    _upsert_prompt_block(destination, _build_prompt_block(text))
    click.echo(f"Installed Kayba prompt ({prompt_ref}) into {destination}")
    click.echo(
        f"Target: {target_label}. Start a new agent session so it reloads the file."
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@click.command()
@click.argument("job_id")
@click.option("--wait", is_flag=True, help="Poll until the job completes.")
@click.option(
    "--interval", type=int, default=5, help="Poll interval in seconds (default 5)."
)
@_api_key_option
@_base_url_option
def status(job_id, wait, interval, api_key, base_url):
    """Check the status of an analysis job."""
    client = _client(api_key, base_url)

    if wait:
        _poll_job(client, job_id, interval=interval)
    else:
        try:
            job = client.get_job(job_id)
        except KaybaAPIError as exc:
            raise click.ClickException(str(exc))
        _print_job(job)


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


@click.command()
@click.argument("job_id")
@_api_key_option
@_base_url_option
def materialize(job_id, api_key, base_url):
    """Materialize completed job results into the skillbook."""
    client = _client(api_key, base_url)
    try:
        result = client.materialize_job(job_id)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"Materialized {result.get('skillsGenerated', '?')} skill(s) "
        f"from job {result.get('jobId', job_id)}."
    )


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

DEFAULT_BATCH_PROMPT = """\
You are a trace classification system. Analyze the trace metadata below and group
them into coherent batches for analysis by a Recursive Reflector.

Constraints:
{constraints}

Instructions:
1. Group traces by semantic similarity (similar tasks, tools, domains).
2. Respect min/max batch size constraints.
3. Every trace must be assigned to exactly one batch.
4. Use descriptive batch names (lowercase-with-hyphens).

Output only valid JSON matching this schema:
{{"batches": {{"name": {{"description": "...", "trace_files": [...]}}}}, "summary": {{"total_traces": N, "num_batches": N, "batch_sizes": {{"name": N}}}}}}

Trace metadata:
{traces_json}
"""


def _extract_trace_metadata(filename: str, content: str, file_type: str) -> dict:
    """Extract compact metadata from a trace for prompt context."""
    meta: dict = {
        "filename": filename,
        "type": file_type,
        "size": len(content),
    }

    if file_type == "json":
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if "task_id" in data:
                    meta["task_id"] = data["task_id"]
                if "user_request" in data:
                    meta["user_request"] = str(data["user_request"])[:200]
                if "tools" in data:
                    meta["tools"] = data["tools"]
                steps = data.get("steps") or data.get("events") or []
                if isinstance(steps, list):
                    meta["step_count"] = len(steps)
        except (json.JSONDecodeError, TypeError):
            pass
    elif file_type == "md":
        lines = content.split("\n")
        meta["summary"] = "\n".join(lines[:10])
        headings = [ln for ln in lines if ln.startswith("#")]
        if headings:
            meta["headings"] = headings[:20]
    else:
        lines = content.split("\n")
        meta["summary"] = "\n".join(lines[:5])

    return meta


def _build_classification_prompt(
    traces_metadata: list[dict],
    constraints: str,
    custom_prompt: Optional[str] = None,
) -> str:
    """Build the classification prompt with metadata and constraints."""
    traces_json = json.dumps(traces_metadata, indent=2)
    template = custom_prompt if custom_prompt else DEFAULT_BATCH_PROMPT
    return template.format(traces_json=traces_json, constraints=constraints)


def _validate_batch_plan(
    plan: dict,
    all_filenames: list[str],
    min_size: int,
    max_size: int,
) -> list[str]:
    """Validate a batch plan. Returns list of error strings (empty = valid)."""
    errors: list[str] = []

    batches = plan.get("batches")
    if not isinstance(batches, dict):
        errors.append("Missing or invalid 'batches' key (expected dict).")
        return errors

    assigned: set[str] = set()
    for name, batch in batches.items():
        files = batch.get("trace_files", [])
        if not isinstance(files, list):
            errors.append(f"Batch '{name}': trace_files must be a list.")
            continue

        if len(files) < min_size:
            errors.append(f"Batch '{name}' has {len(files)} traces (min {min_size}).")
        if len(files) > max_size:
            errors.append(f"Batch '{name}' has {len(files)} traces (max {max_size}).")
        for f in files:
            if f in assigned:
                errors.append(f"Trace '{f}' assigned to multiple batches.")
            assigned.add(f)

    missing = set(all_filenames) - assigned
    if missing:
        errors.append(f"Traces not assigned: {sorted(missing)}")

    extra = assigned - set(all_filenames)
    if extra:
        errors.append(f"Unknown traces in plan: {sorted(extra)}")

    return errors


def _upload_batches(
    plan: dict,
    traces_by_name: dict[str, dict[str, str]],
    client: KaybaClient,
) -> None:
    """Upload each batch to the Kayba API."""
    batches = plan.get("batches", {})
    for name, batch in batches.items():
        files = batch.get("trace_files", [])
        batch_traces = [traces_by_name[f] for f in files if f in traces_by_name]
        if not batch_traces:
            click.echo(f"  Skipping empty batch '{name}'.", err=True)
            continue
        try:
            _warn_large_trace_batch(batch_traces)
            result = client.upload_traces(batch_traces)
            count = result.get("count", len(batch_traces))
            click.echo(f"  Uploaded batch '{name}': {count} trace(s).")
        except KaybaAPIError as exc:
            click.echo(f"  Error uploading batch '{name}': {exc}", err=True)


@click.command()
@click.argument("paths", nargs=-1)
@click.option(
    "--prompt",
    "prompt_file",
    type=click.Path(exists=True),
    default=None,
    help="Custom classification prompt file (should contain {traces_json} and {constraints}).",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    default="batches.json",
    show_default=True,
    help="Output batch plan file.",
)
@click.option(
    "--apply",
    "apply_file",
    type=click.Path(exists=True),
    default=None,
    help="Apply an existing batch plan (skip prompt generation).",
)
@click.option(
    "--upload",
    "do_upload",
    is_flag=True,
    help="Upload each batch to the API (requires --apply).",
)
@click.option("--max-batch-size", type=int, default=30, show_default=True)
@click.option("--min-batch-size", type=int, default=10, show_default=True)
@_api_key_option
@_base_url_option
def batch(
    paths,
    prompt_file,
    output_file,
    apply_file,
    do_upload,
    max_batch_size,
    min_batch_size,
    api_key,
    base_url,
):
    """Pre-batch traces for the Recursive Reflector.

    Two modes:

      Prepare (default): collect traces, extract metadata, print a classification
      prompt to stdout for Claude Code to process.

      Apply (--apply FILE): validate a batch plan JSON and optionally upload.

    PATHS can be files or directories (walked recursively).
    """
    if not paths:
        raise click.ClickException("Provide at least one path.")

    # ---- Collect traces ----
    traces: list[dict[str, str]] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    _add_file(traces, child, None)
        elif p.is_file():
            _add_file(traces, p, None)
        else:
            click.echo(f"Warning: skipping {item} (not found)", err=True)

    if not traces:
        raise click.ClickException("No trace files found.")

    all_filenames = [t["filename"] for t in traces]

    # ---- Mode 2: Apply ----
    if apply_file:
        plan_text = Path(apply_file).read_text(encoding="utf-8")
        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in {apply_file}: {exc}")

        errors = _validate_batch_plan(
            plan, all_filenames, min_batch_size, max_batch_size
        )
        if errors:
            for err in errors:
                click.echo(f"  Error: {err}", err=True)
            raise click.ClickException("Batch plan validation failed.")

        num_batches = len(plan.get("batches", {}))
        click.echo(
            f"Batch plan valid: {num_batches} batch(es), {len(traces)} trace(s)."
        )

        if do_upload:
            client = _client(api_key, base_url)
            traces_by_name = {t["filename"]: t for t in traces}
            _upload_batches(plan, traces_by_name, client)
            click.echo("Upload complete.")
        return

    # ---- Mode 1: Prepare ----
    if do_upload:
        raise click.ClickException("--upload requires --apply.")

    metadata = [
        _extract_trace_metadata(t["filename"], t["content"], t["fileType"])
        for t in traces
    ]

    constraints = f"min_batch_size={min_batch_size}, max_batch_size={max_batch_size}"
    custom_prompt = None
    if prompt_file:
        custom_prompt = Path(prompt_file).read_text(encoding="utf-8")
    prompt_text = _build_classification_prompt(metadata, constraints, custom_prompt)

    # Write metadata to output file as starting point
    starter = {
        "batches": {},
        "summary": {"total_traces": len(traces), "num_batches": 0, "batch_sizes": {}},
    }
    out_path = Path(output_file)
    out_path.write_text(json.dumps(starter, indent=2), encoding="utf-8")
    click.echo(f"Wrote metadata to {out_path}", err=True)
    click.echo(f"Found {len(traces)} trace(s).", err=True)

    # Print prompt to stdout for Claude Code
    click.echo(prompt_text)


# ---------------------------------------------------------------------------
# integrations
# ---------------------------------------------------------------------------


def _mask_token(token: str) -> str:
    """Mask a token/key for display, showing first 4 + last 4 chars."""
    if not token or len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


@click.group()
def integrations():
    """Manage platform integrations (MLflow, LangSmith)."""
    pass


@integrations.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@_api_key_option
@_base_url_option
def integrations_list(as_json, api_key, base_url):
    """Show configured integrations."""
    client = _client(api_key, base_url)
    try:
        result = client.get_integrations()
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    for name in ("mlflow", "langsmith"):
        config = result.get(name, {})
        enabled = config.get("enabled", False)
        status_str = "enabled" if enabled else "disabled"

        click.echo(f"\n  {name}")
        click.echo(f"    Status: {status_str}")

        if name == "mlflow":
            uri = config.get("trackingUri", "")
            auth = config.get("authType", "none")
            experiment = config.get("experimentName", "")
            if uri:
                click.echo(f"    Tracking URI: {uri}")
            click.echo(f"    Auth type: {auth}")
            if config.get("token"):
                click.echo(f"    Token: {_mask_token(config['token'])}")
            if config.get("username"):
                click.echo(f"    Username: {config['username']}")
            if experiment:
                click.echo(f"    Experiment: {experiment}")
        elif name == "langsmith":
            api_url = config.get("apiUrl", "")
            project = config.get("projectName", "")
            if api_url:
                click.echo(f"    API URL: {api_url}")
            if config.get("apiKey"):
                click.echo(f"    API key: {_mask_token(config['apiKey'])}")
            if project:
                click.echo(f"    Project: {project}")

    click.echo()


@integrations.command("configure")
@click.argument("name", type=click.Choice(["mlflow", "langsmith"]))
@_api_key_option
@_base_url_option
def integrations_configure(name, api_key, base_url):
    """Interactively configure an integration."""
    client = _client(api_key, base_url)

    # Fetch current config for defaults
    try:
        current = client.get_integrations()
    except KaybaAPIError:
        current = {}

    existing = current.get(name, {})

    if name == "mlflow":
        tracking_uri = click.prompt(
            "MLflow tracking URI",
            default=existing.get("trackingUri", ""),
        )
        auth_type = click.prompt(
            "Auth type",
            type=click.Choice(["none", "basic", "bearer", "databricks"]),
            default=existing.get("authType", "none"),
        )

        token = ""
        username = ""
        if auth_type in ("basic", "bearer", "databricks"):
            token = click.prompt(
                "Token / password",
                default="",
                hide_input=True,
                show_default=False,
            )
        if auth_type == "basic":
            username = click.prompt(
                "Username",
                default=existing.get("username", ""),
            )

        experiment_name = click.prompt(
            "Experiment name (optional filter)",
            default=existing.get("experimentName", ""),
        )

        config = {
            "enabled": True,
            "trackingUri": tracking_uri,
            "authType": auth_type,
            "token": token,
            "username": username,
            "experimentName": experiment_name,
        }

    elif name == "langsmith":
        api_url = click.prompt(
            "LangSmith API URL",
            default=existing.get("apiUrl", "https://api.smith.langchain.com"),
        )
        langsmith_key = click.prompt(
            "LangSmith API key",
            default="",
            hide_input=True,
            show_default=False,
        )
        project_name = click.prompt(
            "Project name (optional filter)",
            default=existing.get("projectName", ""),
        )

        config = {
            "enabled": True,
            "apiUrl": api_url,
            "apiKey": langsmith_key,
            "projectName": project_name,
        }

    try:
        client.update_integration(name, config)
        click.echo(f"\n  {name} configuration saved.")
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    # Auto-test the connection
    click.echo(f"  Testing {name} connection...")
    try:
        test_result = client.test_integration(name)
        if test_result.get("connected"):
            click.echo(f"  Connected successfully.")
            if name == "mlflow" and test_result.get("mlflowVersion"):
                click.echo(f"  MLflow version: {test_result['mlflowVersion']}")
        else:
            click.echo(f"  Warning: connection test returned unexpected result.")
    except KaybaAPIError as exc:
        click.echo(f"  Warning: connection test failed: {exc}", err=True)


@integrations.command("test")
@click.argument("name", type=click.Choice(["mlflow", "langsmith"]))
@_api_key_option
@_base_url_option
def integrations_test(name, api_key, base_url):
    """Test an integration connection."""
    client = _client(api_key, base_url)
    try:
        result = client.test_integration(name)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))

    if result.get("connected"):
        click.echo(f"  {name}: connected")
        if name == "mlflow" and result.get("mlflowVersion"):
            click.echo(f"  MLflow version: {result['mlflowVersion']}")
        if name == "mlflow" and result.get("experimentCount") is not None:
            click.echo(f"  Experiments found: {result['experimentCount']}")
    else:
        error = result.get("error", "Unknown error")
        raise click.ClickException(f"{name}: connection failed — {error}")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--append-to",
    type=click.Path(),
    default=None,
    help="Append instructions to this file (default: print to stdout). "
    "Recommended: AGENTS.md (universal), CLAUDE.md, .cursorrules.",
)
@click.option(
    "--skills/--no-skills",
    default=True,
    help="Install Claude Code skills (default: enabled).",
)
@click.option(
    "--project-dir",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Project root directory (default: current directory).",
)
def setup(append_to, skills, project_dir):
    """Print or install Kayba CLI instructions and skills for coding agents."""
    snippet = (
        importlib.resources.files("ace.cli.commands")
        .joinpath("kayba-agent-instructions.md")
        .read_text(encoding="utf-8")
    )

    if append_to:
        path = Path(append_to)
        mode = "a" if path.exists() else "w"
        with path.open(mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n")
            f.write(snippet)
        click.echo(f"Appended Kayba CLI instructions to {path}")
    else:
        click.echo(snippet)

    if skills:
        target = Path(project_dir) / ".claude" / "skills"
        _install_skills(target)


def _install_skills(target_dir: Path) -> None:
    """Copy bundled skill files to the target .claude/skills/ directory."""
    skills_pkg = importlib.resources.files("ace.cli.skills")
    installed = []

    for skill_dir in skills_pkg.iterdir():
        if skill_dir.name.startswith("_") or not skill_dir.is_dir():
            continue

        dest = target_dir / skill_dir.name
        dest.mkdir(parents=True, exist_ok=True)

        # Copy top-level SKILL.md
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_file():
            (dest / "SKILL.md").write_bytes(skill_file.read_bytes())

        # Copy stage subdirectories
        for sub in skill_dir.iterdir():
            if sub.is_dir() and not sub.name.startswith("_"):
                sub_dest = dest / sub.name
                sub_dest.mkdir(parents=True, exist_ok=True)
                sub_skill = sub / "SKILL.md"
                if sub_skill.is_file():
                    (sub_dest / "SKILL.md").write_bytes(sub_skill.read_bytes())

        stages = [d.name for d in dest.iterdir() if d.is_dir()]
        installed.append((skill_dir.name, len(stages)))

    if installed:
        click.echo(f"\nInstalled skills to {target_dir}/:")
        for name, stage_count in installed:
            click.echo(f"  {name} ({stage_count} stages)")
    else:
        click.echo("\nNo skills found to install.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interactive_trace_select(traces: list[dict]) -> list[str]:
    """Visual checkbox selector for traces."""
    try:
        import questionary
    except ImportError:
        raise click.ClickException(
            "Interactive mode requires 'questionary'. "
            "Install with: pip install 'ace-framework[cloud]'"
        )

    choices = []
    for t in traces:
        size = _format_size(t.get("size", 0))
        age = _format_age(t.get("uploadedAt", ""))
        label = f"{t['filename']:<40} {size:>8}  {age}"
        choices.append(questionary.Choice(title=label, value=t["id"]))

    selected = questionary.checkbox(
        "Select traces (space to toggle, a to toggle all, enter to confirm):",
        choices=choices,
    ).ask()

    if selected is None:  # Ctrl-C
        return []
    return selected


def _no_traces_message() -> str:
    """Explain why a new hosted account often has no traces yet."""
    return (
        "No traces found in your Kayba account.\n"
        "Kayba does not auto-import local agent transcripts yet.\n"
        "If you're using Claude Code, upload its local .jsonl files first, for example:\n"
        "  kayba traces upload ~/.claude/projects/<project>/*.jsonl\n"
        "Docs: https://kayba.ai/docs/integrations/hosted-api/#where-do-traces-come-from"
    )


def _fetch_prompt_text(client: KaybaClient, prompt_id: Optional[str]) -> tuple[str, str]:
    """Load a prompt body from the API."""
    if prompt_id:
        result = client.get_prompt(prompt_id)
        prompt_ref = prompt_id
    else:
        listing = client.list_prompts()
        items = listing if isinstance(listing, list) else listing.get("prompts", [])
        if not items:
            raise click.ClickException(
                "No prompts available. Generate one first with `kayba prompts generate`."
            )
        first = items[0]
        prompt_ref = str(first.get("id", first.get("promptId", "latest")))
        result = client.get_prompt(prompt_ref)

    text = result.get("content", {}).get("text", "")
    if not text.strip():
        raise click.ClickException(f"Prompt {prompt_ref} is empty.")
    return prompt_ref, text


def _build_prompt_block(prompt_text: str) -> str:
    """Wrap prompt text in a managed block so repeated installs replace cleanly."""
    body = prompt_text.strip()
    return (
        f"{PROMPT_BLOCK_START}\n"
        "## Kayba Prompt\n"
        "_Managed by `kayba prompts install`. Re-run the command to update this block._\n\n"
        f"{body}\n"
        f"{PROMPT_BLOCK_END}\n"
    )


def _upsert_prompt_block(path: Path, block: str) -> None:
    """Replace an existing managed prompt block or append a new one."""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(
        rf"{re.escape(PROMPT_BLOCK_START)}.*?{re.escape(PROMPT_BLOCK_END)}\n?",
        re.DOTALL,
    )

    if pattern.search(existing):
        updated = pattern.sub(block, existing, count=1)
    elif existing.strip():
        updated = existing.rstrip() + "\n\n" + block
    else:
        updated = block

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_age(iso_str: str) -> str:
    """Format ISO datetime as relative time."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = diff.total_seconds()
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        else:
            return f"{int(seconds / 86400)}d ago"
    except (ValueError, TypeError):
        return iso_str


def _poll_job(client: KaybaClient, job_id: str, *, interval: int = 5):
    """Poll a job until it reaches a terminal state."""
    terminal = {"completed", "failed"}
    while True:
        try:
            job = client.get_job(job_id)
        except KaybaAPIError as exc:
            raise click.ClickException(str(exc))

        st = job.get("status", "unknown")
        click.echo(f"  {job_id}  {st}")

        if st in terminal:
            _print_job(job)
            if st == "completed":
                click.echo(f"\nRun: kayba materialize {job_id}")
            return

        time.sleep(interval)


def _print_job(job: dict):
    """Pretty-print a job status dict."""
    click.echo(f"Job:    {job.get('jobId', '?')}")
    click.echo(f"Status: {job.get('status', '?')}")
    if job.get("startedAt"):
        click.echo(f"Started: {job['startedAt']}")
    if job.get("completedAt"):
        click.echo(f"Completed: {job['completedAt']}")
    if job.get("error"):
        click.echo(f"Error: {job['error']}")
    result = job.get("result")
    if result:
        click.echo(f"Skills generated: {result.get('skillsGenerated', '?')}")
        if result.get("summary"):
            click.echo(f"Summary: {result['summary']}")
        click.echo(f"Materialized: {result.get('materialized', False)}")
