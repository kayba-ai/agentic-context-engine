"""Kayba CLI — commands for the Kayba hosted API."""

from __future__ import annotations

import json
import sys
import time
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


def _client(api_key: Optional[str], base_url: Optional[str]) -> KaybaClient:
    """Build a KaybaClient, surfacing auth errors as click failures."""
    try:
        return KaybaClient(api_key=api_key, base_url=base_url)
    except KaybaAPIError as exc:
        raise click.ClickException(str(exc))


def _detect_file_type(filename: str) -> str:
    """Infer fileType from extension."""
    ext = Path(filename).suffix.lower()
    return {"md": "md", "json": "json"}.get(ext.lstrip("."), "txt")


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
    traces = []

    items = list(paths) if paths else ["-"]

    for item in items:
        if item == "-":
            content = sys.stdin.read()
            ft = file_type or "txt"
            traces.append({"filename": "stdin.txt", "content": content, "fileType": ft})
            continue

        p = Path(item)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    _add_file(traces, child, file_type)
        elif p.is_file():
            _add_file(traces, p, file_type)
        else:
            click.echo(f"Warning: skipping {item} (not found)", err=True)

    if not traces:
        raise click.ClickException("No traces to upload.")

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
    if len(content) > 350_000:
        click.echo(f"Warning: {path.name} is {len(content)} chars (>350k)", err=True)
    ft = forced_type or _detect_file_type(path.name)
    traces.append({"filename": path.name, "content": content, "fileType": ft})


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
# Helpers
# ---------------------------------------------------------------------------


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
