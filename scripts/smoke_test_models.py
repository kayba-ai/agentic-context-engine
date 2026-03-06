#!/usr/bin/env python3
"""Smoke test — verify API connectivity and response parsing for each target model.

Runs a minimal experiment (2 traces, 1 budget, 1 run) per model and checks
that run_info.json is produced with the expected number of traces.

Usage:
    uv run python scripts/smoke_test_models.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "label": "sonnet-4.6",
        "model": "bedrock/us.anthropic.claude-sonnet-4-6",
    },
    {
        "label": "gpt5-mini",
        "model": "gpt-5-mini",
    },
]

SCRIPT = ROOT / "scripts" / "run_variance_experiment.py"
MAX_TRACES = 2
BUDGET = "2000"
RUNS = "1"


def run_smoke_test(label: str, model: str, output_dir: Path) -> bool:
    """Run a minimal experiment and return True if it succeeded."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--model",
        model,
        "--output-dir",
        str(output_dir),
        "--max-traces",
        str(MAX_TRACES),
        "--budgets",
        BUDGET,
        "--runs",
        RUNS,
    ]
    print(f"\n{'='*60}")
    print(f"  Smoke test: {label}")
    print(f"  Model: {model}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=False,
        timeout=300,  # 5 min timeout
    )

    if result.returncode != 0:
        print(f"  FAIL: process exited with code {result.returncode}")
        return False

    # Check run_info.json
    run_info_path = output_dir / f"budget-{BUDGET}" / f"run_{RUNS}" / "run_info.json"
    if not run_info_path.exists():
        print(f"  FAIL: {run_info_path} not found")
        return False

    try:
        info = json.loads(run_info_path.read_text())
        n_traces = len(info.get("traces", []))
        if n_traces != MAX_TRACES:
            print(f"  FAIL: expected {MAX_TRACES} traces, got {n_traces}")
            return False
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  FAIL: could not parse run_info.json: {exc}")
        return False

    print(f"  PASS: {n_traces} traces, {info.get('final_skills', '?')} skills")
    return True


def main() -> None:
    results: dict[str, bool] = {}

    for spec in MODELS:
        label = spec["label"]
        model = spec["model"]
        output_dir = Path(tempfile.mkdtemp(prefix=f"smoke_test_{label}_"))

        try:
            ok = run_smoke_test(label, model, output_dir)
        except subprocess.TimeoutExpired:
            print(f"  FAIL: timed out after 300s")
            ok = False
        except Exception as exc:
            print(f"  FAIL: {exc}")
            ok = False
        finally:
            # Clean up temp dir
            shutil.rmtree(output_dir, ignore_errors=True)

        results[label] = ok

    # Summary
    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for label, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {label}: {status}")
        if not ok:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
