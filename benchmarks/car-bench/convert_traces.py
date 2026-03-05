#!/usr/bin/env python3
"""
Extract traj from CAR-bench result JSON files into individual JSON files for ACE.

Usage:
    python convert_traces.py results/base_train/model.json -o traces/
    python convert_traces.py results/base_train/*.json results/hallucination_train/*.json -o traces/
"""

import argparse
import json
from pathlib import Path


def process_file(filepath, output_dir):
    filepath = Path(filepath)
    with open(filepath) as f:
        results = json.load(f)

    count = 0
    for entry in results:
        task_id = entry.get("task_id", "unknown")
        trial = entry.get("trial", 0)
        traj = entry.get("traj", [])

        if not traj:
            continue

        out = {"task_id": task_id, "traj": traj}

        out_path = output_dir / f"{task_id}_trial{trial}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("traces"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for filepath in args.result_files:
        if not filepath.exists():
            print(f"Not found: {filepath}")
            continue
        print(f"Processing {filepath.name}...")
        count = process_file(filepath, args.output_dir)
        print(f"  Wrote {count} files")
        total += count

    print(f"\nDone. {total} files in {args.output_dir}/")


if __name__ == "__main__":
    main()
