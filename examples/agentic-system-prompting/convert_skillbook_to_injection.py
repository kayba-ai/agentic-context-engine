#!/usr/bin/env python3
"""
Convert a skillbook JSON file to an external agent injection text file.

Usage:
    python convert_skillbook_to_injection.py /path/to/skillbook.json
    python convert_skillbook_to_injection.py /path/to/skillbook.json -o output.txt
"""

import argparse
from pathlib import Path

from ace import Skillbook
from ace.prompts_v3 import wrap_skillbook_for_external_agent


def main():
    parser = argparse.ArgumentParser(
        description="Convert a skillbook JSON to external agent injection format"
    )
    parser.add_argument("skillbook", type=Path, help="Path to skillbook JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <skillbook_name>_injection.txt)",
    )
    args = parser.parse_args()

    if not args.skillbook.exists():
        print(f"Error: Skillbook not found: {args.skillbook}")
        return 1

    skillbook = Skillbook.load_from_file(str(args.skillbook))
    print(f"Loaded skillbook with {len(skillbook.skills())} skills")

    injection_text = wrap_skillbook_for_external_agent(skillbook)

    if not injection_text:
        print("No skills in skillbook - nothing to output")
        return 1

    output_path = args.output or args.skillbook.with_name(
        f"{args.skillbook.stem}_injection.txt"
    )
    output_path.write_text(injection_text)
    print(f"Wrote injection to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
