#!/usr/bin/env python3
"""
JSON to TOON Converter

Converts all JSON files in a directory to TOON format (token-efficient encoding).

Usage:
    python convert.py /path/to/json/files
"""

import argparse
import json
from pathlib import Path
from toon import encode

parser = argparse.ArgumentParser(description="Convert JSON files to TOON format")
parser.add_argument("input_dir", type=Path, help="Directory containing JSON files")
args = parser.parse_args()

input_dir = args.input_dir
json_files = list(input_dir.glob("*.json"))
print(f"Converting {len(json_files)} JSON files to TOON format...")

for json_path in sorted(json_files):
    toon_path = input_dir / (json_path.stem + ".toon")
    original_size = json_path.stat().st_size

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    toon_content = encode(data)

    with open(toon_path, "w", encoding="utf-8") as f:
        f.write(toon_content)

    new_size = toon_path.stat().st_size
    savings = ((original_size - new_size) / original_size) * 100
    print(f"  {json_path.name} -> {toon_path.name} ({savings:.1f}% smaller)")

print("Done!")
