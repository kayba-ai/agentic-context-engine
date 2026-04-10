#!/usr/bin/env python3
"""Extract metadata from CAR-bench traces for batch classification.

Parses all .toon trace files and outputs a JSON summary with:
- task_id, task_type, file size
- tools used by the agent
- user request (first user message)
- number of turns, tool calls
- whether the agent succeeded or had errors

Output: results/traces_car_bench/trace_metadata.json
"""

import json
import re
from pathlib import Path

TRACES_DIR = Path(__file__).resolve().parents[1] / "results" / "traces_car_bench"


def parse_toon(path: Path) -> dict:
    """Parse a .toon trace file and extract metadata."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    task_id = ""
    tools_used: list[str] = []
    user_messages: list[str] = []
    assistant_messages: list[str] = []
    tool_call_count = 0
    has_error = False
    error_details: list[str] = []

    # Extract task_id from first line
    if lines and lines[0].startswith("task_id:"):
        task_id = lines[0].split(":", 1)[1].strip()

    # Parse roles and content
    current_role = None
    current_content = ""
    in_content = False

    for line in lines:
        # Detect role transitions
        role_match = re.match(r'\s+- role: (\w+)', line)
        if role_match:
            # Save previous
            if current_role == "user" and current_content:
                user_messages.append(current_content.strip())
            elif current_role == "assistant" and current_content:
                assistant_messages.append(current_content.strip())
            current_role = role_match.group(1)
            current_content = ""
            in_content = False
            continue

        # Detect content lines
        content_match = re.match(r'\s+content: ["\']?(.+)', line)
        if content_match and current_role:
            current_content = content_match.group(1).rstrip('"\'')
            in_content = True
            continue

        # Detect tool names
        tool_match = re.match(r'\s+name: (\w+)', line)
        if tool_match:
            tool_name = tool_match.group(1)
            if current_role == "tool":
                # This is a tool response
                pass
            else:
                # This is a tool call from assistant
                tools_used.append(tool_name)
                tool_call_count += 1

        # Detect errors in tool responses
        if current_role == "tool" and '"status": "ERROR"' in line:
            has_error = True
        if current_role == "tool" and '"error"' in line.lower():
            error_details.append(line.strip()[:200])

    # Save last message
    if current_role == "user" and current_content:
        user_messages.append(current_content.strip())
    elif current_role == "assistant" and current_content:
        assistant_messages.append(current_content.strip())

    # Filter out ###STOP### and system prompt from user messages
    user_messages = [
        m for m in user_messages
        if m != "###STOP###" and not m.startswith("# In-Car Assistant")
    ]

    # Deduplicate tools (keep order)
    seen = set()
    unique_tools = []
    for t in tools_used:
        if t not in seen:
            seen.add(t)
            unique_tools.append(t)

    # Infer task type from task_id
    task_type = task_id.split("_")[0] if task_id else "unknown"

    # Infer domain from tools used
    domain_tags = set()
    nav_tools = {
        "search_location", "get_routes", "set_navigation", "get_navigation_status",
        "search_poi", "search_poi_along_the_route", "add_waypoint",
        "delete_waypoint", "replace_waypoint", "add_destination",
        "replace_destination", "delete_destination",
    }
    vehicle_tools = {
        "open_close_sunroof", "open_close_sunshade", "get_sunroof_and_sunshade_position",
        "open_close_window", "get_window_position",
        "set_fan_speed", "get_fan_speed", "set_fan_airflow_direction",
        "get_fan_airflow_direction", "set_temperature", "get_temperature",
        "set_air_conditioning", "get_air_conditioning_status",
        "set_window_defrost", "get_window_defrost_status",
        "set_fog_lights", "get_fog_lights_status",
        "set_headlights", "get_headlights_status",
        "set_low_beam_headlights", "get_low_beam_headlights_status",
        "set_high_beam_headlights", "get_high_beam_headlights_status",
    }
    productivity_tools = {
        "get_calendar", "get_contacts", "send_email", "call_phone_number",
        "search_contacts",
    }
    charging_tools = {
        "get_charging_status", "get_battery_status", "search_charging_station",
        "get_range",
    }
    weather_tools = {"get_weather"}
    utility_tools = {
        "planning", "get_user_preferences", "think", "calculate",
    }

    for t in unique_tools:
        if t in nav_tools:
            domain_tags.add("navigation")
        elif t in vehicle_tools:
            domain_tags.add("vehicle_control")
        elif t in productivity_tools:
            domain_tags.add("productivity")
        elif t in charging_tools:
            domain_tags.add("charging")
        elif t in weather_tools:
            domain_tags.add("weather")
        elif t in utility_tools:
            domain_tags.add("utility")
        else:
            domain_tags.add(f"unknown:{t}")

    return {
        "task_id": task_id,
        "task_type": task_type,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "user_request": user_messages[0][:300] if user_messages else "",
        "num_user_messages": len(user_messages),
        "num_assistant_messages": len(assistant_messages),
        "num_tool_calls": tool_call_count,
        "tools_used": unique_tools,
        "domain_tags": sorted(domain_tags),
        "has_error": has_error,
        "error_details": error_details[:3],
    }


def main():
    traces = sorted(TRACES_DIR.glob("*.toon"))
    print(f"Found {len(traces)} trace files")

    metadata = []
    for t in traces:
        meta = parse_toon(t)
        metadata.append(meta)

    # Summary stats
    by_type = {}
    by_domain = {}
    for m in metadata:
        by_type.setdefault(m["task_type"], []).append(m["task_id"])
        for d in m["domain_tags"]:
            by_domain.setdefault(d, []).append(m["task_id"])

    print(f"\nBy task type:")
    for t, ids in sorted(by_type.items()):
        print(f"  {t}: {len(ids)}")

    print(f"\nBy domain (may overlap):")
    for d, ids in sorted(by_domain.items()):
        print(f"  {d}: {len(ids)}")

    errors = [m for m in metadata if m["has_error"]]
    print(f"\nTraces with errors: {len(errors)}")

    # Save full metadata
    out_path = TRACES_DIR / "trace_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Also save a compact summary for Opus classification
    compact = []
    for m in metadata:
        compact.append({
            "task_id": m["task_id"],
            "task_type": m["task_type"],
            "user_request": m["user_request"],
            "tools": m["tools_used"],
            "domains": m["domain_tags"],
            "turns": m["num_user_messages"],
            "tool_calls": m["num_tool_calls"],
            "has_error": m["has_error"],
        })

    compact_path = TRACES_DIR / "trace_summary_for_classification.json"
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)
    print(f"Saved compact summary: {compact_path}")


if __name__ == "__main__":
    main()
