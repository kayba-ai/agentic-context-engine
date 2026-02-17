#!/usr/bin/env python3
"""Clean v4 Haiku skillbook: remove harmful/duplicate skills, merge near-duplicates, update incomplete ones.

Changes from v3 (79 skills):
- Remove 2 harmful skills contradicting airline policy (00001, 00002)
- Remove 5 duplicates (00068, 00080, 00078, 00020, 00101)
- Merge 00091+00092 into single name-change verification skill
- Update 00041 with urgency from deleted 00101
- Update 00009 with full baggage allowance table (3 tiers × 3 cabins)
- Update 00007 with cancellation eligibility rules

Expected result: 72 skills (79 - 7 removed)
"""

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path("tau_benchmark_results/cleaned_haiku_skillbook.json")
DEFAULT_OUTPUT = Path("tau_benchmark_results/cleaned_v4_haiku_skillbook.json")

# Skills to remove entirely
REMOVE_IDS = {
    # Harmful: contradicts airline policy
    "reservation_management-00001",  # "Accept contextual justifications for policy exceptions"
    "reservation_management-00002",  # "Process cancellations without additional verification"
    # Duplicate of 00004+00011: "request IDs upfront"
    "reservation_management-00068",
    "task_management-00080",
    # Duplicate of 00033: "modification sequencing"
    "flight_modification-00078",
    # Duplicate of 00069: "identity verification"
    "fraud_prevention-00020",
    # Merged into 00041: "task transition timing"
    "task_management-00101",
}

# Merge 00091 + 00092 → keep 00092 with merged content, remove 00091
MERGE_REMOVE = "passenger_management-00091"


def main():
    parser = argparse.ArgumentParser(description="Clean skillbook v4")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    before = len(data["skills"])
    assert before == 79, f"Expected 79 skills, got {before}"

    # 1. Remove harmful and duplicate skills
    for sid in REMOVE_IDS:
        removed = data["skills"].pop(sid, None)
        assert removed is not None, f"Skill {sid} not found"

    # 2. Merge 00091 into 00092
    assert MERGE_REMOVE in data["skills"], f"{MERGE_REMOVE} not found"
    data["skills"].pop(MERGE_REMOVE)
    s92 = data["skills"]["passenger_management-00092"]
    s92["content"] = (
        "Verify old and new passenger names explicitly before executing name change: "
        "confirm current name on reservation matches, then confirm new name with user"
    )
    s92["helpful"] = 2  # sum of both

    # 3. Update 00041: add urgency from 00101
    s41 = data["skills"]["task_management-00041"]
    s41["content"] = (
        "Transition explicitly between tasks with confirmation before moving to the next task; "
        "move from information gathering to action execution within the first few conversation steps"
    )

    # 4. Update 00009: full baggage allowance table (3 tiers × 3 cabins)
    s09 = data["skills"]["baggage_policy-00009"]
    s09["content"] = (
        "Baggage allowance (checked bags per person): "
        "Regular — economy:1, business:2, first:3; "
        "Silver — economy:2, business:3, first:3; "
        "Gold — economy:3, business:3, first:3. "
        "All tiers get 1 free carry-on and 1 personal item."
    )

    # 5. Update 00007: add cancellation eligibility rules
    s07 = data["skills"]["reservation_management-00007"]
    s07["content"] = (
        "Follow all required flight_cancellation workflow steps sequentially: "
        "identify_reservation, check_cancellation_eligibility, confirm_cancellation_policy, "
        "process_cancellation, provide_confirmation. "
        "Cancellation is only allowed if: booked within last 24 hours, OR flight was cancelled by airline, "
        "OR cabin is business/first, OR reservation has travel insurance. "
        "The API does NOT enforce eligibility — the agent must check these conditions."
    )

    # 6. Rebuild sections index
    new_sections: dict[str, list[str]] = {}
    for sid, skill in data["skills"].items():
        sec = skill["section"]
        new_sections.setdefault(sec, []).append(sid)
    data["sections"] = new_sections

    after = len(data["skills"])
    expected = before - len(REMOVE_IDS) - 1  # -1 for merge
    assert after == expected, f"Expected {expected} skills, got {after}"

    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Cleaned: {before} -> {after} skills ({before - after} removed/merged)")
    print(f"Sections: {sorted(data['sections'].keys())}")
    print(f"Updated skills: 00007, 00009, 00041, 00092")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
