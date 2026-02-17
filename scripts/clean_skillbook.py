#!/usr/bin/env python3
"""Clean v3 Haiku skillbook by removing 29 problematic skills.

Removes skills that:
- Reference hallucinated tools (get_available_flights, rebook_passenger, etc.)
- Encode fabricated policies (loyalty points, weather monitoring, insurance)
- Are harmful/counterproductive (false fraud assumptions, blocking alternatives)
- Were never validated as helpful (0 helpful, only neutral counts)

Also edits skill 00024 to remove reference to nonexistent cancellation_reason param.
"""

import json
from pathlib import Path

INPUT = Path(
    "tau_benchmark_results/tau_airline_claude-haiku-4-5-20251001_ace_20260209_154102_skillbook.json"
)
OUTPUT = Path("tau_benchmark_results/cleaned_haiku_skillbook.json")

# 29 skills to remove
REMOVE_IDS = {
    # Hallucinated tool names (16)
    "flight_rebooking-00017",  # get_available_flights
    "flight_rebooking-00022",  # rebook_passenger
    "reservation_management-00026",  # cancellation_reason param
    "reservation_management-00035",  # get_reservations_by_user_id
    "flight_modification-00047",  # change_reservation
    "reservation_management-00051",  # update_reservation (combined)
    "balance_management-00061",  # get_gift_card_balance
    "balance_management-00062",  # get_certificate_balance
    "balance_management-00063",  # depends on 00061
    "balance_management-00064",  # depends on 00062
    "flight_modification-00072",  # modify_reservation
    "flight_modification-00074",  # modify_reservation (verify date)
    "flight_modification-00075",  # modify_reservation (confirm itinerary)
    "reservation_management-00087",  # get_reservations_by_user_id (bulk)
    "reservation_management-00094",  # get_current_time
    "reservation_management-00095",  # depends on get_current_time
    # Hallucinated policies/systems (6)
    "loyalty_compensation-00019",  # 5,000 loyalty points system
    "complaint_resolution-00085",  # weather monitoring, 30-min buffer
    "reservation_management-00105",  # insurance claims eligibility
    "reservation_management-00106",  # 5-7 business days insurance refund
    "reservation_management-00107",  # medical documentation requirement
    "reservation_management-00108",  # insurance vs airline distinction
    # Harmful/counterproductive (2)
    "fraud_prevention-00016",  # assumes fraud on missing details
    "complaint_resolution-00104",  # blocks offering alternatives
    # Never validated (5)
    "reservation_management-00027",  # helpful=0, neutral=1
    "flight_rebooking-00028",  # helpful=0, neutral=1
    "complaint_resolution-00029",  # helpful=0, neutral=1
    "reservation_management-00030",  # helpful=0, neutral=1
    "reservation_management-00096",  # all zeros
}


def main():
    data = json.loads(INPUT.read_text())

    before = len(data["skills"])
    assert before == 108, f"Expected 108 skills, got {before}"

    # Remove skills
    for sid in REMOVE_IDS:
        removed = data["skills"].pop(sid, None)
        assert removed is not None, f"Skill {sid} not found"

    # Edit skill 00024: remove "and cancellation_reason parameter"
    s24 = data["skills"]["reservation_management-00024"]
    old = s24["content"]
    s24["content"] = old.replace(" and cancellation_reason parameter", "")
    assert "cancellation_reason" not in s24["content"], "Edit failed"

    # Rebuild sections index
    new_sections = {}
    for sid, skill in data["skills"].items():
        sec = skill["section"]
        new_sections.setdefault(sec, []).append(sid)
    data["sections"] = new_sections

    after = len(data["skills"])
    assert after == 79, f"Expected 79 skills, got {after}"

    # Verify empty sections are gone
    assert (
        "balance_management" not in data["sections"]
    ), "balance_management should be removed"
    assert (
        "loyalty_compensation" not in data["sections"]
    ), "loyalty_compensation should be removed"

    # Verify no hallucinated tool references remain in any skill
    hallucinated = [
        "get_gift_card_balance",
        "get_certificate_balance",
        "get_available_flights",
        "rebook_passenger",
        "change_reservation",
        "modify_reservation",
        "get_current_time",
        "get_reservations_by_user_id",
    ]
    for sid, skill in data["skills"].items():
        for h in hallucinated:
            assert h not in skill["content"], f"Skill {sid} still references {h}"

    OUTPUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Cleaned: {before} -> {after} skills ({before - after} removed)")
    print(f"Sections: {sorted(data['sections'].keys())}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
