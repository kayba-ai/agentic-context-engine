"""Backfill embeddings into skillbook snapshots that are missing them.

Computes text-embedding-3-small embeddings via litellm for all skills
with embedding=None in final snapshots (skillbook_24.json) of budget runs.

Usage:
    uv run python scripts/backfill_embeddings.py
"""

import json
import sys
from pathlib import Path

import litellm

EXPERIMENT_DIR = Path(__file__).parent.parent / "results" / "variance_experiment"
EMBEDDING_MODEL = "text-embedding-3-small"
BUDGETS_TO_BACKFILL = [
    "budget-500",
    "budget-1000",
    "budget-2000",
    "budget-3000",
    "budget-5000",
    "budget-10000",
]


def skill_text(skill: dict) -> str:
    """Build the text to embed — same as deduplication uses: section + content."""
    parts = [skill.get("section", ""), skill.get("content", "")]
    if skill.get("justification"):
        parts.append(skill["justification"])
    return " | ".join(parts)


def backfill_file(path: Path) -> int:
    """Compute and store missing embeddings in a single skillbook JSON. Returns count."""
    with open(path) as f:
        raw = json.load(f)

    skills = raw.get("skills", {})
    to_embed = {sid: s for sid, s in skills.items() if s.get("embedding") is None}

    if not to_embed:
        return 0

    sids = list(to_embed.keys())
    texts = [skill_text(to_embed[sid]) for sid in sids]

    # Batch embed
    response = litellm.embedding(model=EMBEDDING_MODEL, input=texts)
    embeddings = [item["embedding"] for item in response.data]

    for sid, emb in zip(sids, embeddings):
        raw["skills"][sid]["embedding"] = emb

    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    return len(sids)


def main():
    total = 0
    for budget in BUDGETS_TO_BACKFILL:
        budget_dir = EXPERIMENT_DIR / budget
        if not budget_dir.is_dir():
            continue
        for run_dir in sorted(budget_dir.glob("run_*")):
            path = run_dir / "skillbook_24.json"
            if not path.exists():
                continue
            n = backfill_file(path)
            total += n
            print(f"  {budget}/{run_dir.name}: {n} embeddings computed")

    print(
        f"\nDone — {total} embeddings backfilled across {len(BUDGETS_TO_BACKFILL)} budgets"
    )


if __name__ == "__main__":
    main()
