"""Skillbook history analysis for variance experiment results.

Provides data loading, diffing, metrics, cross-run convergence analysis,
embedding-based clustering, plotting utilities, and automated report
generation for the 7-budget × 5-run × 25-snapshot experiment.

Usage (from scripts/ directory):
    from analysis.skillbook_history import load_experiment
    exp = load_experiment(Path("../results/variance_experiment"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TRACES = 25

BUDGET_ORDER = [
    "budget-500",
    "budget-1000",
    "budget-2000",
    "budget-3000",
    "budget-5000",
    "budget-10000",
    "no-budget",
]

BUDGET_COLORS: dict[str, str] = {
    "budget-500": "#dc2626",
    "budget-1000": "#ea580c",
    "budget-2000": "#ca8a04",
    "budget-3000": "#16a34a",
    "budget-5000": "#2563eb",
    "budget-10000": "#7c3aed",
    "no-budget": "#6b7280",
}

BUDGET_VALUES: dict[str, int | None] = {
    "budget-500": 500,
    "budget-1000": 1000,
    "budget-2000": 2000,
    "budget-3000": 3000,
    "budget-5000": 5000,
    "budget-10000": 10000,
    "no-budget": None,
}

_enc = tiktoken.encoding_for_model("gpt-4")


def _tok(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Topic mapping: 207 raw section names → 18 canonical topics
# ---------------------------------------------------------------------------

TOPIC_MAP: dict[str, str] = {
    # Core domain
    "cancellation": "cancellation",
    "compensation": "compensation",
    "delay": "compensation",
    "modification": "modification",
    "passenger": "modification",
    "cabin": "modification",
    "reservation": "reservation",
    "booking": "reservation",
    "conflict": "reservation",
    "flight": "flight_search",
    "search": "flight_search",
    "destination": "flight_search",
    "return": "flight_search",
    "payment": "payment",
    "refund": "payment",
    "pricing": "pricing",
    "cost": "pricing",
    "financial": "pricing",
    "calculation": "pricing",
    "calculations": "pricing",
    "baggage": "baggage",
    "membership": "baggage",
    "insurance": "insurance",
    # Process
    "escalation": "escalation",
    "handoff": "escalation",
    "transfer": "escalation",
    "triage": "escalation",
    "confirmation": "confirmation",
    "consent": "confirmation",
    "preference": "confirmation",
    "policy": "policy",
    "constraint": "policy",
    "eligibility": "policy",
    "boundary": "policy",
    "authorization": "policy",
    "technical": "policy",
    "airline": "policy",
    "tool": "tool_usage",
    "tools": "tool_usage",
    "transaction": "tool_usage",
    "execution": "tool_usage",
    "action": "tool_usage",
    "interaction": "tool_usage",
    "api": "tool_usage",
    "customer": "customer_comms",
    "communication": "customer_comms",
    "user": "customer_comms",
    "message": "customer_comms",
    "capability": "customer_comms",
    "clarification": "customer_comms",
    "data": "data_retrieval",
    "information": "data_retrieval",
    "discovery": "data_retrieval",
    "workflow": "workflow",
    "task": "workflow",
    "decision": "decision_support",
    "option": "decision_support",
    "alternative": "decision_support",
    "alternatives": "decision_support",
    "problem": "decision_support",
    "temporal": "temporal",
    "date": "temporal",
}


def section_to_topic(section_name: str) -> str:
    """Map a raw section name to its canonical topic via prefix lookup."""
    prefix = section_name.split("_")[0] if "_" in section_name else section_name
    return TOPIC_MAP.get(prefix, prefix)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class DeltaType(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    REMOVE = "REMOVE"


@dataclass
class SkillRecord:
    """Lightweight representation of a single skill (no embedding)."""

    id: str
    section: str
    content: str
    justification: str | None = None
    evidence: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SkillDelta:
    delta_type: DeltaType
    skill_id: str
    trace_index: int
    section: str
    old_content: str | None = None
    new_content: str | None = None


@dataclass
class SnapshotMetrics:
    trace_index: int
    skill_count: int
    section_count: int
    section_names: list[str]
    next_id: int
    toon_tokens: int
    injection_tokens: int
    avg_content_len: float
    avg_justification_len: float
    avg_evidence_len: float


@dataclass
class RunHistory:
    budget_label: str
    run_num: int
    run_dir: Path
    snapshots: list[dict[str, SkillRecord]] = field(default_factory=list)
    metrics: list[SnapshotMetrics] = field(default_factory=list)
    deltas: list[SkillDelta] = field(default_factory=list)
    run_info: dict[str, Any] = field(default_factory=dict)
    next_ids: list[int] = field(default_factory=list)


@dataclass
class BudgetGroup:
    budget_label: str
    budget_value: int | None
    runs: list[RunHistory] = field(default_factory=list)

    def mean_metric(self, fn: Callable[[RunHistory], Sequence[float]]) -> np.ndarray:
        """Mean of a per-run metric (each fn(run) returns a sequence)."""
        arrays = [np.array(fn(r)) for r in self.runs]
        return np.mean(arrays, axis=0)

    def std_metric(self, fn: Callable[[RunHistory], Sequence[float]]) -> np.ndarray:
        arrays = [np.array(fn(r)) for r in self.runs]
        return np.std(arrays, axis=0)


@dataclass
class Experiment:
    budget_groups: list[BudgetGroup] = field(default_factory=list)
    experiment_dir: Path = field(default_factory=Path)

    def get_budget(self, label: str) -> BudgetGroup | None:
        for g in self.budget_groups:
            if g.budget_label == label:
                return g
        return None

    @property
    def labels(self) -> list[str]:
        return [g.budget_label for g in self.budget_groups]


# ---------------------------------------------------------------------------
# Loading pipeline
# ---------------------------------------------------------------------------


def load_snapshot_raw(path: Path, skip_embeddings: bool = True) -> dict[str, Any]:
    """Parse a skillbook JSON file, optionally stripping embeddings."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if skip_embeddings:
        for skill in raw.get("skills", {}).values():
            skill["embedding"] = None
    return raw


def parse_skills(raw: dict[str, Any]) -> dict[str, SkillRecord]:
    """Extract lightweight SkillRecords from raw JSON dict."""
    records: dict[str, SkillRecord] = {}
    for sid, s in raw.get("skills", {}).items():
        if s.get("status", "active") != "active":
            continue
        records[sid] = SkillRecord(
            id=sid,
            section=s.get("section", ""),
            content=s.get("content", ""),
            justification=s.get("justification"),
            evidence=s.get("evidence"),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
        )
    return records


def compute_snapshot_metrics(
    skills: dict[str, SkillRecord],
    next_id: int,
    trace_index: int,
    run_info_trace: dict[str, Any] | None = None,
) -> SnapshotMetrics:
    """Compute metrics for a snapshot.  Uses run_info_trace for token counts
    (already computed during the experiment) and falls back to recomputing."""
    sections = sorted({s.section for s in skills.values()})
    content_lens = [len(s.content) for s in skills.values()]
    just_lens = [len(s.justification) for s in skills.values() if s.justification]
    ev_lens = [len(s.evidence) for s in skills.values() if s.evidence]

    if run_info_trace:
        toon_tok = run_info_trace.get("toon_tokens", 0)
        inj_tok = run_info_trace.get("injection_tokens", 0)
    else:
        toon_tok = 0
        inj_tok = 0

    return SnapshotMetrics(
        trace_index=trace_index,
        skill_count=len(skills),
        section_count=len(sections),
        section_names=sections,
        next_id=next_id,
        toon_tokens=toon_tok,
        injection_tokens=inj_tok,
        avg_content_len=np.mean(content_lens) if content_lens else 0.0,
        avg_justification_len=np.mean(just_lens) if just_lens else 0.0,
        avg_evidence_len=np.mean(ev_lens) if ev_lens else 0.0,
    )


def diff_snapshots(
    prev: dict[str, SkillRecord],
    curr: dict[str, SkillRecord],
    trace_index: int,
) -> list[SkillDelta]:
    """Diff consecutive snapshots → list of SkillDeltas."""
    deltas: list[SkillDelta] = []
    prev_ids = set(prev.keys())
    curr_ids = set(curr.keys())

    for sid in curr_ids - prev_ids:
        deltas.append(
            SkillDelta(
                delta_type=DeltaType.ADD,
                skill_id=sid,
                trace_index=trace_index,
                section=curr[sid].section,
                new_content=curr[sid].content,
            )
        )
    for sid in prev_ids - curr_ids:
        deltas.append(
            SkillDelta(
                delta_type=DeltaType.REMOVE,
                skill_id=sid,
                trace_index=trace_index,
                section=prev[sid].section,
                old_content=prev[sid].content,
            )
        )
    for sid in prev_ids & curr_ids:
        if prev[sid].content != curr[sid].content:
            deltas.append(
                SkillDelta(
                    delta_type=DeltaType.UPDATE,
                    skill_id=sid,
                    trace_index=trace_index,
                    section=curr[sid].section,
                    old_content=prev[sid].content,
                    new_content=curr[sid].content,
                )
            )
    return deltas


def load_run(run_dir: Path, budget_label: str, run_num: int) -> RunHistory:
    """Load all snapshots + run_info for a single run."""
    history = RunHistory(budget_label=budget_label, run_num=run_num, run_dir=run_dir)

    # Load run_info.json
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        with open(info_path) as f:
            history.run_info = json.load(f)

    traces_info = history.run_info.get("traces", [])

    # Load snapshots
    snap_paths = sorted(run_dir.glob("skillbook_*.json"))
    for i, path in enumerate(snap_paths):
        raw = load_snapshot_raw(path)
        skills = parse_skills(raw)
        history.snapshots.append(skills)
        history.next_ids.append(raw.get("next_id", 0))

        trace_info = traces_info[i] if i < len(traces_info) else None
        metrics = compute_snapshot_metrics(skills, raw.get("next_id", 0), i, trace_info)
        history.metrics.append(metrics)

    # Compute deltas between consecutive snapshots
    for i in range(1, len(history.snapshots)):
        deltas = diff_snapshots(history.snapshots[i - 1], history.snapshots[i], i)
        history.deltas.extend(deltas)

    # Also record initial snapshot as ADDs
    if history.snapshots:
        for sid, skill in history.snapshots[0].items():
            history.deltas.append(
                SkillDelta(
                    delta_type=DeltaType.ADD,
                    skill_id=sid,
                    trace_index=0,
                    section=skill.section,
                    new_content=skill.content,
                )
            )

    return history


def load_budget_group(budget_dir: Path, budget_label: str) -> BudgetGroup:
    """Load all runs for a single budget level."""
    group = BudgetGroup(
        budget_label=budget_label,
        budget_value=BUDGET_VALUES.get(budget_label),
    )
    run_dirs = sorted(budget_dir.glob("run_*"))
    for rd in run_dirs:
        run_num = int(rd.name.split("_")[1])
        group.runs.append(load_run(rd, budget_label, run_num))
    return group


def load_experiment(
    experiment_dir: Path, budgets: list[str] | None = None
) -> Experiment:
    """Load the full experiment.  *budgets* filters to specific labels."""
    exp = Experiment(experiment_dir=experiment_dir)
    labels = budgets or BUDGET_ORDER
    for label in labels:
        bd = experiment_dir / label
        if bd.is_dir():
            exp.budget_groups.append(load_budget_group(bd, label))
    return exp


def load_compression_metrics(
    experiment_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Load Opus compression metrics from ``opus_compressed/compression_metrics.json``.

    Returns the raw dict keyed by entry name (e.g. ``"budget-500_run_1"``,
    ``"consensus_budget-500"``).  Each value contains ``type``, ``budget``,
    ``sections``, ``skills``, ``md_chars``, ``md_tokens_tiktoken``, and
    optionally ``run``.
    """
    path = experiment_dir / "opus_compressed" / "compression_metrics.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compression_distribution(
    experiment_dir: Path,
) -> dict[str, dict[str, float]]:
    """Return mean ± std of compression metrics per budget (individual runs only).

    Includes both compressed (Opus) metrics and raw (uncompressed) metrics from
    the source ``skills_final.md`` files.  Also computes compression percentage
    (``compression_pct``) per run.

    Returns ``{budget: {metric_mean, metric_std, ...}}`` where metrics include:

    * Opus-compressed: ``skills``, ``md_chars``, ``md_tokens_tiktoken``
    * Raw source: ``raw_skills``, ``raw_md_tokens``
    * Ratio: ``compression_pct`` (opus_tokens / raw_tokens * 100)
    """
    import re
    import statistics as _st

    try:
        import tiktoken

        _enc = tiktoken.encoding_for_model("gpt-4")
    except ImportError:
        _enc = None

    metrics = load_compression_metrics(experiment_dir)
    by_budget: dict[str, list[dict]] = {}
    for entry in metrics.values():
        if entry["type"] != "individual":
            continue
        by_budget.setdefault(entry["budget"], []).append(entry)

    # Enrich each entry with raw source metrics
    for budget, entries in by_budget.items():
        for entry in entries:
            src = experiment_dir / budget / f"run_{entry['run']}" / "skills_final.md"
            if src.exists():
                text = src.read_text(encoding="utf-8")
                entry["raw_skills"] = len(re.findall(r"^- ", text, re.MULTILINE))
                if _enc is not None:
                    raw_toks = len(_enc.encode(text))
                    entry["raw_md_tokens"] = raw_toks
                    opus_toks = entry["md_tokens_tiktoken"]
                    entry["compression_pct"] = (
                        (opus_toks / raw_toks * 100) if raw_toks > 0 else 0.0
                    )

    result: dict[str, dict[str, float]] = {}
    keys = ["sections", "skills", "md_chars", "md_tokens_tiktoken"]
    if _enc is not None:
        keys += ["raw_skills", "raw_md_tokens", "compression_pct"]
    else:
        keys += ["raw_skills"]

    for budget in BUDGET_ORDER:
        entries = by_budget.get(budget, [])
        if not entries:
            continue
        n = len(entries)
        for key in keys:
            vals = [e[key] for e in entries if key in e]
            if not vals:
                continue
            result.setdefault(budget, {})[f"{key}_mean"] = _st.mean(vals)
            result[budget][f"{key}_std"] = _st.stdev(vals) if len(vals) > 1 else 0.0
        result[budget]["n"] = n
    return result


# ---------------------------------------------------------------------------
# Per-run metric extractors  (return list[int|float] of length NUM_TRACES)
# ---------------------------------------------------------------------------


def skill_counts(run: RunHistory) -> list[int]:
    return [m.skill_count for m in run.metrics]


def section_counts(run: RunHistory) -> list[int]:
    return [m.section_count for m in run.metrics]


def toon_token_counts(run: RunHistory) -> list[int]:
    return [m.toon_tokens for m in run.metrics]


def next_id_counts(run: RunHistory) -> list[int]:
    return run.next_ids


def tokens_per_skill(run: RunHistory) -> list[float]:
    out: list[float] = []
    for m in run.metrics:
        out.append(m.toon_tokens / m.skill_count if m.skill_count else 0.0)
    return out


def avg_content_lengths(run: RunHistory) -> list[float]:
    return [m.avg_content_len for m in run.metrics]


# ---------------------------------------------------------------------------
# Skill lifecycle analysis
# ---------------------------------------------------------------------------


def delta_counts_by_trace(run: RunHistory) -> dict[DeltaType, np.ndarray]:
    """Count ADD/UPDATE/REMOVE deltas at each trace index."""
    counts = {dt: np.zeros(NUM_TRACES) for dt in DeltaType}
    for d in run.deltas:
        if d.trace_index < NUM_TRACES:
            counts[d.delta_type][d.trace_index] += 1
    return counts


def skill_survival(run: RunHistory) -> dict[int, float]:
    """For skills added at each trace_index, what fraction survives to the final snapshot?"""
    if not run.snapshots:
        return {}
    final_ids = set(run.snapshots[-1].keys())
    added_at: dict[int, list[str]] = {}
    for d in run.deltas:
        if d.delta_type == DeltaType.ADD:
            added_at.setdefault(d.trace_index, []).append(d.skill_id)
    result: dict[int, float] = {}
    for ti, sids in sorted(added_at.items()):
        surviving = sum(1 for s in sids if s in final_ids)
        result[ti] = surviving / len(sids) if sids else 0.0
    return result


def skill_lifespans(run: RunHistory) -> list[int]:
    """Lifespan of each skill that was added and later removed (in trace steps)."""
    add_trace: dict[str, int] = {}
    remove_trace: dict[str, int] = {}
    for d in run.deltas:
        if d.delta_type == DeltaType.ADD:
            add_trace.setdefault(d.skill_id, d.trace_index)
        elif d.delta_type == DeltaType.REMOVE:
            remove_trace[d.skill_id] = d.trace_index
    spans: list[int] = []
    for sid, added in add_trace.items():
        if sid in remove_trace:
            spans.append(remove_trace[sid] - added)
    return spans


def churn_per_trace(run: RunHistory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(adds, removes, net_change) arrays of length NUM_TRACES."""
    adds = np.zeros(NUM_TRACES)
    removes = np.zeros(NUM_TRACES)
    for d in run.deltas:
        if d.trace_index < NUM_TRACES:
            if d.delta_type == DeltaType.ADD:
                adds[d.trace_index] += 1
            elif d.delta_type == DeltaType.REMOVE:
                removes[d.trace_index] += 1
    return adds, removes, adds - removes


# ---------------------------------------------------------------------------
# Section evolution
# ---------------------------------------------------------------------------


def section_first_appearance(run: RunHistory) -> dict[str, int]:
    """Section name → first trace_index where it appears."""
    first: dict[str, int] = {}
    for i, m in enumerate(run.metrics):
        for s in m.section_names:
            if s not in first:
                first[s] = i
    return first


def section_sizes_over_time(run: RunHistory) -> dict[str, list[int]]:
    """Section name → list[int(NUM_TRACES)] of skill counts in that section."""
    all_sections: set[str] = set()
    for snap in run.snapshots:
        for skill in snap.values():
            all_sections.add(skill.section)

    result: dict[str, list[int]] = {s: [] for s in sorted(all_sections)}
    for snap in run.snapshots:
        sec_counts: dict[str, int] = {}
        for skill in snap.values():
            sec_counts[skill.section] = sec_counts.get(skill.section, 0) + 1
        for s in result:
            result[s].append(sec_counts.get(s, 0))
    return result


# ---------------------------------------------------------------------------
# Cross-run convergence
# ---------------------------------------------------------------------------


def _section_names_to_topics(section_names: Sequence[str]) -> set[str]:
    """Map a list of raw section names to the set of canonical topics."""
    return {section_to_topic(s) for s in section_names}


# ---------------------------------------------------------------------------
# Embedding analysis (lazy — only loads final snapshots)
# ---------------------------------------------------------------------------


def load_final_embeddings(run: RunHistory) -> dict[str, np.ndarray]:
    """Load embeddings from the last skillbook snapshot file."""
    snap_paths = sorted(run.run_dir.glob("skillbook_*.json"))
    if not snap_paths:
        return {}
    with open(snap_paths[-1]) as f:
        raw = json.load(f)
    embeddings: dict[str, np.ndarray] = {}
    for sid, s in raw.get("skills", {}).items():
        emb = s.get("embedding")
        if emb:
            embeddings[sid] = np.array(emb, dtype=np.float32)
    return embeddings


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of A (m,d) and B (n,d) → (m,n)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def cross_run_embedding_overlap(
    group: BudgetGroup,
) -> dict[str, Any]:
    """Pairwise nearest-neighbor similarity stats across runs."""
    embs = []
    for run in group.runs:
        embs.append(load_final_embeddings(run))

    n = len(embs)
    pair_stats: list[dict[str, float]] = []
    for i in range(n):
        if not embs[i]:
            continue
        ids_i = list(embs[i].keys())
        mat_i = np.stack([embs[i][k] for k in ids_i])
        for j in range(i + 1, n):
            if not embs[j]:
                continue
            ids_j = list(embs[j].keys())
            mat_j = np.stack([embs[j][k] for k in ids_j])
            sim = cosine_similarity_matrix(mat_i, mat_j)
            nn_i_to_j = sim.max(axis=1)
            nn_j_to_i = sim.max(axis=0)
            pair_stats.append(
                {
                    "run_i": group.runs[i].run_num,
                    "run_j": group.runs[j].run_num,
                    "mean_nn_i_to_j": float(nn_i_to_j.mean()),
                    "mean_nn_j_to_i": float(nn_j_to_i.mean()),
                    "mean_nn": float((nn_i_to_j.mean() + nn_j_to_i.mean()) / 2),
                    "median_nn": float(
                        np.median(np.concatenate([nn_i_to_j, nn_j_to_i]))
                    ),
                }
            )
    overall_nn = [p["mean_nn"] for p in pair_stats]
    return {
        "pairs": pair_stats,
        "overall_mean_nn": float(np.mean(overall_nn)) if overall_nn else 0.0,
        "overall_std_nn": float(np.std(overall_nn)) if overall_nn else 0.0,
    }


def cluster_final_skills(group: BudgetGroup, n_clusters: int = 10) -> dict[str, Any]:
    """KMeans cluster all final-snapshot skills across runs.

    Returns cluster labels, core clusters (present in majority of runs),
    and per-cluster run coverage.
    """
    from sklearn.cluster import KMeans

    all_embs: list[np.ndarray] = []
    all_run_ids: list[int] = []
    all_skill_ids: list[str] = []

    for run in group.runs:
        emb = load_final_embeddings(run)
        for sid, vec in emb.items():
            all_embs.append(vec)
            all_run_ids.append(run.run_num)
            all_skill_ids.append(sid)

    if len(all_embs) < n_clusters:
        return {"error": "too few skills for clustering", "n_skills": len(all_embs)}

    X = np.stack(all_embs)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Per-cluster: which runs contribute
    n_runs = len(group.runs)
    cluster_info: list[dict[str, Any]] = []
    for c in range(n_clusters):
        mask = labels == c
        runs_in_cluster = set(np.array(all_run_ids)[mask].tolist())
        cluster_info.append(
            {
                "cluster": c,
                "size": int(mask.sum()),
                "runs_present": sorted(runs_in_cluster),
                "run_coverage": len(runs_in_cluster) / n_runs,
            }
        )

    core = [ci for ci in cluster_info if ci["run_coverage"] >= 0.6]
    return {
        "n_clusters": n_clusters,
        "total_skills": len(all_embs),
        "clusters": cluster_info,
        "core_clusters": len(core),
        "labels": labels.tolist(),
        "skill_ids": all_skill_ids,
        "run_ids": all_run_ids,
    }


# ---------------------------------------------------------------------------
# Cross-budget comparison
# ---------------------------------------------------------------------------


def budget_comparison_table(
    experiment: Experiment,
    metric_fn: Callable[[RunHistory], Sequence[float]],
    final_only: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute mean/std of a metric across runs for each budget.

    If final_only=True, uses only the last value of each run's metric.
    Otherwise, uses all values (returns mean of means).
    """
    table: dict[str, dict[str, float]] = {}
    for group in experiment.budget_groups:
        vals: list[float] = []
        for run in group.runs:
            seq = metric_fn(run)
            if not seq:
                continue
            vals.append(float(seq[-1]) if final_only else float(np.mean(seq)))
        table[group.budget_label] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std": float(np.std(vals)) if vals else 0.0,
            "min": float(np.min(vals)) if vals else 0.0,
            "max": float(np.max(vals)) if vals else 0.0,
            "n": len(vals),
        }
    return table


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _color(label: str) -> str:
    return BUDGET_COLORS.get(label, "#333333")


def plot_growth_curves(
    experiment: Experiment,
    metric_fn: Callable[[RunHistory], Sequence[float]] = skill_counts,
    ylabel: str = "Skills",
    title: str = "Skill Growth",
) -> plt.Figure:
    """Per-budget panels (2×4 grid) showing individual runs + mean±std."""
    groups = experiment.budget_groups
    ncols = 4
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 4.5 * nrows), squeeze=False, sharey=True
    )

    x = np.arange(NUM_TRACES)
    for idx, group in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        color = _color(group.budget_label)
        arrays = [np.array(metric_fn(r)) for r in group.runs]
        for arr in arrays:
            ax.plot(x[: len(arr)], arr, color=color, alpha=0.25, linewidth=1)
        if arrays:
            mean = np.mean(arrays, axis=0)
            std = np.std(arrays, axis=0)
            ax.plot(x[: len(mean)], mean, color=color, linewidth=2)
            ax.fill_between(
                x[: len(mean)], mean - std, mean + std, color=color, alpha=0.15
            )
        ax.set_title(group.budget_label, fontsize=10)
        ax.set_xlabel("Trace")
        if idx % ncols == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(groups), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_cross_budget_overlay(
    experiment: Experiment,
    metric_fn: Callable[[RunHistory], Sequence[float]] = skill_counts,
    ylabel: str = "Skills",
    title: str = "Cross-Budget Overlay",
) -> plt.Figure:
    """Single panel: mean±std for each budget overlaid."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_TRACES)
    for group in experiment.budget_groups:
        color = _color(group.budget_label)
        arrays = [np.array(metric_fn(r)) for r in group.runs]
        if not arrays:
            continue
        mean = np.mean(arrays, axis=0)
        std = np.std(arrays, axis=0)
        ax.plot(
            x[: len(mean)], mean, color=color, linewidth=2, label=group.budget_label
        )
        ax.fill_between(x[: len(mean)], mean - std, mean + std, color=color, alpha=0.1)
    ax.set_xlabel("Trace")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_delta_bars(
    run: RunHistory,
    title: str | None = None,
) -> plt.Figure:
    """Stacked bars: ADD/UPDATE/REMOVE per trace for one run."""
    counts = delta_counts_by_trace(run)
    x = np.arange(NUM_TRACES)
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(x, counts[DeltaType.ADD], color="#16a34a", label="ADD")
    ax.bar(
        x,
        counts[DeltaType.UPDATE],
        bottom=counts[DeltaType.ADD],
        color="#2563eb",
        label="UPDATE",
    )
    ax.bar(
        x,
        -counts[DeltaType.REMOVE],
        color="#dc2626",
        label="REMOVE",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Deltas")
    ax.set_title(title or f"Deltas — {run.budget_label}/run_{run.run_num}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_survival_curve(experiment: Experiment) -> plt.Figure:
    """Survival curves: fraction of skills surviving vs. trace added."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for group in experiment.budget_groups:
        color = _color(group.budget_label)
        all_surv: list[dict[int, float]] = [skill_survival(r) for r in group.runs]
        # Collect per-trace-index across runs
        max_t = max((max(s.keys()) for s in all_surv if s), default=0)
        mean_surv = np.zeros(max_t + 1)
        cnt = np.zeros(max_t + 1)
        for s_dict in all_surv:
            for t, v in s_dict.items():
                mean_surv[t] += v
                cnt[t] += 1
        mask = cnt > 0
        mean_surv[mask] /= cnt[mask]
        x_vals = np.where(mask)[0]
        ax.plot(
            x_vals,
            mean_surv[x_vals],
            color=color,
            linewidth=2,
            label=group.budget_label,
        )
    ax.set_xlabel("Trace where skill was added")
    ax.set_ylabel("Fraction surviving to final")
    ax.set_title("Skill Survival by Cohort")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_section_timeline(run: RunHistory, title: str | None = None) -> plt.Figure:
    """Horizontal bar chart: section first appearance and lifetime."""
    first = section_first_appearance(run)
    sizes = section_sizes_over_time(run)
    sections_sorted = sorted(first.keys(), key=lambda s: first[s])

    fig, ax = plt.subplots(figsize=(12, max(4, len(sections_sorted) * 0.4)))
    for i, sec in enumerate(sections_sorted):
        start = first[sec]
        # Find last trace where section has skills
        sz = sizes.get(sec, [])
        end = start
        for t in range(len(sz) - 1, -1, -1):
            if sz[t] > 0:
                end = t
                break
        ax.barh(
            i,
            end - start + 1,
            left=start,
            height=0.6,
            color=_color(run.budget_label),
            alpha=0.7,
        )
    ax.set_yticks(range(len(sections_sorted)))
    ax.set_yticklabels(sections_sorted, fontsize=8)
    ax.set_xlabel("Trace")
    ax.set_title(title or f"Section Timeline — {run.budget_label}/run_{run.run_num}")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_churn_analysis(experiment: Experiment) -> plt.Figure:
    """Scatter (next_id vs final count) + bar (churn rate by budget)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: next_id vs final skill count
    ax = axes[0]
    for group in experiment.budget_groups:
        color = _color(group.budget_label)
        for run in group.runs:
            if not run.metrics:
                continue
            final_nid = run.next_ids[-1] if run.next_ids else 0
            final_sk = run.metrics[-1].skill_count
            ax.scatter(final_nid, final_sk, color=color, s=40, alpha=0.7)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="No churn")
    ax.set_xlabel("next_id (total skills ever created)")
    ax.set_ylabel("Final skill count (surviving)")
    ax.set_title("Churn: Skills Created vs Surviving")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bar chart: churn rate per budget
    ax = axes[1]
    labels = []
    rates = []
    stds = []
    colors = []
    for group in experiment.budget_groups:
        run_rates = []
        for run in group.runs:
            if not run.next_ids or not run.metrics:
                continue
            nid = run.next_ids[-1]
            sk = run.metrics[-1].skill_count
            run_rates.append((nid - sk) / nid if nid > 0 else 0)
        labels.append(group.budget_label)
        rates.append(np.mean(run_rates) if run_rates else 0)
        stds.append(np.std(run_rates) if run_rates else 0)
        colors.append(_color(group.budget_label))

    ax.bar(range(len(labels)), rates, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Churn rate (removed / created)")
    ax.set_title("Skill Churn Rate by Budget")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_lifespan_distributions(experiment: Experiment) -> plt.Figure:
    """2x4 grid of lifespan histograms, one per budget."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False, sharey=True)

    for idx, group in enumerate(experiment.budget_groups):
        ax = axes[idx // 4][idx % 4]
        all_spans = []
        for run in group.runs:
            all_spans.extend(skill_lifespans(run))
        if all_spans:
            ax.hist(
                all_spans,
                bins=range(0, NUM_TRACES + 1),
                color=_color(group.budget_label),
                alpha=0.7,
            )
        ax.set_title(group.budget_label, fontsize=10)
        ax.set_xlabel("Lifespan (traces)")
        if idx % 4 == 0:
            ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    for idx in range(len(experiment.budget_groups), 8):
        axes[idx // 4][idx % 4].set_visible(False)

    fig.suptitle(
        "Skill Lifespan Distributions (removed skills only)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_section_timelines(experiment: Experiment) -> plt.Figure:
    """2x4 grid of horizontal bar charts (run 1 per budget)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), squeeze=False, sharex=True)

    for idx, group in enumerate(experiment.budget_groups):
        ax = axes[idx // 4][idx % 4]
        if not group.runs:
            continue
        run = group.runs[0]
        first = section_first_appearance(run)
        sizes = section_sizes_over_time(run)
        sections_sorted = sorted(first.keys(), key=lambda s: first[s])

        for i, sec in enumerate(sections_sorted):
            start = first[sec]
            sz = sizes.get(sec, [])
            end = start
            for t in range(len(sz) - 1, -1, -1):
                if sz[t] > 0:
                    end = t
                    break
            ax.barh(
                i,
                end - start + 1,
                left=start,
                height=0.6,
                color=_color(group.budget_label),
                alpha=0.7,
            )

        ax.set_yticks(range(len(sections_sorted)))
        ax.set_yticklabels(sections_sorted, fontsize=6)
        ax.set_xlabel("Trace")
        ax.set_title(f"{group.budget_label} (run 1)", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")

    for idx in range(len(experiment.budget_groups), 8):
        axes[idx // 4][idx % 4].set_visible(False)

    fig.suptitle(
        "Section Appearance Timelines",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_section_heatmap(run: RunHistory) -> plt.Figure:
    """Imshow heatmap of section sizes over time for a single run."""
    sizes = section_sizes_over_time(run)
    sections = sorted(sizes.keys())
    mat = np.array([sizes[s] for s in sections])

    fig, ax = plt.subplots(figsize=(14, max(4, len(sections) * 0.4)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(sections)))
    ax.set_yticklabels(sections, fontsize=8)
    ax.set_xlabel("Trace")
    ax.set_title(f"Section Sizes \u2014 {run.budget_label}/run_{run.run_num}")
    fig.colorbar(im, ax=ax, label="Skills in section")
    fig.tight_layout()
    return fig


def plot_cluster_coverage(experiment: Experiment) -> plt.Figure:
    """2x4 grid of bar charts showing KMeans cluster run coverage."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False, sharey=True)

    for idx, group in enumerate(experiment.budget_groups):
        ax = axes[idx // 4][idx % 4]
        result = cluster_final_skills(group, n_clusters=10)
        if "error" in result:
            ax.text(
                0.5,
                0.5,
                result["error"],
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group.budget_label)
            continue

        clusters = result["clusters"]
        coverage = [c["run_coverage"] for c in clusters]
        colors_bar = [
            _color(group.budget_label) if c >= 0.6 else "#d1d5db" for c in coverage
        ]

        ax.bar(range(len(clusters)), coverage, color=colors_bar)
        ax.axhline(0.6, color="red", linestyle="--", alpha=0.5, label="Core threshold")
        ax.set_xlabel("Cluster")
        if idx % 4 == 0:
            ax.set_ylabel("Run coverage")
        ax.set_title(
            f"{group.budget_label} ({result['core_clusters']} core)",
            fontsize=10,
        )
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

    for idx in range(len(experiment.budget_groups), 8):
        axes[idx // 4][idx % 4].set_visible(False)

    fig.suptitle(
        "Skill Cluster Run Coverage (KMeans k=10)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_budget_saturation(experiment: Experiment) -> plt.Figure:
    """1x3 panels: skills, TOON tokens, sections vs budget (log x-axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, fn, ylabel in [
        (axes[0], skill_counts, "Final Skills"),
        (axes[1], toon_token_counts, "Final TOON Tokens"),
        (axes[2], section_counts, "Final Sections"),
    ]:
        budget_vals = []
        means = []
        stds = []
        colors = []
        for group in experiment.budget_groups:
            vals = [fn(r)[-1] for r in group.runs if r.metrics]
            bv = group.budget_value if group.budget_value is not None else 20000
            budget_vals.append(bv)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors.append(_color(group.budget_label))

        ax.errorbar(budget_vals, means, yerr=stds, fmt="o-", capsize=4)
        for bv, m, c in zip(budget_vals, means, colors):
            ax.scatter([bv], [m], color=c, s=60, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Token Budget")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if means:
            ax.annotate(
                "no-budget",
                (20000, means[-1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        ax.grid(True, alpha=0.3)

    fig.suptitle("Budget Saturation", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_compression_distribution(experiment_dir: Path) -> plt.Figure:
    """1x3 panels: skills, tokens, compression % (raw -> opus -> consensus)."""
    comp_metrics_data = load_compression_metrics(experiment_dir)
    comp_dist_data = compression_distribution(experiment_dir)

    budget_vals = []
    for budget in BUDGET_ORDER:
        bv = 20000 if budget == "no-budget" else int(budget.split("-")[1])
        budget_vals.append(bv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Skills
    ax = axes[0]
    raw_sk = [comp_dist_data[b]["raw_skills_mean"] for b in BUDGET_ORDER]
    raw_sk_std = [comp_dist_data[b]["raw_skills_std"] for b in BUDGET_ORDER]
    opus_sk = [comp_dist_data[b]["skills_mean"] for b in BUDGET_ORDER]
    opus_sk_std = [comp_dist_data[b]["skills_std"] for b in BUDGET_ORDER]
    cons_sk = [
        comp_metrics_data.get(f"consensus_{b}", {}).get("skills", 0)
        for b in BUDGET_ORDER
    ]

    ax.errorbar(
        budget_vals,
        raw_sk,
        yerr=raw_sk_std,
        fmt="o-",
        capsize=4,
        label="Raw (uncompressed)",
        color="#9ca3af",
    )
    ax.errorbar(
        budget_vals,
        opus_sk,
        yerr=opus_sk_std,
        fmt="o-",
        capsize=4,
        label="Opus (individual)",
        color="#2563eb",
    )
    ax.plot(
        budget_vals,
        cons_sk,
        "s--",
        color="#dc2626",
        label="Opus (consensus)",
        markersize=7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Skills")
    ax.set_title("Skill Counts")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: MD Tokens
    ax = axes[1]
    raw_toks = [comp_dist_data[b]["raw_md_tokens_mean"] for b in BUDGET_ORDER]
    raw_toks_std = [comp_dist_data[b]["raw_md_tokens_std"] for b in BUDGET_ORDER]
    opus_toks = [comp_dist_data[b]["md_tokens_tiktoken_mean"] for b in BUDGET_ORDER]
    opus_toks_std = [comp_dist_data[b]["md_tokens_tiktoken_std"] for b in BUDGET_ORDER]
    cons_toks = [
        comp_metrics_data.get(f"consensus_{b}", {}).get("md_tokens_tiktoken", 0)
        for b in BUDGET_ORDER
    ]

    ax.errorbar(
        budget_vals,
        raw_toks,
        yerr=raw_toks_std,
        fmt="o-",
        capsize=4,
        label="Raw (uncompressed)",
        color="#9ca3af",
    )
    ax.errorbar(
        budget_vals,
        opus_toks,
        yerr=opus_toks_std,
        fmt="o-",
        capsize=4,
        label="Opus (individual)",
        color="#2563eb",
    )
    ax.plot(
        budget_vals,
        cons_toks,
        "s--",
        color="#dc2626",
        label="Opus (consensus)",
        markersize=7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Tiktoken Tokens")
    ax.set_title("MD Token Counts")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Compression %
    ax = axes[2]
    comp_pcts = [comp_dist_data[b]["compression_pct_mean"] for b in BUDGET_ORDER]
    comp_pcts_std = [comp_dist_data[b]["compression_pct_std"] for b in BUDGET_ORDER]

    cons_comp = []
    for b in BUDGET_ORDER:
        c = comp_metrics_data.get(f"consensus_{b}", {})
        raw = c.get("raw_md_tokens", 0)
        opus = c.get("md_tokens_tiktoken", 0)
        cons_comp.append(opus / raw * 100 if raw > 0 else 0)

    ax.errorbar(
        budget_vals,
        comp_pcts,
        yerr=comp_pcts_std,
        fmt="o-",
        capsize=4,
        label="Individual runs (mean \u00b1 std)",
        color="#2563eb",
    )
    ax.plot(
        budget_vals,
        cons_comp,
        "s--",
        color="#dc2626",
        label="Consensus",
        markersize=7,
    )
    ax.axhline(45, color="#9ca3af", linestyle="--", alpha=0.5, label="~45% average")
    ax.set_xscale("log")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Compression %")
    ax.set_title("Opus Compression Ratio (compressed/raw)")
    ax.set_ylim(0, 70)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Opus Compression: Individual Runs vs Consensus",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def format_embedding_overlap_text(experiment: Experiment) -> str:
    """Format per-budget embedding NN cosine similarity stats as markdown."""
    lines = []
    for group in experiment.budget_groups:
        stats = cross_run_embedding_overlap(group)
        lines.append(
            f"- **{group.budget_label}**: mean NN cosine "
            f"{stats['overall_mean_nn']:.3f} \u00b1 {stats['overall_std_nn']:.3f}"
        )
        for p in stats["pairs"]:
            lines.append(
                f"  - run_{p['run_i']} \u2194 run_{p['run_j']}: " f"{p['mean_nn']:.3f}"
            )
    return "\n".join(lines)


def format_compression_table(experiment_dir: Path) -> str:
    """Format raw -> opus compression table as markdown."""
    import pandas as pd

    comp_metrics_data = load_compression_metrics(experiment_dir)
    comp_dist_data = compression_distribution(experiment_dir)

    rows = []
    for budget in BUDGET_ORDER:
        d = comp_dist_data.get(budget)
        if not d:
            continue
        ckey = f"consensus_{budget}"
        c = comp_metrics_data.get(ckey, {})
        rows.append(
            {
                "Budget": budget,
                "Raw Skills": f"{d['raw_skills_mean']:.1f} \u00b1 {d['raw_skills_std']:.1f}",
                "Raw MD Tokens": f"{d['raw_md_tokens_mean']:.0f} \u00b1 {d['raw_md_tokens_std']:.0f}",
                "Opus Skills": f"{d['skills_mean']:.1f} \u00b1 {d['skills_std']:.1f}",
                "Opus MD Tokens": f"{d['md_tokens_tiktoken_mean']:.0f} \u00b1 {d['md_tokens_tiktoken_std']:.0f}",
                "Compression %": f"{d['compression_pct_mean']:.1f}% \u00b1 {d['compression_pct_std']:.1f}%",
                "Consensus Skills": c.get("skills", ""),
                "Consensus Tokens": c.get("md_tokens_tiktoken", ""),
            }
        )

    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    experiment: Experiment,
    output_dir: Path,
) -> Path:
    """Generate SKILLBOOK_HISTORY_ANALYSIS.md with computed values and figures."""
    import pandas as pd

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save figures
    fig_files: dict[str, str] = {}

    # --- 1. Growth curves ---
    fig = plot_growth_curves(
        experiment, skill_counts, "Skills", "Skill Growth per Budget"
    )
    fig.savefig(figures_dir / "growth_skills.png", dpi=150, bbox_inches="tight")
    fig_files["growth_skills"] = "figures/growth_skills.png"
    plt.close(fig)

    fig = plot_cross_budget_overlay(
        experiment, skill_counts, "Skills", "Skills \u2014 Cross-Budget"
    )
    fig.savefig(figures_dir / "overlay_skills.png", dpi=150, bbox_inches="tight")
    fig_files["overlay_skills"] = "figures/overlay_skills.png"
    plt.close(fig)

    fig = plot_cross_budget_overlay(
        experiment, toon_token_counts, "TOON Tokens", "TOON Tokens \u2014 Cross-Budget"
    )
    fig.savefig(figures_dir / "overlay_toon.png", dpi=150, bbox_inches="tight")
    fig_files["overlay_toon"] = "figures/overlay_toon.png"
    plt.close(fig)

    fig = plot_cross_budget_overlay(
        experiment, section_counts, "Sections", "Sections \u2014 Cross-Budget"
    )
    fig.savefig(figures_dir / "overlay_sections.png", dpi=150, bbox_inches="tight")
    fig_files["overlay_sections"] = "figures/overlay_sections.png"
    plt.close(fig)

    # --- 2. Skill lifecycle ---
    rep = experiment.get_budget("budget-3000")
    if rep and rep.runs:
        fig = plot_delta_bars(rep.runs[0])
        fig.savefig(
            figures_dir / "deltas_representative.png", dpi=150, bbox_inches="tight"
        )
        fig_files["deltas"] = "figures/deltas_representative.png"
        plt.close(fig)

    fig = plot_survival_curve(experiment)
    fig.savefig(figures_dir / "survival.png", dpi=150, bbox_inches="tight")
    fig_files["survival"] = "figures/survival.png"
    plt.close(fig)

    fig = plot_churn_analysis(experiment)
    fig.savefig(figures_dir / "churn.png", dpi=150, bbox_inches="tight")
    fig_files["churn"] = "figures/churn.png"
    plt.close(fig)

    fig = plot_lifespan_distributions(experiment)
    fig.savefig(figures_dir / "lifespans.png", dpi=150, bbox_inches="tight")
    fig_files["lifespans"] = "figures/lifespans.png"
    plt.close(fig)

    # --- 3. Section evolution ---
    fig = plot_section_timelines(experiment)
    fig.savefig(figures_dir / "section_timelines.png", dpi=150, bbox_inches="tight")
    fig_files["section_timelines"] = "figures/section_timelines.png"
    plt.close(fig)

    if rep and rep.runs:
        fig = plot_section_heatmap(rep.runs[0])
        fig.savefig(figures_dir / "section_heatmap.png", dpi=150, bbox_inches="tight")
        fig_files["section_heatmap"] = "figures/section_heatmap.png"
        plt.close(fig)

    # --- 4. Cross-run convergence (optional) ---
    try:
        embedding_text = format_embedding_overlap_text(experiment)
    except Exception:
        embedding_text = None

    try:
        fig = plot_cluster_coverage(experiment)
        fig.savefig(figures_dir / "cluster_coverage.png", dpi=150, bbox_inches="tight")
        fig_files["cluster_coverage"] = "figures/cluster_coverage.png"
        plt.close(fig)
    except Exception:
        pass

    # --- 5. Conciseness ---
    fig = plot_cross_budget_overlay(
        experiment, tokens_per_skill, "Tokens/Skill", "Conciseness \u2014 Cross-Budget"
    )
    fig.savefig(figures_dir / "conciseness.png", dpi=150, bbox_inches="tight")
    fig_files["conciseness"] = "figures/conciseness.png"
    plt.close(fig)

    # --- 6. Budget saturation ---
    fig = plot_budget_saturation(experiment)
    fig.savefig(figures_dir / "saturation.png", dpi=150, bbox_inches="tight")
    fig_files["saturation"] = "figures/saturation.png"
    plt.close(fig)

    # --- 7. Opus compression (optional) ---
    try:
        compression_table_text = format_compression_table(experiment.experiment_dir)
    except Exception:
        compression_table_text = None

    try:
        fig = plot_compression_distribution(experiment.experiment_dir)
        fig.savefig(
            figures_dir / "compression_distribution.png", dpi=150, bbox_inches="tight"
        )
        fig_files["compression_distribution"] = "figures/compression_distribution.png"
        plt.close(fig)
    except Exception:
        pass

    # Summary table
    rows: list[dict[str, Any]] = []
    for group in experiment.budget_groups:
        sk = budget_comparison_table(
            Experiment(budget_groups=[group], experiment_dir=experiment.experiment_dir),
            skill_counts,
        )[group.budget_label]
        tt = budget_comparison_table(
            Experiment(budget_groups=[group], experiment_dir=experiment.experiment_dir),
            toon_token_counts,
        )[group.budget_label]
        sc = budget_comparison_table(
            Experiment(budget_groups=[group], experiment_dir=experiment.experiment_dir),
            section_counts,
        )[group.budget_label]
        ni = budget_comparison_table(
            Experiment(budget_groups=[group], experiment_dir=experiment.experiment_dir),
            next_id_counts,
        )[group.budget_label]

        # Churn
        total_adds = []
        total_removes = []
        for run in group.runs:
            a, r, _ = churn_per_trace(run)
            total_adds.append(float(a.sum()))
            total_removes.append(float(r.sum()))

        rows.append(
            {
                "Budget": group.budget_label,
                "Skills (mean)": f"{sk['mean']:.1f}",
                "Skills (std)": f"{sk['std']:.1f}",
                "Sections (mean)": f"{sc['mean']:.1f}",
                "TOON (mean)": f"{tt['mean']:.0f}",
                "next_id (mean)": f"{ni['mean']:.0f}",
                "Total ADDs (mean)": f"{np.mean(total_adds):.0f}",
                "Total REMOVEs (mean)": f"{np.mean(total_removes):.0f}",
            }
        )

    df = pd.DataFrame(rows)
    table_md = df.to_markdown(index=False)

    # Build markdown section fragments
    delta_img = (
        f"![Delta Events]({fig_files['deltas']})"
        if "deltas" in fig_files
        else "*(No representative budget found)*"
    )
    churn_img = (
        f"\n![Churn Analysis]({fig_files['churn']})" if "churn" in fig_files else ""
    )
    lifespans_img = (
        f"\n![Lifespan Distributions]({fig_files['lifespans']})"
        if "lifespans" in fig_files
        else ""
    )
    section_timelines_img = (
        f"![Section Timelines]({fig_files['section_timelines']})"
        if "section_timelines" in fig_files
        else ""
    )
    section_heatmap_img = (
        f"\n![Section Heatmap]({fig_files['section_heatmap']})"
        if "section_heatmap" in fig_files
        else ""
    )
    embedding_section = (
        embedding_text if embedding_text else "*(Embeddings not available)*"
    )
    cluster_img = (
        f"\n![Cluster Coverage]({fig_files['cluster_coverage']})"
        if "cluster_coverage" in fig_files
        else "\n*(Embeddings not available)*"
    )
    saturation_img = (
        f"\n![Budget Saturation]({fig_files['saturation']})"
        if "saturation" in fig_files
        else ""
    )
    compression_table_section = (
        compression_table_text
        if compression_table_text
        else "*(Compression metrics not available)*"
    )
    compression_dist_img = (
        f"\n![Compression Distribution]({fig_files['compression_distribution']})"
        if "compression_distribution" in fig_files
        else "\n*(Compression metrics not available)*"
    )

    report = f"""\
# Skillbook History Analysis

Analysis of skillbook evolution across the variance experiment:
7 budget levels x 5 runs x 25 traces = 875 snapshots.

## Experiment Setup

- **Model**: Claude Haiku 4.5 (Bedrock)
- **Budgets**: 500, 1000, 2000, 3000, 5000, 10000, unlimited
- **Runs per budget**: 5
- **Traces per run**: 25
- **Dedup threshold**: 0.7

## Summary Table

{table_md}

## 1. Growth Curves

How skills accumulate over the 25-trace sequence.

![Skill Growth]({fig_files['growth_skills']})

![Skills Overlay]({fig_files['overlay_skills']})

![TOON Tokens Overlay]({fig_files['overlay_toon']})

![Sections Overlay]({fig_files['overlay_sections']})

## 2. Skill Lifecycle

ADD/UPDATE/REMOVE events and skill survival analysis.

{delta_img}

![Survival Curves]({fig_files['survival']})
{churn_img}
{lifespans_img}

## 3. Section Evolution

When do sections first appear? How do their sizes change over time?

{section_timelines_img}
{section_heatmap_img}

## 4. Cross-Run Convergence

Do independent runs converge on similar skill structures?

### Embedding Nearest-Neighbor Overlap

{embedding_section}

### Skill Clustering (KMeans k=10)
{cluster_img}

## 5. Conciseness

Token efficiency per skill over time.

![Conciseness]({fig_files['conciseness']})

## 6. Cross-Budget Summary

Final metrics across all budgets.

{table_md}
{saturation_img}

## 7. Opus Compression

Raw skillbook vs Opus-compressed individual runs vs consensus.

{compression_table_section}
{compression_dist_img}

---
*Generated by `scripts/analysis/skillbook_history.py`*
"""

    out_path = output_dir / "SKILLBOOK_HISTORY_ANALYSIS.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path
