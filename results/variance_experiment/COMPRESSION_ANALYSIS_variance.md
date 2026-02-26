# Compression Analysis: Variance Experiment (7 Budgets x 5 Runs)

## Motivation

The [dedupped compression analysis](COMPRESSION_ANALYSIS_dedupped.md) compared a single no-budget skillbook against budget-4000 and an Opus-compressed version. That analysis showed Opus compresses to 63% of TOON tokens vs budget-4000's 89%, with both converging on ~90% topical overlap despite different section naming.

**Limitation**: A single run per budget is a sample size of 1. The SkillManager's section naming and skill wording are stochastic — different runs with identical inputs produce different skillbooks purely from LLM non-determinism.

The variance experiment addresses this with **7 budget levels x 5 runs = 35 skillbooks** (34 valid after excluding `no-budget/run_1` which used `dedup_interval=1` vs 5). All runs process the same 25 traces in identical sorted order, same model (Haiku 4.5), dedup threshold 0.7, dedup interval 5.

**This analysis covers three angles:**

1. **Cross-budget content convergence** — what knowledge is stable across all budgets?
2. **Within-budget consistency** — how much variance exists across runs at the same budget?
3. **Opus compression of median and consensus skillbooks** — how much of the variance is "noise Opus would remove anyway"?

## Experiment Configuration

| Parameter | Value |
|---|---|
| Model | Haiku 4.5 (`bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`) |
| Traces | 25 (identical sorted order across all runs) |
| Dedup threshold | 0.7 |
| Dedup interval | 5 (except `no-budget/run_1` = 1, excluded) |
| Budget levels | 500, 1000, 2000, 3000, 5000, 10000, None |
| Runs per budget | 5 (4 for no-budget after exclusion) |
| Total valid runs | 34 |

## Part 1: Cross-Budget Content Convergence

### Section Names vs Topics

Across 34 skillbooks, the SkillManager generated **207 unique section names**. By exact name matching, only **1 section** (`cancellation_policy`) appears consistently across budgets — suggesting near-total fragmentation.

However, exact name matching is misleading. The SkillManager assigns different qualifiers to the same underlying concept:

| Topic | Example Section Names (all expressing same concept) |
|---|---|
| Escalation | `escalation`, `escalation_protocol`, `escalation_workflow`, `escalation_decision`, `escalation_handoff`, `escalation_judgment` |
| Payment | `payment_handling`, `payment_processing`, `payment_verification`, `payment_routing`, `payment_methods`, `payment_allocation` |
| Tool usage | `tool_sequencing`, `tool_execution`, `tool_usage`, `tool_workflows`, `tools_and_apis`, `transaction_execution` |

Grouping section names by their prefix stem (e.g., `payment_*` → `payment`) and manually merging obvious synonyms (e.g., `tool` + `tools` + `transaction`, `cost` + `pricing` + `financial`) collapses 207 names into **18 canonical topics**. This reveals a fundamentally different picture.

### Topic Presence Heatmap

| Topic | 500 | 1000 | 2000 | 3000 | 5000 | 10000 | None | Coverage |
|---|---|---|---|---|---|---|---|---|
| `cancellation` | ▓▓ | ██ | ▓▓ | ██ | ██ | ██ | ██ | **Core** |
| `customer_comms` | ██ | ▓▓ | ▓▓ | ▓▓ | ██ | ██ | ▓▓ | **Core** |
| `escalation` | ██ | ██ | ██ | ██ | ██ | ▓▓ | ▓▓ | **Core** |
| `policy` | ██ | ██ | ██ | ██ | ▓▓ | ▓▓ | ▓▓ | **Core** |
| `pricing` | ██ | ▓▓ | ██ | ▓▓ | ██ | ▓▓ | ▓▓ | **Core** |
| `reservation` | ██ | ██ | ▓▓ | ██ | ░ | ██ | ██ | **Core** |
| `tool_usage` | ▓▓ | ██ | ██ | ░ | ██ | ▓▓ | ▓▓ | **Core** |
| `flight_search` | ██ | ██ | ░ | ██ | ░ | ██ | ▓▓ | Common |
| `payment` | ▓▓ | ▓▓ | ░ | ▓▓ | ░ | ▓▓ | ▓▓ | Common |
| `baggage` | ░ | ▓▓ | ░ | ▓▓ | ░ | ░ | ██ | Sparse |
| `compensation` | ░ | ░ | ▓▓ | ██ | ░ | ░ | ██ | Sparse |
| `modification` | | ░ | ░ | ▓▓ | ░ | ░ | ██ | Sparse |
| `confirmation` | ▓▓ | | ░ | | ░ | ░ | | Rare |
| `data_retrieval` | ░ | ░ | ░ | ░ | ░ | ▓▓ | ░ | Rare |
| `decision_support` | ░ | ░ | ░ | ░ | | ░ | | Rare |
| `insurance` | | | ░ | ░ | | | | Rare |
| `temporal` | ░ | | | ░ | | | | Rare |
| `workflow` | ░ | | ░ | ░ | | ░ | | Rare |

(██ = ≥80% runs, ▓▓ = ≥50% runs, ░ = <50% but present, blank = absent)

**Key finding**: **7 out of 18 topics are "core"** — discovered by ≥6/7 budgets at ≥50% within-budget frequency. These cover the fundamental domain concepts: cancellation policy, customer communication, escalation, policy constraints, pricing, reservations, and tool usage. The remaining 11 topics are domain-relevant but not universally prioritized across budgets.

This contrasts sharply with exact name matching (1/207 core) and shows that the SkillManager **does converge on topics** — it's the naming, not the knowledge discovery, that's stochastic.

## Part 2: Within-Budget Consistency

### Embedding-Based Skill Similarity

Q: do runs produce the same specific skill formulations? (with ≥0.7 cosine similarity)

All skills across all runs within each budget were embedded (OpenAI `text-embedding-3-small`) and clustered greedily at similarity ≥ 0.7 (matching the dedup threshold). A cluster is "stable" if it contains skills from ≥3 runs.

| Budget | Total Skills | Unique Clusters | Stable Clusters (≥3 runs) | Stability % |
|---|---|---|---|---|
| 500 | 209 | 126 | 16 | 12.7% |
| 1000 | 255 | 136 | 25 | 18.4% |
| 2000 | 290 | 166 | 22 | 13.3% |
| 3000 | 333 | 184 | 20 | 10.9% |
| 5000 | 343 | 178 | 28 | 15.7% |
| 10000 | 370 | 191 | 30 | 15.7% |
| None | 296 | 160 | 23 | 14.4% |

**Key finding**: Only **11-18% of skill clusters are stable** across runs. The remaining 82-89% appear in ≤2 runs and are effectively stochastic — generated by LLM non-determinism in one or two runs but not consistently discovered.
ACCORDIN TO EMBEDDING. IF YOU DO WITH OPUS MAYBE IT SEES SOMETHING ELSE?

### Stability Breakdown by Run Coverage

Representative breakdown (budget-1000, 5 runs, 136 clusters):

| Coverage | Clusters | % | Category |
|---|---|---|---|
| 1/5 runs | 82 | 60.3% | Stochastic — unique to single run |
| 2/5 runs | 29 | 21.3% | Stochastic — appears in minority |
| 3/5 runs | 16 | 11.8% | **Stable** |
| 4/5 runs | 6 | 4.4% | **Stable** |
| 5/5 runs | 3 | 2.2% | **Core** — every run discovers this |

This pattern is remarkably consistent across all budgets: ~62-68% of clusters appear in exactly 1 run, ~18-23% appear in 2 runs, and only ~11-18% appear in ≥3 runs.

**Two-layer consistency picture**:
- **Topic discovery** is moderately stable (7/18 core topics, see Part 1 heatmap) — runs converge on the same domain areas
- **Skill articulation** is highly stochastic (11-18% stable) — how each topic's knowledge gets worded and split into discrete skills varies dramatically

A single run captures the stable core (~16-30 skills) plus a large amount of stochastic variation (~40-60 additional skills). The consensus skillbook (intersection of ≥3 runs) extracts just the signal.


Problems with this greedy clustering + 0.7 threshold:

1. Greedy order dependence — the first skill in the iteration becomes the
representative, and everything ≥0.7 to it gets pulled in. A different iteration
order could produce different clusters.
2. 0.7 is a content-level threshold, but skills can be semantically equivalent
at lower cosine similarity — e.g., "Always confirm before executing cancellation
API" and "Require explicit user approval prior to any state-changing tool call"
express the same principle at different abstraction levels. Embeddings might
score these at 0.55-0.65, missing the match.
3. No semantic judgment — embeddings measure lexical/surface similarity. Opus
could identify that two skills with different wording encode the same
operational rule.

## Part 3: Opus Compression — Individual Runs vs Consensus Skillbook

### Methodology

For each of 7 budgets:
1. **Individual runs**: All 5 runs per budget (35 total), each Opus-compressed independently
2. **Consensus skillbook**: Intersection approach — only skills from clusters appearing in ≥3 runs, representative selected by highest helpful count then longest content, then Opus-compressed

Opus compression follows the same methodology as the dedupped analysis (merge near-duplicates, tighten wording, consolidate sections, preserve high-value content).

All token counts use tiktoken `cl100k_base`.

### Individual Runs — Opus Compression

All 35 individual runs (5 per budget) were Opus-compressed. Mean ± std per budget:

| Budget | Raw Skills | Original MD Tokens | Opus Skills | Opus MD Tokens | Compression % |
|---|---|---|---|---|---|
| 500 | 41.8 ± 11.6 | 5,479 ± 1,327 | 22.0 ± 3.5 | 2,458 ± 528 | 45.2% ± 4.8% |
| 1000 | 51.0 ± 2.2 | 6,547 ± 338 | 33.0 ± 7.5 | 2,994 ± 549 | 45.7% ± 7.5% |
| 2000 | 58.0 ± 2.5 | 7,298 ± 678 | 38.4 ± 4.7 | 3,259 ± 320 | 44.9% ± 5.6% |
| 3000 | 66.6 ± 8.4 | 8,061 ± 595 | 43.0 ± 5.5 | 3,600 ± 441 | 44.8% ± 5.4% |
| 5000 | 68.6 ± 9.7 | 8,444 ± 1,337 | 46.8 ± 9.2 | 3,615 ± 604 | 43.5% ± 8.9% |
| 10000 | 74.0 ± 3.8 | 8,886 ± 533 | 55.2 ± 6.8 | 4,259 ± 301 | 48.1% ± 5.0% |
| None | 75.2 ± 8.7 | 9,201 ± 1,160 | 51.6 ± 12.1 | 4,156 ± 870 | 45.8% ± 11.0% |

**Observations**:

1. **Compression ratio is remarkably stable across budgets**: Opus consistently reduces skillbooks to 43-48% of original MD tokens regardless of budget level. The ~45% mean with CoV of 5-11% suggests a stable compression "constant" for this domain and methodology.

2. **Skills scale with budget**: Raw skill counts grow from 42 (budget-500) to 75 (no-budget). After Opus compression, this becomes 22→52 — roughly 50-70% of raw counts retained as distinct compressed skills.

3. **Token counts scale sub-linearly**: Compressed tokens grow from ~2,500 to ~4,300 as budget increases from 500 to 10000 — a 1.7x range vs the raw 1.6x range (5,479→8,886). Higher budgets produce proportionally more redundancy that Opus absorbs.

4. **Within-budget variance is moderate**: CoV (std/mean) for compression % ranges from 5-11%, with no-budget showing the highest variance (11.0%) and budget-500/budget-3000 the lowest (~5%). Opus compression normalizes some of the stochastic variation in raw skillbooks.

5. **no-budget ≈ budget-10000**: After Opus compression, no-budget (4,156 toks, 45.8%) is comparable to budget-10000 (4,259 toks, 48.1%), confirming that the extra skills from unlimited budgets are mostly noise that Opus removes.

### Consensus Skillbook — Opus Compression

| Budget | Consensus Skills | TOON Tokens | MD Tokens | Reduction vs Indiv. Mean (Skills) | Opus Skills | Opus MD Tokens | Compression % |
|---|---|---|---|---|---|---|---|
| 500 | 16 | 1,608 | 1,724 | -62% | 14 | 957 | 55.5% |
| 1000 | 25 | 2,135 | 2,299 | -51% | 20 | 1,047 | 45.5% |
| 2000 | 22 | 1,781 | 1,918 | -62% | 17 | 1,007 | 52.5% |
| 3000 | 20 | 1,625 | 1,763 | -70% | 19 | 1,030 | 58.4% |
| 5000 | 28 | 2,080 | 2,264 | -59% | 23 | 1,273 | 56.2% |
| 10000 | 30 | 2,221 | 2,414 | -59% | 17 | 981 | 40.6% |
| None | 23 | 1,748 | 1,909 | -69% | 21 | 1,059 | 55.5% |

**Key finding**: Consensus skillbooks are **51-70% smaller** (by skill count) than mean individual runs before any Opus compression. The intersection filter removes the ~82-89% of stochastic clusters, retaining only the 16-30 skills that multiple runs independently discovered.

After Opus compression, consensus reaches **~950-1,300 MD tokens** — the absolute floor. Consensus Opus compression ratios (41-58%) are higher than individual-run ratios (~45%), because consensus has already removed inter-run noise, leaving less redundancy for Opus to compress. In other words: consensus has less fluff, so Opus cuts proportionally less.

### Individual vs Consensus — Side-by-Side Comparison

| Budget | Source | Skills | MD Tokens | Opus Skills | Opus MD Tokens | Compression |
|---|---|---|---|---|---|---|
| **500** | Individual (mean) | 41.8 | 5,479 | 22.0 | 2,458 | 45.2% |
| | Consensus | 16 | 1,724 | 14 | 957 | 55.5% |
| **1000** | Individual (mean) | 51.0 | 6,547 | 33.0 | 2,994 | 45.7% |
| | Consensus | 25 | 2,299 | 20 | 1,047 | 45.5% |
| **2000** | Individual (mean) | 58.0 | 7,298 | 38.4 | 3,259 | 44.9% |
| | Consensus | 22 | 1,918 | 17 | 1,007 | 52.5% |
| **3000** | Individual (mean) | 66.6 | 8,061 | 43.0 | 3,600 | 44.8% |
| | Consensus | 20 | 1,763 | 19 | 1,030 | 58.4% |
| **5000** | Individual (mean) | 68.6 | 8,444 | 46.8 | 3,615 | 43.5% |
| | Consensus | 28 | 2,264 | 23 | 1,273 | 56.2% |
| **10000** | Individual (mean) | 74.0 | 8,886 | 55.2 | 4,259 | 48.1% |
| | Consensus | 30 | 2,414 | 17 | 981 | 40.6% |
| **None** | Individual (mean) | 75.2 | 9,201 | 51.6 | 4,156 | 45.8% |
| | Consensus | 23 | 1,909 | 21 | 1,059 | 55.5% |

The gap between Opus(individual) and Opus(consensus) reveals how much stochastic content Opus retains from individual runs. For budget-500: Opus keeps 2,458 tokens from individual runs vs 957 from consensus — meaning ~61% of what Opus preserves from individual runs is content that wasn't stable across runs. This gap grows with budget: at no-budget, 4,156 vs 1,059 means ~75% is unstable content.

### Compression Curve Summary

| Budget | Avg TOON | Avg Raw MD | Consensus MD | Opus(Individual) MD | Opus(Consensus) MD |
|---|---|---|---|---|---|
| 500 | 3,077 | 5,479 | 1,724 | 2,458 | 957 |
| 1000 | 3,664 | 6,547 | 2,299 | 2,994 | 1,047 |
| 2000 | 3,936 | 7,298 | 1,918 | 3,259 | 1,007 |
| 3000 | 4,352 | 8,061 | 1,763 | 3,600 | 1,030 |
| 5000 | 4,425 | 8,444 | 2,264 | 3,615 | 1,273 |
| 10000 | 4,559 | 8,886 | 2,414 | 4,259 | 981 |
| None | 4,840 | 9,201 | 1,909 | 4,156 | 1,059 |

**Observations**:

1. **Opus compression of individual runs achieves ~45% of original MD tokens** — consistent across all budgets. This is stronger than the 63% TOON ratio from the dedupped analysis, because MD tokens include J/E metadata that Opus aggressively shortens.

2. **Consensus skillbooks are already highly compressed** — at 1,700-2,400 MD tokens, they're smaller than Opus-compressed individual runs (2,458-4,259). The intersection filter provides "free" compression by removing stochastic content.

3. **Opus compression of consensus skillbooks reaches ~950-1,300 MD tokens** — the absolute floor. At this point, skills are maximally deduplicated and verbally tightened.

4. **The gap between Opus(individual) and Opus(consensus) reveals noise**: Opus-compressed individual runs retain 2-4x more tokens than Opus-compressed consensus, meaning 60-75% of what Opus keeps from individual runs is stochastic content that wasn't discovered by multiple runs. This gap widens at higher budgets, where more stochastic skills accumulate.

## Cross-Reference with Dedupped Analysis

| Finding | Dedupped Analysis | Variance Analysis |
|---|---|---|
| Opus compression ratio (MD tokens) | 44% (of no-budget) | ~45% across all budgets (mean of 5 runs each) |
| Section name convergence | 1 shared name (of 10 vs 12) | 1 core section name, but 7/18 core topics after prefix grouping |
| Topical overlap | ~90% between budget and no-budget | Confirmed: 7/18 core topics cross-budget (see heatmap) |
| Within-budget variance | N/A (single run) | Skills: only 11-18% stable across runs |
| Consensus vs single run | N/A | Consensus is 51-70% smaller (skills), captures stable core |
| Primary compression source | Structural (skill count) + J/E length | Confirmed: stochastic noise removal + J/E tightening |

**New insight from variance analysis**: The dedupped analysis compared two single runs and found ~90% topical overlap. The variance analysis confirms this is genuine at the topic level — 7/18 topics are core across all budgets (see Part 1 heatmap). However, at the skill level only 11-18% of formulations are stable. The topical convergence is real; the skill-level variance is where the stochastic noise lives.

## Conclusions

### 1. Topic Discovery Is Stable, Skill Articulation Is Not

LLM non-determinism affects the two layers differently:

- **Topics**: Runs converge on the same domain areas (7/18 topics core across all budgets, see Part 1 heatmap). The SkillManager consistently discovers cancellation, escalation, pricing, policy, reservations, customer communication, and tool usage.
- **Skills**: Only 11-18% of specific skill formulations are reproducibly discovered across runs. The remaining 82-89% are stochastic. A single run's skillbook is ~60% noise (unique to that run) and ~15% stable signal (discovered by ≥3/5 runs), with ~22% in a gray zone (2/5 runs).

The variance is in *how* knowledge gets articulated, not *what* knowledge gets discovered.

### 2. Section Names Require Topic-Level Grouping

207 unique section names across 34 runs collapse to **18 topics** via prefix-based grouping. By exact name matching, only `cancellation_policy` appears consistently. By topic grouping, 7/18 topics are core. The SkillManager converges on a stable set of domain concepts but assigns ad-hoc qualifiers each run (`escalation_protocol` vs `escalation_workflow` vs `escalation_handoff`). Any cross-run analysis should operate at the topic level, not exact section names.

### 3. Consensus Skillbooks Are a "Free" Compression Technique

Running multiple runs and taking the intersection (skills appearing in ≥3 runs) produces a 51-70% smaller skillbook (by skill count) with no Opus cost. The consensus approach:
- Removes stochastic noise (single-run artifacts)
- Retains domain-critical knowledge (cancellation eligibility, escalation triggers, tool sequencing)
- Produces skillbooks of 16-30 skills vs 42-75 mean individual-run skills

### 4. Opus Compression and Consensus Are Complementary

| Compression Stage | MD Tokens | Compression | Cumulative |
|---|---|---|---|
| Single run (mean of all 35) | ~7,400 avg | Baseline | 100% |
| Opus(individual) | ~3,300 avg | -55% | 45% |
| Consensus (≥3/5 runs) | ~2,000 avg | -73% | 27% |
| Opus(consensus) | ~1,050 avg | -48% of consensus | 14% |

The full pipeline (consensus + Opus) achieves ~86% total compression. Each stage removes different content:
- **Consensus** removes stochastic skills (appear in ≤2 runs)
- **Opus** removes within-skill verbosity and merges near-duplicates that survived the intersection

### 5. Budget Level Has Limited Impact on Stable Core

The number of stable skills is remarkably consistent across budgets:

| Budget | Stable Skills | Avg Total Skills per Run |
|---|---|---|
| 500 | 16 | 41.8 |
| 1000 | 25 | 51.0 |
| 2000 | 22 | 58.0 |
| 3000 | 20 | 66.6 |
| 5000 | 28 | 68.6 |
| 10000 | 30 | 74.0 |
| None | 23 | 75.2 |

Higher budgets produce more total skills per run (42→75), but the stable core only grows modestly (16→30). The extra skills at higher budgets are predominantly stochastic — budget pressure at lower levels filters out noise that higher budgets allow through.

### 6. Practical Recommendation

For production deployment with token-constrained contexts:

1. **Best quality/cost ratio**: Run 3-5 times at any budget, take consensus (≥3 runs). Cost: 3-5x the Haiku generation cost, zero Opus cost. Result: ~20-30 skills, ~2,000 MD tokens.

2. **Maximum compression**: Consensus + Opus post-processing. Cost: 3-5x Haiku + 1 Opus call. Result: ~15-20 skills, ~1,000 MD tokens.

3. **Single-run baseline**: Budget-1000 to budget-3000 provides a reasonable tradeoff. The budget constrains stochastic growth while capturing the stable core. Result: ~50-65 skills, ~7,000-8,000 MD tokens.

For 200K context windows, even the single-run baseline (~4,000-5,000 TOON tokens) is negligible. The consensus approach becomes valuable when targeting smaller context windows or when skillbook quality matters more than generation cost.

## Limitations & Open Questions

### 1. No Downstream Task Evaluation

This is the largest gap. All analysis is structural — token counts, cluster stability, compression ratios. There is **no measurement of whether any of this matters for agent performance**. Unanswered questions:

- Does a consensus skillbook produce better agent answers than a single-run skillbook?
- Does budget-500's skillbook (41 skills) perform worse than no-budget's (75 skills)?
- Does Opus compression lose useful information or just noise?
- Is the "stable core" actually the most task-relevant content, or could rare stochastic skills carry disproportionate value?

Without task evaluation, the compression pipeline is optimizing a proxy (structural stability) rather than the objective (agent quality).

### 2. No Content Quality Assessment

The analysis counts skills and clusters but does not evaluate whether the **stable core is actually the most useful content**. The 82-89% of "stochastic" skills could include rare but high-value insights (e.g., an edge case in cancellation policy that only one run articulated clearly). A consensus filter discards these by design. The assumption that stability ≈ quality is untested.

### 3. Budget-500 Variance Is Unexplained

The [skillbook history analysis](../SKILLBOOK_HISTORY_ANALYSIS.md) shows budget-500 has std=10.4 in final skill count vs std=2.0 at budget-1000 — a 5x difference. Budget-500 also has the highest REMOVE count (10 avg vs 0 at budget-10000). The hypothesis is that tight budget pressure creates chaotic churn: small LLM phrasing differences push skills over/under the token limit, triggering cascading adds and removes. But this is unverified — it could also be an artifact of how the SkillManager prioritizes removals under pressure.

### 4. No Analysis of REMOVE Dynamics

The history analysis shows REMOVEs happen (10 avg at budget-500, 6 at budget-1000, 0 at budget-10000) but does not analyze **what gets removed**. Key questions:

- Are removals consistent across runs? (i.e., do different runs remove the same low-value skills?)
- Does the SkillManager remove the oldest skills, the shortest, or the least-helpful?
- Are removed skills genuinely lower quality, or are they victims of arbitrary budget pressure?

If removals are consistent, the budget mechanism is functioning as intelligent pruning. If random, it's just noise injection.

### 5. Embedding Clustering Limitations

The stability analysis (Part 2) uses greedy clustering at cosine similarity ≥ 0.7 with `text-embedding-3-small`. This has three known problems (noted inline in Part 2):

1. **Greedy order dependence** — iteration order affects cluster membership.
2. **Threshold rigidity** — semantically equivalent skills at different abstraction levels (e.g., "confirm before executing cancellation API" vs "require explicit user approval prior to state-changing tool calls") may score 0.55-0.65, missing the match.
3. **No semantic judgment** — embeddings measure surface similarity, not operational equivalence.

The 11-18% stability figure may **undercount** true semantic stability. Using Opus as a judge instead of embeddings could yield significantly higher stability rates, changing the interpretation of how much content is truly "stochastic."

### 6. No Trace Ordering Sensitivity

All 34 runs process the same 25 traces in identical sorted order. The SkillManager processes traces sequentially — early traces shape the skillbook that influences how later traces get integrated. The stable core might be partially an artifact of fixed ordering (early traces always getting priority) rather than genuine content importance. No shuffled-order runs exist to test this.

### 7. Dedup Interval Interaction Unexplored

The excluded `no-budget/run_1` (dedup_interval=1 vs 5) hints that dedup interval affects outcomes, but this parameter was held constant at 5 for all valid runs. The interaction between budget level and dedup interval is unknown — tight budgets with frequent deduplication might behave very differently from tight budgets with infrequent deduplication.

### 8. No Budget Efficiency Curve

The data to compute marginal value per budget token exists but isn't synthesized. Key ratios:

| Budget | Avg TOON | Stable Skills | TOON per Stable Skill | Marginal TOON per Extra Budget Token |
|---|---|---|---|---|
| 500 | 3,077 | 16 | 192 | — |
| 1000 | 3,664 | 25 | 147 | 1.17 |
| 2000 | 3,936 | 22 | 179 | 0.27 |
| 3000 | 4,352 | 20 | 218 | 0.42 |
| 5000 | 4,425 | 28 | 158 | 0.04 |
| 10000 | 4,559 | 30 | 152 | 0.03 |
| None | 4,840 | 23 | 210 | — |

Above budget-1000, each additional budget token yields diminishing returns in TOON output. The jump from 500→1000 is the most efficient (+9 stable skills for +587 TOON). Above 5000, extra budget is almost entirely absorbed by stochastic growth.

## Future Experiments

### High Priority — Directly Actionable

#### 1. TAU-bench Evaluation with Different Skillbooks

Take the consensus, median, and Opus-compressed skillbooks from this experiment and run them through actual task evaluation (TAU-bench airline domain). This is the only way to resolve whether structural compression preserves task performance. Compare:

- Single-run skillbook (median) vs consensus skillbook vs Opus(consensus)
- Budget-500 vs budget-1000 vs no-budget (all single-run)
- No skillbook baseline

This directly addresses Limitation #1 (no downstream evaluation) and #2 (quality of stable core).

#### 2. REMOVE Analysis

For budget-500 and budget-1000, extract what skills get removed at each trace step and whether removals are consistent across runs. Specifically:

- Map removed skills to their topic clusters
- Compute removal consistency: do ≥3/5 runs remove skills from the same clusters?
- Compare helpful/harmful counts of removed vs surviving skills

This addresses Limitation #4 and informs whether budget pressure is intelligent pruning or random churn.

#### 3. Trace Order Sensitivity

Run 5 shuffled orderings at budget-1000 (the tightest stable budget, std=2.0) and compare final skillbooks against the sorted-order runs. Measure:

- Topic overlap between shuffled and sorted runs (heatmap comparison)
- Whether the same stable clusters (≥3/5 runs) emerge
- Whether early-trace topics dominate regardless of order

If the stable core survives shuffling, it's genuinely robust. If it shifts, the "core" is partly an ordering artifact.

### Medium Priority — Informative but More Work

#### 4. Opus Semantic Clustering

Re-run the stability analysis from Part 2 using Opus as a semantic judge instead of embedding cosine similarity. For each pair of skills, ask Opus: "Do these encode the same operational rule?" This addresses Limitation #5 directly. If stability rates jump from 11-18% to 30-40%+, the "stochastic noise" estimate needs significant revision, and the consensus approach may be discarding less noise than assumed.

#### 5. Dedup Interval x Budget Grid

Run dedup_interval in {1, 3, 5, 10} x budget in {500, 1000, 5000} — a 12-cell grid, 3 runs each (36 runs). This addresses Limitation #7 and could reveal whether frequent deduplication compensates for tight budgets (or makes churn worse).

#### 6. Budget Efficiency Deep Dive

Formalize the marginal analysis from Limitation #8 into proper diminishing-returns curves. Plot stable_skills/budget and consensus_toon/budget. Identify the "elbow" where additional budget stops yielding stable content. The raw data already exists; this is primarily an analysis task.

### Lower Priority — Research Extensions

#### 7. Different SkillManager Models

All runs use Haiku 4.5. Running the same experiment with Sonnet would show whether the stochastic noise level is model-dependent. If Sonnet produces higher stability rates (e.g., 30% vs 15%), the variance problem may be partially a model capability issue rather than fundamental to the approach.

#### 8. More Traces (Saturation Test)

25 traces may not be enough for full convergence. Run to 50 or 100 traces at budget-1000 and no-budget, tracking stable cluster count at each step. If stable skills plateau by trace 25, the current experiment is sufficient. If they're still growing, the 11-18% stability figure underestimates the converged value.

#### 9. Adaptive Budgets (Generate Then Compress)

Compare two strategies: (a) generate with budget constraint from the start, vs (b) generate with no budget, then retroactively compress to a target token count. Strategy (b) lets the SkillManager see all content before deciding what to keep, potentially making better pruning decisions. This tests whether it's better to constrain during generation or compress afterward.
