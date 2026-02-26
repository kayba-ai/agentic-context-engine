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

| Budget       |   Skills (mean) |   Skills (std) |   Sections (mean) |   TOON (mean) |   next_id (mean) |   Total ADDs (mean) |   Total REMOVEs (mean) |
|:-------------|----------------:|---------------:|------------------:|--------------:|-----------------:|--------------------:|-----------------------:|
| budget-500   |            41.8 |           10.4 |              11.8 |          3077 |               52 |                  52 |                     10 |
| budget-1000  |            51   |            2   |               9.8 |          3664 |               57 |                  57 |                      6 |
| budget-2000  |            58   |            2.3 |              10.8 |          3936 |               65 |                  65 |                      7 |
| budget-3000  |            66.6 |            7.5 |              13.8 |          4352 |               73 |                  73 |                      6 |
| budget-5000  |            68.6 |            8.6 |               9.2 |          4425 |               72 |                  72 |                      3 |
| budget-10000 |            74   |            3.4 |              12.6 |          4559 |               74 |                  74 |                      0 |
| no-budget    |            75.2 |            7.8 |              15.2 |          4844 |               76 |                  76 |                      0 |

## 1. Growth Curves

How skills accumulate over the 25-trace sequence.

![Skill Growth](figures/growth_skills.png)

![Skills Overlay](figures/overlay_skills.png)

![TOON Tokens Overlay](figures/overlay_toon.png)

![Sections Overlay](figures/overlay_sections.png)

## 2. Skill Lifecycle

ADD/UPDATE/REMOVE events and skill survival analysis.

![Delta Events](figures/deltas_representative.png)

![Survival Curves](figures/survival.png)

![Churn Analysis](figures/churn.png)

![Lifespan Distributions](figures/lifespans.png)

## 3. Section Evolution

When do sections first appear? How do their sizes change over time?

![Section Timelines](figures/section_timelines.png)

![Section Heatmap](figures/section_heatmap.png)

## 4. Cross-Run Convergence

Do independent runs converge on similar skill structures?

### Embedding Nearest-Neighbor Overlap

- **budget-500**: mean NN cosine 0.669 ± 0.013
  - run_1 ↔ run_2: 0.683
  - run_1 ↔ run_3: 0.665
  - run_1 ↔ run_4: 0.693
  - run_1 ↔ run_5: 0.659
  - run_2 ↔ run_3: 0.681
  - run_2 ↔ run_4: 0.668
  - run_2 ↔ run_5: 0.664
  - run_3 ↔ run_4: 0.667
  - run_3 ↔ run_5: 0.665
  - run_4 ↔ run_5: 0.644
- **budget-1000**: mean NN cosine 0.695 ± 0.014
  - run_1 ↔ run_2: 0.692
  - run_1 ↔ run_3: 0.691
  - run_1 ↔ run_4: 0.682
  - run_1 ↔ run_5: 0.698
  - run_2 ↔ run_3: 0.669
  - run_2 ↔ run_4: 0.698
  - run_2 ↔ run_5: 0.713
  - run_3 ↔ run_4: 0.701
  - run_3 ↔ run_5: 0.687
  - run_4 ↔ run_5: 0.719
- **budget-2000**: mean NN cosine 0.678 ± 0.010
  - run_1 ↔ run_2: 0.670
  - run_1 ↔ run_3: 0.665
  - run_1 ↔ run_4: 0.677
  - run_1 ↔ run_5: 0.682
  - run_2 ↔ run_3: 0.673
  - run_2 ↔ run_4: 0.681
  - run_2 ↔ run_5: 0.663
  - run_3 ↔ run_4: 0.682
  - run_3 ↔ run_5: 0.691
  - run_4 ↔ run_5: 0.695
- **budget-3000**: mean NN cosine 0.687 ± 0.008
  - run_1 ↔ run_2: 0.674
  - run_1 ↔ run_3: 0.689
  - run_1 ↔ run_4: 0.694
  - run_1 ↔ run_5: 0.688
  - run_2 ↔ run_3: 0.681
  - run_2 ↔ run_4: 0.699
  - run_2 ↔ run_5: 0.672
  - run_3 ↔ run_4: 0.695
  - run_3 ↔ run_5: 0.684
  - run_4 ↔ run_5: 0.692
- **budget-5000**: mean NN cosine 0.687 ± 0.011
  - run_1 ↔ run_2: 0.679
  - run_1 ↔ run_3: 0.693
  - run_1 ↔ run_4: 0.678
  - run_1 ↔ run_5: 0.681
  - run_2 ↔ run_3: 0.706
  - run_2 ↔ run_4: 0.679
  - run_2 ↔ run_5: 0.702
  - run_3 ↔ run_4: 0.682
  - run_3 ↔ run_5: 0.695
  - run_4 ↔ run_5: 0.673
- **budget-10000**: mean NN cosine 0.691 ± 0.010
  - run_1 ↔ run_2: 0.691
  - run_1 ↔ run_3: 0.679
  - run_1 ↔ run_4: 0.686
  - run_1 ↔ run_5: 0.696
  - run_2 ↔ run_3: 0.689
  - run_2 ↔ run_4: 0.715
  - run_2 ↔ run_5: 0.699
  - run_3 ↔ run_4: 0.684
  - run_3 ↔ run_5: 0.688
  - run_4 ↔ run_5: 0.685
- **no-budget**: mean NN cosine 0.687 ± 0.028
  - run_1 ↔ run_2: 0.643
  - run_1 ↔ run_3: 0.658
  - run_1 ↔ run_4: 0.667
  - run_1 ↔ run_5: 0.653
  - run_2 ↔ run_3: 0.702
  - run_2 ↔ run_4: 0.710
  - run_2 ↔ run_5: 0.690
  - run_3 ↔ run_4: 0.713
  - run_3 ↔ run_5: 0.726
  - run_4 ↔ run_5: 0.711

### Skill Clustering (KMeans k=10)

![Cluster Coverage](figures/cluster_coverage.png)

## 5. Conciseness

Token efficiency per skill over time.

![Conciseness](figures/conciseness.png)

## 6. Cross-Budget Summary

Final metrics across all budgets.

| Budget       |   Skills (mean) |   Skills (std) |   Sections (mean) |   TOON (mean) |   next_id (mean) |   Total ADDs (mean) |   Total REMOVEs (mean) |
|:-------------|----------------:|---------------:|------------------:|--------------:|-----------------:|--------------------:|-----------------------:|
| budget-500   |            41.8 |           10.4 |              11.8 |          3077 |               52 |                  52 |                     10 |
| budget-1000  |            51   |            2   |               9.8 |          3664 |               57 |                  57 |                      6 |
| budget-2000  |            58   |            2.3 |              10.8 |          3936 |               65 |                  65 |                      7 |
| budget-3000  |            66.6 |            7.5 |              13.8 |          4352 |               73 |                  73 |                      6 |
| budget-5000  |            68.6 |            8.6 |               9.2 |          4425 |               72 |                  72 |                      3 |
| budget-10000 |            74   |            3.4 |              12.6 |          4559 |               74 |                  74 |                      0 |
| no-budget    |            75.2 |            7.8 |              15.2 |          4844 |               76 |                  76 |                      0 |

![Budget Saturation](figures/saturation.png)

## 7. Opus Compression

Raw skillbook vs Opus-compressed individual runs vs consensus.

| Budget       | Raw Skills   | Raw MD Tokens   | Opus Skills   | Opus MD Tokens   | Compression %   |   Consensus Skills |   Consensus Tokens |
|:-------------|:-------------|:----------------|:--------------|:-----------------|:----------------|-------------------:|-------------------:|
| budget-500   | 41.8 ± 11.6  | 5479 ± 1327     | 22.0 ± 3.5    | 2458 ± 528       | 45.2% ± 4.8%    |                 14 |                957 |
| budget-1000  | 51.0 ± 2.2   | 6547 ± 338      | 33.0 ± 7.5    | 2994 ± 549       | 45.7% ± 7.5%    |                 20 |               1047 |
| budget-2000  | 58.0 ± 2.5   | 7298 ± 678      | 38.4 ± 4.7    | 3259 ± 320       | 44.9% ± 5.6%    |                 17 |               1007 |
| budget-3000  | 66.6 ± 8.4   | 8061 ± 595      | 43.0 ± 5.5    | 3600 ± 441       | 44.8% ± 5.4%    |                 19 |               1030 |
| budget-5000  | 68.6 ± 9.7   | 8444 ± 1337     | 46.8 ± 9.2    | 3615 ± 604       | 43.5% ± 8.9%    |                 23 |               1273 |
| budget-10000 | 74.0 ± 3.8   | 8886 ± 533      | 55.2 ± 6.8    | 4259 ± 301       | 48.1% ± 5.0%    |                 17 |                981 |
| no-budget    | 75.2 ± 8.7   | 9201 ± 1160     | 51.6 ± 12.1   | 4156 ± 870       | 45.8% ± 11.0%   |                 21 |               1059 |

![Compression Distribution](figures/compression_distribution.png)

---
*Generated by `scripts/analysis/skillbook_history.py`*
