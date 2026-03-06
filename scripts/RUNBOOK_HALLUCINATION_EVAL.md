# Runbook: Hallucination-Only Evaluation

Task-separated skillbook evaluation for the mixed-tasks hypothesis.
Tests whether skillbooks trained exclusively on hallucination traces
outperform mixed-trained skillbooks on hallucination test tasks.

## Comparison Matrix

| Config | Training Data | Test Data |
|--------|--------------|-----------|
| Baseline | None | Hallucination (50 tasks) |
| Mixed-trained (Section 12) | All 129 traces (base+disamb+halluc) | Hallucination (50 tasks) |
| Hallucination-only (this eval) | 48 hallucination traces | Hallucination (50 tasks) |

Mixed-trained results already exist in `results/car_bench_eval/` — no need to re-run.

---

## Step 1: Training — Variance Experiments (48 traces)

Run 4 variance experiments in parallel (2 models × 2 budgets, 5 runs each = 20 runs total).

Traces directory: `results/traces_car_bench_hallucination/` (48 symlinks, already created).

Launch all 4 in parallel tmux sessions to minimize wall clock time:

```bash
# From project root
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"

# Haiku no-budget
tmux new-session -d -s train-halluc-haiku-nb \
  "set -a && source $ENV_FILE && set +a && \
   uv run python scripts/run_variance_experiment.py \
     --model 'bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0' \
     --traces-dir results/traces_car_bench_hallucination \
     --output-dir results/variance_experiment_car_hallucination_haiku_4.5 \
     --budgets none \
     --resume"

# Haiku budget-500
tmux new-session -d -s train-halluc-haiku-500 \
  "set -a && source $ENV_FILE && set +a && \
   uv run python scripts/run_variance_experiment.py \
     --model 'bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0' \
     --traces-dir results/traces_car_bench_hallucination \
     --output-dir results/variance_experiment_car_hallucination_haiku_4.5 \
     --budgets 500 \
     --resume"

# Sonnet no-budget
tmux new-session -d -s train-halluc-sonnet-nb \
  "set -a && source $ENV_FILE && set +a && \
   uv run python scripts/run_variance_experiment.py \
     --model 'bedrock/us.anthropic.claude-sonnet-4-6' \
     --traces-dir results/traces_car_bench_hallucination \
     --output-dir results/variance_experiment_car_hallucination_sonnet_4.6 \
     --budgets none \
     --resume"

# Sonnet budget-500
tmux new-session -d -s train-halluc-sonnet-500 \
  "set -a && source $ENV_FILE && set +a && \
   uv run python scripts/run_variance_experiment.py \
     --model 'bedrock/us.anthropic.claude-sonnet-4-6' \
     --traces-dir results/traces_car_bench_hallucination \
     --output-dir results/variance_experiment_car_hallucination_sonnet_4.6 \
     --budgets 500 \
     --resume"
```

Monitor:
```bash
tmux ls
tmux attach -t train-halluc-haiku-nb
```

**Output**: 20 runs total across:
- `results/variance_experiment_car_hallucination_haiku_4.5/{no-budget,budget-500}/run_{1..5}/`
- `results/variance_experiment_car_hallucination_sonnet_4.6/{no-budget,budget-500}/run_{1..5}/`

Each run produces `skillbook_00.json` through `skillbook_47.json` + `skills_final.md` + `run_info.json`.

**Note**: The `--resume` flag skips completed runs, so it's safe to re-run if interrupted.

**Caveat**: Both budget levels for the same model share the same `--output-dir`, so both tmux sessions write into the same directory. This is fine — they write into separate subdirectories (`no-budget/` vs `budget-500/`). However, both will try to write `config.json` and `SUMMARY.md` at the top level. The last one to finish wins. This is harmless since both contain the same model info and the summary is regenerated anyway.

---

## Step 2: Build Consensus + Median Skillbooks

After all 20 training runs complete:

```bash
uv run python scripts/prepare_hallucination_only_eval.py
```

**What it does**:
1. Loads `skillbook_47.json` (final snapshot after 48 traces) from all 5 runs per config
2. Identifies the median run per config (closest to mean TOON tokens)
3. Builds consensus skillbooks (t≥3 of 5 runs, cosine threshold 0.7)
4. Saves markdown files + prepared wiki variants

**Output**:
- `results/car_bench_eval_hallucination_only/skillbooks/` — 8 markdown files:
  - `{haiku,sonnet}-{nobudget,500}-halluc48-{median,consensus}.md`
- `benchmarks/car-bench/prepared_wikis/` — 8 wiki variants:
  - `wiki_{haiku,sonnet}-{nobudget,500}-halluc48-{median,consensus}.md`
- `results/car_bench_eval_hallucination_only/skillbooks/halluc48_metadata.json`

---

## Step 3: Opus Compression of Median Skillbooks (Interactive)

This step is done interactively in a Claude chat session. Each median skillbook
is compressed by Opus in a **clean context** (separate subagent / fresh conversation).

There are 4 median skillbooks to compress:
- `haiku-nobudget-halluc48-median.md`
- `haiku-500-halluc48-median.md`
- `sonnet-nobudget-halluc48-median.md`
- `sonnet-500-halluc48-median.md`

### Compression Instructions (for each)

Provide the median skillbook content and ask Opus to compress it with these instructions:

> Compress this skillbook by merging near-duplicate skills, tightening wording,
> consolidating related sections, and preserving all high-value content.
>
> **J/E retention rule**: Retain Justification/Evidence metadata — do NOT strip it.
> For near-duplicate merges, keep one representative J/E pair.
> For multi-skill merges, keep the most representative J/E (occasionally multiple
> when sub-concepts are genuinely distinct).
> Shorten labels from `Justification:` / `Evidence:` to `J:` / `E:`,
> and tighten the text itself.
>
> Output the compressed skillbook in the same markdown format.

### Save Compressed Files

Save each compressed output to:
- `results/car_bench_eval_hallucination_only/skillbooks/{label}-halluc48-opus-median.md`

Then create the corresponding wiki variants:
```bash
# For each of the 4 opus-median files:
WIKI_ORIG="benchmarks/car-bench/car-bench/car_bench/envs/car_voice_assistant/wiki.md"
WIKIS_DIR="benchmarks/car-bench/prepared_wikis"

for label in haiku-nobudget haiku-500 sonnet-nobudget sonnet-500; do
    opus_file="results/car_bench_eval_hallucination_only/skillbooks/${label}-halluc48-opus-median.md"
    wiki_out="${WIKIS_DIR}/wiki_${label}-halluc48-opus-median.md"
    cat "$WIKI_ORIG" > "$wiki_out"
    echo "" >> "$wiki_out"
    echo "" >> "$wiki_out"
    echo "## ACE Learned Skills" >> "$wiki_out"
    echo "" >> "$wiki_out"
    cat "$opus_file" >> "$wiki_out"
    echo "Created: $wiki_out"
done
```

After this step, all 12 wiki variants + baseline should exist in `prepared_wikis/`.

---

## Step 4: Run Evaluation

Verify all wiki files exist:
```bash
./scripts/run_car_bench_eval_hallucination_only.sh --dry-run
```

Launch all 13 configs in parallel tmux sessions:
```bash
./scripts/run_car_bench_eval_hallucination_only.sh
```

**Configuration**:
- Agent model: Haiku 4.5 (Bedrock)
- User simulator: Gemini 2.5 Flash (Vertex AI)
- Policy evaluator: Gemini 2.5 Flash (Vertex AI)
- Task type: hallucination only
- K=4 trials per task, 50 test tasks = 200 trials per config
- Max concurrency: 5

**Output**: `results/car_bench_eval_hallucination_only/{config}/hallucination_test/*.json`

Monitor:
```bash
tmux ls
tail -f results/car_bench_eval_hallucination_only/baseline/hallucination.log

# Check all results when done:
for d in results/car_bench_eval_hallucination_only/*/; do
    echo $(basename $d)
    cat $d/*.log | grep -E 'reward|pass'
done
```

---

## Step 5: Analysis

Compare hallucination-only trained results against:
1. **Baseline** (no skillbook) — from this eval
2. **Mixed-trained** (all 129 traces) — from `results/car_bench_eval/` (already exists)

The mixed-trained hallucination results to compare against (from Section 12):

| Config | Hallucination Reward | Δ Baseline |
|--------|---------------------|------------|
| baseline | 0.485 | — |
| sonnet-500-consensus | 0.490 | +0.005 |
| sonnet-500-median | 0.490 | +0.005 |
| haiku-500-opus-median | 0.465 | -0.020 |
| sonnet-nobudget-opus-median | 0.360 | -0.125 |
| sonnet-nobudget-median | 0.230 | -0.255 |

Key questions to answer:
- Do hallucination-only skillbooks beat the baseline on hallucination tasks?
- Do they beat mixed-trained skillbooks on hallucination tasks?
- Is the interference from base/disambiguation skills the cause of poor mixed-trained hallucination performance?

---

## File Inventory

### Scripts
| File | Purpose |
|------|---------|
| `scripts/prepare_hallucination_only_eval.py` | Build consensus + median from hallucination-only runs |
| `scripts/run_car_bench_eval_hallucination_only.sh` | Launch 13 eval configs in parallel tmux |
| `scripts/RUNBOOK_HALLUCINATION_EVAL.md` | This file |

### Data directories
| Directory | Contents |
|-----------|----------|
| `results/traces_car_bench_hallucination/` | 48 hallucination trace symlinks |
| `results/variance_experiment_car_hallucination_haiku_4.5/` | Haiku training runs (10 runs) |
| `results/variance_experiment_car_hallucination_sonnet_4.6/` | Sonnet training runs (10 runs) |
| `results/car_bench_eval_hallucination_only/skillbooks/` | Prepared skillbooks (median, consensus, opus-median) |
| `results/car_bench_eval_hallucination_only/{config}/` | Eval results per config |
| `benchmarks/car-bench/prepared_wikis/wiki_*-halluc48-*.md` | Wiki variants for injection |
