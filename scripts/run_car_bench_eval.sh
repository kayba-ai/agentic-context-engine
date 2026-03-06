#!/usr/bin/env bash
# =============================================================================
# CAR-bench pass^4 Evaluation: 13 configs in parallel tmux sessions
#
# Evaluates 3 skillbook types x 2 budgets x 2 source models + 1 baseline
# All runs use Haiku as the test agent, k=4, Gemini 2.5 Flash as user sim
#
# Skillbook types:
#   consensus       - Raw consensus markdown (appended to wiki.md)
#   median          - Raw median run markdown (appended to wiki.md)
#   opus-median     - Opus-compressed median run (appended to wiki.md)
#
# Usage:
#   chmod +x scripts/run_car_bench_eval.sh
#   ./scripts/run_car_bench_eval.sh          # Launch all 13 configs
#   ./scripts/run_car_bench_eval.sh --dry-run  # Print commands without executing
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
CAR_BENCH_DIR="$PROJECT_DIR/benchmarks/car-bench/car-bench"
WIKI_ORIG="$CAR_BENCH_DIR/car_bench/envs/car_voice_assistant/wiki.md"
PREPARED_WIKIS_DIR="$PROJECT_DIR/benchmarks/car-bench/prepared_wikis"
OUTPUT_DIR="$PROJECT_DIR/results/car_bench_eval"
TMUX_SESSION_PREFIX="car-eval"
MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_PROVIDER="bedrock"
USER_MODEL="gemini-2.5-flash"
K=4
MAX_CONCURRENCY=5
TASK_TYPES=("disambiguation")
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — commands will be printed but not executed ==="
    echo ""
fi

# --- Validate prerequisites ---
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: API keys file not found at $ENV_FILE"
    exit 1
fi

if [[ ! -f "$WIKI_ORIG" ]]; then
    echo "ERROR: wiki.md not found at $WIKI_ORIG"
    echo "  Clone car-bench: git clone https://github.com/CAR-bench/car-bench.git $CAR_BENCH_DIR"
    exit 1
fi

# Verify car-bench is importable via uv
if ! uv run --project "$PROJECT_DIR" python -c "import car_bench" 2>/dev/null; then
    echo "ERROR: car_bench not importable. Install with: uv pip install -e $CAR_BENCH_DIR"
    exit 1
fi

# --- Skillbook paths ---
HAIKU_DIR="$PROJECT_DIR/results/variance_experiment_car_haiku_4.5"
SONNET_DIR="$PROJECT_DIR/results/variance_experiment_car_sonnet_4.6"

# Median run numbers (from analysis_results.json):
#   Haiku:  no-budget=run_2, budget-500=run_3
#   Sonnet: no-budget=run_1, budget-500=run_5

# --- Define configurations ---
# Format: CONFIGS["label"]="skillbook_path_or_empty"
declare -A CONFIGS

# Baseline (no skillbook injection)
CONFIGS["baseline"]=""

# Haiku source: consensus (markdown)
CONFIGS["haiku-nobudget-consensus"]="$HAIKU_DIR/consensus/consensus_no-budget.md"
CONFIGS["haiku-500-consensus"]="$HAIKU_DIR/consensus/consensus_budget-500.md"

# Haiku source: median run (raw markdown)
CONFIGS["haiku-nobudget-median"]="$HAIKU_DIR/no-budget/run_2/skills_final.md"
CONFIGS["haiku-500-median"]="$HAIKU_DIR/budget-500/run_3/skills_final.md"

# Haiku source: opus-compressed median (markdown)
CONFIGS["haiku-nobudget-opus-median"]="$HAIKU_DIR/opus_compressed/opus_median_no-budget.md"
CONFIGS["haiku-500-opus-median"]="$HAIKU_DIR/opus_compressed/opus_median_budget-500.md"

# Sonnet source: consensus (markdown)
CONFIGS["sonnet-nobudget-consensus"]="$SONNET_DIR/consensus/consensus_no-budget.md"
CONFIGS["sonnet-500-consensus"]="$SONNET_DIR/consensus/consensus_budget-500.md"

# Sonnet source: median run (raw markdown)
CONFIGS["sonnet-nobudget-median"]="$SONNET_DIR/no-budget/run_1/skills_final.md"
CONFIGS["sonnet-500-median"]="$SONNET_DIR/budget-500/run_5/skills_final.md"

# Sonnet source: opus-compressed median (markdown)
CONFIGS["sonnet-nobudget-opus-median"]="$SONNET_DIR/opus_compressed/opus_median_no-budget.md"
CONFIGS["sonnet-500-opus-median"]="$SONNET_DIR/opus_compressed/opus_median_budget-500.md"

# --- Validate all skillbook files exist ---
echo "Validating skillbook files..."
MISSING=0
for label in "${!CONFIGS[@]}"; do
    path="${CONFIGS[$label]}"
    if [[ -z "$path" ]]; then
        continue  # baseline
    fi
    if [[ ! -f "$path" ]]; then
        echo "  MISSING: $label -> $path"
        MISSING=$((MISSING + 1))
    fi
done
if [[ $MISSING -gt 0 ]]; then
    echo "ERROR: $MISSING skillbook files missing. Aborting."
    exit 1
fi
echo "  All ${#CONFIGS[@]} configurations validated."
echo ""

# --- Prepare wiki variants ---
echo "Preparing wiki variants..."
mkdir -p "$PREPARED_WIKIS_DIR"

for label in "${!CONFIGS[@]}"; do
    wiki_out="$PREPARED_WIKIS_DIR/wiki_${label}.md"
    path="${CONFIGS[$label]}"

    if [[ -z "$path" ]]; then
        # Baseline: original wiki.md only
        cp "$WIKI_ORIG" "$wiki_out"
    else
        # Append skillbook to wiki.md
        cat "$WIKI_ORIG" > "$wiki_out"
        echo "" >> "$wiki_out"
        echo "" >> "$wiki_out"
        echo "## ACE Learned Skills" >> "$wiki_out"
        echo "" >> "$wiki_out"
        cat "$path" >> "$wiki_out"
    fi
    echo "  Created: wiki_${label}.md"
done
echo ""

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

# --- Build and launch tmux sessions ---
echo "Launching ${#CONFIGS[@]} evaluation configs..."
echo "  Output:         $OUTPUT_DIR"
echo "  Agent model:    $MODEL ($MODEL_PROVIDER)"
echo "  User model:     $USER_MODEL"
echo "  K=$K, concurrency=$MAX_CONCURRENCY"
echo "  Task types:     ${TASK_TYPES[*]}"
echo ""

# Sort labels for consistent ordering
SORTED_LABELS=($(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort))

for label in "${SORTED_LABELS[@]}"; do
    wiki_path="$PREPARED_WIKIS_DIR/wiki_${label}.md"
    log_dir="$OUTPUT_DIR/$label"

    # Build command that runs all 3 task types sequentially
    CMD="set -a && source $ENV_FILE && set +a && "
    CMD+="unset HF_DATASETS_CACHE HF_HUB_CACHE BENCHMARK_CACHE_DIR HUGGINGFACE_API_KEY APPWORLD_ROOT && "
    CMD+="export PYTHONUNBUFFERED=1 && "
    CMD+="export CAR_BENCH_WIKI_PATH=$wiki_path && "
    CMD+="mkdir -p $log_dir && "

    for task_type in "${TASK_TYPES[@]}"; do
        CMD+="echo '=== [$label] Running $task_type ===' && "
        CMD+="uv run --project $PROJECT_DIR python $CAR_BENCH_DIR/run.py "
        CMD+="--model $MODEL "
        CMD+="--model-provider $MODEL_PROVIDER "
        CMD+="--user-model $USER_MODEL "
        CMD+="--user-model-provider vertex_ai "
        CMD+="--policy-evaluator-model $USER_MODEL "
        CMD+="--policy-evaluator-model-provider vertex_ai "
        CMD+="--task-split test "
        CMD+="--task-type $task_type "
        CMD+="--num-trials $K "
        CMD+="--max-concurrency $MAX_CONCURRENCY "
        CMD+="--temperature 0.0 "
        CMD+="--log-dir $log_dir "
        CMD+="2>&1 | tee -a $log_dir/${task_type}.log && "
    done

    # Remove trailing ' && '
    CMD="${CMD% && }"

    SESSION_NAME="${TMUX_SESSION_PREFIX}-${label}"

    if $DRY_RUN; then
        echo "[$label]"
        echo "  tmux:     $SESSION_NAME"
        echo "  wiki:     $wiki_path"
        echo "  log_dir:  $log_dir"
        echo "  tasks:    ${TASK_TYPES[*]}"
        echo ""
    else
        # Kill existing session with same name if any
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        tmux new-session -d -s "$SESSION_NAME" "$CMD"
        echo "  Started: $SESSION_NAME"
    fi
done

echo ""
if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE — ${#CONFIGS[@]} configs would be launched ==="
else
    echo "=== All ${#CONFIGS[@]} tmux sessions launched ==="
    echo ""
    echo "Monitor progress:"
    echo "  tmux ls                                    # List all sessions"
    echo "  tmux attach -t ${TMUX_SESSION_PREFIX}-baseline    # Attach to a session"
    echo "  tail -f $OUTPUT_DIR/baseline/base.log     # Follow log"
    echo ""
    echo "Check results when done:"
    echo "  ls $OUTPUT_DIR/*/base_test/*.json"
    echo "  for d in $OUTPUT_DIR/*/; do echo \$(basename \$d); cat \$d/*.log | grep -E 'reward|pass'; done"
fi
