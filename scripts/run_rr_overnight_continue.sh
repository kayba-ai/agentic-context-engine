#!/usr/bin/env bash
# =============================================================================
# RR Overnight — Continue: poll, merge, consensus, eval
#
# The 24 tmux training sessions are already running. This script:
# 1. Polls until all 24 batches produce skillbook.json
# 2. Merges each of the 4 new runs
# 3. Builds consensus from all 5 runs
# 4. Launches consensus eval automatically
#
# Opus compression is a manual step — do it tomorrow, then run eval via:
#   ./scripts/run_rr_overnight.sh --phase3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
DEDUP_THRESHOLD=0.7

EXISTING_RUN="$PROJECT_DIR/results/rr_car_bench_20260309_162529"

# The 4 new run dirs (already created by the first launch)
RUN_DIRS=(
    "$PROJECT_DIR/results/rr_car_bench_20260309_232848_run1"
    "$PROJECT_DIR/results/rr_car_bench_20260309_232849_run2"
    "$PROJECT_DIR/results/rr_car_bench_20260309_232850_run3"
    "$PROJECT_DIR/results/rr_car_bench_20260309_232851_run4"
)

ALL_RUN_DIRS=("$EXISTING_RUN" "${RUN_DIRS[@]}")
CONSENSUS_DIR="$PROJECT_DIR/results/rr_car_bench_consensus"

BATCHES=(
    "base_vehicle_climate"
    "base_nav_charging_prod"
    "halluc_vehicle_climate"
    "halluc_nav_charging_prod"
    "disambig_vehicle_climate"
    "disambig_nav_charging_prod"
)

TASK_TYPES=("base" "hallucination" "disambiguation")

# Eval config
EVAL_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
EVAL_MODEL_PROVIDER="bedrock"
EVAL_USER_MODEL="gemini-2.5-flash"
EVAL_K=4
EVAL_MAX_CONCURRENCY=5
CAR_BENCH_DIR="$PROJECT_DIR/benchmarks/car-bench/car-bench"
WIKI_ORIG="$CAR_BENCH_DIR/car_bench/envs/car_voice_assistant/wiki.md"
PREPARED_WIKIS_DIR="$PROJECT_DIR/benchmarks/car-bench/prepared_wikis"
EVAL_OUTPUT_DIR="$PROJECT_DIR/results/rr_car_bench_eval"

echo "============================================================"
echo "RR Overnight — Continue (poll → merge → consensus → eval)"
echo "============================================================"
echo "Existing run: $(basename $EXISTING_RUN)"
echo "New runs:     ${#RUN_DIRS[@]}"
echo ""

# =====================================================================
# PHASE 1: Poll for completion
# =====================================================================
echo "=== Polling for batch completion ==="
echo ""

while true; do
    ALL_DONE=true
    COMPLETED=0
    TOTAL=$((${#RUN_DIRS[@]} * ${#BATCHES[@]}))

    for RUN_DIR in "${RUN_DIRS[@]}"; do
        for batch in "${BATCHES[@]}"; do
            if [[ -f "$RUN_DIR/$batch/skillbook.json" ]]; then
                COMPLETED=$((COMPLETED + 1))
            else
                ALL_DONE=false
            fi
        done
    done

    echo "  $(date +%H:%M:%S) — $COMPLETED/$TOTAL batches complete"

    if $ALL_DONE; then
        echo ""
        echo "=== All batches complete! ==="
        break
    fi

    sleep 60
done

# =====================================================================
# PHASE 2: Merge each run + build consensus
# =====================================================================
echo ""
echo "=== Merging runs ==="
echo ""

set -a && source "$ENV_FILE" && set +a

for RUN_DIR in "${RUN_DIRS[@]}"; do
    if [[ -f "$RUN_DIR/skillbook_merged.json" ]]; then
        echo "  Already merged: $(basename $RUN_DIR)"
        continue
    fi

    echo "  Merging: $(basename $RUN_DIR)..."
    uv run --project "$PROJECT_DIR" python "$SCRIPT_DIR/run_rr_car_bench_merge.py" \
        --output-dir "$RUN_DIR" \
        --threshold "$DEDUP_THRESHOLD"
done

echo ""
echo "=== Building consensus from ${#ALL_RUN_DIRS[@]} runs ==="
echo ""

uv run --project "$PROJECT_DIR" python "$SCRIPT_DIR/run_rr_car_bench_consensus.py" \
    --run-dirs "${ALL_RUN_DIRS[@]}" \
    --output-dir "$CONSENSUS_DIR" \
    --threshold 3 \
    --similarity 0.7

# =====================================================================
# PHASE 3: Launch consensus eval
# =====================================================================
echo ""
echo "=== Launching consensus eval ==="
echo ""

declare -A EVAL_CONFIGS
EVAL_CONFIGS["rr-consensus"]="$CONSENSUS_DIR/consensus.md"

mkdir -p "$PREPARED_WIKIS_DIR"

for label in "${!EVAL_CONFIGS[@]}"; do
    wiki_out="$PREPARED_WIKIS_DIR/wiki_${label}.md"
    skillbook_path="${EVAL_CONFIGS[$label]}"

    cat "$WIKI_ORIG" > "$wiki_out"
    echo "" >> "$wiki_out"
    echo "" >> "$wiki_out"
    echo "## ACE Learned Skills" >> "$wiki_out"
    echo "" >> "$wiki_out"
    cat "$skillbook_path" >> "$wiki_out"

    echo "  Wiki: wiki_${label}.md ($(wc -c < "$wiki_out") bytes)"
done
echo ""

for label in "${!EVAL_CONFIGS[@]}"; do
    wiki_path="$PREPARED_WIKIS_DIR/wiki_${label}.md"
    log_dir="$EVAL_OUTPUT_DIR/$label"
    SESSION_NAME="rr-eval-${label}"

    CMD="set -a && source $ENV_FILE && set +a && "
    CMD+="unset HF_DATASETS_CACHE HF_HUB_CACHE BENCHMARK_CACHE_DIR HUGGINGFACE_API_KEY APPWORLD_ROOT && "
    CMD+="export PYTHONUNBUFFERED=1 && "
    CMD+="export CAR_BENCH_WIKI_PATH=$wiki_path && "
    CMD+="mkdir -p $log_dir && "

    for task_type in "${TASK_TYPES[@]}"; do
        CMD+="echo '=== [$label] Running $task_type ===' && "
        CMD+="uv run --project $PROJECT_DIR python $CAR_BENCH_DIR/run.py "
        CMD+="--model $EVAL_MODEL "
        CMD+="--model-provider $EVAL_MODEL_PROVIDER "
        CMD+="--user-model $EVAL_USER_MODEL "
        CMD+="--user-model-provider vertex_ai "
        CMD+="--policy-evaluator-model $EVAL_USER_MODEL "
        CMD+="--policy-evaluator-model-provider vertex_ai "
        CMD+="--task-split test "
        CMD+="--task-type $task_type "
        CMD+="--num-trials $EVAL_K "
        CMD+="--max-concurrency $EVAL_MAX_CONCURRENCY "
        CMD+="--temperature 0.0 "
        CMD+="--log-dir $log_dir "
        CMD+="2>&1 | tee -a $log_dir/${task_type}.log && "
    done

    CMD="${CMD% && }"

    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    tmux new-session -d -s "$SESSION_NAME" "$CMD"
    echo "  Started: $SESSION_NAME"
done

echo ""
echo "=== Consensus eval launched ==="
echo ""
echo "Tomorrow:"
echo "  1. Check eval: tmux attach -t rr-eval-rr-consensus"
echo "  2. Opus-compress median: cat $CONSENSUS_DIR/median.md"
echo "     → paste into Claude chat with compression prompt"
echo "     → save to $CONSENSUS_DIR/opus_median.md"
echo "  3. Run opus-median eval: ./scripts/run_rr_overnight.sh --phase3"
echo ""
echo "Done. Good night!"
