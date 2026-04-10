#!/usr/bin/env bash
# =============================================================================
# RR CAR-bench Overnight Pipeline
#
# Phase 1: Launch 4 RR training runs (24 parallel tmux sessions)
# Phase 2: Poll for completion, then merge each run
# Phase 3: Build consensus from all 5 runs (including existing run)
# Phase 4: Launch evaluation on consensus + opus-median configs
#
# All 4 runs launch in parallel (6 batches each = 24 sessions).
# Each run gets its own output directory and tmux session prefix.
#
# Usage:
#   chmod +x scripts/run_rr_overnight.sh
#   ./scripts/run_rr_overnight.sh              # Launch full pipeline
#   ./scripts/run_rr_overnight.sh --dry-run    # Print plan only
#   ./scripts/run_rr_overnight.sh --phase2     # Skip to merge + consensus
#   ./scripts/run_rr_overnight.sh --phase3     # Skip to eval only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
MODEL="bedrock/us.anthropic.claude-sonnet-4-6"
SUBAGENT_MODEL="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
MAX_ITERATIONS=60
MAX_LLM_CALLS=60
DEDUP_THRESHOLD=0.7
N_NEW_RUNS=4

# Existing run (run 1 of 5)
EXISTING_RUN="$PROJECT_DIR/results/rr_car_bench_20260309_162529"

# Eval config
EVAL_MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
EVAL_MODEL_PROVIDER="bedrock"
EVAL_USER_MODEL="gemini-2.5-flash"
EVAL_K=4
EVAL_MAX_CONCURRENCY=5

BATCHES=(
    "base_vehicle_climate"
    "base_nav_charging_prod"
    "halluc_vehicle_climate"
    "halluc_nav_charging_prod"
    "disambig_vehicle_climate"
    "disambig_nav_charging_prod"
)

TASK_TYPES=("base" "hallucination" "disambiguation")

DRY_RUN=false
SKIP_TO_PHASE2=false
SKIP_TO_PHASE3=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --phase2) SKIP_TO_PHASE2=true ;;
        --phase3) SKIP_TO_PHASE3=true ;;
    esac
done

# --- Validate ---
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Env file not found at $ENV_FILE"; exit 1
fi
source "$ENV_FILE"
if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    echo "ERROR: AWS_ACCESS_KEY_ID not set"; exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set (needed for embeddings)"; exit 1
fi
if [[ ! -f "$EXISTING_RUN/skillbook_merged.json" ]]; then
    echo "ERROR: Existing run not found at $EXISTING_RUN"; exit 1
fi

# Generate output dirs for 4 new runs
declare -a RUN_DIRS
for i in $(seq 1 $N_NEW_RUNS); do
    TS=$(date +"%Y%m%d_%H%M%S")
    # Add index suffix to avoid timestamp collisions
    RUN_DIRS+=("$PROJECT_DIR/results/rr_car_bench_${TS}_run${i}")
    sleep 1  # ensure unique timestamps
done

# All 5 run dirs (for consensus)
ALL_RUN_DIRS=("$EXISTING_RUN" "${RUN_DIRS[@]}")
CONSENSUS_DIR="$PROJECT_DIR/results/rr_car_bench_consensus"

echo "============================================================"
echo "RR CAR-bench Overnight Pipeline"
echo "============================================================"
echo "Model:           $MODEL"
echo "Sub-agent:       $SUBAGENT_MODEL"
echo "Max iterations:  $MAX_ITERATIONS"
echo "Max LLM calls:   $MAX_LLM_CALLS"
echo "Existing run:    $(basename $EXISTING_RUN)"
echo "New runs:        $N_NEW_RUNS (24 parallel tmux sessions)"
echo "Consensus dir:   $CONSENSUS_DIR"
echo ""
echo "New run dirs:"
for d in "${RUN_DIRS[@]}"; do
    echo "  $(basename $d)"
done
echo ""

# =====================================================================
# PHASE 1: Launch 4 RR training runs in parallel
# =====================================================================
if ! $SKIP_TO_PHASE2 && ! $SKIP_TO_PHASE3; then
    echo "=== PHASE 1: Launching $N_NEW_RUNS RR training runs ==="
    echo ""

    for run_i in $(seq 0 $((N_NEW_RUNS - 1))); do
        RUN_DIR="${RUN_DIRS[$run_i]}"
        RUN_NUM=$((run_i + 2))  # runs 2-5 (run 1 is existing)
        TMUX_PREFIX="rr-run${RUN_NUM}"

        if $DRY_RUN; then
            echo "  [Run $RUN_NUM] → $TMUX_PREFIX-* (6 sessions)"
            echo "    Output: $(basename $RUN_DIR)"
            continue
        fi

        mkdir -p "$RUN_DIR"

        # Save run config
        cat > "$RUN_DIR/run_config.json" << EOFCONFIG
{
  "model": "$MODEL",
  "subagent_model": "$SUBAGENT_MODEL",
  "max_iterations": $MAX_ITERATIONS,
  "max_llm_calls": $MAX_LLM_CALLS,
  "dedup_threshold": $DEDUP_THRESHOLD,
  "prompt": "REFLECTOR_RECURSIVE_V5_PROMPT",
  "run_number": $RUN_NUM,
  "batches": ["${BATCHES[0]}", "${BATCHES[1]}", "${BATCHES[2]}", "${BATCHES[3]}", "${BATCHES[4]}", "${BATCHES[5]}"]
}
EOFCONFIG

        # Launch 6 batch sessions for this run
        for batch in "${BATCHES[@]}"; do
            SESSION_NAME="${TMUX_PREFIX}-${batch}"

            CMD="set -a && source $ENV_FILE && set +a && "
            CMD+="export PYTHONUNBUFFERED=1 && "
            CMD+="uv run --project $PROJECT_DIR python $SCRIPT_DIR/run_rr_car_bench.py "
            CMD+="--model $MODEL "
            CMD+="--subagent-model $SUBAGENT_MODEL "
            CMD+="--max-iterations $MAX_ITERATIONS "
            CMD+="--max-llm-calls $MAX_LLM_CALLS "
            CMD+="--threshold $DEDUP_THRESHOLD "
            CMD+="--output-dir $RUN_DIR "
            CMD+="--batch $batch "
            CMD+="2>&1 | tee $RUN_DIR/${batch}.log"

            tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
            tmux new-session -d -s "$SESSION_NAME" "$CMD"
        done
        echo "  Started: $TMUX_PREFIX-* (6 sessions for run $RUN_NUM)"
    done

    if $DRY_RUN; then
        echo ""
        echo "=== DRY RUN — would launch $((N_NEW_RUNS * 6)) tmux sessions ==="
        exit 0
    fi

    echo ""
    echo "=== $((N_NEW_RUNS * 6)) tmux sessions launched ==="
    echo ""

    # --- Poll for completion ---
    echo "Waiting for all batches to complete..."
    echo "  (checking every 60s for skillbook.json in each batch dir)"
    echo ""

    while true; do
        ALL_DONE=true
        COMPLETED=0
        TOTAL=$((N_NEW_RUNS * 6))

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
fi

# =====================================================================
# PHASE 2: Merge each run + build consensus
# =====================================================================
if ! $SKIP_TO_PHASE3; then
    echo ""
    echo "=== PHASE 2: Merging runs + building consensus ==="
    echo ""

    # Merge each new run
    for RUN_DIR in "${RUN_DIRS[@]}"; do
        if [[ -f "$RUN_DIR/skillbook_merged.json" ]]; then
            echo "  Already merged: $(basename $RUN_DIR)"
            continue
        fi

        echo "  Merging: $(basename $RUN_DIR)"
        set -a && source "$ENV_FILE" && set +a
        uv run --project "$PROJECT_DIR" python "$SCRIPT_DIR/run_rr_car_bench_merge.py" \
            --output-dir "$RUN_DIR" \
            --threshold "$DEDUP_THRESHOLD"
    done

    # Build consensus from all 5 runs
    echo ""
    echo "Building consensus from ${#ALL_RUN_DIRS[@]} runs..."
    set -a && source "$ENV_FILE" && set +a
    uv run --project "$PROJECT_DIR" python "$SCRIPT_DIR/run_rr_car_bench_consensus.py" \
        --run-dirs "${ALL_RUN_DIRS[@]}" \
        --output-dir "$CONSENSUS_DIR" \
        --threshold 3 \
        --similarity 0.7

    echo ""
    echo "=== Phase 2 complete ==="
    echo ""
    echo "Proceeding to Phase 3 (consensus eval)..."
    echo "Opus compression of median is a MANUAL step — do it tomorrow:"
    echo "  1. cat $CONSENSUS_DIR/median.md"
    echo "  2. Paste into Claude chat (fresh context) with compression prompt"
    echo "  3. Save to: $CONSENSUS_DIR/opus_median.md"
    echo "  4. Run: ./scripts/run_rr_overnight.sh --phase3  (for opus-median eval)"
fi

# =====================================================================
# PHASE 3: Evaluation
# =====================================================================
# Auto-run after Phase 2, or manually via --phase3
if $SKIP_TO_PHASE3 || ! $SKIP_TO_PHASE2; then
    echo ""
    echo "=== PHASE 3: Launching evaluation ==="
    echo ""

    CAR_BENCH_DIR="$PROJECT_DIR/benchmarks/car-bench/car-bench"
    WIKI_ORIG="$CAR_BENCH_DIR/car_bench/envs/car_voice_assistant/wiki.md"
    PREPARED_WIKIS_DIR="$PROJECT_DIR/benchmarks/car-bench/prepared_wikis"
    EVAL_OUTPUT_DIR="$PROJECT_DIR/results/rr_car_bench_eval"

    # Define eval configs: label -> skillbook path
    declare -A EVAL_CONFIGS
    EVAL_CONFIGS["rr-consensus"]="$CONSENSUS_DIR/consensus.md"

    # Only add opus-median if it exists
    if [[ -f "$CONSENSUS_DIR/opus_median.md" ]]; then
        EVAL_CONFIGS["rr-opus-median"]="$CONSENSUS_DIR/opus_median.md"
    else
        echo "WARNING: $CONSENSUS_DIR/opus_median.md not found — skipping opus-median eval"
        echo "  Run Opus compression first, then re-run --phase3"
    fi

    if [[ ${#EVAL_CONFIGS[@]} -eq 0 ]]; then
        echo "ERROR: No eval configs available"; exit 1
    fi

    echo "Eval configs: ${!EVAL_CONFIGS[@]}"
    echo ""

    # Prepare wikis
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

    # Launch eval sessions (one tmux per config, sequential task types within)
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
    echo "=== ${#EVAL_CONFIGS[@]} eval sessions launched ==="
    echo ""
    echo "Monitor:"
    echo "  tmux ls"
    for label in "${!EVAL_CONFIGS[@]}"; do
        echo "  tmux attach -t rr-eval-${label}"
    done
    echo ""
    echo "Each config runs 3 task types × ${EVAL_K} trials sequentially."
    echo "Expected: ~1-2 hours per config."
fi
