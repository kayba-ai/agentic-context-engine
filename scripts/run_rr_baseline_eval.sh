#!/usr/bin/env bash
# =============================================================================
# CAR-bench Baseline Evaluation (vertex_ai provider)
#
# Runs a no-skillbook baseline using the SAME setup as the RR per-run evals:
#   - Agent: Haiku 4.5 (Bedrock)
#   - User sim + policy eval: Gemini 2.5 Flash (vertex_ai)
#   - k=4 trials, temperature 0.0
#
# Purpose: Provide a directly comparable baseline for all RR eval results.
#
# Usage:
#   chmod +x scripts/run_rr_baseline_eval.sh
#   ./scripts/run_rr_baseline_eval.sh              # Launch
#   ./scripts/run_rr_baseline_eval.sh --dry-run    # Print commands only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration (identical to run_rr_per_run_eval.sh) ---
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
CAR_BENCH_DIR="$PROJECT_DIR/benchmarks/car-bench/car-bench"
WIKI_ORIG="$CAR_BENCH_DIR/car_bench/envs/car_voice_assistant/wiki.md"
OUTPUT_DIR="$PROJECT_DIR/results/rr_car_bench_eval/baseline"
TMUX_SESSION_PREFIX="rr-baseline"
MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_PROVIDER="bedrock"
USER_MODEL="gemini-2.5-flash"
K=4
MAX_CONCURRENCY=5
TASK_TYPES=("base" "hallucination" "disambiguation")

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
    echo ""
fi

# --- Validate prerequisites ---
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Env file not found at $ENV_FILE"; exit 1
fi

if [[ ! -f "$WIKI_ORIG" ]]; then
    echo "ERROR: wiki.md not found at $WIKI_ORIG"; exit 1
fi

if ! uv run --project "$PROJECT_DIR" python -c "import car_bench" 2>/dev/null; then
    echo "ERROR: car_bench not importable"; exit 1
fi

echo "============================================================"
echo "RR Baseline Evaluation (vertex_ai provider)"
echo "============================================================"
echo "Agent model:    $MODEL ($MODEL_PROVIDER)"
echo "User model:     $USER_MODEL (vertex_ai)"
echo "K=$K, concurrency=$MAX_CONCURRENCY"
echo "Task types:     ${TASK_TYPES[*]}"
echo "Output:         $OUTPUT_DIR"
echo "Wiki:           $WIKI_ORIG (no skillbook injection)"
echo ""

mkdir -p "$OUTPUT_DIR"

# --- Launch one tmux session per task type (maximally parallel) ---
SESSION_COUNT=0

for task_type in "${TASK_TYPES[@]}"; do
    SESSION_NAME="${TMUX_SESSION_PREFIX}-${task_type}"

    CMD="set -a && source $ENV_FILE && set +a && "
    CMD+="unset HF_DATASETS_CACHE HF_HUB_CACHE BENCHMARK_CACHE_DIR HUGGINGFACE_API_KEY APPWORLD_ROOT && "
    CMD+="export PYTHONUNBUFFERED=1 && "
    CMD+="export CAR_BENCH_WIKI_PATH=$WIKI_ORIG && "
    CMD+="mkdir -p $OUTPUT_DIR && "
    CMD+="echo '=== [baseline] Running $task_type ===' && "
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
    CMD+="--log-dir $OUTPUT_DIR "
    CMD+="2>&1 | tee $OUTPUT_DIR/${task_type}.log"

    if $DRY_RUN; then
        echo "[baseline / $task_type]"
        echo "  session: $SESSION_NAME"
        echo "  log:     $OUTPUT_DIR/${task_type}.log"
        echo ""
    else
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        tmux new-session -d -s "$SESSION_NAME" "$CMD"
        echo "  Started: $SESSION_NAME"
    fi
    SESSION_COUNT=$((SESSION_COUNT + 1))
done

echo ""
if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE ($SESSION_COUNT sessions would launch) ==="
else
    echo "=== $SESSION_COUNT baseline sessions launched ==="
    echo ""
    echo "Monitor:"
    echo "  tmux ls | grep $TMUX_SESSION_PREFIX"
    echo "  tail -f $OUTPUT_DIR/base.log"
    echo ""
    echo "Expected: ~20-40 min per task type."
fi
