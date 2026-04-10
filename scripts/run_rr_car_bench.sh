#!/usr/bin/env bash
# =============================================================================
# RR CAR-bench: Launch 6 batches in parallel tmux sessions, then merge
#
# Each batch runs independently via run_rr_car_bench.py --batch <name>.
# After all complete, run the merge step to consolidate skillbooks.
#
# Usage:
#   chmod +x scripts/run_rr_car_bench.sh
#   ./scripts/run_rr_car_bench.sh              # Launch all 6 batches
#   ./scripts/run_rr_car_bench.sh --dry-run    # Print commands only
#   ./scripts/run_rr_car_bench.sh --merge-only # Skip batches, just merge
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
TMUX_PREFIX="rr-car"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$PROJECT_DIR/results/rr_car_bench_${TIMESTAMP}"

BATCHES=(
    "base_vehicle_climate"
    "base_nav_charging_prod"
    "halluc_vehicle_climate"
    "halluc_nav_charging_prod"
    "disambig_vehicle_climate"
    "disambig_nav_charging_prod"
)

DRY_RUN=false
MERGE_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --merge-only) MERGE_ONLY=true ;;
    esac
done

# --- Validate prerequisites ---
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Env file not found at $ENV_FILE"
    exit 1
fi

# Quick check that keys are present
source "$ENV_FILE"
if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    echo "ERROR: AWS_ACCESS_KEY_ID not set in $ENV_FILE"
    exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set in $ENV_FILE (needed for dedup embeddings)"
    exit 1
fi

echo "============================================================"
echo "RR CAR-bench Experiment"
echo "============================================================"
echo "Model:           $MODEL"
echo "Sub-agent:       $SUBAGENT_MODEL"
echo "Max iterations:  $MAX_ITERATIONS"
echo "Max LLM calls:   $MAX_LLM_CALLS"
echo "Dedup threshold: $DEDUP_THRESHOLD"
echo "Output:          $OUTPUT_DIR"
echo "Batches:         ${#BATCHES[@]}"
echo ""

if $MERGE_ONLY; then
    echo "=== MERGE ONLY MODE ==="
    echo ""
    # Find most recent output dir if not specified
    LATEST=$(ls -dt "$PROJECT_DIR/results/rr_car_bench_"* 2>/dev/null | head -1)
    if [[ -z "$LATEST" ]]; then
        echo "ERROR: No rr_car_bench_* directories found"
        exit 1
    fi
    OUTPUT_DIR="$LATEST"
    echo "Using: $OUTPUT_DIR"

    # Check all batch skillbooks exist
    MISSING=0
    for batch in "${BATCHES[@]}"; do
        if [[ ! -f "$OUTPUT_DIR/$batch/skillbook.json" ]]; then
            echo "  MISSING: $batch/skillbook.json"
            MISSING=$((MISSING + 1))
        else
            echo "  OK: $batch/skillbook.json"
        fi
    done
    if [[ $MISSING -gt 0 ]]; then
        echo ""
        echo "ERROR: $MISSING batch skillbooks missing. Wait for all batches to finish."
        exit 1
    fi

    echo ""
    echo "Running merge + dedup..."
    set -a && source "$ENV_FILE" && set +a
    uv run --project "$PROJECT_DIR" python "$SCRIPT_DIR/run_rr_car_bench_merge.py" \
        --output-dir "$OUTPUT_DIR" \
        --threshold "$DEDUP_THRESHOLD"
    exit 0
fi

# --- Create output dir and save config ---
mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/run_config.json" << EOFCONFIG
{
  "model": "$MODEL",
  "subagent_model": "$SUBAGENT_MODEL",
  "max_iterations": $MAX_ITERATIONS,
  "max_llm_calls": $MAX_LLM_CALLS,
  "dedup_threshold": $DEDUP_THRESHOLD,
  "prompt": "REFLECTOR_RECURSIVE_V5_PROMPT",
  "batches": [$(printf '"%s",' "${BATCHES[@]}" | sed 's/,$//')],
  "timestamp": "$TIMESTAMP",
  "merge_strategy": "concatenate + embedding dedup (Option A)"
}
EOFCONFIG

# --- Launch parallel tmux sessions ---
if $DRY_RUN; then
    echo "=== DRY RUN ==="
    echo ""
fi

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
    CMD+="--output-dir $OUTPUT_DIR "
    CMD+="--batch $batch "
    CMD+="2>&1 | tee $OUTPUT_DIR/${batch}.log"

    if $DRY_RUN; then
        echo "[$batch]"
        echo "  session: $SESSION_NAME"
        echo "  cmd: uv run ... --batch $batch"
        echo ""
    else
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        tmux new-session -d -s "$SESSION_NAME" "$CMD"
        echo "  Started: $SESSION_NAME"
    fi
done

echo ""
if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE — ${#BATCHES[@]} sessions would be launched ==="
else
    echo "=== All ${#BATCHES[@]} tmux sessions launched ==="
    echo ""
    echo "Monitor:"
    echo "  tmux ls                                       # List sessions"
    echo "  tmux attach -t ${TMUX_PREFIX}-base_vehicle_climate  # Attach"
    echo "  tail -f $OUTPUT_DIR/base_vehicle_climate.log  # Follow log"
    echo ""
    echo "When ALL batches complete, run merge:"
    echo "  ./scripts/run_rr_car_bench.sh --merge-only"
    echo ""
    echo "Or merge manually:"
    echo "  uv run python scripts/run_rr_car_bench_merge.py --output-dir $OUTPUT_DIR"
fi
