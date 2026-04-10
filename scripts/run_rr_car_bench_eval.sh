#!/usr/bin/env bash
# =============================================================================
# CAR-bench Evaluation: RR-generated skillbook vs baseline
#
# Runs RR-merged config only (baseline already exists in results/car_bench_eval/).
# Launches 1 tmux session running 3 task types sequentially.
#
# All runs use Haiku as test agent, k=4, Gemini 2.5 Flash as user sim
# Tests all 3 task types: base, hallucination, disambiguation
#
# Usage:
#   chmod +x scripts/run_rr_car_bench_eval.sh
#   ./scripts/run_rr_car_bench_eval.sh              # Launch
#   ./scripts/run_rr_car_bench_eval.sh --dry-run    # Print commands only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
CAR_BENCH_DIR="$PROJECT_DIR/benchmarks/car-bench/car-bench"
WIKI_ORIG="$CAR_BENCH_DIR/car_bench/envs/car_voice_assistant/wiki.md"
PREPARED_WIKIS_DIR="$PROJECT_DIR/benchmarks/car-bench/prepared_wikis"
OUTPUT_DIR="$PROJECT_DIR/results/rr_car_bench_eval"
TMUX_SESSION_PREFIX="rr-eval"
MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_PROVIDER="bedrock"
USER_MODEL="gemini-2.5-flash"
K=4
MAX_CONCURRENCY=5
TASK_TYPES=("base" "hallucination" "disambiguation")

# RR skillbook
RR_SKILLBOOK="$PROJECT_DIR/results/rr_car_bench_20260309_162529/skills_merged.md"

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

if [[ ! -f "$RR_SKILLBOOK" ]]; then
    echo "ERROR: RR skillbook not found at $RR_SKILLBOOK"; exit 1
fi

# --- Define configs ---
# Baseline already exists at results/car_bench_eval/baseline/ — skip rerunning.
declare -A CONFIGS

# RR merged skillbook only
CONFIGS["rr-merged"]="$RR_SKILLBOOK"

echo "============================================================"
echo "RR CAR-bench Downstream Evaluation"
echo "============================================================"
echo "Agent model:    $MODEL ($MODEL_PROVIDER)"
echo "User model:     $USER_MODEL"
echo "K=$K, concurrency=$MAX_CONCURRENCY"
echo "Task types:     ${TASK_TYPES[*]}"
echo "Configs:        ${#CONFIGS[@]}"
echo "Output:         $OUTPUT_DIR"
echo ""

# --- Prepare wiki variants ---
mkdir -p "$PREPARED_WIKIS_DIR"
mkdir -p "$OUTPUT_DIR"

for label in "${!CONFIGS[@]}"; do
    wiki_out="$PREPARED_WIKIS_DIR/wiki_rr_${label}.md"
    path="${CONFIGS[$label]}"

    if [[ -z "$path" ]]; then
        cp "$WIKI_ORIG" "$wiki_out"
    else
        cat "$WIKI_ORIG" > "$wiki_out"
        echo "" >> "$wiki_out"
        echo "" >> "$wiki_out"
        echo "## ACE Learned Skills (RR-generated)" >> "$wiki_out"
        echo "" >> "$wiki_out"
        cat "$path" >> "$wiki_out"
    fi
    echo "  Wiki: wiki_rr_${label}.md ($(wc -c < "$wiki_out") bytes)"
done
echo ""

# --- Launch tmux sessions ---
SORTED_LABELS=($(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort))

for label in "${SORTED_LABELS[@]}"; do
    wiki_path="$PREPARED_WIKIS_DIR/wiki_rr_${label}.md"
    log_dir="$OUTPUT_DIR/$label"

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

    CMD="${CMD% && }"
    SESSION_NAME="${TMUX_SESSION_PREFIX}-${label}"

    if $DRY_RUN; then
        echo "[$label]"
        echo "  session: $SESSION_NAME"
        echo "  wiki:    $wiki_path"
        echo "  log_dir: $log_dir"
        echo "  tasks:   ${TASK_TYPES[*]}"
        echo ""
    else
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
        tmux new-session -d -s "$SESSION_NAME" "$CMD"
        echo "  Started: $SESSION_NAME"
    fi
done

echo ""
if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE ==="
else
    echo "=== ${#CONFIGS[@]} eval sessions launched ==="
    echo ""
    echo "Monitor:"
    echo "  tmux ls"
    echo "  tmux attach -t ${TMUX_SESSION_PREFIX}-rr-merged"
    echo "  tail -f $OUTPUT_DIR/rr-merged/base.log"
    echo ""
    echo "Each config runs 3 task types × ${K} trials sequentially."
    echo "Expected: ~1-2 hours per config."
fi
