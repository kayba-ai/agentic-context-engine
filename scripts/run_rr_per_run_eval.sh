#!/usr/bin/env bash
# =============================================================================
# CAR-bench Evaluation: Per-run RR skillbook downstream eval
#
# Evaluates each of the 5 individual RR runs separately to measure
# downstream performance variance across RR runs.
#
# Maximally parallel: 1 tmux session per (run × task_type) = 15 sessions.
#
# All runs use Haiku as test agent, k=4, Gemini 2.5 Flash as user sim.
# Tests all 3 task types: base, hallucination, disambiguation.
#
# Usage:
#   chmod +x scripts/run_rr_per_run_eval.sh
#   ./scripts/run_rr_per_run_eval.sh              # Launch all 15 sessions
#   ./scripts/run_rr_per_run_eval.sh --dry-run    # Print commands only
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
TMUX_SESSION_PREFIX="rr-perrun"
MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_PROVIDER="bedrock"
USER_MODEL="gemini-2.5-flash"
K=4
MAX_CONCURRENCY=5
TASK_TYPES=("base" "hallucination" "disambiguation")

# --- The 5 individual RR runs ---
declare -A RUNS
RUNS["rr-run1"]="$PROJECT_DIR/results/rr_car_bench_20260309_162529/skills_merged.md"
RUNS["rr-run2"]="$PROJECT_DIR/results/rr_car_bench_20260309_232848_run1/skills_merged.md"
RUNS["rr-run3"]="$PROJECT_DIR/results/rr_car_bench_20260309_232849_run2/skills_merged.md"
RUNS["rr-run4"]="$PROJECT_DIR/results/rr_car_bench_20260309_232850_run3/skills_merged.md"
RUNS["rr-run5"]="$PROJECT_DIR/results/rr_car_bench_20260309_232851_run4/skills_merged.md"

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

for label in "${!RUNS[@]}"; do
    if [[ ! -f "${RUNS[$label]}" ]]; then
        echo "ERROR: Skillbook not found for $label: ${RUNS[$label]}"; exit 1
    fi
done

echo "============================================================"
echo "RR Per-Run Downstream Evaluation"
echo "============================================================"
echo "Agent model:    $MODEL ($MODEL_PROVIDER)"
echo "User model:     $USER_MODEL"
echo "K=$K, concurrency=$MAX_CONCURRENCY"
echo "Task types:     ${TASK_TYPES[*]}"
echo "Runs:           ${#RUNS[@]}"
echo "Sessions:       $((${#RUNS[@]} * ${#TASK_TYPES[@]})) (maximally parallel)"
echo "Output:         $OUTPUT_DIR"
echo ""

# --- Prepare wiki variants ---
mkdir -p "$PREPARED_WIKIS_DIR"
mkdir -p "$OUTPUT_DIR"

for label in "${!RUNS[@]}"; do
    wiki_out="$PREPARED_WIKIS_DIR/wiki_${label}.md"
    path="${RUNS[$label]}"

    cat "$WIKI_ORIG" > "$wiki_out"
    echo "" >> "$wiki_out"
    echo "" >> "$wiki_out"
    echo "## ACE Learned Skills (RR-generated)" >> "$wiki_out"
    echo "" >> "$wiki_out"
    cat "$path" >> "$wiki_out"

    echo "  Wiki: wiki_${label}.md ($(wc -c < "$wiki_out") bytes)"
done
echo ""

# --- Launch one tmux session per (run × task_type) ---
SORTED_LABELS=($(echo "${!RUNS[@]}" | tr ' ' '\n' | sort))
SESSION_COUNT=0

for label in "${SORTED_LABELS[@]}"; do
    wiki_path="$PREPARED_WIKIS_DIR/wiki_${label}.md"
    log_dir="$OUTPUT_DIR/$label"

    for task_type in "${TASK_TYPES[@]}"; do
        SESSION_NAME="${TMUX_SESSION_PREFIX}-${label}-${task_type}"

        CMD="set -a && source $ENV_FILE && set +a && "
        CMD+="unset HF_DATASETS_CACHE HF_HUB_CACHE BENCHMARK_CACHE_DIR HUGGINGFACE_API_KEY APPWORLD_ROOT && "
        CMD+="export PYTHONUNBUFFERED=1 && "
        CMD+="export CAR_BENCH_WIKI_PATH=$wiki_path && "
        CMD+="mkdir -p $log_dir && "
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
        CMD+="2>&1 | tee $log_dir/${task_type}.log"

        if $DRY_RUN; then
            echo "[$label / $task_type]"
            echo "  session: $SESSION_NAME"
            echo "  wiki:    $wiki_path"
            echo "  log_dir: $log_dir"
            echo ""
        else
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
            tmux new-session -d -s "$SESSION_NAME" "$CMD"
            echo "  Started: $SESSION_NAME"
        fi
        SESSION_COUNT=$((SESSION_COUNT + 1))
    done
done

echo ""
if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE ($SESSION_COUNT sessions would launch) ==="
else
    echo "=== $SESSION_COUNT tmux sessions launched ==="
    echo ""
    echo "Monitor:"
    echo "  tmux ls | grep $TMUX_SESSION_PREFIX"
    echo "  tail -f $OUTPUT_DIR/rr-run1/base.log"
    echo ""
    echo "Each session runs 1 task type × ${K} trials."
    echo "Expected: ~20-40 min per session."
fi
