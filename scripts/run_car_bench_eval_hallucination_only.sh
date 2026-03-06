#!/usr/bin/env bash
# =============================================================================
# CAR-bench Hallucination-Only Evaluation: 13 configs in parallel tmux sessions
#
# Tests the task-separated skillbook hypothesis: do skillbooks trained on
# hallucination traces only outperform mixed-trained skillbooks on
# hallucination test tasks?
#
# Evaluates 3 skillbook types x 2 budgets x 2 source models + 1 baseline
# All runs use Haiku as the test agent, k=4, Gemini 2.5 Flash as user sim
#
# Skillbook types:
#   median          - Hallucination-48 median run markdown
#   consensus       - Hallucination-48 consensus (t>=3) markdown
#   opus-median     - Opus-compressed hallucination-48 median markdown
#
# Prerequisites:
#   1. Run variance experiments on hallucination-only traces (see RUNBOOK)
#   2. Run: uv run python scripts/prepare_hallucination_only_eval.py
#   3. Run Opus compression on median skillbooks (subagent step)
#   4. Prepare opus-median wiki variants
#   5. Verify wiki files in benchmarks/car-bench/prepared_wikis/
#
# Usage:
#   chmod +x scripts/run_car_bench_eval_hallucination_only.sh
#   ./scripts/run_car_bench_eval_hallucination_only.sh          # Launch all
#   ./scripts/run_car_bench_eval_hallucination_only.sh --dry-run  # Print only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
CAR_BENCH_DIR="$PROJECT_DIR/benchmarks/car-bench/car-bench"
PREPARED_WIKIS_DIR="$PROJECT_DIR/benchmarks/car-bench/prepared_wikis"
OUTPUT_DIR="$PROJECT_DIR/results/car_bench_eval_hallucination_only"
TMUX_SESSION_PREFIX="car-halluc48"
MODEL="us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_PROVIDER="bedrock"
USER_MODEL="gemini-2.5-flash"
K=4
MAX_CONCURRENCY=5
# Hallucination tasks only (the whole point of this experiment)
TASK_TYPES=("hallucination")
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

if ! uv run --project "$PROJECT_DIR" python -c "import car_bench" 2>/dev/null; then
    echo "ERROR: car_bench not importable. Install with: uv pip install -e $CAR_BENCH_DIR"
    exit 1
fi

# --- Define configurations ---
# Format: CONFIGS["label"]="wiki_file_name"
declare -A CONFIGS

# Baseline (no skillbook injection)
CONFIGS["baseline"]="wiki_baseline.md"

# Haiku source: hallucination-48 variants
CONFIGS["haiku-nobudget-halluc48-median"]="wiki_haiku-nobudget-halluc48-median.md"
CONFIGS["haiku-nobudget-halluc48-consensus"]="wiki_haiku-nobudget-halluc48-consensus.md"
CONFIGS["haiku-nobudget-halluc48-opus-median"]="wiki_haiku-nobudget-halluc48-opus-median.md"
CONFIGS["haiku-500-halluc48-median"]="wiki_haiku-500-halluc48-median.md"
CONFIGS["haiku-500-halluc48-consensus"]="wiki_haiku-500-halluc48-consensus.md"
CONFIGS["haiku-500-halluc48-opus-median"]="wiki_haiku-500-halluc48-opus-median.md"

# Sonnet source: hallucination-48 variants
CONFIGS["sonnet-nobudget-halluc48-median"]="wiki_sonnet-nobudget-halluc48-median.md"
CONFIGS["sonnet-nobudget-halluc48-consensus"]="wiki_sonnet-nobudget-halluc48-consensus.md"
CONFIGS["sonnet-nobudget-halluc48-opus-median"]="wiki_sonnet-nobudget-halluc48-opus-median.md"
CONFIGS["sonnet-500-halluc48-median"]="wiki_sonnet-500-halluc48-median.md"
CONFIGS["sonnet-500-halluc48-consensus"]="wiki_sonnet-500-halluc48-consensus.md"
CONFIGS["sonnet-500-halluc48-opus-median"]="wiki_sonnet-500-halluc48-opus-median.md"

# --- Validate all wiki files exist ---
echo "Validating wiki files..."
MISSING=0
for label in "${!CONFIGS[@]}"; do
    wiki_file="${CONFIGS[$label]}"
    wiki_path="$PREPARED_WIKIS_DIR/$wiki_file"
    if [[ ! -f "$wiki_path" ]]; then
        echo "  MISSING: $label -> $wiki_path"
        MISSING=$((MISSING + 1))
    fi
done
if [[ $MISSING -gt 0 ]]; then
    echo "ERROR: $MISSING wiki files missing. Complete all preparation steps first."
    exit 1
fi
echo "  All ${#CONFIGS[@]} configurations validated."
echo ""

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

# --- Build and launch tmux sessions ---
echo "Launching ${#CONFIGS[@]} evaluation configs..."
echo "  Output:         $OUTPUT_DIR"
echo "  Agent model:    $MODEL ($MODEL_PROVIDER)"
echo "  User model:     $USER_MODEL (vertex_ai)"
echo "  K=$K, concurrency=$MAX_CONCURRENCY"
echo "  Task types:     ${TASK_TYPES[*]} (hallucination only)"
echo ""

# Sort labels for consistent ordering
SORTED_LABELS=($(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort))

for label in "${SORTED_LABELS[@]}"; do
    wiki_file="${CONFIGS[$label]}"
    wiki_path="$PREPARED_WIKIS_DIR/$wiki_file"
    log_dir="$OUTPUT_DIR/$label"

    # Build command
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
    echo "  tmux ls                                              # List all sessions"
    echo "  tmux attach -t ${TMUX_SESSION_PREFIX}-baseline       # Attach to a session"
    echo "  tail -f $OUTPUT_DIR/baseline/hallucination.log       # Follow log"
    echo ""
    echo "Check results when done:"
    echo "  ls $OUTPUT_DIR/*/hallucination_test/*.json"
    echo "  for d in $OUTPUT_DIR/*/; do echo \$(basename \$d); cat \$d/*.log | grep -E 'reward|pass'; done"
fi
