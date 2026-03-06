#!/usr/bin/env bash
# =============================================================================
# TAU-bench pass^4 Evaluation: 17 runs in parallel tmux sessions
#
# Evaluates 4 skillbook types x 2 budgets x 2 source models + 1 baseline
# All runs use Haiku as the test agent, k=4, airline domain
#
# Skillbook types:
#   consensus       - Raw consensus JSON (--skillbook)
#   opus-consensus  - Opus-compressed consensus markdown (--playbook)
#   median          - Raw median run JSON (--skillbook)
#   opus-median     - Opus-compressed median run markdown (--playbook)
#
# Usage:
#   chmod +x scripts/run_pass4_eval.sh
#   ./scripts/run_pass4_eval.sh          # Launch all 17 runs
#   ./scripts/run_pass4_eval.sh --dry-run  # Print commands without executing
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
ENV_FILE="/scratch/tzerweck/other/Kayba/.env"
TAU2_DATA_DIR="/tmp/tau2-bench/data"
OUTPUT_DIR="$PROJECT_DIR/tau_benchmark_results/pass4_eval"
TMUX_SESSION_PREFIX="tau-eval"
CONFIG="haiku"
MODEL="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
DOMAIN="airline"
K=4
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

if [[ ! -d "$TAU2_DATA_DIR" ]]; then
    echo "ERROR: TAU2 data directory not found at $TAU2_DATA_DIR"
    echo "  Clone it: git clone --depth 1 https://github.com/sierra-research/tau2-bench.git /tmp/tau2-bench"
    exit 1
fi

# --- Skillbook paths ---
HAIKU_DIR="$PROJECT_DIR/results/variance_experiment_haiku_4.5"
SONNET_DIR="$PROJECT_DIR/results/variance_experiment_sonnet_4.6"

# Median run numbers (from analysis_results.json):
#   Haiku:  no-budget=run_5, budget-500=run_4
#   Sonnet: no-budget=run_5, budget-500=run_4

declare -A RUNS
# Format: RUNS["label"]="flag|path"
#   flag = --skillbook (JSON) or --playbook (markdown)

# --- Baseline ---
RUNS["baseline"]="--skip-ace|"

# --- Haiku source, consensus (JSON) ---
RUNS["haiku-nobudget-consensus"]="--skillbook|$HAIKU_DIR/consensus/consensus_no-budget.json"
RUNS["haiku-500-consensus"]="--skillbook|$HAIKU_DIR/consensus/consensus_budget-500.json"

# --- Haiku source, opus-compressed consensus (markdown) ---
RUNS["haiku-nobudget-opus-consensus"]="--playbook|$HAIKU_DIR/opus_compressed/opus_consensus_no-budget.md"
RUNS["haiku-500-opus-consensus"]="--playbook|$HAIKU_DIR/opus_compressed/opus_consensus_budget-500.md"

# --- Haiku source, median run (raw JSON) ---
RUNS["haiku-nobudget-median"]="--skillbook|$HAIKU_DIR/no-budget/run_5/skillbook_24.json"
RUNS["haiku-500-median"]="--skillbook|$HAIKU_DIR/budget-500/run_4/skillbook_24.json"

# --- Haiku source, opus-compressed median run (markdown) ---
RUNS["haiku-nobudget-opus-median"]="--playbook|$HAIKU_DIR/opus_compressed/opus_no-budget_run_5.md"
RUNS["haiku-500-opus-median"]="--playbook|$HAIKU_DIR/opus_compressed/opus_budget-500_run_4.md"

# --- Sonnet source, consensus (JSON) ---
RUNS["sonnet-nobudget-consensus"]="--skillbook|$SONNET_DIR/consensus/consensus_no-budget.json"
RUNS["sonnet-500-consensus"]="--skillbook|$SONNET_DIR/consensus/consensus_budget-500.json"

# --- Sonnet source, opus-compressed consensus (markdown) ---
RUNS["sonnet-nobudget-opus-consensus"]="--playbook|$SONNET_DIR/opus_compressed/opus_consensus_no-budget.md"
RUNS["sonnet-500-opus-consensus"]="--playbook|$SONNET_DIR/opus_compressed/opus_consensus_budget-500.md"

# --- Sonnet source, median run (raw JSON) ---
RUNS["sonnet-nobudget-median"]="--skillbook|$SONNET_DIR/no-budget/run_5/skillbook_24.json"
RUNS["sonnet-500-median"]="--skillbook|$SONNET_DIR/budget-500/run_4/skillbook_24.json"

# --- Sonnet source, opus-compressed median run (markdown) ---
RUNS["sonnet-nobudget-opus-median"]="--playbook|$SONNET_DIR/opus_compressed/opus_no-budget_run_5.md"
RUNS["sonnet-500-opus-median"]="--playbook|$SONNET_DIR/opus_compressed/opus_budget-500_run_4.md"

# --- Validate all skillbook files exist ---
echo "Validating skillbook files..."
MISSING=0
for label in "${!RUNS[@]}"; do
    IFS='|' read -r flag path <<< "${RUNS[$label]}"
    if [[ "$flag" == "--skip-ace" ]]; then
        continue
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
echo "  All ${#RUNS[@]} configurations validated."
echo ""

# --- Create output directory ---
mkdir -p "$OUTPUT_DIR"

# --- Build and launch tmux sessions ---
echo "Launching ${#RUNS[@]} evaluation runs..."
echo "  Output: $OUTPUT_DIR"
echo "  Agent model: $MODEL"
echo "  Config: $CONFIG (GPT-4.1 user sim)"
echo "  K=$K, domain=$DOMAIN"
echo ""

# Sort labels for consistent ordering
SORTED_LABELS=($(echo "${!RUNS[@]}" | tr ' ' '\n' | sort))

for label in "${SORTED_LABELS[@]}"; do
    IFS='|' read -r flag path <<< "${RUNS[$label]}"

    # Build the command
    CMD="set -a && source $ENV_FILE && set +a && "
    CMD+="export TAU2_DATA_DIR=$TAU2_DATA_DIR && "
    CMD+="cd $PROJECT_DIR && "
    CMD+="uv run python scripts/run_tau_benchmark.py "
    CMD+="--config $CONFIG "
    CMD+="--model $MODEL "
    CMD+="--domain $DOMAIN "
    CMD+="-k $K "
    CMD+="--label $label "
    CMD+="--save-detailed "
    CMD+="--output $OUTPUT_DIR "

    if [[ "$flag" == "--skip-ace" ]]; then
        CMD+="--skip-ace "
    else
        CMD+="$flag $path "
    fi

    # Log file for this run
    CMD+="2>&1 | tee $OUTPUT_DIR/${label}.log"

    SESSION_NAME="${TMUX_SESSION_PREFIX}-${label}"

    if $DRY_RUN; then
        echo "[$label]"
        echo "  tmux: $SESSION_NAME"
        echo "  cmd:  uv run python scripts/run_tau_benchmark.py --config $CONFIG $flag ${path:-} --label $label -k $K"
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
    echo "=== DRY RUN COMPLETE — ${#RUNS[@]} runs would be launched ==="
else
    echo "=== All ${#RUNS[@]} tmux sessions launched ==="
    echo ""
    echo "Monitor progress:"
    echo "  tmux ls                           # List all sessions"
    echo "  tmux attach -t ${TMUX_SESSION_PREFIX}-baseline  # Attach to a session"
    echo "  tail -f $OUTPUT_DIR/baseline.log  # Follow log"
    echo ""
    echo "Check results when done:"
    echo "  ls $OUTPUT_DIR/*_summary.json"
    echo "  cat $OUTPUT_DIR/*.log | grep 'pass\^'"
fi
