#!/bin/bash
# ace-learn — wrapper for learn_from_traces.py
#
# Writes skillbook output to the persistent workspace volume so it
# survives container restarts. The agent calls this as: ace-learn [args]

set -euo pipefail

# Resolve OPENCLAW_HOME once — the Python script also reads this env var
# to find the sessions directory. Without it, the script falls back to a
# path relative to its own location which is wrong inside Docker.
export OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"

OUTPUT_DIR="$OPENCLAW_HOME/workspace/skills/kayba-ace"
mkdir -p "$OUTPUT_DIR"

cd /opt/ace
exec .venv/bin/python examples/openclaw/kayba-ace/learn_from_traces.py \
    --output "$OUTPUT_DIR" \
    "$@"
