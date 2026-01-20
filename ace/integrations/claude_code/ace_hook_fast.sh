#!/bin/bash
# Fast ACE hook handler - queue only, daemon handles processing
# Target: < 50ms exit time
#
# The daemon (ace-daemon) watches the queue directory and processes files.
# This script only writes to the queue and exits immediately.

QUEUE_DIR="${ACE_QUEUE_DIR:-$HOME/.ace/queue}"
mkdir -p "$QUEUE_DIR"

# Generate unique filename with nanoseconds and PID
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
QUEUE_FILE="$QUEUE_DIR/hook-${TIMESTAMP}-$$.json"

# Atomic write: write to temp file then rename
TEMP_FILE=$(mktemp)
cat > "$TEMP_FILE"
mv "$TEMP_FILE" "$QUEUE_FILE"

exit 0
