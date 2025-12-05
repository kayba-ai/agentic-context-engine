#!/bin/bash
#
# Reset workspace for clean ACE loop runs
#
# This script:
# 1. Archives and resets workspace as separate git repository
# 2. Archives and cleans .agent/ directory (Claude Code working files)
# 3. Archives and resets skillbook
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"
DATA_DIR="${ACE_DEMO_DATA_DIR:-$SCRIPT_DIR/.data}"
SKILLBOOK_FILE="$DATA_DIR/skillbooks/skillbook.json"
LOGS_DIR="$DATA_DIR/logs"
TEMPLATE_DIR="$SCRIPT_DIR/workspace_template"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo "RESETTING WORKSPACE"
echo "========================================================================"
echo ""

# Create logs directory for this reset
mkdir -p "$LOGS_DIR/$TIMESTAMP"

# Step 1: Archive and reset workspace git repo
if [ -d "$WORKSPACE_DIR" ]; then
    echo "Step 1: Archiving and resetting workspace..."
    # Archive entire workspace
    cp -r "$WORKSPACE_DIR" "$LOGS_DIR/$TIMESTAMP/workspace"
    echo "   Archived to $LOGS_DIR/$TIMESTAMP/workspace"
    # Delete workspace
    rm -rf "$WORKSPACE_DIR"
    echo "   Deleted workspace"
fi

# Create fresh workspace from template
echo "   Creating fresh workspace..."
cp -r "$TEMPLATE_DIR" "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Copy .env.example to .env if needed
if [ -f "$WORKSPACE_DIR/.env.example" ] && [ ! -f "$WORKSPACE_DIR/.env" ]; then
    cp "$WORKSPACE_DIR/.env.example" "$WORKSPACE_DIR/.env"
    echo "   Created .env from .env.example"
fi

git init
git add .
git commit -m "Initial workspace setup"
echo "   Done"

# Create timestamped branch for this run
BRANCH_NAME="run-$TIMESTAMP"
git checkout -b "$BRANCH_NAME" 2>/dev/null || true
echo "   Branch: $BRANCH_NAME"
echo ""

# Step 2: Clean .agent directory (already done by deleting workspace, but handle if exists)
echo "Step 2: Cleaning .agent directory..."
if [ -d "$WORKSPACE_DIR/.agent" ]; then
    rm -rf "$WORKSPACE_DIR/.agent"
    echo "   Cleaned"
else
    echo "   Nothing to clean"
fi
echo ""

# Step 3: Archive and reset skillbook
echo "Step 3: Skillbook setup..."
mkdir -p "$DATA_DIR/skillbooks"
if [ -f "$SKILLBOOK_FILE" ]; then
    # Archive existing skillbook
    cp "$SKILLBOOK_FILE" "$LOGS_DIR/$TIMESTAMP/skillbook.json"
    if command -v jq &> /dev/null; then
        SKILL_COUNT=$(jq '.skills | length' "$SKILLBOOK_FILE" 2>/dev/null || echo "?")
        echo "   Archived skillbook ($SKILL_COUNT strategies) to $LOGS_DIR/$TIMESTAMP/"
    else
        echo "   Archived skillbook to $LOGS_DIR/$TIMESTAMP/"
    fi
    # Delete and create fresh
    rm "$SKILLBOOK_FILE"
fi
echo '{"skills": {}, "sections": {}, "next_id": 1}' > "$SKILLBOOK_FILE"
echo "   Created fresh skillbook"
echo ""

# Done
echo "========================================================================"
echo "READY"
echo "========================================================================"
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo "Skillbook: $SKILLBOOK_FILE"
echo "Archive:   $LOGS_DIR/$TIMESTAMP"
echo ""
echo "Next: python ace_loop.py"
echo ""