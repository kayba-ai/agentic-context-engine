#!/bin/bash
#
# Reset workspace for clean ACE loop runs
#
# This script:
# 1. Initializes or resets workspace as separate git repository
# 2. Cleans .agent/ directory (Claude Code working files)
# 3. Keeps existing skillbook (learned strategies persist)
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"
DATA_DIR="${ACE_DEMO_DATA_DIR:-$SCRIPT_DIR/.data}"
SKILLBOOK_FILE="$DATA_DIR/skillbooks/ace_typescript.json"
LOGS_DIR="$DATA_DIR/logs"
TEMPLATE_DIR="$SCRIPT_DIR/workspace_template"

echo "========================================================================"
echo "RESETTING WORKSPACE"
echo "========================================================================"
echo ""

# Step 1: Initialize or reset workspace git repo
if [ ! -d "$WORKSPACE_DIR/.git" ]; then
    echo "Step 1: Creating new workspace git repository..."
    if [ -d "$WORKSPACE_DIR" ]; then
        echo "   Old workspace exists - backing up..."
        mv "$WORKSPACE_DIR" "$WORKSPACE_DIR.backup.$(date +%Y%m%d_%H%M%S)"
    fi
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
else
    echo "Step 1: Resetting existing workspace..."
    cd "$WORKSPACE_DIR"
    if [ -n "$(git status --porcelain)" ]; then
        git stash push -m "Auto-stash $(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    fi
    git reset --hard HEAD > /dev/null 2>&1
    git clean -fd > /dev/null 2>&1
    echo "   Done"
fi

# Create timestamped branch for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BRANCH_NAME="run-$TIMESTAMP"
git checkout -b "$BRANCH_NAME" 2>/dev/null || true
echo "   Branch: $BRANCH_NAME"
echo ""

# Step 2: Clean .agent directory
echo "Step 2: Cleaning .agent directory..."
if [ -d "$WORKSPACE_DIR/.agent" ]; then
    # Archive TODO.md before deleting
    if [ -f "$WORKSPACE_DIR/.agent/TODO.md" ]; then
        mkdir -p "$LOGS_DIR"
        cp "$WORKSPACE_DIR/.agent/TODO.md" "$LOGS_DIR/TODO_$TIMESTAMP.md"
    fi
    rm -rf "$WORKSPACE_DIR/.agent"
    echo "   Cleaned"
else
    echo "   Nothing to clean"
fi
echo ""

# Step 3: Setup skillbook
echo "Step 3: Skillbook setup..."
mkdir -p "$DATA_DIR/skillbooks"
if [ ! -f "$SKILLBOOK_FILE" ]; then
    echo '{"skills": {}, "sections": {}, "next_id": 1}' > "$SKILLBOOK_FILE"
    echo "   Created fresh skillbook"
else
    if command -v jq &> /dev/null; then
        SKILL_COUNT=$(jq '.skills | length' "$SKILLBOOK_FILE" 2>/dev/null || echo "?")
        echo "   Keeping existing skillbook ($SKILL_COUNT strategies)"
    else
        echo "   Keeping existing skillbook"
    fi
fi
echo ""

# Done
echo "========================================================================"
echo "READY"
echo "========================================================================"
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo "Skillbook: $SKILLBOOK_FILE"
echo ""
echo "Next: python ace_loop.py"
echo ""
