#!/bin/bash
#
# Reset workspace and playbook for clean ACE loop runs
#
# This script:
# 1. Initializes or resets workspace as separate git repository
# 2. Migrates old playbook data to .data/ directory
# 3. Clones fresh source code
# 4. Archives previous run logs
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"
DATA_DIR="${ACE_DEMO_DATA_DIR:-$SCRIPT_DIR/.data}"
PLAYBOOK_FILE="$DATA_DIR/playbooks/ace_typescript.json"
LOGS_DIR="$DATA_DIR/logs"
TEMPLATE_DIR="$SCRIPT_DIR/workspace_template"

echo "========================================================================"
echo "🔄 RESETTING WORKSPACE FOR CLEAN ACE LOOP RUN"
echo "========================================================================"
echo ""

# Step 0: Migrate old data if exists
if [ -f "$SCRIPT_DIR/playbooks/ace_typescript.json" ]; then
    echo "🔄 Step 0: Migrating old playbook data..."
    mkdir -p "$DATA_DIR/playbooks"
    cp "$SCRIPT_DIR/playbooks/ace_typescript.json" "$DATA_DIR/playbooks/"
    echo "   ✅ Migrated playbook from old location to .data/"
    echo "   💡 Old playbooks/ directory is deprecated (safe to delete)"
    echo ""
fi

# Step 1: Initialize or reset workspace git repo
if [ ! -d "$WORKSPACE_DIR/.git" ]; then
    echo "🆕 Step 1: Creating new workspace git repository..."
    if [ -d "$WORKSPACE_DIR" ]; then
        echo "   ⚠️  Old workspace directory exists without git - backing up..."
        mv "$WORKSPACE_DIR" "$WORKSPACE_DIR.backup.$(date +%Y%m%d_%H%M%S)"
        echo "   ✅ Backed up old workspace"
    fi
    cp -r "$TEMPLATE_DIR" "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"

    # Copy .env.example to .env if needed
    if [ -f "$WORKSPACE_DIR/.env.example" ]; then
        if [ ! -f "$WORKSPACE_DIR/.env" ]; then
            cp "$WORKSPACE_DIR/.env.example" "$WORKSPACE_DIR/.env"
            echo "   ✅ Created .env from .env.example"
            echo "   💡 Edit workspace/.env to add your ANTHROPIC_API_KEY"
        fi
    fi

    git init
    git add .
    git commit -m "Initial workspace setup

Generated from workspace_template/ by reset_workspace.sh
This is a separate git repository for ACE + Claude Code work."
    echo "   ✅ Workspace git repository initialized"
else
    echo "🔄 Step 1: Resetting existing workspace git repository..."
    cd "$WORKSPACE_DIR"
    # Stash any uncommitted work (in case user made changes)
    if [ -n "$(git status --porcelain)" ]; then
        echo "   💾 Stashing uncommitted changes..."
        git stash push -m "Auto-stash before reset $(date +%Y%m%d_%H%M%S)"
    fi
    git reset --hard HEAD > /dev/null 2>&1
    git clean -fd > /dev/null 2>&1  # Clean untracked files
    echo "   ✅ Workspace git reset to clean state"

    # Copy .env.example to .env if needed
    if [ -f "$WORKSPACE_DIR/.env.example" ]; then
        if [ ! -f "$WORKSPACE_DIR/.env" ]; then
            cp "$WORKSPACE_DIR/.env.example" "$WORKSPACE_DIR/.env"
            echo "   ✅ Created .env from .env.example"
            echo "   💡 Edit workspace/.env to add your ANTHROPIC_API_KEY"
        else
            echo "   ℹ️  .env already exists (keeping existing)"
        fi
    fi
fi

# Create timestamped branch for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BRANCH_NAME="run-$TIMESTAMP"
git checkout -b "$BRANCH_NAME"
echo "   ✅ Created branch: $BRANCH_NAME"
echo ""

# Step 2: Clone source code (outside workspace git tracking)
echo "📥 Step 2: Getting fresh agentic-context-engine source..."
SOURCE_DIR="$WORKSPACE_DIR/source"

# Remove existing source
if [ -d "$SOURCE_DIR" ]; then
    rm -rf "$SOURCE_DIR"
    echo "   ✅ Removed existing source/"
fi

# Clone fresh from GitHub
echo "   → Cloning from https://github.com/kayba-ai/agentic-context-engine..."
git clone https://github.com/kayba-ai/agentic-context-engine "$SOURCE_DIR" --quiet

# Clean any build artifacts
find "$SOURCE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SOURCE_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$SOURCE_DIR" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf "$SOURCE_DIR/build" "$SOURCE_DIR/dist" 2>/dev/null || true

echo "   ✅ Fresh source code cloned and cleaned"
echo ""

# Step 3: Clean .agent directory (Claude Code working files)
echo "🧹 Step 3: Cleaning .agent directory..."
if [ -d "$WORKSPACE_DIR/.agent" ]; then
    # Archive TODO.md before deleting
    if [ -f "$WORKSPACE_DIR/.agent/TODO.md" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        ARCHIVE_DIR="$LOGS_DIR/archive_$TIMESTAMP"
        mkdir -p "$ARCHIVE_DIR"
        cp "$WORKSPACE_DIR/.agent/TODO.md" "$ARCHIVE_DIR/TODO.md"
        echo "   ✅ Archived TODO.md to $ARCHIVE_DIR"
    fi
    rm -rf "$WORKSPACE_DIR/.agent"
    echo "   ✅ Removed .agent/ directory"
else
    echo "   ℹ️  .agent/ does not exist (will be created by Claude Code)"
fi
echo ""

# Step 4: Archive old run logs (if any exist)
echo "💾 Step 4: Archiving logs from previous run..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="$LOGS_DIR/archive_$TIMESTAMP"

ARCHIVED_FILES=0

# Archive playbook
if [ -f "$PLAYBOOK_FILE" ]; then
    mkdir -p "$ARCHIVE_DIR"
    cp "$PLAYBOOK_FILE" "$ARCHIVE_DIR/playbook.json"
    echo "   ✅ Archived playbook.json"
    ARCHIVED_FILES=$((ARCHIVED_FILES + 1))
fi

# Archive other logs from old location
if [ -f "$SCRIPT_DIR/logs/last_output.txt" ]; then
    mkdir -p "$ARCHIVE_DIR"
    cp "$SCRIPT_DIR/logs/last_output.txt" "$ARCHIVE_DIR/last_output.txt"
    ARCHIVED_FILES=$((ARCHIVED_FILES + 1))
fi

if [ $ARCHIVED_FILES -eq 0 ]; then
    echo "   ℹ️  No files to archive (fresh run)"
else
    echo "   📦 Archived $ARCHIVED_FILES files to: .data/logs/archive_$TIMESTAMP/"
fi
echo ""

# Step 5: Create or keep existing playbook
echo "📚 Step 5: Playbook setup..."
mkdir -p "$DATA_DIR/playbooks"
if [ ! -f "$PLAYBOOK_FILE" ]; then
    echo '{"bullets": {}, "sections": {}, "next_id": 1}' > "$PLAYBOOK_FILE"
    echo "   ✅ Created fresh playbook (empty)"
else
    # Count strategies
    if command -v jq &> /dev/null; then
        BULLET_COUNT=$(jq '.bullets | length' "$PLAYBOOK_FILE")
        echo "   ✅ Keeping existing playbook ($BULLET_COUNT strategies)"
    else
        echo "   ✅ Keeping existing playbook"
    fi
fi
echo ""

# Step 6: Commit initial workspace state
echo "💾 Step 6: Committing workspace state..."
cd "$WORKSPACE_DIR"
if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "Reset workspace for new ACE loop run

Timestamp: $(date +%Y-%m-%d\ %H:%M:%S)
Fresh source code cloned
.agent/ directory cleaned
Ready for new tasks"
    echo "   ✅ Workspace changes committed"
else
    echo "   ℹ️  No changes to commit"
fi
echo ""

# Verification
echo "========================================================================"
echo "✅ WORKSPACE RESET COMPLETE"
echo "========================================================================"
echo ""

echo "📊 Configuration:"
echo "   Workspace: $WORKSPACE_DIR"
echo "   Data directory: $DATA_DIR"
echo "   Playbook: $PLAYBOOK_FILE"
echo ""

echo "📊 Workspace status:"
cd "$WORKSPACE_DIR"
git log --oneline -3 2>/dev/null || echo "   (no commits yet)"
echo ""

echo "📊 Directory structure:"
echo "   specs/ - Project specification ✅"
echo "   source/ - Python code (git ignored) ✅"
echo "   target/ - TypeScript output (will be tracked)"
echo "   .agent/ - Claude Code working files (will be created)"
echo ""

echo "========================================================================"
echo "🚀 READY FOR CLEAN RUN"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run: python ace_loop.py"
echo "  2. Claude Code will create .agent/TODO.md with translation tasks"
echo "  3. Each task will result in commits to workspace git repo"
echo "  4. Playbook will learn from successful patterns"
echo ""
echo "To inspect agent's work:"
echo "  cd workspace && git log --oneline"
echo ""
