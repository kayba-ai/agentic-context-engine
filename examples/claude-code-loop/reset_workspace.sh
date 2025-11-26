#!/bin/bash
#
# Reset workspace and playbook for clean ACE loop runs
#
# This script:
# 1. Deletes workspace/.agent/ directory
# 2. Creates fresh seed playbook with initial strategies
# 3. Resets workspace git to clean state
# 4. Verifies everything is clean
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"
PLAYBOOK_FILE="$SCRIPT_DIR/playbooks/ace_typescript.json"

echo "========================================================================"
echo "🔄 RESETTING WORKSPACE FOR CLEAN ACE LOOP RUN"
echo "========================================================================"
echo ""

# 1. Reset workspace git FIRST (before modifying source/)
echo "🔧 Step 1: Resetting workspace git to clean state..."
cd "$WORKSPACE_DIR"
git reset --hard HEAD > /dev/null 2>&1
# NOTE: We do NOT run 'git clean -fd' because it deletes untracked files
# like specs/ directory that we need to keep
echo "   ✅ Workspace git reset (tracked files only)"
echo ""

# 2. NOW replace source/ with fresh clone from GitHub (after git reset)
echo "📥 Step 2: Getting fresh agentic-context-engine source from GitHub..."
SOURCE_DIR="$WORKSPACE_DIR/source"

# Remove existing symlink or directory
if [ -L "$SOURCE_DIR" ]; then
    rm "$SOURCE_DIR"
    echo "   ✅ Removed existing symlink"
elif [ -d "$SOURCE_DIR" ]; then
    rm -rf "$SOURCE_DIR"
    echo "   ✅ Removed existing directory"
fi

# Clone fresh from GitHub
echo "   → Cloning from https://github.com/kayba-ai/agentic-context-engine..."
git clone https://github.com/kayba-ai/agentic-context-engine "$SOURCE_DIR" --quiet
echo "   ✅ Fresh clone complete"

# Clean any build artifacts that might be in the repo
echo "   → Cleaning build artifacts from clone..."
CLEANED=false
if [ -d "$SOURCE_DIR/build" ]; then
    rm -rf "$SOURCE_DIR/build"
    echo "      ✅ Removed build/ directory"
    CLEANED=true
fi
if [ -d "$SOURCE_DIR/dist" ]; then
    rm -rf "$SOURCE_DIR/dist"
    echo "      ✅ Removed dist/ directory"
    CLEANED=true
fi
find "$SOURCE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SOURCE_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$SOURCE_DIR" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
if [ "$CLEANED" = false ]; then
    echo "      ✅ No build artifacts found"
fi
echo ""

# 3. Archive logs FIRST (before deleting .agent)
echo "💾 Step 3: Archiving logs from previous run..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_DIR="$SCRIPT_DIR/logs/archive_$TIMESTAMP"
mkdir -p "$ARCHIVE_DIR"

ARCHIVED_FILES=0

# Archive playbook
if [ -f "$PLAYBOOK_FILE" ]; then
    cp "$PLAYBOOK_FILE" "$ARCHIVE_DIR/playbook.json"
    echo "   ✅ Archived playbook.json"
    ARCHIVED_FILES=$((ARCHIVED_FILES + 1))
fi

# Archive Claude Code output
if [ -f "$WORKSPACE_DIR/.agent/last_claude_output.txt" ]; then
    cp "$WORKSPACE_DIR/.agent/last_claude_output.txt" "$ARCHIVE_DIR/last_claude_output.txt"
    echo "   ✅ Archived last_claude_output.txt"
    ARCHIVED_FILES=$((ARCHIVED_FILES + 1))
fi

# Archive execution trace
if [ -f "$WORKSPACE_DIR/.agent/last_execution_trace.md" ]; then
    cp "$WORKSPACE_DIR/.agent/last_execution_trace.md" "$ARCHIVE_DIR/last_execution_trace.md"
    echo "   ✅ Archived last_execution_trace.md"
    ARCHIVED_FILES=$((ARCHIVED_FILES + 1))
fi

# Archive TODO.md if it exists
if [ -f "$WORKSPACE_DIR/.agent/TODO.md" ]; then
    cp "$WORKSPACE_DIR/.agent/TODO.md" "$ARCHIVE_DIR/TODO.md"
    echo "   ✅ Archived TODO.md"
    ARCHIVED_FILES=$((ARCHIVED_FILES + 1))
fi

if [ $ARCHIVED_FILES -eq 0 ]; then
    rm -rf "$ARCHIVE_DIR"
    echo "   ℹ️  No files to archive (fresh run)"
else
    echo "   📦 Archived $ARCHIVED_FILES files to: logs/archive_$TIMESTAMP/"
fi
echo ""

# 4. Delete .agent directory (AFTER archiving)
echo "📁 Step 4: Removing workspace/.agent/ directory..."
if [ -d "$WORKSPACE_DIR/.agent" ]; then
    rm -rf "$WORKSPACE_DIR/.agent"
    echo "   ✅ Removed $WORKSPACE_DIR/.agent/"
else
    echo "   ℹ️  Directory $WORKSPACE_DIR/.agent/ does not exist (already clean)"
fi
echo ""

# 5. Delete playbook (fresh start with no strategies)
echo "📚 Step 5: Deleting playbook for fresh start..."
if [ -f "$PLAYBOOK_FILE" ]; then
    rm "$PLAYBOOK_FILE"
    echo "   ✅ Deleted $PLAYBOOK_FILE"
else
    echo "   ℹ️  Playbook file does not exist (already clean)"
fi
echo ""

# 6. Remove .github directory (CI/CD templates shouldn't persist)
echo "📁 Step 6: Removing workspace/.github/ directory..."
if [ -d "$WORKSPACE_DIR/.github" ]; then
    rm -rf "$WORKSPACE_DIR/.github"
    echo "   ✅ Removed $WORKSPACE_DIR/.github/"
else
    echo "   ℹ️  Directory $WORKSPACE_DIR/.github/ does not exist (already clean)"
fi
echo ""

# 7. Remove prompt.md (learned strategies shouldn't persist)
echo "📁 Step 7: Removing workspace/prompt.md..."
if [ -f "$WORKSPACE_DIR/prompt.md" ]; then
    rm "$WORKSPACE_DIR/prompt.md"
    echo "   ✅ Removed $WORKSPACE_DIR/prompt.md"
else
    echo "   ℹ️  File $WORKSPACE_DIR/prompt.md does not exist (already clean)"
fi
echo ""

# 8. Remove workspace/TODO.md (should be in .agent/TODO.md instead)
echo "📁 Step 8: Removing workspace/TODO.md..."
if [ -f "$WORKSPACE_DIR/TODO.md" ]; then
    rm "$WORKSPACE_DIR/TODO.md"
    echo "   ✅ Removed $WORKSPACE_DIR/TODO.md"
else
    echo "   ℹ️  File $WORKSPACE_DIR/TODO.md does not exist (already clean)"
fi
echo ""

# 9. Clean workspace/target/ directory (keep only .gitignore)
echo "🧹 Step 9: Cleaning workspace/target/ directory..."
if [ -d "$WORKSPACE_DIR/target" ]; then
    # Remove everything except .gitignore
    find "$WORKSPACE_DIR/target" -mindepth 1 ! -name '.gitignore' -exec rm -rf {} + 2>/dev/null || true
    echo "   ✅ Cleaned target/ directory (kept .gitignore)"
else
    echo "   ℹ️  Directory $WORKSPACE_DIR/target/ does not exist"
fi
echo ""

# 10. Remove erroneous workspace/workspace directory (from path bug)
echo "📁 Step 10: Removing workspace/workspace/ if exists..."
if [ -d "$WORKSPACE_DIR/workspace" ]; then
    rm -rf "$WORKSPACE_DIR/workspace"
    echo "   ✅ Removed erroneous workspace/workspace/ directory"
else
    echo "   ℹ️  No workspace/workspace/ directory found"
fi
echo ""

# 11. Verify clean state
echo "========================================================================"
echo "✅ WORKSPACE RESET COMPLETE"
echo "========================================================================"
echo ""

echo "📊 Verification:"
echo ""

echo "1. Workspace git status:"
cd "$WORKSPACE_DIR"
git status --short
if [ $? -eq 0 ] && [ -z "$(git status --short)" ]; then
    echo "   ✅ Git working tree is clean"
else
    echo "   ⚠️  Git has uncommitted changes"
fi
echo ""

echo "2. .agent directory:"
if [ -d "$WORKSPACE_DIR/.agent" ]; then
    echo "   ⚠️  WARNING: .agent/ still exists!"
    ls -la "$WORKSPACE_DIR/.agent/"
else
    echo "   ✅ .agent/ does not exist (will be created by Task 1)"
fi
echo ""

echo "3. Playbook file:"
if [ -f "$PLAYBOOK_FILE" ]; then
    echo "   ⚠️  WARNING: Playbook still exists!"
    cat "$PLAYBOOK_FILE" | head -10
else
    echo "   ✅ Playbook deleted (will be created fresh by ACE)"
fi
echo ""

echo "4. Critical directories:"
if [ -d "$WORKSPACE_DIR/source" ] && [ -d "$WORKSPACE_DIR/specs" ]; then
    echo "   ✅ source/ (fresh clone) and specs/ exist"
    # Check that source is a git repo with correct remote
    if [ -d "$WORKSPACE_DIR/source/.git" ]; then
        cd "$WORKSPACE_DIR/source"
        REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "none")
        if [[ "$REMOTE_URL" == *"agentic-context-engine"* ]]; then
            echo "   ✅ source/ is correctly cloned from agentic-context-engine"
        else
            echo "   ⚠️  WARNING: source/ has unexpected remote: $REMOTE_URL"
        fi
        cd "$SCRIPT_DIR"
    fi
else
    echo "   ⚠️  WARNING: source/ or specs/ missing!"
    if [ ! -d "$WORKSPACE_DIR/source" ]; then
        echo "      source/ directory is missing"
    fi
    if [ ! -d "$WORKSPACE_DIR/specs" ]; then
        echo "      specs/ directory is missing"
        echo "      Run: ./restore_workspace.sh to fix specs/"
    fi
fi
echo ""

echo "========================================================================"
echo "🚀 READY FOR CLEAN RUN"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run: python ace_loop.py"
echo "  2. Task 1 will be: 'Create minimal TODO.md with Python-to-TypeScript translation tasks'"
echo "  3. Claude Code will create .agent/TODO.md focused on translation (not infrastructure)"
echo "  4. You should see substantial output and real work"
echo ""
echo "Expected behavior:"
echo "  - Empty playbook → ACE starts with no strategies"
echo "  - Claude Code creates .agent/TODO.md with translation tasks"
echo "  - Reflector analyzes actual code creation (not just prompt updates)"
echo "  - Curator creates first strategies based on Claude Code's approach"
echo "  - Playbook grows organically from actual successful patterns"
echo ""
