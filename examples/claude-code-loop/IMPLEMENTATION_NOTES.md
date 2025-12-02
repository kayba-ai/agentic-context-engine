# Implementation Notes: Autonomous Translation with Testing Loop

## Overview

Enhanced the ACE + Claude Code loop to include automatic validation and error feedback for Python-to-TypeScript translation. This creates a complete autonomous system that translates code, validates it, and iteratively fixes errors.

## Changes Made

### 1. Updated Workspace Template

**workspace_template/specs/project.md**
- Added comprehensive validation requirements section
- Specified 4 validation checks: TypeScript compilation, unit tests, simple example, seahorse example
- Added architectural changes documentation (LiteLLM → Vercel AI SDK)
- Listed required dependencies (TypeScript, Vercel AI SDK, Jest, etc.)
- Added data structures migration guide (Python dataclass → TypeScript interface + Zod)

**workspace_template/specs/rules.md**
- Added detailed Jest testing standards with configuration example
- Added test structure examples (describe/test blocks)
- Added comprehensive LiteLLM → Vercel AI SDK migration guide
- Included side-by-side Python/TypeScript examples
- Documented key differences (async/await, import structure, response format)
- Added testing patterns with mock LLM implementation

### 2. Enhanced ace_loop.py

**Added Imports:**
- `import subprocess` for running npm commands

**Added Validation Functions:**
1. `run_npm_command(workspace_dir, command, description)` - Execute npm scripts and capture output
2. `extract_tsc_errors(output, max_lines)` - Parse TypeScript compilation errors
3. `extract_jest_errors(output, max_lines)` - Parse Jest test errors
4. `validate_typescript_compilation(workspace_dir)` - Run `npm run build`
5. `validate_unit_tests(workspace_dir)` - Run `npm test`
6. `validate_example(workspace_dir, example_name)` - Run `npm run example:{name}`
7. `run_full_validation(workspace_dir)` - Execute all checks in sequence

**Modified main() Loop:**

**Phase 1: Translation (Existing)**
- Unchanged behavior: Read TODO.md, execute tasks, commit results
- Added phase label for clarity

**Phase 2 & 3: Validation + Fix Loop (NEW)**
- Checks if target/package.json exists (TypeScript project setup)
- Runs 4 validation checks in sequence:
  1. TypeScript compilation (`npm run build`)
  2. Unit tests (`npm test`)
  3. Simple example (`npm run example:simple`)
  4. Seahorse example (`npm run example:seahorse`)
- On success: Celebrates and exits
- On failure: Creates fix prompt with error messages
- Feeds errors back to Claude Code for fixing
- Repeats validation after fixes (max 5 attempts)
- Interactive mode: Asks for confirmation before each fix attempt

**Updated Final Summary:**
- Shows translation tasks count
- Shows validation attempts count
- Shows playbook strategies count

## Three-Phase Architecture

### Phase 1: TRANSLATION
- Claude Code translates Python files task-by-task
- Works through TODO.md until all tasks marked [x]
- Commits to workspace git repo after each task
- **Trigger for Phase 2**: TODO.md has no more [ ] items

### Phase 2: VALIDATION
Run 4 automated checks in sequence:
1. TypeScript Compilation: `npm run build`
2. Unit Tests: `npm test`
3. Example 1: `npm run example:simple`
4. Example 2: `npm run example:seahorse`

**Success**: All 4 checks pass → Done!
**Failure**: Any check fails → Proceed to Phase 3

### Phase 3: FIX LOOP
- Extract error messages from failed check
- Create fix prompt with errors
- Feed back to Claude Code via `agent.run(fix_prompt)`
- Claude Code analyzes errors and fixes issues
- Commits fixes to workspace git
- Return to Phase 2 (re-run validation)
- **Max 5 iterations** to prevent infinite loops

## Expected Workflow

### First Run (Learning)
1. User runs `python ace_loop.py` (or AUTO_MODE=true)
2. Task 1: Claude Code creates TODO.md with ~20-25 translation tasks
3. Tasks 2-25: Translate Python files to TypeScript
4. Each task: Claude Code commits tested changes to workspace git
5. Phase 2: Run validation checks
6. Phase 3: If validation fails, Claude Code fixes errors iteratively
7. Result: Working TypeScript translation + 15-30 learned strategies in playbook

### Second Run (Optimized)
1. Reset workspace: `./reset_workspace.sh` (keeps playbook)
2. Run: `AUTO_MODE=true python ace_loop.py`
3. Claude Code starts with 15-30 strategies from first run
4. Expected: Consolidates work into 3-8 tasks (vs 20-25 first run)
5. Expected: Validation passes on first attempt (vs 3-5 attempts first run)
6. Potential: One-shot success (<5 tasks, <30 minutes, <$0.20 API cost)

## Error Feedback Structure

When validation fails, Claude Code receives:
```
VALIDATION FAILED - FIX REQUIRED

The TypeScript translation has validation errors. Please analyze and fix them.

❌ TypeScript compilation failed:
src/roles.ts:45:12 - error TS2322: Type 'string | undefined' is not assignable to type 'string'

INSTRUCTIONS:
1. Read the error messages carefully
2. Identify which files need fixes
3. Make the necessary corrections
4. Test your changes (run the failing command to verify fix)
5. Commit your fixes with: git add . && git commit -m "Fix validation errors"
6. Respond with a summary of what you fixed

Focus only on fixing the specific errors shown above.
Do NOT move on to other tasks or improvements.
```

This structured feedback gives Reflector clear context to tag helpful/harmful strategies.

## Files Modified

1. **examples/claude-code-loop/ace_loop.py**
   - Added 7 validation functions (135 lines)
   - Modified main() loop for validation phases (65 lines)
   - Added `import subprocess`

2. **examples/claude-code-loop/workspace_template/specs/project.md**
   - Added "Validation Requirements" section
   - Added "Key Architectural Changes" section
   - Added "Required Dependencies" section

3. **examples/claude-code-loop/workspace_template/specs/rules.md**
   - Expanded "Testing Standards" with Jest examples
   - Added "LiteLLM → Vercel AI SDK Migration" section
   - Added side-by-side Python/TypeScript examples

## Testing

✅ Python syntax check passed: `python -m py_compile ace_loop.py`
✅ Workspace reset works: `./reset_workspace.sh`
✅ Updated template files deployed to workspace
✅ Git repository structure intact

## Next Steps for Users

1. **First Run**:
   ```bash
   ./reset_workspace.sh
   AUTO_MODE=true python ace_loop.py
   ```
   - Let it run end-to-end (~1.5-2.5 hours)
   - Observe validation failures and fixes
   - Check playbook: `cat .data/playbooks/ace_typescript.json`

2. **Second Run** (with learned strategies):
   ```bash
   ./reset_workspace.sh  # Keeps playbook
   AUTO_MODE=true python ace_loop.py
   ```
   - Should complete faster (~15-45 minutes)
   - Should pass validation in 1-2 attempts

3. **Inspect Results**:
   ```bash
   cd workspace
   git log --oneline          # See all commits
   git log -p -5              # See last 5 commits with diffs
   npm run build              # Test TypeScript compilation
   npm test                   # Run unit tests
   npm run example:simple     # Test simple example
   npm run example:seahorse   # Test seahorse example
   ```

## Success Metrics

### First Run
- ✅ All 20-25 tasks completed
- ✅ TypeScript compiles without errors
- ✅ All Jest tests pass
- ✅ Both examples run successfully
- ✅ Playbook contains 15-30 learned strategies

### Second Run
- ✅ Completes in ≤8 tasks (vs 20-25 first run)
- ✅ Validation passes in ≤2 attempts (vs 3-5 first run)
- ✅ Total time <45 minutes (vs 1.5-2.5 hours first run)
- ✅ Potential one-shot success

## Estimated Costs

**First Run**:
- Duration: 1.5-2.5 hours
- API Cost: ~$0.80-$1.20 (Reflector + Curator on ~25 tasks)
- Validation iterations: 3-5
- Claude Code: Subscription usage only

**Second Run**:
- Duration: 15-45 minutes
- API Cost: ~$0.08-$0.15 (Reflector + Curator on ~5 tasks)
- Validation iterations: 1-2
- Claude Code: Subscription usage only

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Infinite validation loop | MAX_VALIDATION_ATTEMPTS = 5 hard limit |
| npm commands not available | Check for npm/node at startup, fail fast |
| Examples require API keys | Provide .env.example, check for ANTHROPIC_API_KEY |
| Too many tasks overwhelm Claude Code | Break into phases, ≤5 tasks per phase |
| Validation errors too verbose | Extract only relevant error lines (max 50 lines) |

## Known Issues

- Pydantic compatibility issue with browser-use dependency (unrelated to changes)
- Workspace template only deployed on first initialization (by design)
- Validation requires target/package.json to exist (graceful skip if missing)
