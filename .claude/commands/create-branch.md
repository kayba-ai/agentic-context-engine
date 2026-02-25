Create a new git branch with an associated worktree following the project naming convention.

**Format:** `<type>/<developer>/<description>`
**Worktree:** `../<sanitized-branch-name>` (sibling to current worktree)

**Arguments:** $ARGUMENTS should be in format: `<type> <description>`

**Examples:**
- `/create-branch feature add-caching` → branch: `feature/<dev>/add-caching`, worktree: `../feature-<dev>-add-caching`
- `/create-branch fix login-error` → branch: `fix/<dev>/login-error`, worktree: `../fix-<dev>-login-error`

**Steps:**
1. Parse type and description from arguments (validate type is one of: feature, fix, docs, refactor, test, chore)
2. Get developer name from `git config user.name` (sanitize: lowercase, replace spaces with hyphens)
3. Construct branch name: `<type>/<developer>/<description>`
4. Construct worktree path: `../<type>-<developer>-<description>` (replace all `/` with `-`)
5. Create branch and worktree atomically: `git worktree add -b <branch> <worktree-path>`
6. Symlink `.env` from the main worktree into the new worktree:
   - Get the main worktree path: `git worktree list --porcelain | head -1` (first `worktree` line)
   - If `<main-worktree>/.env` exists, create symlink: `ln -s <main-worktree>/.env <new-worktree>/.env`
   - If `.env` doesn't exist in main worktree, skip silently
7. Report success with the created branch name and worktree path

**Valid types:** feature, fix, docs, refactor, test, chore

**On success, output:**
```
✓ Created branch: <branch-name>
✓ Created worktree: <worktree-path>
✓ Linked .env → <main-worktree>/.env

To switch to the new worktree:
  cd <worktree-path>
```

**Error handling:**
- If type is invalid, show valid types and abort
- If branch already exists, suggest checking it out instead
- If worktree path exists, suggest using existing worktree
