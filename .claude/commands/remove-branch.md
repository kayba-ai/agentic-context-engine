Remove a branch and its associated worktree.

**Arguments:** $ARGUMENTS should be the branch name (full or partial match)
- `--force`: Skip confirmations and force delete unmerged branches

**Examples:**
- `/remove-branch feature/john/add-caching` → remove branch and worktree
- `/remove-branch add-caching` → partial match, will prompt to confirm
- `/remove-branch feature/john/add-caching --force` → skip all confirmations

**Steps:**
1. Parse branch name and flags from arguments
2. Resolve branch name (support partial matching if unique)
3. Safety checks:
   - Abort if trying to remove main/master
   - Abort if trying to remove current branch (must switch first)
   - Warn if branch has unmerged commits (show `git log main..<branch> --oneline`)
4. Check if branch has an associated worktree: `git worktree list`
5. If worktree exists:
   - Remove worktree first: `git worktree remove <path>` (or `--force` if needed)
   - Prune worktree list: `git worktree prune`
6. Delete the branch: `git branch -d <branch>` (or `-D` with `--force`)
7. Confirm success

**On success, output:**
```
✓ Removed worktree: <worktree-path>
✓ Removed branch: <branch-name>
```

**Error handling:**
- Multiple partial matches: list matches and ask user to be more specific
- Unmerged commits without --force: show commits and ask for confirmation
- Protected branches (main/master): refuse with explanation
- Current branch: instruct user to switch branches first

**Protected branches:** main, master

**Confirmation prompts (unless --force):**
- "Branch has N unmerged commits. Remove anyway? (show commits first)"
- For partial match: "Did you mean <full-branch-name>?"
