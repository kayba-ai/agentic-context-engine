List project branches with their worktree status.

**Arguments:** $ARGUMENTS (optional filters)
- No args: List all branches matching naming convention `<type>/<developer>/<description>`
- `--all`: Include all branches (not just convention-named)
- `--worktrees`: Only show branches with active worktrees
- `<type>`: Filter by type (feature, fix, docs, refactor, test, chore)

**Examples:**
- `/list-branches` → all convention-named branches
- `/list-branches feature` → only feature branches
- `/list-branches --worktrees` → only branches with worktrees
- `/list-branches --all` → all branches including main

**Steps:**
1. Get all local branches: `git branch --format='%(refname:short)'`
2. Get worktree list: `git worktree list --porcelain`
3. Parse worktree output to map branches to paths
4. Filter branches based on arguments
5. Format and display results

**Output format:**
```
Branch                              Worktree                         Status
──────────────────────────────────────────────────────────────────────────────
* main                              .                                current
  feature/john/add-caching          ../feature-john-add-caching      active
  fix/jane/login-error              (no worktree)                    -

Summary: 3 branches, 2 with worktrees
```

**Column meanings:**
- `*` indicates current branch
- Worktree shows path relative to repo root, or "(no worktree)" if none
- Status: "current" (HEAD), "active" (has worktree), "-" (no worktree)

**Tips shown after output:**
- To create a worktree for a branch: `git worktree add ../<path> <branch>`
- To remove a branch with worktree: use `/remove-branch <branch>`
