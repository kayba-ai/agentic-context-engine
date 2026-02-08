Create a pull request for the current branch against main.

**Arguments:** $ARGUMENTS (optional) — base branch override (defaults to `main`)

**Steps:**

1. **Validate branch state:**
   - Get current branch: `git branch --show-current`
   - If on `main`, abort with error: "You're on main. Switch to a feature branch first."
   - Determine base branch: use $ARGUMENTS if provided, otherwise `main`

2. **Check for unmerged commits:**
   - Run `git log <base>..HEAD --oneline`
   - If no commits, abort: "No unmerged commits against <base>. Nothing to PR."

3. **Gather context (run in parallel):**
   - `git diff <base>...HEAD --stat` — file change summary
   - `git log <base>..HEAD --format='%h %s'` — commit list
   - `git diff <base>...HEAD` — full diff for understanding changes

4. **Draft PR title and body:**
   - Analyze the commits and diff to understand the change
   - Write a short PR title (under 70 chars, imperative mood)
   - Write the body using this format:
     ```
     ## Summary
     <1-3 bullet points explaining the changes>

     ## Changes
     <bulleted list of key file/module changes>

     ## Test plan
     <how to verify the changes work>
     ```

5. **Present draft to user:**
   - Show the proposed title and body
   - Ask: "Push and create this PR?" with options: Yes (create), Edit (let me revise), Cancel

6. **On approval:**
   - Check if branch has upstream: `git rev-parse --abbrev-ref @{upstream}`
   - Push branch: `git push -u origin HEAD`
   - Create PR: `gh pr create --title "<title>" --body "<body>" --base <base>`
   - Show the resulting PR URL

**On success, output:**
```
✓ Pushed branch: <branch-name>
✓ Created PR: <pr-url>
```

**Error handling:**
- If `gh` is not installed: "GitHub CLI (gh) is required. Install it: https://cli.github.com"
- If not authenticated: "Run `gh auth login` first."
- If PR already exists: show the existing PR URL with `gh pr view --web`
- If push fails: show the git error and abort
