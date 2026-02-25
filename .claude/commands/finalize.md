Finalize the current work: format, test, fix, commit, and push.

Run this command after making code changes to complete the development cycle. It handles formatting, testing (with auto-fix retries), committing, and pushing.

**Arguments:** $ARGUMENTS is an optional commit message override. If not provided, compose one automatically from the diff.

**Workflow:**

1. **Format code**
   - Run `uv run black ace/ tests/ examples/`

2. **Update CLAUDE.md**
   - Run `/init` to update the project's CLAUDE.md file with current codebase context

3. **Review test coverage**
   - Check if the changes have adequate test coverage
   - Look at which lines/branches are untested for the modified files
   - If coverage gaps exist for the changed code, add targeted tests before proceeding
   - Focus on: new functions, error paths, edge cases, and branches introduced by this changeset

4. **Run tests**
   - Run `uv run pytest`

5. **Test-fix loop (max 3 retries)**
   - If tests pass, continue to step 6
   - If tests fail:
     - Analyze the failure output
     - Fix the failing code or tests
     - Add missing tests if the failures reveal gaps
     - Re-run formatter: `uv run black ace/ tests/ examples/`
     - Re-run tests: `uv run pytest`
     - If tests still fail after 3 total attempts, **stop entirely** and report the failures. Never commit broken code.

6. **Security review**
   - Run `/security-review` to scan changed code for vulnerabilities
   - If Critical or High severity issues are found, fix them before proceeding
   - After fixing, re-run formatter (`uv run black ace/ tests/ examples/`) and tests (`uv run pytest`)
   - If issues can't be auto-fixed, report them and stop

7. **Review changes**
   - Run `git diff` to review all changes
   - Run `git status` to see untracked/modified files
   - Determine which files to stage

8. **Stage files selectively**
   - Use explicit `git add <file>` for each file. **Never use `git add -A` or `git add .`**
   - **Never stage:** `.env`, credentials, secrets, `__pycache__/`, `*.pyc`, large binaries, `.DS_Store`
   - **Only stage `uv.lock`** if dependency changes in `pyproject.toml` were intentional
   - If unsure about a file, ask the user

9. **Compose commit message**
   - If $ARGUMENTS was provided, use it as the commit message
   - Otherwise, compose a Conventional Commit message: `<type>(<scope>): <short description>`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`
   - Derive scope from the primary changed file path (e.g., `ace/skillbook.py` -> `skillbook`, `ace/integrations/litellm.py` -> `integrations`, `tests/test_foo.py` -> `tests`)
   - Keep messages short and imperative

10. **Check branch safety**
    - Run `git branch --show-current` to get the current branch
    - If on `main` or `master`, **warn the user** and ask for explicit confirmation before committing
    - If denied, stop without committing

11. **Commit and push**
    - Commit with the composed message
    - Push to remote: `git push` (or `git push -u origin <branch>` if no upstream is set)

12. **Report summary**
    - Commit hash (short)
    - Branch name
    - Files changed count
    - Test results (pass count)
    - Any warnings encountered

**Error handling:**
- If no changes exist (clean working tree), report "Nothing to finalize" and stop
- If tests fail after 3 retries, report failures and stop without committing
- If push fails, report the error but keep the local commit
- If formatter fails, report the error and stop
- If security review finds unfixable Critical/High issues, report and stop without committing
