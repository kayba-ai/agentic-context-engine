Release a new version: bump version, update changelog, tag, push, and create a GitHub release.

**Arguments:** $ARGUMENTS — the new version number (e.g. `0.9.0`). Required.

**Steps:**

1. **Validate inputs**
   - If $ARGUMENTS is empty, abort: "Usage: /release <version> (e.g. /release 0.9.0)"
   - Strip leading `v` if present (e.g. `v0.9.0` → `0.9.0`)
   - Validate format matches `X.Y.Z` (semver)

2. **Validate branch state**
   - Must be on `main`: `git branch --show-current`
   - If not on main, abort: "Switch to main first."
   - Pull latest: `git pull origin main`
   - Working tree must be clean: `git status --porcelain`

3. **Check version isn't already used**
   - Read current version from `pyproject.toml` (line with `version = "..."`)
   - If new version equals current version, abort: "Version <ver> is already set."
   - Check tag doesn't exist: `git tag -l v<version>`
   - If tag exists, abort: "Tag v<version> already exists."

4. **Build changelog entry from git history**
   - Find the latest tag: `git describe --tags --abbrev=0`
   - Get commits since that tag: `git log <last-tag>..HEAD --format='%s'`
   - Get merged PRs since that tag: `gh pr list --state merged --base main --search "merged:>=$(git log -1 --format=%ci <last-tag> | cut -d' ' -f1)" --json title,number --limit 50`
   - From the commits/PRs, compose a changelog section with **only `### Added` items** — user-facing features. Skip fixes, refactors, chores, docs-only changes, and CI changes.
   - Format:
     ```
     ## [X.Y.Z] - YYYY-MM-DD

     ### Added
     - **Feature name** — short description
     - **Feature name** — short description
     ```
   - Also prepare a compare link for the bottom of CHANGELOG.md:
     `[X.Y.Z]: https://github.com/Kayba-ai/agentic-context-engine/compare/v<prev>...vX.Y.Z`

5. **Present draft to user**
   - Show: new version, changelog entry, and the release note (same as changelog "Added" bullets)
   - Ask: "Create this release?" with options: Yes, Edit (let me revise), Cancel

6. **On approval — apply changes**
   - Update `pyproject.toml`: replace `version = "<old>"` with `version = "<new>"`
   - Insert the changelog entry in `CHANGELOG.md` after line 7 (before the previous release)
   - Add the compare link at the bottom of CHANGELOG.md
   - Stage files: `git add pyproject.toml CHANGELOG.md`
   - Commit: `git commit -m "chore(release): bump version to <version>"`

7. **Tag and push**
   - `git tag v<version>`
   - `git push origin main --tags`

8. **Create GitHub release**
   - Title: `v<version>`
   - Notes: **only the "Added" bullets** from the changelog entry — short and clean, no preamble
   - Append: `**Full Changelog**: https://github.com/kayba-ai/agentic-context-engine/compare/v<prev>...v<version>`
   - Run: `gh release create v<version> --title "v<version>" --notes "<notes>"`
   - This triggers `.github/workflows/publish.yml` → PyPI publish

9. **Report summary**
   ```
   Released v<version>
   - Commit: <short-hash>
   - Tag: v<version>
   - Release: <github-release-url>
   - PyPI: publishing via workflow (check Actions tab)
   ```

**Error handling:**
- If `gh` is not installed: "GitHub CLI (gh) is required. Install it: https://cli.github.com"
- If not authenticated: "Run `gh auth login` first."
- If push fails: report error, keep local commit (user can retry)
- If `gh release create` fails: show the error, suggest manual creation
