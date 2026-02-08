# Contributing to ACE Framework

Thank you for your interest in contributing to the Agentic Context Engine! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Environment details (OS, Python version, package versions)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear description of the enhancement
- Use cases and benefits
- Possible implementation approach (optional)
- Any potential drawbacks or considerations

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure nothing breaks
5. Commit your changes using conventional commits (see below)
6. Push to your branch
7. Open a Pull Request

## Branch Naming Convention

Use consistent prefixes for branch names:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/john/benchmarks` |
| `fix/` | Bug fixes | `fix/jane/memory-leak` |
| `docs/` | Documentation changes | `docs/john/api-reference` |
| `refactor/` | Code refactoring | `refactor/jane/llm-client` |
| `test/` | Test additions/fixes | `test/john/integration-suite` |
| `chore/` | Maintenance tasks | `chore/jane/update-deps` |

**Format:** `<type>/<developer>/<description>`

**Rules:**
- Use lowercase with hyphens (kebab-case)
- Use your GitHub username or first name as developer identifier
- Keep descriptions short but descriptive
- Include issue number if applicable: `fix/john/123-login-error`
- Never push directly to `main` - always use feature branches

## Worktree Workflow

We use git worktrees to work on multiple branches simultaneously without switching. Each branch gets its own directory.

### Claude Code Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/create-branch` | Create branch + worktree | `/create-branch feature add-caching` |
| `/checkout-branch` | Switch to branch (creates worktree if needed) | `/checkout-branch add-caching` |
| `/list-branches` | List branches with worktree status | `/list-branches` or `/list-branches feature` |
| `/remove-branch` | Remove branch + worktree | `/remove-branch feature/john/add-caching` |

### Worktree Path Convention

Worktrees are created as siblings to the main worktree:
- Branch: `feature/john/add-caching`
- Worktree: `../feature-john-add-caching`

### Manual Worktree Commands

```bash
# List all worktrees
git worktree list

# Add worktree for existing branch
git worktree add ../path-name branch-name

# Add worktree with new branch
git worktree add -b new-branch ../path-name

# Remove worktree
git worktree remove ../path-name

# Prune stale worktree references
git worktree prune
```

### Benefits

- **Parallel development**: Work on multiple features without stashing
- **Faster context switching**: No need to rebuild dependencies
- **Cleaner git history**: No accidental commits to wrong branch
- **IDE-friendly**: Open each worktree in separate IDE windows

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/agentic-context-engine.git
cd agentic-context-engine

# Install all dependencies (uses UV - 10-100x faster than pip)
uv sync

# Run tests
uv run pytest

# Run linting and formatting
uv run black ace/ tests/ examples/
uv run mypy ace/

# Run specific test files
uv run pytest tests/test_skillbook.py
uv run pytest -m unit  # Only unit tests
uv run pytest -m integration  # Only integration tests
```

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/) for clear commit history and automatic changelog generation.

Format: `<type>(<scope>): <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(llm): add support for new LLM provider
fix(adapter): resolve memory leak in online mode
docs(readme): update installation instructions
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all public functions and classes
- Keep line length under 100 characters
- Use Black for automatic formatting

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use meaningful test names

## Documentation

- Update README.md if adding new features
- Add docstrings to new code
- Update CHANGELOG.md following Keep a Changelog format
- Include examples for new functionality

## Questions?

Feel free to open an issue for any questions or join the discussion in [GitHub Discussions](https://github.com/Kayba-ai/agentic-context-engine/discussions).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.