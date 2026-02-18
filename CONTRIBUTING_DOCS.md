# Contributing to the Docs

The docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and deployed via GitHub Actions using [mike](https://github.com/jimporter/mike) for versioning.

## Local Preview

```bash
uv sync --group docs
mkdocs serve
```

Open [http://localhost:8000](http://localhost:8000). Changes reload automatically.

## Making Changes

All source files are in `docs/`. The site structure is defined in `mkdocs.yml`.

1. Edit or add `.md` files in `docs/`
2. If adding a new page, register it under `nav:` in `mkdocs.yml`
3. Preview locally with `mkdocs serve`

## Publishing to the Live Site

Once your changes are on GitHub (any branch):

1. Go to **Actions → Docs → Run workflow**
2. Select your branch
3. Enter a version (e.g. `0.8`) — this appears in the version dropdown on the site
4. Click **Run workflow**

The site will be live at https://kayba-ai.github.io/agentic-context-engine/ within a minute.

> **Note:** Pushing to `main` automatically deploys a `dev` version. Pushing a `v*` tag (e.g. `v0.8.0`) deploys a numbered release version.