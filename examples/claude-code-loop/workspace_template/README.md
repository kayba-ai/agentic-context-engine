# ACE Workspace

This workspace is managed by ACE + Claude Code.

## Structure

- `.agent/` - Claude Code working files (notes, plans, todos - git ignored)
- Your project files go here

## Git Repository

This workspace is its own git repository. All work is committed here.

## Managed by ACE

This workspace is orchestrated by `ace_loop.py` which:
- Injects learned strategies into Claude Code's context
- Learns from execution to improve over time
