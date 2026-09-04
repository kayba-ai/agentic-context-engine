# Decisions and current work

<!-- Written by `knos export`. Commit this file. -->

A second clone reads this on its first question - it is one of the decision
records knos looks for. Nothing here is private: secrets and private paths
never reach it.


## Decisions

- **pipeline-first is mandatory** - Do not write a function that calls multiple steps manually instead of composing them in a Pipeline, and do not inline reflection or evaluation logic instead of creating a `ReflectStep` or `EvaluateStep`.  _(AGENTS.md)_
- **no ad-hoc concurrency** - Use `async_boundary` and `max_workers` on steps rather than reaching for `ThreadPoolExecutor` directly.  _(AGENTS.md)_
- **respect requires/provides** - Do not bypass the contracts by accessing context fields a step has not declared in `requires`.  _(AGENTS.md)_
- **no standalone duplicates** - Scripts that duplicate pipeline functionality without using the pipeline engine are not accepted.  _(AGENTS.md)_
- **design decisions have a home** - `docs/design/ACE_DECISIONS.md` holds decisions and rejected alternatives; architecture is in `ACE_ARCHITECTURE.md`, the code reference in `ACE_REFERENCE.md`, the pipeline engine in `PIPELINE_DESIGN.md`.  _(AGENTS.md)_

## Being worked on right now

_Nothing claimed._

---
<sub>knos export. Claims lapse after 30 minutes.</sub>
