Run a TAU-bench evaluation end-to-end and present results.

**Syntax:** `/benchmark <config> [mode] [extra-args]`

**Arguments:**

- `config` (from $ARGUMENTS, first word) — profile name: `haiku`, `sonnet`, `gpt4.1-mini`, `gpt4.1`, `fast`, `default`
- `mode` (from $ARGUMENTS, second word, optional) — `baseline` (default), `compare`, or `ace-only`
- `extra-args` (from $ARGUMENTS, remaining words) — forwarded verbatim to the CLI

**Workflow:**

1. **Parse arguments** from `$ARGUMENTS`:
   - Split into: `config` (first word), `mode` (second word if it matches baseline/compare/ace-only, else default to baseline), and `extra-args` (the rest)

2. **Build the command:**
   ```
   uv run python scripts/run_tau_benchmark.py --config <config> --save-detailed <mode-flag> <extra-args>
   ```
   Mode flags:
   - `baseline` → `--skip-ace`
   - `compare` → `--compare`
   - `ace-only` → (no flag)

3. **Show the command** to the user before running

4. **Run the command** with a 10-minute timeout (TAU-bench runs are long)

5. **Find the latest result**: list `tau_benchmark_results/` sorted by modification time, pick the newest `*_summary.json`

6. **Read the summary JSON** and present results using this format:

   For baseline runs:
   ```
   ## <Mode>: <Model Short Name> — <Domain> (test split, k=<k>)

   | Setting | Value |
   |---------|-------|
   | Model | <exact model id> |
   | User LLM | <user_llm> |
   | Domain | <domain> |
   | Split | <split> (<N> tasks) |
   | Max steps | <max_steps> |
   | Seed | <seed> |

   | Metric | Score |
   |--------|-------|
   | pass^1 | XX.XX% |
   | pass^2 | XX.XX% |
   | ... | ... |
   ```

   For comparison runs, add Baseline / ACE / Delta columns.

**Examples:**
- `/benchmark haiku` → baseline haiku run
- `/benchmark haiku compare` → baseline vs ACE comparison
- `/benchmark fast` → quick smoke test (3 tasks, k=1)
- `/benchmark sonnet compare --domain retail` → sonnet comparison on retail

**Key fields** to always include in the results table: exact model ID, user LLM, domain, split + task count, skillbook status, and all pass^k metrics.
