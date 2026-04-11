## Kayba CLI

The `kayba` CLI interacts with the Kayba hosted API (https://use.kayba.ai).
Auth: set `KAYBA_API_KEY` env var or pass `--api-key` to every command.

### Commands

```
kayba traces list [--json]               List uploaded traces
kayba traces show <id> [--meta] [--json] View a trace
kayba traces upload <paths...>           Upload trace files/dirs (or - for stdin)
  --type [md|json|txt]                   Force file type (auto-detected by default)
kayba traces delete <ids...> [--force]   Delete traces

kayba run                                Run pipeline (interactive trace selector)
  --traces ID  --all  --model MODEL  --epochs N
  --reflector-mode [recursive|standard]  --anthropic-key KEY
  --wait  --json

kayba insights generate                  Trigger insight generation
  --traces ID  --model MODEL  --epochs N  --reflector-mode [recursive|standard]
  --anthropic-key KEY  --wait

kayba insights list                      List insights
  --status [pending|new|accepted|rejected]  --section NAME  --json

kayba insights triage                    Accept/reject insights
  --accept ID  --reject ID  --accept-all  --note TEXT

kayba prompts generate                   Generate prompt from accepted insights
  --insights ID  --label NAME  -o FILE

kayba prompts list                       List prompt versions

kayba prompts pull                       Download a prompt
  --id ID  -o FILE  --pretty
kayba prompts install                   Install a generated prompt into an agent file
  --target TARGET  --file PATH  --id ID  --input FILE

kayba status <job-id>                    Check job status
  --wait  --interval N

kayba materialize <job-id>               Materialize results into skillbook

kayba integrations list [--json]         Show configured integrations
kayba integrations configure <name>      Configure mlflow or langsmith
kayba integrations test <name>           Test integration connection

kayba batch <paths...>                   Pre-batch traces for Recursive Reflector
  --apply FILE  --upload  --min-batch-size N  --max-batch-size N
```

### Typical workflow

```
kayba traces upload traces/
kayba run --all --wait
kayba insights triage --accept-all
kayba prompts generate -o prompt.md
kayba prompts install --target claude-code
```

### Programmatic workflow (for agents)

```
kayba traces list --json
kayba run --traces ID1 --traces ID2 --json --wait
kayba insights triage --accept-all
kayba prompts generate -o prompt.md
```
