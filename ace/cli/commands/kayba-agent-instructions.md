## Kayba CLI

The `kayba` CLI interacts with the Kayba hosted API (https://use.kayba.ai).
Auth: set `KAYBA_API_KEY` env var or pass `--api-key` to every command.

### Commands

```
kayba upload <paths...>                  Upload trace files/dirs (or - for stdin)
  --type [md|json|txt]                   Force file type (auto-detected by default)

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

kayba status <job-id>                    Check job status
  --wait  --interval N

kayba materialize <job-id>               Materialize results into skillbook

kayba batch <paths...>                   Pre-batch traces for Recursive Reflector
  --apply FILE  --upload  --min-batch-size N  --max-batch-size N
```

### Typical workflow

```
kayba upload traces/
kayba insights generate --wait
kayba insights triage --accept-all
kayba prompts generate -o prompt.md
```
