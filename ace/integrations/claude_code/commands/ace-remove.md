Remove an ACE learned strategy.

First, show current strategies to find the ID:
```bash
ace-learn insights
```

Then ask the user which strategy to remove (by ID or keyword).

Remove it with:
```bash
ace-learn remove "<id-or-keyword>"
```

For example:
```bash
ace-learn remove "cli_debugging-00001"
```

Confirm the removal to the user.
