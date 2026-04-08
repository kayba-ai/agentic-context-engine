---
name: Always use Bedrock
description: Never use direct Anthropic API key or fall back to OpenAI — always use Bedrock via AWS_BEARER_TOKEN_BEDROCK
type: feedback
---

Always use Bedrock for LLM calls. Never use the Anthropic API key directly, never fall back to OpenAI or any other provider.

**Why:** The user has Bedrock configured with `AWS_BEARER_TOKEN_BEDROCK` and does not want direct Anthropic API usage (burns quota/money on the wrong account). Fallback logic is unacceptable — it silently uses the wrong provider.

**How to apply:** In integration tests and any code that needs an LLM model string, use `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` (or similar Bedrock model). Never write fallback chains like "if ANTHROPIC_KEY else OPENAI". Just use Bedrock, period.
