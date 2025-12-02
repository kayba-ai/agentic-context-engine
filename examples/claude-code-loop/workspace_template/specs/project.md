# Python to TypeScript Translation Project

Translate the ACE framework from Python to TypeScript.

## Source
Python code in `source/ace/` directory (cloned from agentic-context-engine repo)

## Target
TypeScript code in `target/ace/` directory

## Goals
- Maintain feature parity with Python version
- Use TypeScript best practices
- Include tests for each module
- Keep code clean and well-documented

## Git Workflow
This workspace is a git repository. After translating and testing each module:
1. Run tests to verify correctness
2. Commit with: `git add target/ && git commit -m "Translate module X"`
3. Continue to next module

## Validation Requirements

After ALL translation tasks are complete, the code must pass these checks:
1. **TypeScript Compilation**: `npm run build` - Must compile with strict mode, no errors
2. **Unit Tests**: `npm test` - All Jest tests must pass (25%+ coverage)
3. **Simple Example**: `npm run example:simple` - Basic ACE usage example must run successfully
4. **Seahorse Example**: `npm run example:seahorse` - Seahorse emoji challenge must run successfully

If any check fails, you'll receive error messages to fix before proceeding.

## Key Architectural Changes

### LiteLLM → Vercel AI SDK
Replace all LiteLLM calls with Vercel AI SDK:
- Use `@ai-sdk/anthropic` for Claude models
- Replace `litellm.completion()` with `generateText()` from `ai` package
- Handle async/await throughout (all LLM calls are async in TypeScript)

### Data Classes → TypeScript Interfaces
- Python `@dataclass` → TypeScript `interface` + factory functions
- Pydantic models → Zod schemas for runtime validation
- Maintain exact same data structure and field names

### File Organization
- Source: `src/` directory (not `ace/`)
- Tests: `tests/` directory with `.test.ts` files
- Examples: `examples/` directory with runnable `.ts` files

## Required Dependencies

Create `package.json` with these essential packages:

**Core**:
- `typescript` (^5.3.0) - TypeScript compiler
- `@ai-sdk/anthropic` or `@anthropic-ai/sdk` - For Claude API
- `ai` - Vercel AI SDK
- `zod` (^3.22.0) - Schema validation

**Testing**:
- `jest` (^29.0.0) - Test framework
- `ts-jest` (^29.0.0) - TypeScript support for Jest
- `@types/jest` - Jest type definitions

**Development**:
- `ts-node` (^10.9.0) - Run TypeScript directly
- `dotenv` (^16.0.0) - Environment variables

**Configuration Files Needed**:
- `tsconfig.json` - TypeScript config with strict mode
- `jest.config.js` - Jest configuration
- `.env.example` - Example environment variables (ANTHROPIC_API_KEY)
