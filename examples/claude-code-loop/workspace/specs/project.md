# ACE TypeScript Translation Project

## Overview

Translate the **ACE (Agentic Context Engine)** framework from Python to TypeScript.

**Source:** `workspace/source/` (Python ACE - symlinked)
**Target:** `workspace/target/` (TypeScript output)

## Goal

Create a functionally equivalent TypeScript implementation of ACE that:
- Maintains the same API surface as Python version
- Works with Node.js >= 18
- Uses modern TypeScript best practices
- Can be published as npm package

## Source Files to Translate

Priority order:

1. **`ace/playbook.py`** → `target/src/playbook.ts`
   - Bullet & Playbook data structures
   - JSON serialization

2. **`ace/delta.py`** → `target/src/delta.ts`
   - DeltaOperation & DeltaBatch
   - Operation types: ADD, UPDATE, TAG, REMOVE

3. **`ace/llm.py`** → `target/src/llm.ts`
   - LLMClient interface
   - DummyLLMClient for testing

4. **`ace/roles.py`** → `target/src/roles.ts`
   - Generator, Reflector, Curator
   - ReplayGenerator

5. **`ace/adaptation.py`** → `target/src/adaptation.ts`
   - OnlineAdapter, OfflineAdapter
   - Learning loop logic

## Translation Guidelines

### Type Safety
- Use strict TypeScript mode
- No `any` types (use `unknown` if needed)
- Proper type inference
- Zod schemas for runtime validation

### Code Structure
- Follow Python's class structure closely
- Use ES6 classes
- Async/await for async operations
- Proper error handling

### Testing
- Use vitest for testing
- Aim for similar test coverage as Python
- Port existing Python tests

### Dependencies
- Minimize external dependencies
- Use standard Node.js APIs where possible
- Match Python library choices (e.g., use `dotenv` like Python's `python-dotenv`)

## Success Criteria

- All TypeScript compiles with strict mode
- All tests pass
- Functionally equivalent to Python version
- Can run the same examples as Python ACE
