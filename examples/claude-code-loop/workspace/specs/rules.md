# TypeScript Coding Standards

## Type Safety Rules

1. **Strict Mode**: Always use strict TypeScript (`"strict": true`)
2. **No Any**: Avoid `any` type - use `unknown` and type guards instead
3. **Explicit Returns**: Always specify return types for functions
4. **Null Safety**: Use `null` and `undefined` appropriately, not interchangeably

## Code Style

1. **Naming Conventions**:
   - Classes: PascalCase (`class Playbook`)
   - Functions/methods: camelCase (`addBullet()`)
   - Constants: UPPER_SNAKE_CASE (`MAX_RETRIES`)
   - Private members: prefix with `_` (`_internalState`)

2. **File Organization**:
   - One class per file
   - Exports at bottom of file
   - Imports at top, grouped (external, internal, types)

3. **Documentation**:
   - JSDoc comments for public APIs
   - Inline comments for complex logic
   - README for each module

## Testing Standards

1. **Test Coverage**: Aim for >80% coverage
2. **Test Structure**: Arrange-Act-Assert pattern
3. **Test Files**: `*.test.ts` alongside source files
4. **Mocking**: Use vitest mocking utilities

## Error Handling

1. **Errors**: Extend `Error` class for custom errors
2. **Validation**: Use Zod for runtime validation
3. **Async Errors**: Always handle promise rejections

## Dependencies

1. **Minimal Dependencies**: Only add if necessary
2. **Type Definitions**: Always install `@types/*` packages
3. **Version Pinning**: Use exact versions in package.json
