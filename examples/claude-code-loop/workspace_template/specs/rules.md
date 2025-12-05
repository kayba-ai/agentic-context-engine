# Coding Standards

## TypeScript Style Guide

### General Principles
- Write clean, readable code
- Follow TypeScript best practices
- Use meaningful variable and function names
- Add comments for complex logic

### Code Organization
- One class per file (when possible)
- Group related functionality together
- Use ES6 modules (import/export)
- Keep files under 300 lines when possible

### TypeScript Specifics
- Use strict mode (`"strict": true` in tsconfig.json)
- Prefer interfaces over type aliases for objects
- Use explicit return types for public functions
- Avoid `any` type - use `unknown` or proper types

### Testing Standards

- Write Jest tests for all public functions
- Use `describe`/`test` blocks for test organization
- Mock LLM calls in tests (don't make real API calls)
- Aim for 25%+ code coverage minimum
- Use descriptive test names that explain what's being tested
- Test edge cases and error conditions

**Jest Configuration Example:**
```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.ts'],
  collectCoverageFrom: ['src/**/*.ts'],
  coverageThreshold: {
    global: {
      statements: 25,
      branches: 25,
      functions: 25,
      lines: 25
    }
  }
};
```

**Test Structure Example:**
```typescript
// tests/skillbook.test.ts
import { Skillbook, Skill } from '../src/skillbook';

describe('Skillbook', () => {
  test('should create empty skillbook', () => {
    const skillbook = new Skillbook();
    expect(skillbook.getSkills()).toHaveLength(0);
  });

  test('should add skill successfully', () => {
    const skillbook = new Skillbook();
    const skill = new Skill('test-001', 'Test strategy');
    skillbook.addSkill(skill);
    expect(skillbook.getSkills()).toHaveLength(1);
  });
});
```

### Git Commits
- Write clear, descriptive commit messages
- Commit after each logical unit of work
- Test before committing
- Keep commits atomic (one feature/fix per commit)

## Translation Guidelines

### Naming Conventions
- Python snake_case → TypeScript camelCase for variables/functions
- Python PascalCase → TypeScript PascalCase for classes
- Preserve intent and meaning

### Type Annotations
- Add TypeScript types for all function parameters and return values
- Use generics where appropriate
- Create interfaces for complex objects

### Error Handling
- Convert Python exceptions to TypeScript try/catch
- Use custom Error classes when appropriate
- Preserve error messages and context

## LiteLLM → Vercel AI SDK Migration

The Python ACE framework uses LiteLLM for model calls. TypeScript version must use Vercel AI SDK instead.

### Basic LLM Call Migration

**Python (LiteLLM):**
```python
import litellm

response = litellm.completion(
    model="claude-sonnet-4-5-20250929",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7
)
text = response.choices[0].message.content
```

**TypeScript (Vercel AI SDK):**
```typescript
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const { text } = await generateText({
  model: anthropic('claude-sonnet-4-5-20250929'),
  messages: [
    { role: 'user', content: 'Hello, how are you?' }
  ],
  temperature: 0.7
});
```

### Key Differences

1. **Async/Await**: All LLM calls in TypeScript are async
   - Python: Synchronous by default
   - TypeScript: Must use `async`/`await` everywhere

2. **Import Structure**:
   - Python: `import litellm`
   - TypeScript: `import { generateText } from 'ai'; import { anthropic } from '@ai-sdk/anthropic';`

3. **Response Format**:
   - Python: `response.choices[0].message.content`
   - TypeScript: Destructure `{ text }` from result

4. **Error Handling**:
   - Python: `try/except`
   - TypeScript: `try/catch`

### LLM Client Interface Migration

**Python (ace/llm.py):**
```python
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
```

**TypeScript (src/llm.ts):**
```typescript
export interface LLMClient {
  generate(prompt: string, options?: Record<string, any>): Promise<string>;
}

export class AnthropicLLMClient implements LLMClient {
  async generate(prompt: string, options?: Record<string, any>): Promise<string> {
    const { text } = await generateText({
      model: anthropic('claude-sonnet-4-5-20250929'),
      prompt,
      ...options
    });
    return text;
  }
}
```

### Testing with Mock LLM

**Python:**
```python
class DummyLLMClient(LLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response"
```

**TypeScript:**
```typescript
export class MockLLMClient implements LLMClient {
  async generate(prompt: string, options?: Record<string, any>): Promise<string> {
    return 'Mock response';
  }
}

// In tests:
const mockLLM = new MockLLMClient();
```

### Important Notes

- **ALL** LLM-calling functions must be `async` in TypeScript
- Use `Promise<T>` return types for async functions
- Mock LLM calls in tests to avoid API costs
- Handle errors with try/catch blocks
- Use environment variables for API keys (`process.env.ANTHROPIC_API_KEY`)
