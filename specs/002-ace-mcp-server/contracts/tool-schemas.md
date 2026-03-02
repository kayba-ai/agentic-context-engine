# MCP Tool Schemas: ACE MCP Server

**Feature**: `002-ace-mcp-server`  
**Date**: 2026-03-02  
**Status**: Draft Contract (MVP)

This document defines the canonical request/response schemas for MCP tools exposed by ACE.

## Shared Types

### SessionConfig

```json
{
  "type": "object",
  "properties": {
    "model": { "type": "string", "minLength": 1 },
    "temperature": { "type": "number", "minimum": 0, "maximum": 2 },
    "max_tokens": { "type": "integer", "minimum": 1 }
  },
  "additionalProperties": false
}
```

### ErrorEnvelope

```json
{
  "type": "object",
  "required": ["code", "message"],
  "properties": {
    "code": { "type": "string" },
    "message": { "type": "string" },
    "details": { "type": ["object", "null"], "additionalProperties": true }
  },
  "additionalProperties": false
}
```

---

## Tool: `ace.ask`

### Request

```json
{
  "type": "object",
  "required": ["session_id", "question"],
  "properties": {
    "session_id": { "type": "string", "minLength": 1 },
    "question": { "type": "string", "minLength": 1, "maxLength": 100000 },
    "context": { "type": "string", "default": "" },
    "session_config": { "$ref": "#/definitions/SessionConfig" },
    "metadata": { "type": "object", "additionalProperties": true }
  },
  "additionalProperties": false,
  "definitions": {
    "SessionConfig": {
      "type": "object",
      "properties": {
        "model": { "type": "string", "minLength": 1 },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2 },
        "max_tokens": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    }
  }
}
```

### Response

```json
{
  "type": "object",
  "required": ["session_id", "answer", "skill_count"],
  "properties": {
    "session_id": { "type": "string" },
    "answer": { "type": "string" },
    "skill_count": { "type": "integer", "minimum": 0 },
    "applied_skill_ids": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    }
  },
  "additionalProperties": false
}
```

---

## Tool: `ace.learn.sample`

### Request

```json
{
  "type": "object",
  "required": ["session_id", "samples"],
  "properties": {
    "session_id": { "type": "string", "minLength": 1 },
    "samples": {
      "type": "array",
      "minItems": 1,
      "maxItems": 25,
      "items": {
        "type": "object",
        "required": ["question"],
        "properties": {
          "question": { "type": "string", "minLength": 1 },
          "context": { "type": "string", "default": "" },
          "ground_truth": { "type": ["string", "null"], "default": null },
          "metadata": { "type": "object", "additionalProperties": true }
        },
        "additionalProperties": false
      }
    },
    "epochs": { "type": "integer", "minimum": 1, "maximum": 20, "default": 1 },
    "session_config": { "$ref": "#/definitions/SessionConfig" }
  },
  "additionalProperties": false,
  "definitions": {
    "SessionConfig": {
      "type": "object",
      "properties": {
        "model": { "type": "string", "minLength": 1 },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2 },
        "max_tokens": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    }
  }
}
```

### Response

```json
{
  "type": "object",
  "required": ["session_id", "processed", "skill_count_before", "skill_count_after"],
  "properties": {
    "session_id": { "type": "string" },
    "processed": { "type": "integer", "minimum": 0 },
    "failed": { "type": "integer", "minimum": 0, "default": 0 },
    "skill_count_before": { "type": "integer", "minimum": 0 },
    "skill_count_after": { "type": "integer", "minimum": 0 },
    "new_skill_count": { "type": "integer", "minimum": 0 }
  },
  "additionalProperties": false
}
```

---

## Tool: `ace.learn.feedback`

### Request

```json
{
  "type": "object",
  "required": ["session_id", "question", "answer", "feedback"],
  "properties": {
    "session_id": { "type": "string", "minLength": 1 },
    "question": { "type": "string", "minLength": 1 },
    "answer": { "type": "string", "minLength": 1 },
    "feedback": { "type": "string", "minLength": 1 },
    "context": { "type": "string", "default": "" },
    "ground_truth": { "type": ["string", "null"], "default": null },
    "session_config": { "$ref": "#/definitions/SessionConfig" }
  },
  "additionalProperties": false,
  "definitions": {
    "SessionConfig": {
      "type": "object",
      "properties": {
        "model": { "type": "string", "minLength": 1 },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2 },
        "max_tokens": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    }
  }
}
```

### Response

```json
{
  "type": "object",
  "required": ["session_id", "learned", "skill_count_before", "skill_count_after"],
  "properties": {
    "session_id": { "type": "string" },
    "learned": { "type": "boolean" },
    "skill_count_before": { "type": "integer", "minimum": 0 },
    "skill_count_after": { "type": "integer", "minimum": 0 },
    "new_skill_count": { "type": "integer", "minimum": 0 }
  },
  "additionalProperties": false
}
```

---

## Tool: `ace.skillbook.get`

### Request

```json
{
  "type": "object",
  "required": ["session_id"],
  "properties": {
    "session_id": { "type": "string", "minLength": 1 },
    "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 20 },
    "include_invalid": { "type": "boolean", "default": false }
  },
  "additionalProperties": false
}
```

### Response

```json
{
  "type": "object",
  "required": ["session_id", "stats", "skills"],
  "properties": {
    "session_id": { "type": "string" },
    "stats": { "type": "object", "additionalProperties": true },
    "skills": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "content"],
        "properties": {
          "id": { "type": "string" },
          "content": { "type": "string" },
          "topic": { "type": ["string", "null"] },
          "helpful": { "type": ["integer", "null"] },
          "harmful": { "type": ["integer", "null"] },
          "neutral": { "type": ["integer", "null"] }
        },
        "additionalProperties": true
      }
    }
  },
  "additionalProperties": false
}
```

---

## Tool: `ace.skillbook.save`

### Request

```json
{
  "type": "object",
  "required": ["session_id", "path"],
  "properties": {
    "session_id": { "type": "string", "minLength": 1 },
    "path": { "type": "string", "minLength": 1 }
  },
  "additionalProperties": false
}
```

Path policy: if server config sets `ACE_MCP_SKILLBOOK_ROOT`, `path` MUST resolve inside that directory; otherwise return `ACE_MCP_VALIDATION_ERROR`.

### Response

```json
{
  "type": "object",
  "required": ["session_id", "path", "saved_skill_count"],
  "properties": {
    "session_id": { "type": "string" },
    "path": { "type": "string" },
    "saved_skill_count": { "type": "integer", "minimum": 0 }
  },
  "additionalProperties": false
}
```

---

## Tool: `ace.skillbook.load`

### Request

```json
{
  "type": "object",
  "required": ["session_id", "path"],
  "properties": {
    "session_id": { "type": "string", "minLength": 1 },
    "path": { "type": "string", "minLength": 1 }
  },
  "additionalProperties": false
}
```

Path policy: if server config sets `ACE_MCP_SKILLBOOK_ROOT`, `path` MUST resolve inside that directory; otherwise return `ACE_MCP_VALIDATION_ERROR`.

### Response

```json
{
  "type": "object",
  "required": ["session_id", "path", "skill_count"],
  "properties": {
    "session_id": { "type": "string" },
    "path": { "type": "string" },
    "skill_count": { "type": "integer", "minimum": 0 }
  },
  "additionalProperties": false
}
```

---

## Safe Mode Policy Matrix

| Tool | Allowed in `safe_mode=true` |
|------|-----------------------------|
| `ace.ask` | ✅ |
| `ace.skillbook.get` | ✅ |
| `ace.learn.sample` | ❌ |
| `ace.learn.feedback` | ❌ |
| `ace.skillbook.save` | ❌ |
| `ace.skillbook.load` | ❌ |

Blocked calls must return `ACE_MCP_FORBIDDEN_IN_SAFE_MODE`.
