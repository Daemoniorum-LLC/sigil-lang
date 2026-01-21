# Sigil Playground Backend Integration Specification

## Overview

This specification defines the API contract between the Sigil Playground frontend and a backend execution service. The backend provides real Sigil code execution, type checking, and IR generation.

---

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │  HTTP   │                 │  exec   │                 │
│   Playground    │◄───────►│   API Server    │◄───────►│  Sigil Binary   │
│   (Browser)     │  JSON   │   (Node/Rust)   │         │                 │
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

### Components

1. **Playground Frontend** - Browser-based editor (this repo: `playground/`)
2. **API Server** - HTTP service that accepts code and returns results
3. **Sigil Binary** - The actual Sigil interpreter/compiler (`parser/target/release/sigil`)

---

## API Endpoints

### Base URL

```
Production: https://api.sigil-lang.org
Development: http://localhost:8080
```

### Common Headers

**Request:**
```
Content-Type: application/json
```

**Response:**
```
Content-Type: application/json
X-Request-Id: <uuid>
X-Execution-Time-Ms: <milliseconds>
```

---

## Endpoints

### POST /run

Execute Sigil code and return output.

**Request:**
```json
{
  "code": "fn main() {\n    print(\"Hello\");\n    return 0;\n}",
  "backend": "interpreter",
  "timeout_ms": 5000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | Yes | Sigil source code |
| `backend` | string | No | `"interpreter"` (default), `"jit"`, or `"llvm"` |
| `timeout_ms` | integer | No | Execution timeout (default: 5000, max: 30000) |

**Response (Success):**
```json
{
  "success": true,
  "output": "Hello\n",
  "exit_code": 0,
  "execution_time_ms": 12.5,
  "memory_used_bytes": 1048576
}
```

**Response (Runtime Error):**
```json
{
  "success": false,
  "output": "partial output before error\n",
  "error": {
    "type": "runtime",
    "message": "Division by zero",
    "location": {
      "line": 5,
      "column": 12
    }
  },
  "execution_time_ms": 8.2
}
```

**Response (Timeout):**
```json
{
  "success": false,
  "error": {
    "type": "timeout",
    "message": "Execution exceeded 5000ms limit"
  }
}
```

---

### POST /check

Type-check code with evidentiality enforcement.

**Request:**
```json
{
  "code": "fn main() {\n    let x: int! = get_input()~;\n}",
  "format": "detailed"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | Yes | Sigil source code |
| `format` | string | No | `"simple"` (default) or `"detailed"` |

**Response (No Errors):**
```json
{
  "success": true,
  "errors": [],
  "warnings": [],
  "info": {
    "functions": 1,
    "lines": 3,
    "evidence_annotations": 2
  }
}
```

**Response (With Errors):**
```json
{
  "success": false,
  "errors": [
    {
      "code": "E0003",
      "severity": "error",
      "message": "evidence mismatch: expected known (!), found reported (~)",
      "location": {
        "line": 2,
        "column": 18,
        "length": 12
      },
      "hint": "Use |validate?{...} to promote reported data to uncertain",
      "related": [
        {
          "message": "function expects known evidence here",
          "location": { "line": 2, "column": 8 }
        }
      ]
    }
  ],
  "warnings": []
}
```

---

### POST /ir

Generate AI-readable intermediate representation.

**Request:**
```json
{
  "code": "fn add(a: int, b: int) -> int { a + b }",
  "format": "json"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | Yes | Sigil source code |
| `format` | string | No | `"json"` (default) or `"compact"` |

**Response:**
```json
{
  "success": true,
  "ir": {
    "version": "1.0.0",
    "source": "playground",
    "functions": [
      {
        "name": "add",
        "id": "fn_001",
        "params": [
          { "name": "a", "type": { "kind": "int", "size": 64 } },
          { "name": "b", "type": { "kind": "int", "size": 64 } }
        ],
        "return_type": { "kind": "int", "size": 64 },
        "body": {
          "kind": "binary",
          "op": "add",
          "left": { "kind": "var", "name": "a" },
          "right": { "kind": "var", "name": "b" }
        }
      }
    ]
  }
}
```

---

### POST /format

Format Sigil code.

**Request:**
```json
{
  "code": "fn main(){let x=1;print(x);return 0;}",
  "options": {
    "indent_size": 4,
    "max_line_length": 100
  }
}
```

**Response:**
```json
{
  "success": true,
  "formatted": "fn main() {\n    let x = 1;\n    print(x);\n    return 0;\n}"
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "sigil_version": "0.1.0",
  "backends": ["interpreter", "jit"],
  "uptime_seconds": 3600
}
```

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "success": false,
  "error": {
    "type": "parse_error | type_error | runtime | timeout | internal",
    "message": "Human-readable error message",
    "code": "E0001",
    "location": {
      "line": 1,
      "column": 5,
      "length": 10
    }
  }
}
```

### Error Types

| Type | HTTP Status | Description |
|------|-------------|-------------|
| `parse_error` | 200 | Syntax error in code |
| `type_error` | 200 | Type checking failed |
| `runtime` | 200 | Runtime error during execution |
| `timeout` | 200 | Execution exceeded time limit |
| `rate_limit` | 429 | Too many requests |
| `invalid_request` | 400 | Malformed request |
| `internal` | 500 | Server error |

---

## Rate Limiting

| Limit | Value |
|-------|-------|
| Requests per minute | 60 |
| Requests per hour | 500 |
| Max code size | 64 KB |
| Max execution time | 30 seconds |
| Max memory | 256 MB |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1701590400
```

---

## Security

### Sandboxing

All code execution MUST be sandboxed:

1. **Container isolation** - Each execution runs in an isolated container
2. **No network access** - Sandboxed code cannot make network requests
3. **No filesystem access** - Only `/tmp` with size limits
4. **Resource limits** - CPU, memory, and time limits enforced
5. **Syscall filtering** - seccomp or similar

### Recommended Technologies

- **gVisor** - User-space kernel for container isolation
- **Firecracker** - Lightweight microVMs
- **nsjail** - Process isolation with namespaces

### Input Validation

- Maximum code length: 64 KB
- UTF-8 validation required
- Strip null bytes
- Sanitize error messages (no path disclosure)

---

## CORS Configuration

```
Access-Control-Allow-Origin: https://sigil-lang.org, http://localhost:*
Access-Control-Allow-Methods: POST, GET, OPTIONS
Access-Control-Allow-Headers: Content-Type
Access-Control-Max-Age: 86400
```

---

## WebSocket API (Optional)

For real-time output streaming during long-running executions:

### Connection

```
wss://api.sigil-lang.org/ws
```

### Messages

**Client → Server (Execute):**
```json
{
  "type": "run",
  "id": "req_123",
  "code": "...",
  "backend": "interpreter"
}
```

**Server → Client (Output):**
```json
{
  "type": "output",
  "id": "req_123",
  "data": "Hello\n"
}
```

**Server → Client (Complete):**
```json
{
  "type": "complete",
  "id": "req_123",
  "exit_code": 0,
  "execution_time_ms": 150
}
```

---

## Implementation Reference

### Node.js (Express)

```javascript
const express = require('express');
const { execFile } = require('child_process');
const { writeFile, unlink } = require('fs/promises');
const { v4: uuid } = require('uuid');

const app = express();
app.use(express.json({ limit: '64kb' }));

const SIGIL_BIN = process.env.SIGIL_BIN || 'sigil';
const TIMEOUT_MS = 5000;

app.post('/run', async (req, res) => {
  const { code, backend = 'interpreter', timeout_ms = TIMEOUT_MS } = req.body;
  const tempFile = `/tmp/sigil-${uuid()}.sigil`;

  try {
    await writeFile(tempFile, code);
    const command = backend === 'jit' ? 'jit' : 'run';

    const start = Date.now();
    execFile(SIGIL_BIN, [command, tempFile], {
      timeout: Math.min(timeout_ms, 30000),
      maxBuffer: 1024 * 1024,
    }, (error, stdout, stderr) => {
      const executionTime = Date.now() - start;
      unlink(tempFile).catch(() => {});

      if (error?.killed) {
        return res.json({
          success: false,
          error: { type: 'timeout', message: `Exceeded ${timeout_ms}ms limit` }
        });
      }

      res.json({
        success: !error,
        output: stdout,
        error: error ? { type: 'runtime', message: stderr } : undefined,
        execution_time_ms: executionTime
      });
    });
  } catch (e) {
    res.status(500).json({
      success: false,
      error: { type: 'internal', message: 'Server error' }
    });
  }
});
```

### Rust (Axum)

```rust
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

#[derive(Deserialize)]
struct RunRequest {
    code: String,
    backend: Option<String>,
    timeout_ms: Option<u64>,
}

#[derive(Serialize)]
struct RunResponse {
    success: bool,
    output: Option<String>,
    error: Option<ErrorInfo>,
    execution_time_ms: f64,
}

async fn run_code(Json(req): Json<RunRequest>) -> Json<RunResponse> {
    let temp_file = NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), &req.code).unwrap();

    let backend = req.backend.unwrap_or_else(|| "run".to_string());
    let start = Instant::now();

    let output = Command::new("sigil")
        .arg(&backend)
        .arg(temp_file.path())
        .output()
        .expect("Failed to execute");

    let execution_time = start.elapsed().as_secs_f64() * 1000.0;

    Json(RunResponse {
        success: output.status.success(),
        output: Some(String::from_utf8_lossy(&output.stdout).to_string()),
        error: if !output.status.success() {
            Some(ErrorInfo {
                r#type: "runtime".to_string(),
                message: String::from_utf8_lossy(&output.stderr).to_string(),
            })
        } else { None },
        execution_time_ms: execution_time,
    })
}
```

---

## Deployment Checklist

- [ ] Set up container orchestration (Kubernetes, ECS, etc.)
- [ ] Configure sandboxing (gVisor, Firecracker)
- [ ] Set up rate limiting (Redis, in-memory)
- [ ] Configure CORS for playground domains
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling for execution workers
- [ ] Set up CDN for static assets
- [ ] Configure SSL/TLS
- [ ] Set up logging and tracing
- [ ] Load testing (target: 100 req/s sustained)

---

## Future Enhancements

1. **Persistent sessions** - Save code to user accounts
2. **Collaborative editing** - Real-time multiplayer
3. **Package support** - Import external Sigil modules
4. **Debugging** - Step-through execution with breakpoints
5. **Profiling** - Performance analysis of code
6. **AI assistance** - Integrated code suggestions

---

*Version: 1.0.0*
*Last Updated: December 2024*
