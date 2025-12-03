# Sigil AI-Facing IR Specification

This document defines the JSON Intermediate Representation (IR) format designed for AI agents and tooling to consume, generate, and reason about Sigil programs.

## Design Goals

1. **AI-Readable**: Structured for LLMs to parse, understand, and generate
2. **Complete**: Captures all semantic information from the AST
3. **Self-Describing**: Types embedded in the IR, not requiring external context
4. **Bidirectional**: Can be compiled back to Sigil source or forward to LLVM/Cranelift
5. **Evidentiality-Aware**: First-class representation of the evidence lattice

---

## 1. Top-Level IR Structure

```json
{
  "$schema": "https://sigil-lang.dev/schemas/ir/v1.json",
  "version": "1.0.0",
  "source": {
    "file": "example.sigil",
    "hash": "sha256:abc123...",
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "module": {
    "name": "example",
    "imports": [...],
    "exports": [...],
    "items": [...]
  },
  "types": { ... },
  "evidence_constraints": [ ... ],
  "metadata": { ... }
}
```

### 1.1 Module Structure

```json
{
  "module": {
    "name": "persona_builder",
    "imports": [
      { "path": "std::collections", "items": ["HashMap", "Vec"] },
      { "path": "protocols::http", "items": ["*"] }
    ],
    "exports": ["build_persona_sigil", "PersonaIntent", "Sigil"],
    "items": [
      { "kind": "struct", ... },
      { "kind": "function", ... },
      { "kind": "trait", ... },
      { "kind": "impl", ... }
    ]
  }
}
```

---

## 2. Type System Representation

### 2.1 Core Type Schema

```json
{
  "types": {
    "PersonaIntent": {
      "kind": "struct",
      "fields": [
        { "name": "name", "type": { "kind": "primitive", "name": "str" } },
        { "name": "traits", "type": { "kind": "array", "element": { "kind": "primitive", "name": "str" } } },
        { "name": "context", "type": { "kind": "evidential", "inner": { "kind": "primitive", "name": "str" }, "evidence": "reported" } }
      ]
    },
    "Sigil": {
      "kind": "struct",
      "fields": [
        { "name": "geometry", "type": { "kind": "path", "name": "Geometry" } },
        { "name": "features", "type": { "kind": "path", "name": "FeatureVector" } }
      ]
    }
  }
}
```

### 2.2 Type Kinds

| Kind | JSON Representation |
|------|---------------------|
| Primitive | `{ "kind": "primitive", "name": "i64" }` |
| Array | `{ "kind": "array", "element": <Type>, "size": null }` |
| Tuple | `{ "kind": "tuple", "elements": [<Type>, ...] }` |
| Function | `{ "kind": "function", "params": [...], "return": <Type>, "async": false }` |
| Evidential | `{ "kind": "evidential", "inner": <Type>, "evidence": "known" }` |
| Reference | `{ "kind": "ref", "mutable": false, "inner": <Type> }` |
| Generic | `{ "kind": "generic", "name": "T", "bounds": [...] }` |
| Path | `{ "kind": "path", "name": "MyStruct", "args": [...] }` |
| Option | `{ "kind": "option", "inner": <Type> }` |
| Result | `{ "kind": "result", "ok": <Type>, "err": <Type> }` |
| Never | `{ "kind": "never" }` |
| Unit | `{ "kind": "unit" }` |

### 2.3 Evidentiality Lattice

The evidence lattice represents data provenance and trust levels:

```
        Paradox (‽)     ← Contradictory/unsafe boundary
           ↑
       Reported (~)     ← External/network data
           ↑
      Uncertain (?)     ← May be absent, inferred
           ↑
        Known (!)       ← Verified, computed locally
```

**JSON Representation:**

```json
{
  "evidence_lattice": {
    "levels": ["known", "uncertain", "reported", "paradox"],
    "order": {
      "known": 0,
      "uncertain": 1,
      "reported": 2,
      "paradox": 3
    },
    "join": "max",
    "meet": "min"
  }
}
```

**Type with Evidence:**

```json
{
  "kind": "evidential",
  "inner": { "kind": "primitive", "name": "str" },
  "evidence": "reported",
  "source": {
    "origin": "http_fetch",
    "url": "https://api.example.com/data",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### 2.4 Evidence Constraints

```json
{
  "evidence_constraints": [
    {
      "location": { "fn": "process_data", "param": 0 },
      "required": "known",
      "actual": "reported",
      "resolution": "validation_required",
      "suggested_validator": "validate_user_input"
    }
  ]
}
```

---

## 3. Function IR

### 3.1 Function Definition

```json
{
  "kind": "function",
  "name": "build_persona_sigil",
  "visibility": "public",
  "generics": [
    { "name": "T", "bounds": ["Serialize", "Clone"] }
  ],
  "params": [
    {
      "name": "intent",
      "type": { "kind": "path", "name": "PersonaIntent" },
      "evidence": "known"
    }
  ],
  "return_type": {
    "kind": "tuple",
    "elements": [
      { "kind": "evidential", "inner": { "kind": "path", "name": "Sigil" }, "evidence": "known" },
      { "kind": "evidential", "inner": { "kind": "path", "name": "FeatureVector" }, "evidence": "known" }
    ]
  },
  "async": false,
  "body": { ... },
  "attributes": ["#[inline]", "#[must_use]"],
  "span": { "start": 42, "end": 156, "line": 10, "column": 1 }
}
```

### 3.2 Function Body

```json
{
  "body": {
    "kind": "block",
    "statements": [...],
    "tail_expr": { ... },
    "evidence_out": "known"
  }
}
```

---

## 4. Expression IR

### 4.1 Expression Kinds

Every expression has:
- `kind`: The expression type
- `type`: Inferred/declared type (optional in input, required in output)
- `evidence`: Evidence level (optional in input, inferred in output)
- `span`: Source location
- `id`: Unique expression ID for cross-references

### 4.2 Literals

```json
{ "kind": "literal", "value": { "int": 42 }, "type": { "kind": "primitive", "name": "i64" } }
{ "kind": "literal", "value": { "float": 3.14 }, "type": { "kind": "primitive", "name": "f64" } }
{ "kind": "literal", "value": { "string": "hello" }, "type": { "kind": "primitive", "name": "str" } }
{ "kind": "literal", "value": { "bool": true }, "type": { "kind": "primitive", "name": "bool" } }
{ "kind": "literal", "value": { "char": "x" }, "type": { "kind": "primitive", "name": "char" } }
```

### 4.3 Binary Operations

```json
{
  "kind": "binary",
  "op": "+",
  "left": { "kind": "path", "name": "x" },
  "right": { "kind": "literal", "value": { "int": 1 } },
  "type": { "kind": "primitive", "name": "i64" },
  "evidence": "known"
}
```

**Supported Operators:**

| Category | Operators |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `%`, `**` (power) |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Logical | `&&`, `\|\|`, `!` |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |
| Set | `∪` (union), `∩` (intersect), `∖` (diff), `⊂` (subset) |
| Lattice | `⊔` (join), `⊓` (meet) |
| Compose | `∘` (compose), `⊗` (tensor), `⊕` (direct_sum) |

### 4.4 Function Call

```json
{
  "kind": "call",
  "func": { "kind": "path", "name": "process_data" },
  "args": [
    { "kind": "path", "name": "input" },
    { "kind": "literal", "value": { "int": 100 } }
  ],
  "type_args": [{ "kind": "path", "name": "String" }],
  "evidence": "uncertain"
}
```

### 4.5 Method Call

```json
{
  "kind": "method_call",
  "receiver": { "kind": "path", "name": "collection" },
  "method": "filter",
  "args": [
    { "kind": "closure", ... }
  ],
  "evidence": "known"
}
```

---

## 5. Pipeline IR

Pipelines are the core of Sigil's data transformation model.

### 5.1 Pipeline Expression

```json
{
  "kind": "pipeline",
  "source": { "kind": "path", "name": "data" },
  "stages": [
    { "op": "filter", "predicate": { ... } },
    { "op": "transform", "mapper": { ... } },
    { "op": "sort", "key": "name", "descending": false },
    { "op": "reduce", "reducer": { ... }, "initial": { ... } }
  ],
  "type": { "kind": "array", "element": { "kind": "path", "name": "Result" } },
  "evidence": "known"
}
```

### 5.2 Pipeline Operations (Morphemes)

#### Transform (τ / tau)

```json
{
  "op": "transform",
  "symbol": "τ",
  "mapper": {
    "kind": "closure",
    "params": [{ "name": "_", "type": null }],
    "body": {
      "kind": "binary",
      "op": "*",
      "left": { "kind": "path", "name": "_" },
      "right": { "kind": "literal", "value": { "int": 2 } }
    }
  },
  "evidence_propagation": "preserve"
}
```

#### Filter (φ / phi)

```json
{
  "op": "filter",
  "symbol": "φ",
  "predicate": {
    "kind": "closure",
    "params": [{ "name": "x", "type": null }],
    "body": {
      "kind": "binary",
      "op": ">",
      "left": { "kind": "path", "name": "x" },
      "right": { "kind": "literal", "value": { "int": 0 } }
    }
  },
  "evidence_propagation": "preserve"
}
```

#### Sort (σ / sigma)

```json
{
  "op": "sort",
  "symbol": "σ",
  "key": { "kind": "field_access", "field": "age" },
  "descending": false,
  "stable": true
}
```

#### Reduce (ρ / rho)

```json
{
  "op": "reduce",
  "symbol": "ρ",
  "reducer": {
    "kind": "closure",
    "params": [
      { "name": "acc", "type": null },
      { "name": "x", "type": null }
    ],
    "body": {
      "kind": "binary",
      "op": "+",
      "left": { "kind": "path", "name": "acc" },
      "right": { "kind": "path", "name": "x" }
    }
  },
  "initial": { "kind": "literal", "value": { "int": 0 } },
  "evidence_propagation": "join"
}
```

#### Aggregation Morphemes

```json
{ "op": "sum", "symbol": "Σ" }
{ "op": "product", "symbol": "Π" }
{ "op": "first", "symbol": "α" }
{ "op": "last", "symbol": "ω" }
{ "op": "middle", "symbol": "μ" }
{ "op": "choice", "symbol": "χ", "seed": null }
{ "op": "nth", "symbol": "ν", "index": { "kind": "literal", "value": { "int": 5 } } }
{ "op": "next", "symbol": "ξ" }
```

### 5.3 Parallel Execution

```json
{
  "op": "parallel",
  "symbol": "∥",
  "inner_op": {
    "op": "transform",
    "mapper": { ... }
  },
  "thread_pool": "default",
  "chunk_size": null
}
```

### 5.4 GPU Execution

```json
{
  "op": "gpu",
  "symbol": "⊛",
  "inner_op": {
    "op": "transform",
    "mapper": { ... }
  },
  "workgroup_size": [256, 1, 1],
  "shader_hint": "compute"
}
```

### 5.5 Fork/Join (Branching Pipelines)

```json
{
  "kind": "pipeline",
  "source": { "kind": "path", "name": "input" },
  "stages": [
    { "op": "call", "fn": "intent_to_proto" },
    { "op": "call", "fn": "enrich_with_geometry" },
    {
      "op": "fork",
      "branches": [
        {
          "name": "features",
          "stages": [{ "op": "call", "fn": "sigil_to_features" }]
        },
        {
          "name": "identity",
          "stages": [{ "op": "identity" }]
        }
      ],
      "join_strategy": "tuple"
    }
  ],
  "type": {
    "kind": "tuple",
    "elements": [
      { "kind": "evidential", "inner": { "kind": "path", "name": "Sigil" }, "evidence": "known" },
      { "kind": "evidential", "inner": { "kind": "path", "name": "FeatureVector" }, "evidence": "known" }
    ]
  }
}
```

---

## 6. Incorporation IR (Noun·Verb·Verb)

```json
{
  "kind": "incorporation",
  "segments": [
    { "role": "noun", "value": { "kind": "path", "name": "File" } },
    { "role": "verb", "value": "open", "args": [{ "kind": "literal", "value": { "string": "data.txt" } }] },
    { "role": "verb", "value": "read", "args": [] },
    { "role": "verb", "value": "close", "args": [] }
  ],
  "evidence": "uncertain"
}
```

---

## 7. Protocol Operations IR

### 7.1 HTTP Request

```json
{
  "kind": "protocol",
  "protocol": "http",
  "operation": "request",
  "config": {
    "method": "POST",
    "url": { "kind": "path", "name": "api_endpoint" },
    "headers": [
      { "name": "Content-Type", "value": { "kind": "literal", "value": { "string": "application/json" } } },
      { "name": "Authorization", "value": { "kind": "path", "name": "auth_token" } }
    ],
    "body": { "kind": "call", "func": { "kind": "path", "name": "json_encode" }, "args": [{ "kind": "path", "name": "payload" }] },
    "timeout": { "connect": 5000, "read": 30000, "total": 60000 },
    "retry": { "max_attempts": 3, "backoff": "exponential" }
  },
  "type": {
    "kind": "result",
    "ok": { "kind": "evidential", "inner": { "kind": "path", "name": "HttpResponse" }, "evidence": "reported" },
    "err": { "kind": "path", "name": "HttpError" }
  }
}
```

### 7.2 gRPC Call

```json
{
  "kind": "protocol",
  "protocol": "grpc",
  "operation": "call",
  "config": {
    "service": { "kind": "literal", "value": { "string": "persona.PersonaService" } },
    "method": { "kind": "literal", "value": { "string": "CreatePersona" } },
    "message": { "kind": "path", "name": "request_proto" },
    "metadata": [
      { "key": "x-request-id", "value": { "kind": "path", "name": "trace_id" } }
    ],
    "timeout": 30000
  },
  "evidence": "reported"
}
```

### 7.3 WebSocket

```json
{
  "kind": "protocol",
  "protocol": "websocket",
  "operation": "connect",
  "config": {
    "url": { "kind": "literal", "value": { "string": "wss://stream.example.com" } },
    "protocols": ["graphql-ws"],
    "headers": []
  },
  "evidence": "reported"
}
```

### 7.4 Kafka

```json
{
  "kind": "protocol",
  "protocol": "kafka",
  "operation": "produce",
  "config": {
    "topic": { "kind": "literal", "value": { "string": "events" } },
    "key": { "kind": "path", "name": "event_id" },
    "payload": { "kind": "path", "name": "event_data" },
    "partition": null,
    "headers": []
  },
  "evidence": "reported"
}
```

### 7.5 Protocol in Pipeline

```json
{
  "kind": "pipeline",
  "source": { "kind": "literal", "value": { "string": "https://api.example.com/users" } },
  "stages": [
    { "op": "protocol", "protocol": "http", "operation": "get" },
    { "op": "protocol_config", "config": "header", "name": "Accept", "value": "application/json" },
    { "op": "protocol_config", "config": "timeout", "ms": 5000 },
    { "op": "protocol_config", "config": "retry", "count": 3, "strategy": "exponential" },
    { "op": "await" },
    { "op": "validate", "schema": "UserListResponse", "promote_evidence": "known" }
  ],
  "evidence": "known"
}
```

---

## 8. Control Flow IR

### 8.1 If Expression

```json
{
  "kind": "if",
  "condition": { "kind": "binary", "op": ">", "left": { "kind": "path", "name": "x" }, "right": { "kind": "literal", "value": { "int": 0 } } },
  "then_branch": { "kind": "block", "statements": [], "tail_expr": { "kind": "literal", "value": { "string": "positive" } } },
  "else_branch": { "kind": "block", "statements": [], "tail_expr": { "kind": "literal", "value": { "string": "non-positive" } } },
  "evidence": "known"
}
```

### 8.2 Match Expression

```json
{
  "kind": "match",
  "scrutinee": { "kind": "path", "name": "result" },
  "arms": [
    {
      "pattern": { "kind": "variant", "enum": "Result", "variant": "Ok", "bindings": ["value"] },
      "guard": null,
      "body": { "kind": "path", "name": "value" }
    },
    {
      "pattern": { "kind": "variant", "enum": "Result", "variant": "Err", "bindings": ["e"] },
      "guard": null,
      "body": { "kind": "call", "func": { "kind": "path", "name": "handle_error" }, "args": [{ "kind": "path", "name": "e" }] }
    }
  ],
  "evidence": "uncertain"
}
```

### 8.3 Loop Constructs

```json
{
  "kind": "for",
  "pattern": { "kind": "binding", "name": "item" },
  "iter": { "kind": "path", "name": "collection" },
  "body": { "kind": "block", ... },
  "label": null
}

{
  "kind": "while",
  "condition": { "kind": "binary", "op": "<", "left": { "kind": "path", "name": "i" }, "right": { "kind": "literal", "value": { "int": 10 } } },
  "body": { "kind": "block", ... },
  "label": null
}

{
  "kind": "loop",
  "body": { "kind": "block", ... },
  "label": "outer"
}
```

---

## 9. Affective Markers IR

```json
{
  "kind": "affective",
  "inner": { "kind": "literal", "value": { "string": "Great job!" } },
  "affect": {
    "sentiment": { "kind": "positive", "symbol": "⊕" },
    "intensity": { "kind": "high", "symbol": "↑" },
    "emotion": { "kind": "joy", "symbol": "☺" },
    "confidence": { "kind": "high", "symbol": "◉" },
    "formality": { "kind": "informal", "symbol": "♟" },
    "sarcasm": false
  }
}
```

---

## 10. Complete Example: Persona Builder

### 10.1 Source Code

```sigil
fn build_persona_sigil(intent: PersonaIntent!) -> (E<Sigil>!, E<FeatureVector>!) {
    intent
        |intent_to_proto
        |enrich_with_geometry
        |fork {
            |sigil_to_features,
            |identity
        }
}
```

### 10.2 JSON IR

```json
{
  "$schema": "https://sigil-lang.dev/schemas/ir/v1.json",
  "version": "1.0.0",
  "source": {
    "file": "persona_builder.sigil",
    "hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "module": {
    "name": "persona_builder",
    "imports": [],
    "exports": ["build_persona_sigil"],
    "items": [
      {
        "kind": "function",
        "id": "fn_build_persona_sigil",
        "name": "build_persona_sigil",
        "visibility": "public",
        "generics": [],
        "params": [
          {
            "name": "intent",
            "type": {
              "kind": "evidential",
              "inner": { "kind": "path", "name": "PersonaIntent" },
              "evidence": "known"
            }
          }
        ],
        "return_type": {
          "kind": "tuple",
          "elements": [
            {
              "kind": "evidential",
              "inner": { "kind": "path", "name": "Sigil" },
              "evidence": "known"
            },
            {
              "kind": "evidential",
              "inner": { "kind": "path", "name": "FeatureVector" },
              "evidence": "known"
            }
          ]
        },
        "async": false,
        "body": {
          "kind": "pipeline",
          "id": "pipe_0",
          "source": {
            "kind": "path",
            "name": "intent",
            "id": "expr_0"
          },
          "stages": [
            {
              "op": "call",
              "fn": "intent_to_proto",
              "args": [],
              "id": "stage_0",
              "evidence_in": "known",
              "evidence_out": "known"
            },
            {
              "op": "call",
              "fn": "enrich_with_geometry",
              "args": [],
              "id": "stage_1",
              "evidence_in": "known",
              "evidence_out": "known"
            },
            {
              "op": "fork",
              "id": "stage_2",
              "branches": [
                {
                  "name": "branch_0",
                  "stages": [
                    {
                      "op": "call",
                      "fn": "sigil_to_features",
                      "args": [],
                      "id": "stage_2_0",
                      "evidence_in": "known",
                      "evidence_out": "known"
                    }
                  ],
                  "type": {
                    "kind": "evidential",
                    "inner": { "kind": "path", "name": "FeatureVector" },
                    "evidence": "known"
                  }
                },
                {
                  "name": "branch_1",
                  "stages": [
                    {
                      "op": "identity",
                      "id": "stage_2_1"
                    }
                  ],
                  "type": {
                    "kind": "evidential",
                    "inner": { "kind": "path", "name": "Sigil" },
                    "evidence": "known"
                  }
                }
              ],
              "join_strategy": "tuple",
              "evidence_in": "known",
              "evidence_out": "known"
            }
          ],
          "type": {
            "kind": "tuple",
            "elements": [
              {
                "kind": "evidential",
                "inner": { "kind": "path", "name": "Sigil" },
                "evidence": "known"
              },
              {
                "kind": "evidential",
                "inner": { "kind": "path", "name": "FeatureVector" },
                "evidence": "known"
              }
            ]
          }
        },
        "span": { "start": 0, "end": 180, "line": 1, "column": 1 }
      }
    ]
  },
  "types": {
    "PersonaIntent": {
      "kind": "struct",
      "id": "type_PersonaIntent",
      "fields": [
        { "name": "name", "type": { "kind": "primitive", "name": "str" }, "visibility": "public" },
        { "name": "traits", "type": { "kind": "array", "element": { "kind": "primitive", "name": "str" } }, "visibility": "public" }
      ]
    },
    "Sigil": {
      "kind": "struct",
      "id": "type_Sigil",
      "fields": [
        { "name": "geometry", "type": { "kind": "path", "name": "Geometry" }, "visibility": "public" },
        { "name": "data", "type": { "kind": "array", "element": { "kind": "primitive", "name": "u8" } }, "visibility": "public" }
      ]
    },
    "FeatureVector": {
      "kind": "struct",
      "id": "type_FeatureVector",
      "fields": [
        { "name": "values", "type": { "kind": "array", "element": { "kind": "primitive", "name": "f64" } }, "visibility": "public" },
        { "name": "labels", "type": { "kind": "array", "element": { "kind": "primitive", "name": "str" } }, "visibility": "public" }
      ]
    }
  },
  "evidence_constraints": [],
  "metadata": {
    "optimization_level": "O2",
    "target": "x86_64-unknown-linux-gnu"
  }
}
```

---

## 11. Evidence Flow Analysis

The IR includes evidence flow annotations for AI reasoning:

```json
{
  "evidence_flow": {
    "fn_build_persona_sigil": {
      "entry_evidence": { "intent": "known" },
      "stage_evidence": [
        { "stage": "stage_0", "in": "known", "out": "known", "op": "intent_to_proto" },
        { "stage": "stage_1", "in": "known", "out": "known", "op": "enrich_with_geometry" },
        { "stage": "stage_2", "in": "known", "out": "known", "op": "fork" }
      ],
      "exit_evidence": "known",
      "evidence_promotions": [],
      "evidence_demotions": []
    }
  }
}
```

### 11.1 Evidence Promotion (Validation)

When external data is validated:

```json
{
  "kind": "validate",
  "expr": { "kind": "path", "name": "external_data" },
  "validator": "validate_user_input",
  "on_success": {
    "evidence_promotion": {
      "from": "reported",
      "to": "known"
    }
  },
  "on_failure": {
    "kind": "return",
    "value": { "kind": "variant", "enum": "Result", "variant": "Err", "args": [{ "kind": "path", "name": "validation_error" }] }
  }
}
```

### 11.2 Evidence Demotion (Unsafe Boundary)

```json
{
  "kind": "unsafe_block",
  "body": {
    "kind": "deref",
    "expr": { "kind": "path", "name": "raw_ptr" }
  },
  "evidence_demotion": {
    "from": "known",
    "to": "paradox",
    "reason": "unsafe_pointer_dereference"
  }
}
```

---

## 12. Closures and Lambda IR

### 12.1 Full Closure

```json
{
  "kind": "closure",
  "id": "closure_0",
  "params": [
    { "name": "x", "type": { "kind": "primitive", "name": "i64" } },
    { "name": "y", "type": { "kind": "primitive", "name": "i64" } }
  ],
  "body": {
    "kind": "binary",
    "op": "+",
    "left": { "kind": "path", "name": "x" },
    "right": { "kind": "path", "name": "y" }
  },
  "captures": [
    { "name": "multiplier", "capture_mode": "ref" }
  ],
  "return_type": { "kind": "primitive", "name": "i64" },
  "evidence": "known"
}
```

### 12.2 Lambda (Morpheme λ)

```json
{
  "kind": "morpheme",
  "symbol": "λ",
  "body": {
    "kind": "binary",
    "op": "+",
    "left": { "kind": "path", "name": "_" },
    "right": { "kind": "literal", "value": { "int": 1 } }
  }
}
```

---

## 13. Async/Await IR

```json
{
  "kind": "async_block",
  "body": {
    "kind": "block",
    "statements": [
      {
        "kind": "let",
        "pattern": { "kind": "binding", "name": "response" },
        "type": null,
        "value": {
          "kind": "await",
          "future": {
            "kind": "call",
            "func": { "kind": "path", "name": "fetch_data" },
            "args": [{ "kind": "path", "name": "url" }]
          }
        },
        "evidence": "reported"
      }
    ],
    "tail_expr": { "kind": "path", "name": "response" }
  }
}
```

---

## 14. SIMD Operations IR

```json
{
  "kind": "simd",
  "operation": "add",
  "element_type": { "kind": "primitive", "name": "f32" },
  "lanes": 4,
  "operands": [
    { "kind": "path", "name": "vec_a" },
    { "kind": "path", "name": "vec_b" }
  ]
}

{
  "kind": "simd_literal",
  "elements": [
    { "kind": "literal", "value": { "float": 1.0 } },
    { "kind": "literal", "value": { "float": 2.0 } },
    { "kind": "literal", "value": { "float": 3.0 } },
    { "kind": "literal", "value": { "float": 4.0 } }
  ],
  "type": { "kind": "simd", "element": { "kind": "primitive", "name": "f32" }, "lanes": 4 }
}
```

---

## 15. Compiler Integration

### 15.1 CLI Flag

```bash
# Emit IR to stdout
sigil --dump-ir=json source.sigil

# Emit IR to file
sigil --dump-ir=json --output=ir.json source.sigil

# Emit IR with type inference results
sigil --dump-ir=json --include-types source.sigil

# Emit IR with evidence flow analysis
sigil --dump-ir=json --include-evidence-flow source.sigil

# Emit IR optimized at level O2
sigil --dump-ir=json -O2 source.sigil

# Pretty-print (default) vs compact
sigil --dump-ir=json --ir-format=pretty source.sigil
sigil --dump-ir=json --ir-format=compact source.sigil
```

### 15.2 LSP Integration

The Language Server Protocol can emit IR for specific nodes:

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "sigil/getIR",
  "params": {
    "textDocument": { "uri": "file:///path/to/source.sigil" },
    "range": { "start": { "line": 10, "character": 0 }, "end": { "line": 25, "character": 0 } }
  },
  "id": 1
}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "ir": { ... },
    "evidence_analysis": { ... }
  },
  "id": 1
}
```

### 15.3 Programmatic API

```rust
// In Rust
use sigil::ir::{IrEmitter, IrOptions};

let source = std::fs::read_to_string("source.sigil")?;
let ast = sigil::parse(&source)?;
let typed_ast = sigil::typecheck(&ast)?;

let options = IrOptions {
    include_types: true,
    include_evidence_flow: true,
    include_spans: true,
    optimization_level: OptLevel::O2,
};

let ir = IrEmitter::emit(&typed_ast, options)?;
let json = serde_json::to_string_pretty(&ir)?;
```

---

## 16. IR Validation Schema

A JSON Schema for validating IR documents:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://sigil-lang.dev/schemas/ir/v1.json",
  "title": "Sigil IR",
  "type": "object",
  "required": ["version", "module"],
  "properties": {
    "version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "source": {
      "type": "object",
      "properties": {
        "file": { "type": "string" },
        "hash": { "type": "string" },
        "timestamp": { "type": "string", "format": "date-time" }
      }
    },
    "module": { "$ref": "#/definitions/Module" },
    "types": { "type": "object", "additionalProperties": { "$ref": "#/definitions/TypeDef" } },
    "evidence_constraints": { "type": "array", "items": { "$ref": "#/definitions/EvidenceConstraint" } }
  },
  "definitions": {
    "EvidenceLevel": {
      "type": "string",
      "enum": ["known", "uncertain", "reported", "paradox"]
    },
    "Type": {
      "oneOf": [
        { "$ref": "#/definitions/PrimitiveType" },
        { "$ref": "#/definitions/ArrayType" },
        { "$ref": "#/definitions/TupleType" },
        { "$ref": "#/definitions/EvidentialType" },
        { "$ref": "#/definitions/FunctionType" },
        { "$ref": "#/definitions/PathType" }
      ]
    },
    "PrimitiveType": {
      "type": "object",
      "required": ["kind", "name"],
      "properties": {
        "kind": { "const": "primitive" },
        "name": { "type": "string", "enum": ["bool", "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32", "f64", "char", "str", "unit"] }
      }
    },
    "EvidentialType": {
      "type": "object",
      "required": ["kind", "inner", "evidence"],
      "properties": {
        "kind": { "const": "evidential" },
        "inner": { "$ref": "#/definitions/Type" },
        "evidence": { "$ref": "#/definitions/EvidenceLevel" }
      }
    }
  }
}
```

---

## 17. AI Agent Usage Patterns

### 17.1 Generating Sigil Code from IR

AI agents can generate IR and have it compiled:

```python
# Example: AI generates pipeline IR
ir = {
    "kind": "pipeline",
    "source": {"kind": "path", "name": "users"},
    "stages": [
        {"op": "filter", "symbol": "φ", "predicate": {
            "kind": "closure",
            "params": [{"name": "u"}],
            "body": {"kind": "binary", "op": ">",
                     "left": {"kind": "field_access", "object": {"kind": "path", "name": "u"}, "field": "age"},
                     "right": {"kind": "literal", "value": {"int": 18}}}
        }},
        {"op": "transform", "symbol": "τ", "mapper": {
            "kind": "closure",
            "params": [{"name": "u"}],
            "body": {"kind": "field_access", "object": {"kind": "path", "name": "u"}, "field": "name"}
        }}
    ]
}

# Send to Sigil compiler
result = sigil_compile_ir(ir)
```

### 17.2 Analyzing Evidence Flow

```python
# AI inspects evidence flow to find trust boundaries
ir = load_ir("program.json")
for fn in ir["module"]["items"]:
    if fn["kind"] == "function":
        flow = ir["evidence_flow"][fn["name"]]
        for stage in flow["stage_evidence"]:
            if stage["in"] != stage["out"]:
                print(f"Evidence change at {stage['op']}: {stage['in']} -> {stage['out']}")
```

### 17.3 Suggesting Pipeline Optimizations

```python
# AI detects inefficient patterns
def suggest_optimizations(ir):
    suggestions = []
    for fn in ir["module"]["items"]:
        if fn["kind"] == "function" and "body" in fn:
            body = fn["body"]
            if body["kind"] == "pipeline":
                stages = body["stages"]
                # Check for filter-then-transform (can often be fused)
                for i in range(len(stages) - 1):
                    if stages[i]["op"] == "filter" and stages[i+1]["op"] == "transform":
                        suggestions.append({
                            "type": "fusion",
                            "location": f"{fn['name']}:stage_{i}",
                            "message": "Consider fusing filter and transform for better performance"
                        })
    return suggestions
```

---

## 18. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01 | Initial specification |

---

## Appendix A: Quick Reference

### Evidence Symbols

| Symbol | JSON | Level | Meaning |
|--------|------|-------|---------|
| `!` | `"known"` | 0 | Verified, computed locally |
| `?` | `"uncertain"` | 1 | May be absent, inferred |
| `~` | `"reported"` | 2 | External data, untrusted |
| `‽` | `"paradox"` | 3 | Contradictory, unsafe |

### Morpheme Symbols

| Symbol | JSON op | Meaning |
|--------|---------|---------|
| `τ` | `"transform"` | Map elements |
| `φ` | `"filter"` | Filter elements |
| `σ` | `"sort"` | Sort elements |
| `ρ` | `"reduce"` | Fold to single value |
| `λ` | `"lambda"` | Anonymous function |
| `Σ` | `"sum"` | Sum all elements |
| `Π` | `"product"` | Product all elements |
| `α` | `"first"` | First element |
| `ω` | `"last"` | Last element |
| `μ` | `"middle"` | Median element |
| `χ` | `"choice"` | Random selection |
| `ν` | `"nth"` | Get nth element |
| `ξ` | `"next"` | Next element |

### Protocol Operations

| Protocol | Operations |
|----------|------------|
| `http` | `get`, `post`, `put`, `delete`, `patch`, `head`, `options` |
| `grpc` | `call`, `stream`, `bidi_stream` |
| `websocket` | `connect`, `send`, `recv`, `close` |
| `kafka` | `produce`, `consume`, `subscribe` |
| `amqp` | `publish`, `consume`, `declare_queue`, `declare_exchange` |
| `graphql` | `query`, `mutation`, `subscription` |
