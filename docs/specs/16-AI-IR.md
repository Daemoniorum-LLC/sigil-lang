# Sigil AI-Facing Intermediate Representation (IR)

**Version**: 1.0.0
**Status**: Draft
**Author**: Claude Code

## Overview

This specification defines a JSON-based Intermediate Representation (IR) designed for consumption by AI agents, language servers, and external tooling. The IR provides a normalized, semantically-rich view of Sigil programs that captures:

- Function definitions with typed parameters and bodies
- Pipeline operations (morphemes, transformations, forks)
- Evidentiality annotations throughout
- Type information including the evidentiality lattice
- Control flow and data flow structures
- Protocol operations with their trust boundaries

## Design Goals

1. **AI-Parseable**: Simple, consistent JSON structure that LLMs can easily understand
2. **Semantically Complete**: Captures all program semantics, not just syntax
3. **Type-Annotated**: Every expression carries type information
4. **Evidence-Aware**: Evidentiality lattice is explicit at every level
5. **Pipeline-Native**: First-class representation of morpheme pipelines
6. **Tool-Friendly**: Suitable for LSP, linters, code generators, and AI agents

## Top-Level Structure

```json
{
  "version": "1.0.0",
  "source": "path/to/file.sigil",
  "modules": [...],
  "functions": [...],
  "types": [...],
  "traits": [...],
  "impls": [...],
  "constants": [...],
  "evidentiality_lattice": {...}
}
```

## Evidentiality Lattice

The evidentiality system is central to Sigil's type system. The IR makes the lattice structure explicit:

```json
{
  "evidentiality_lattice": {
    "levels": [
      { "name": "known", "symbol": "!", "order": 0, "description": "Direct computation, verified" },
      { "name": "uncertain", "symbol": "?", "order": 1, "description": "Inferred, possibly absent" },
      { "name": "reported", "symbol": "~", "order": 2, "description": "External source, untrusted" },
      { "name": "paradox", "symbol": "‽", "order": 3, "description": "Contradictory, trust boundary" }
    ],
    "join_rules": [
      { "left": "known", "right": "uncertain", "result": "uncertain" },
      { "left": "known", "right": "reported", "result": "reported" },
      { "left": "known", "right": "paradox", "result": "paradox" },
      { "left": "uncertain", "right": "reported", "result": "reported" },
      { "left": "uncertain", "right": "paradox", "result": "paradox" },
      { "left": "reported", "right": "paradox", "result": "paradox" }
    ],
    "meet_rules": [
      { "left": "known", "right": "uncertain", "result": "known" },
      { "left": "known", "right": "reported", "result": "known" },
      { "left": "known", "right": "paradox", "result": "known" },
      { "left": "uncertain", "right": "reported", "result": "uncertain" },
      { "left": "uncertain", "right": "paradox", "result": "uncertain" },
      { "left": "reported", "right": "paradox", "result": "reported" }
    ]
  }
}
```

### Evidence Propagation

Evidence propagates through operations following these rules:

| Operation | Evidence Rule |
|-----------|---------------|
| Binary ops | `join(lhs.evidence, rhs.evidence)` |
| Function call | `join(fn.evidence, args.evidence)` |
| Protocol ops | Always `reported` (external data) |
| Unsafe blocks | Always `paradox` (trust boundary) |
| Trust coercion | Explicit downgrade (requires `trust()`) |
| Verify | Explicit upgrade (requires runtime check) |

## Type Representation

### Primitive Types

```json
{
  "kind": "primitive",
  "name": "i64" | "f64" | "bool" | "char" | "str" | "unit" | "never"
}
```

### Composite Types

```json
{
  "kind": "array",
  "element": { /* type */ },
  "size": 10
}

{
  "kind": "tuple",
  "elements": [{ /* type */ }, { /* type */ }]
}

{
  "kind": "struct",
  "name": "PersonaIntent",
  "generics": [{ /* type */ }],
  "fields": [
    { "name": "field1", "type": { /* type */ } }
  ]
}
```

### Evidential Types

```json
{
  "kind": "evidential",
  "inner": { "kind": "primitive", "name": "i64" },
  "evidence": "known" | "uncertain" | "reported" | "paradox"
}
```

### Function Types

```json
{
  "kind": "function",
  "params": [{ /* type */ }],
  "return": { /* type */ },
  "is_async": false,
  "evidence": "known"
}
```

### Special Types

```json
{
  "kind": "cycle",
  "modulus": 12
}

{
  "kind": "simd",
  "element": { "kind": "primitive", "name": "f32" },
  "lanes": 8
}

{
  "kind": "atomic",
  "inner": { "kind": "primitive", "name": "i64" }
}
```

## Function Representation

```json
{
  "name": "build_persona_sigil",
  "id": "fn_001",
  "visibility": "public",
  "generics": [
    { "name": "T", "bounds": ["Clone", "Debug"] }
  ],
  "params": [
    {
      "name": "intent",
      "type": { "kind": "struct", "name": "PersonaIntent" },
      "evidence": "known"
    }
  ],
  "return_type": {
    "kind": "tuple",
    "elements": [
      { "kind": "evidential", "inner": { "kind": "struct", "name": "Sigil" }, "evidence": "known" },
      { "kind": "evidential", "inner": { "kind": "struct", "name": "FeatureVector" }, "evidence": "known" }
    ]
  },
  "body": { /* operation */ },
  "attributes": ["inline", "pure"],
  "span": { "start": { "line": 10, "column": 1 }, "end": { "line": 25, "column": 2 } }
}
```

## Operation Nodes

Operations are the core IR nodes representing computation. Every operation has:

- `kind`: The operation type
- `type`: The result type (with evidence)
- `evidence`: The evidentiality level
- `span`: Source location

### Pipeline Operations

```json
{
  "kind": "pipeline",
  "steps": [
    { "op": "call", "fn": "intent_to_proto", "args": [...] },
    { "op": "call", "fn": "enrich_with_geometry", "args": [...] },
    { "op": "fork", "branches": [
      { "op": "call", "fn": "sigil_to_features", "args": [...] },
      { "op": "identity" }
    ]}
  ],
  "type": { /* result type */ },
  "evidence": "known"
}
```

### Morpheme Operations

```json
{
  "kind": "morpheme",
  "morpheme": "transform",
  "symbol": "τ",
  "input": { /* expression */ },
  "body": { /* closure expression */ },
  "type": { "kind": "array", "element": { "kind": "primitive", "name": "i64" } },
  "evidence": "known"
}
```

Available morphemes:

| Kind | Symbol | Description |
|------|--------|-------------|
| `transform` | `τ` | Map each element |
| `filter` | `φ` | Keep matching elements |
| `sort` | `σ` | Sort elements |
| `reduce` | `ρ` | Fold to single value |
| `lambda` | `λ` | Anonymous function |
| `sum` | `Σ` | Sum all elements |
| `product` | `Π` | Product of all elements |
| `first` | `α` | First element |
| `last` | `ω` | Last element |
| `middle` | `μ` | Median element |
| `choice` | `χ` | Random choice |
| `nth` | `ν` | Nth element |
| `next` | `ξ` | Iterator next |

### Control Flow

```json
{
  "kind": "if",
  "condition": { /* expression */ },
  "then_branch": { /* block */ },
  "else_branch": { /* block */ },
  "type": { /* result type */ },
  "evidence": "known"
}

{
  "kind": "match",
  "scrutinee": { /* expression */ },
  "arms": [
    {
      "pattern": { /* pattern */ },
      "guard": { /* optional expression */ },
      "body": { /* expression */ }
    }
  ],
  "type": { /* result type */ },
  "evidence": "known"
}

{
  "kind": "loop",
  "variant": "infinite" | "while" | "for",
  "condition": { /* expression for while */ },
  "iterator": { "pattern": {...}, "iterable": {...} },
  "body": { /* block */ },
  "type": { "kind": "primitive", "name": "unit" },
  "evidence": "known"
}
```

### Function Calls

```json
{
  "kind": "call",
  "function": "calculate_geometry",
  "function_id": "fn_002",
  "args": [
    { /* expression */ }
  ],
  "type_args": [{ /* type */ }],
  "type": { /* return type */ },
  "evidence": "known"
}

{
  "kind": "method_call",
  "receiver": { /* expression */ },
  "method": "process",
  "args": [{ /* expression */ }],
  "type": { /* return type */ },
  "evidence": "known"
}
```

### Literals

```json
{
  "kind": "literal",
  "variant": "int",
  "value": 42,
  "base": "decimal",
  "suffix": "i64",
  "type": { "kind": "primitive", "name": "i64" },
  "evidence": "known"
}

{
  "kind": "literal",
  "variant": "string",
  "value": "hello world",
  "type": { "kind": "primitive", "name": "str" },
  "evidence": "known"
}
```

### Binary and Unary Operations

```json
{
  "kind": "binary",
  "operator": "add" | "sub" | "mul" | "div" | "and" | "or" | "eq" | "lt" | ...,
  "left": { /* expression */ },
  "right": { /* expression */ },
  "type": { /* result type */ },
  "evidence": "known"
}

{
  "kind": "unary",
  "operator": "neg" | "not" | "deref" | "ref" | "ref_mut",
  "operand": { /* expression */ },
  "type": { /* result type */ },
  "evidence": "known"
}
```

### Variables and Bindings

```json
{
  "kind": "let",
  "pattern": { /* pattern */ },
  "type_annotation": { /* optional type */ },
  "init": { /* expression */ },
  "evidence": "known"
}

{
  "kind": "var",
  "name": "x",
  "id": "var_001",
  "type": { /* type */ },
  "evidence": "known"
}
```

### Protocol Operations

All protocol operations yield `reported` evidence by default:

```json
{
  "kind": "http_request",
  "method": "GET" | "POST" | "PUT" | "DELETE" | ...,
  "url": { /* expression */ },
  "headers": { /* expression */ },
  "body": { /* expression */ },
  "timeout": { /* expression */ },
  "type": { "kind": "evidential", "inner": {...}, "evidence": "reported" },
  "evidence": "reported"
}

{
  "kind": "grpc_call",
  "service": "PersonaService",
  "method": "GetPersona",
  "message": { /* expression */ },
  "metadata": { /* expression */ },
  "type": { "kind": "evidential", "inner": {...}, "evidence": "reported" },
  "evidence": "reported"
}

{
  "kind": "kafka_op",
  "operation": "produce" | "consume" | "subscribe" | ...,
  "topic": "persona-events",
  "payload": { /* expression */ },
  "key": { /* expression */ },
  "type": { "kind": "evidential", "inner": {...}, "evidence": "reported" },
  "evidence": "reported"
}
```

### Fork and Join

```json
{
  "kind": "fork",
  "branches": [
    { /* operation */ },
    { /* operation */ }
  ],
  "join_strategy": "tuple" | "first" | "all",
  "type": { "kind": "tuple", "elements": [...] },
  "evidence": "known"
}
```

### Evidentiality Coercion

```json
{
  "kind": "evidence_coerce",
  "operation": "trust" | "verify" | "mark",
  "expr": { /* expression */ },
  "from_evidence": "reported",
  "to_evidence": "known",
  "type": { /* type */ },
  "evidence": "known"
}
```

### Incorporation (Noun-Verb Fusion)

```json
{
  "kind": "incorporation",
  "segments": [
    { "kind": "noun", "name": "file" },
    { "kind": "verb", "name": "open" },
    { "kind": "verb", "name": "read" },
    { "kind": "verb", "name": "parse" }
  ],
  "args": [{ /* expression */ }],
  "type": { /* result type */ },
  "evidence": "known"
}
```

### Affect Operations

```json
{
  "kind": "affect",
  "expr": { /* expression */ },
  "affect": {
    "sentiment": "positive" | "negative" | "neutral",
    "intensity": "up" | "down" | "max",
    "formality": "formal" | "informal",
    "emotion": "joy" | "sadness" | "anger" | "fear" | "surprise" | "love",
    "confidence": "high" | "medium" | "low",
    "sarcasm": false
  },
  "type": { /* type with affect */ },
  "evidence": "known"
}
```

## Pattern Representation

```json
{
  "kind": "ident",
  "name": "x",
  "mutable": false,
  "evidence": "known"
}

{
  "kind": "tuple",
  "elements": [{ /* pattern */ }, { /* pattern */ }]
}

{
  "kind": "struct",
  "path": "Point",
  "fields": [
    { "name": "x", "pattern": { /* pattern */ } }
  ],
  "rest": false
}

{
  "kind": "wildcard"
}

{
  "kind": "literal",
  "value": { /* literal */ }
}
```

## Complete Example

Given this Sigil code:

```sigil
fn build_persona_sigil(intent: PersonaIntent!) -> (E<Sigil>!, E<FeatureVector>!) {
    intent
        |intent_to_proto
        |enrich_with_geometry
        |fork{
            |sigil_to_features,
            |identity
        }
}
```

The IR would be:

```json
{
  "version": "1.0.0",
  "source": "persona.sigil",
  "functions": [
    {
      "name": "build_persona_sigil",
      "id": "fn_build_persona_sigil_001",
      "visibility": "public",
      "generics": [],
      "params": [
        {
          "name": "intent",
          "type": {
            "kind": "evidential",
            "inner": { "kind": "struct", "name": "PersonaIntent", "generics": [] },
            "evidence": "known"
          }
        }
      ],
      "return_type": {
        "kind": "tuple",
        "elements": [
          {
            "kind": "evidential",
            "inner": { "kind": "struct", "name": "Sigil", "generics": [] },
            "evidence": "known"
          },
          {
            "kind": "evidential",
            "inner": { "kind": "struct", "name": "FeatureVector", "generics": [] },
            "evidence": "known"
          }
        ]
      },
      "body": {
        "kind": "pipeline",
        "input": {
          "kind": "var",
          "name": "intent",
          "id": "var_intent_001",
          "type": {
            "kind": "evidential",
            "inner": { "kind": "struct", "name": "PersonaIntent", "generics": [] },
            "evidence": "known"
          },
          "evidence": "known"
        },
        "steps": [
          {
            "kind": "call",
            "function": "intent_to_proto",
            "function_id": "fn_intent_to_proto",
            "args": [],
            "type_args": [],
            "type": { "kind": "struct", "name": "Proto", "generics": [] },
            "evidence": "known"
          },
          {
            "kind": "call",
            "function": "enrich_with_geometry",
            "function_id": "fn_enrich_with_geometry",
            "args": [],
            "type_args": [],
            "type": { "kind": "struct", "name": "EnrichedProto", "generics": [] },
            "evidence": "known"
          },
          {
            "kind": "fork",
            "branches": [
              {
                "kind": "call",
                "function": "sigil_to_features",
                "function_id": "fn_sigil_to_features",
                "args": [],
                "type_args": [],
                "type": {
                  "kind": "evidential",
                  "inner": { "kind": "struct", "name": "FeatureVector", "generics": [] },
                  "evidence": "known"
                },
                "evidence": "known"
              },
              {
                "kind": "identity",
                "type": {
                  "kind": "evidential",
                  "inner": { "kind": "struct", "name": "Sigil", "generics": [] },
                  "evidence": "known"
                },
                "evidence": "known"
              }
            ],
            "join_strategy": "tuple",
            "type": {
              "kind": "tuple",
              "elements": [
                {
                  "kind": "evidential",
                  "inner": { "kind": "struct", "name": "FeatureVector", "generics": [] },
                  "evidence": "known"
                },
                {
                  "kind": "evidential",
                  "inner": { "kind": "struct", "name": "Sigil", "generics": [] },
                  "evidence": "known"
                }
              ]
            },
            "evidence": "known"
          }
        ],
        "type": {
          "kind": "tuple",
          "elements": [
            {
              "kind": "evidential",
              "inner": { "kind": "struct", "name": "Sigil", "generics": [] },
              "evidence": "known"
            },
            {
              "kind": "evidential",
              "inner": { "kind": "struct", "name": "FeatureVector", "generics": [] },
              "evidence": "known"
            }
          ]
        },
        "evidence": "known"
      },
      "attributes": [],
      "span": { "start": { "line": 1, "column": 1 }, "end": { "line": 8, "column": 2 } }
    }
  ],
  "types": [
    {
      "kind": "struct_def",
      "name": "PersonaIntent",
      "generics": [],
      "fields": []
    },
    {
      "kind": "struct_def",
      "name": "Sigil",
      "generics": [],
      "fields": []
    },
    {
      "kind": "struct_def",
      "name": "FeatureVector",
      "generics": [],
      "fields": []
    }
  ],
  "evidentiality_lattice": {
    "levels": [
      { "name": "known", "symbol": "!", "order": 0 },
      { "name": "uncertain", "symbol": "?", "order": 1 },
      { "name": "reported", "symbol": "~", "order": 2 },
      { "name": "paradox", "symbol": "‽", "order": 3 }
    ]
  }
}
```

## CLI Usage

```bash
# Dump IR to stdout
sigil dump-ir program.sigil

# Dump IR to file
sigil dump-ir program.sigil -o program.ir.json

# Dump IR with pretty printing
sigil dump-ir program.sigil --pretty

# Dump IR after optimization
sigil dump-ir program.sigil --opt-level=2

# Dump IR with full type information
sigil dump-ir program.sigil --full-types

# Check and dump IR for AI consumption
sigil check program.sigil --format=json --dump-ir
```

## LSP Integration

The IR can be requested via LSP custom commands:

```json
{
  "method": "sigil/dumpIR",
  "params": {
    "textDocument": { "uri": "file:///path/to/file.sigil" },
    "range": { "start": {...}, "end": {...} },
    "options": {
      "pretty": true,
      "full_types": true
    }
  }
}
```

## AI Agent Guidelines

When parsing this IR, AI agents should:

1. **Track Evidence Flow**: Follow `evidence` fields to understand data provenance
2. **Respect Lattice Rules**: Use `join` when combining evidence from multiple sources
3. **Identify Trust Boundaries**: `paradox` evidence indicates unsafe/external boundaries
4. **Parse Pipelines Sequentially**: Steps execute left-to-right, forks execute in parallel
5. **Handle Morphemes**: Map morpheme symbols to their semantic operations
6. **Type-Check Results**: Every operation's output type should match expected input

## Future Extensions

- **Data Flow Graph**: SSA-form IR with explicit data dependencies
- **Control Flow Graph**: Basic blocks with explicit control edges
- **Effect Tracking**: Pure vs effectful operations
- **Provenance Tracking**: Origin of values for debugging
- **Optimization Hints**: Information for AI-driven optimization
