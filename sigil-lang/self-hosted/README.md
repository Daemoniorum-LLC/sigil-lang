# Sigil Self-Hosted Compiler

This directory contains the Sigil compiler written in Sigil itself - the **Jormungandr bootstrap**.

## Status: Phase 4 - Fixed Point (IN PROGRESS)

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `src/span.sg` | ~140 | ✅ Complete | Source span tracking |
| `src/token.sg` | ~450 | ✅ Complete | Token definitions |
| `src/ast.sg` | ~1200 | ✅ Complete | AST node definitions |
| `src/lib.sg` | ~100 | ✅ Complete | Module exports |
| `src/lexer.sg` | ~750 | ✅ Complete | Hand-written tokenization |
| `src/parser.sg` | ~2100 | ✅ Complete | Recursive descent parser |
| `src/typeck.sg` | ~1800 | ✅ Complete | Type checking with evidentiality |
| `src/ir.sg` | ~900 | ✅ Complete | AI-facing intermediate representation |
| `src/lower.sg` | ~800 | ✅ Complete | AST → IR lowering |
| `src/interp.sg` | ~1100 | ✅ Complete | Tree-walking interpreter |
| `src/runtime.sg` | ~600 | ✅ Complete | Runtime system (memory, stdlib) |
| `src/codegen.sg` | ~950 | ✅ Complete | C code generation |
| `src/driver.sg` | ~500 | ✅ Complete | Compiler driver and CLI |

**Total: ~12,000+ lines of Sigil**

## Conversion from Rust

This is a direct conversion of `sigil-lang/parser/src/` from Rust to Sigil:

| Rust File | Sigil File | Original Lines | Notes |
|-----------|------------|----------------|-------|
| `span.rs` | `span.sg` | 58 | Direct conversion |
| `lexer.rs` (Token enum) | `token.sg` | ~800 | Token definitions only |
| `ast.rs` | `ast.sg` | 1,592 | All AST node types |
| `lexer.rs` (Lexer struct) | `lexer.sg` | ~500 | Hand-written (no logos) |
| `parser.rs` | `parser.sg` | 4,462 | Full recursive descent parser |
| `typeck.rs` | `typeck.sg` | 2,560 | Bidirectional inference + evidentiality |
| `ir.rs` | `ir.sg` | 1,243 | AI-facing JSON-serializable IR |
| `lower.rs` (new) | `lower.sg` | N/A | AST → IR transformation |

## Key Differences from Rust Version

1. **Evidentiality Markers**: All types use Sigil's evidentiality system
   - `!Type` for known/verified values
   - `?Type` for optional/uncertain values
   - `~Type` for external/reported values

2. **Default Values**: Struct fields can have defaults inline
   ```sigil
   pub struct Config {
       pub debug: !bool = false,
       pub features: ![String] = [],
   }
   ```

3. **Null Instead of None**: Uses `null` for absent optional values

4. **Method Syntax**: Uses `fn method(self)` pattern

## Bootstrap Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Rust Compiler ✅                                    │
│ sigil-lang/parser/ compiles Sigil source files              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Self-Hosted Core ✅ COMPLETE                        │
│ Write compiler data structures in Sigil                      │
│ - span.sg, token.sg, ast.sg ✅                               │
│ - lexer.sg, parser.sg ✅                                     │
│ - typeck.sg ✅                                               │
│ - ir.sg, lower.sg ✅                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Execution ✅ COMPLETE                               │
│ Execute Sigil code                                           │
│ - interp.sg ✅ (tree-walking interpreter)                    │
│ - runtime.sg ✅ (memory, stdlib, evidence)                   │
│ - codegen.sg ✅ (C code generation)                          │
│ - driver.sg ✅ (compiler CLI)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Fixed Point 🚧 IN PROGRESS                          │
│ - [ ] Create self-compilation test                           │
│ - [ ] Sigil-Sigil compiles Sigil-Sigil → identical output    │
│ Bootstrap complete when output is identical!                 │
└─────────────────────────────────────────────────────────────┘
```

## Testing

Once the lexer and parser are complete, the self-hosted compiler will be tested by:

1. **Differential Testing**: Same input → same AST (Rust vs Sigil)
2. **Bootstrap Identity**: Sigil-compiled-by-Sigil = Sigil-compiled-by-Rust
3. **Test Suite**: Existing tests run against self-hosted version

## Experience Checkpoint (Jormungandr)

This conversion is part of the Jormungandr research initiative. Notes from the conversion:

### Joys 😊
- Evidentiality markers make optionality crystal clear
- Default values in struct definitions reduce boilerplate
- The syntax feels natural for an agent to read and write
- Greek morpheme operators are visually distinct

### Frictions 😤
- Need to implement the actual lexer/parser to test this code
- Some Rust idioms (like `impl Display`) need Sigil equivalents
- The `!Type` syntax can look like factorial in some contexts

### Patterns Discovered
- `Ident::new()` factory pattern works well
- Evidentiality lattice operations (join/meet) are simple to express
- Enum variants with associated data are clean
