# Sigil Self-Hosted Compiler

This directory contains the Sigil compiler written in Sigil itself - the **Jormungandr bootstrap**.

## Status: Phase 4 - Frontend Implementation (COMPLETE!)

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `src/span.sg` | ~140 | ✅ Complete | Source span tracking |
| `src/token.sg` | ~450 | ✅ Complete | Token definitions |
| `src/ast.sg` | ~1200 | ✅ Complete | AST node definitions |
| `src/lib.sg` | ~60 | ✅ Complete | Module exports |
| `src/lexer.sg` | ~750 | ✅ Complete | Hand-written tokenization |
| `src/parser.sg` | ~2100 | ✅ Complete | Recursive descent parser |
| `src/typeck.sg` | - | 🔲 Pending | Type checking |
| `src/ir.sg` | - | 🔲 Pending | Intermediate representation |
| `src/lower.sg` | - | 🔲 Pending | AST → IR lowering |

## Conversion from Rust

This is a direct conversion of `sigil-lang/parser/src/` from Rust to Sigil:

| Rust File | Sigil File | Original Lines | Notes |
|-----------|------------|----------------|-------|
| `span.rs` | `span.sg` | 58 | Direct conversion |
| `lexer.rs` (Token enum) | `token.sg` | ~800 | Token definitions only |
| `ast.rs` | `ast.sg` | 1,592 | All AST node types |
| `lexer.rs` (Lexer struct) | `lexer.sg` | ~500 | Hand-written (no logos) |
| `parser.rs` | `parser.sg` | 4,462 | Full recursive descent parser |

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
│ Phase 1: Rust Compiler                                       │
│ sigil-lang/parser/ compiles Sigil source files              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Self-Hosted Core (CURRENT)                          │
│ Write compiler data structures in Sigil                      │
│ - span.sg, token.sg, ast.sg ✅                               │
│ - lexer.sg, parser.sg, typeck.sg, ir.sg 🔲                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Self-Hosting                                        │
│ Sigil compiler compiles itself                               │
│ Rust version becomes bootstrap-only                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Fixed Point                                         │
│ Sigil-Sigil compiles Sigil-Sigil → identical output          │
│ Bootstrap complete!                                          │
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
