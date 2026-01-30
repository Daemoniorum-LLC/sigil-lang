# Test Coverage Gap Analysis

## Current Test Files

| File | Coverage Area | Status |
|------|--------------|--------|
| test_simple.sg | Basic features (functions, structs, enums) | Partial |
| test_evidentiality.sg | Evidence markers (!, ?, ~) | Partial |
| test_compile.sg | Compilation pipeline | Minimal |
| test_morphemes.sg | Pipeline operators | Partial |
| test_parser_parse.sg | Parser functionality | Minimal |
| test_closure_capture.sg | Closure captures | Single case |
| test_if_ne.sg | If-not-equal | Single case |
| test_try_operator.sg | Try (?) operator | NEW - Good |
| bootstrap_test.sg | Self-compilation | Integration only |

## Gap Analysis by Module

### 1. LEXER (lexer.sg) - ~5% covered

#### CRITICAL GAPS:
- [ ] **Numeric bases**: Binary (0b), Octal (0o), Hex (0x), Vigesimal (0v), Sexagesimal (0s), Duodecimal (0z)
- [ ] **String escapes**: \n, \t, \r, \\, \", \', \0, \xNN, \u{NNNN}
- [ ] **Raw strings**: r"...", r#"..."#, r##"..."##
- [ ] **Unicode identifiers**: Greek letters, other scripts
- [ ] **All 50+ keywords**: Individually tested
- [ ] **Aspect markers**: ·ing, ·ed, ·able, ·ive
- [ ] **Affective markers**: ⊕, ⊖, ⊜, ↑, ↓, ♔, ♟, ☺, ☹, etc.
- [ ] **Math operators**: ∀, ∃, ∈, ∪, ∩, ∧, ∨, ¬, etc.
- [ ] **APL operators**: ⍋, ⍒, ⌽, ↻, ⌺, ⊞, ⍳
- [ ] **Whitespace handling**: Space, tab, newline, CR, form feed

**Needed test file**: `test_lexer.sg` (~100 test cases)

### 2. PARSER (parser.sg) - ~10% covered

#### CRITICAL GAPS:
- [ ] **All expression types**: 45+ forms, only ~10 tested
- [ ] **Pattern types**: Only basic patterns tested (15+ forms exist)
- [ ] **Generics**: Type params, const params, lifetime params, where clauses
- [ ] **Visibility**: pub, pub(crate), pub(super), pub(in path)
- [ ] **Attributes**: #[derive], #[repr], #[inline], custom attributes
- [ ] **Operator precedence**: All precedence levels (10+)
- [ ] **Turbofish syntax**: ::<T, U>
- [ ] **Range expressions**: .., ..=, a..b, a..=b
- [ ] **Cast expressions**: as Type
- [ ] **Unsafe blocks**: unsafe { }
- [ ] **Async/await**: async fn, .await
- [ ] **Error recovery**: Malformed input handling

**Needed test files**:
- `test_parser_expressions.sg` (~80 test cases)
- `test_parser_patterns.sg` (~30 test cases)
- `test_parser_items.sg` (~40 test cases)
- `test_parser_generics.sg` (~20 test cases)

### 3. TYPE CHECKER (typeck.sg) - ~15% covered

#### CRITICAL GAPS:
- [ ] **Type unification**: Matching and inference
- [ ] **Generic instantiation**: Concrete type substitution
- [ ] **Trait bounds**: T: Trait + Bound
- [ ] **Evidence lattice**: Join/meet operations
- [ ] **Pattern exhaustiveness**: Missing arm detection
- [ ] **Mutable borrow checking**: Aliasing rules
- [ ] **Never type propagation**: ! type handling
- [ ] **Error type recovery**: Continuing after errors
- [ ] **Built-in traits**: Clone, Copy, Debug, etc.
- [ ] **Method resolution**: Self type, impl lookup

**Needed test files**:
- `test_typeck_inference.sg` (~50 test cases)
- `test_typeck_traits.sg` (~40 test cases)
- `test_typeck_evidence.sg` (~30 test cases)
- `test_typeck_patterns.sg` (~20 test cases)

### 4. LOWERING (lower.sg) - ~5% covered

#### CRITICAL GAPS:
- [ ] **Pipeline desugaring**: |τ, |φ, |σ, |ρ → explicit calls
- [ ] **Self type resolution**: In impl blocks
- [ ] **Variable ID generation**: Unique naming
- [ ] **Let-else lowering**: To if-else
- [ ] **Match lowering**: Guard handling
- [ ] **Closure capture**: Environment analysis
- [ ] **Type attachment**: Every IR node typed

**Needed test file**: `test_lowering.sg` (~50 test cases)

### 5. CODEGEN (codegen.sg) - ~10% covered

#### CRITICAL GAPS:
- [ ] **C code emission**: Headers, includes, main wrapper
- [ ] **Type definitions**: struct, enum, typedef
- [ ] **Function signatures**: With generics
- [ ] **Control flow**: Labels, gotos for match/loop
- [ ] **Evidence tagging**: Runtime SIGIL_KNOWN, etc.
- [ ] **Closure generation**: Separate functions
- [ ] **Temporary variables**: Naming and management
- [ ] **Protocol marshaling**: HTTP, WebSocket, etc.

**Needed test file**: `test_codegen.sg` (~60 test cases)

### 6. IR (ir.sg) - ~5% covered

#### CRITICAL GAPS:
- [ ] **All 50+ IR operations**: Only ~10 tested
- [ ] **Evidence lattice**: Join, meet, cmp
- [ ] **Type information**: On all nodes
- [ ] **Pipeline steps**: All step types
- [ ] **Morpheme operations**: All 40+ variants
- [ ] **Protocol operations**: HTTP, WS, Kafka, etc.
- [ ] **JSON serialization**: Pretty and compact

**Needed test file**: `test_ir.sg` (~80 test cases)

### 7. SPECIAL FEATURES - ~0% covered

#### Protocol Operations:
- [ ] HTTP: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
- [ ] WebSocket: Connect, Text, Binary, Ping, Pong, Close
- [ ] Kafka: Produce, Consume, Subscribe, Commit
- [ ] gRPC: Unary, ServerStream, ClientStream, BiDi
- [ ] GraphQL: Query, Mutation, Subscription

**Needed test file**: `test_protocols.sg` (~50 test cases)

#### SIMD Operations:
- [ ] Arithmetic: Add, Sub, Mul, Div
- [ ] Horizontal: HAdd, Dot
- [ ] Shuffle/Blend
- [ ] Load/Store (aligned and unaligned)

**Needed test file**: `test_simd.sg` (~30 test cases)

#### Atomic Operations:
- [ ] Load, Store, Swap
- [ ] CompareExchange
- [ ] FetchAdd, FetchSub, FetchAnd, FetchOr
- [ ] Memory orderings

**Needed test file**: `test_atomics.sg` (~25 test cases)

#### Inline Assembly:
- [ ] Template strings
- [ ] Input/output operands
- [ ] Clobbers
- [ ] Options

**Needed test file**: `test_asm.sg` (~15 test cases)

## Summary: Test Files Needed

### HIGH PRIORITY (Foundation):
1. `test_lexer.sg` - Token recognition (~100 cases)
2. `test_parser_expressions.sg` - All expression forms (~80 cases)
3. `test_typeck_inference.sg` - Type inference (~50 cases)
4. `test_ir_operations.sg` - IR operation types (~80 cases)

### MEDIUM PRIORITY (Common Patterns):
5. `test_parser_patterns.sg` - Pattern matching (~30 cases)
6. `test_parser_items.sg` - Top-level items (~40 cases)
7. `test_typeck_traits.sg` - Trait system (~40 cases)
8. `test_codegen.sg` - C code generation (~60 cases)
9. `test_lowering.sg` - AST→IR conversion (~50 cases)

### LOWER PRIORITY (Advanced Features):
10. `test_protocols.sg` - Network protocols (~50 cases)
11. `test_simd.sg` - SIMD operations (~30 cases)
12. `test_atomics.sg` - Atomic operations (~25 cases)
13. `test_parser_generics.sg` - Generic syntax (~20 cases)
14. `test_asm.sg` - Inline assembly (~15 cases)

## Estimated Total Test Cases Needed

| Category | Current | Needed | Gap |
|----------|---------|--------|-----|
| Lexer | ~10 | 100 | 90 |
| Parser | ~30 | 170 | 140 |
| Type Checker | ~20 | 140 | 120 |
| IR | ~10 | 80 | 70 |
| Lowering | ~5 | 50 | 45 |
| Codegen | ~15 | 60 | 45 |
| Protocols | 0 | 50 | 50 |
| SIMD | 0 | 30 | 30 |
| Atomics | 0 | 25 | 25 |
| ASM | 0 | 15 | 15 |
| Integration | ~30 | 80 | 50 |
| **TOTAL** | **~120** | **~800** | **~680** |

## Recommended Implementation Order

### Phase 1: Core Language (Week 1-2)
1. test_lexer.sg
2. test_parser_expressions.sg
3. test_parser_patterns.sg

### Phase 2: Type System (Week 2-3)
4. test_typeck_inference.sg
5. test_typeck_traits.sg
6. test_typeck_evidence.sg

### Phase 3: Compilation (Week 3-4)
7. test_ir_operations.sg
8. test_lowering.sg
9. test_codegen.sg

### Phase 4: Advanced Features (Week 4+)
10. test_protocols.sg
11. test_simd.sg
12. test_atomics.sg
13. test_asm.sg

## Notes

- Each test case should be independent and self-documenting
- Tests should cover both success and failure cases
- Edge cases and boundary conditions are critical
- Regression tests should be added for each bug fix
- Integration tests validate end-to-end behavior
