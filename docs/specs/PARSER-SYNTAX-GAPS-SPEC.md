# Parser Syntax Gaps Spec

**Version:** 0.3.0
**Status:** Mostly Resolved
**Date:** 2026-02-04
**Discovery:** Running 37 uncounted top-level test files revealed parse failures on syntax constructs that should be valid Sigil.

**Update (v0.3.0):** Resolved 4 of 5 original parse failures:
- Renamed identifiers with Sigil symbols (3 files)
- Implemented multi-param morpheme closures (`a, b => expr`), σ{comparator}, ρ{init, closure}, δ distinct
- Only `bootstrap_test.sg` remains (requires `each...of` loop syntax)

---

## 1. Gap Discovery

### 1.1 Context

Five top-level test files originally failed with parse errors. Four have been fixed. One remains.

### 1.2 Affected Files — Current Status

| File | Original Error | Status | Fix Applied |
|------|-------|--------|-------------|
| `option_syntax_test.sg` | `⤺` in identifier | ✅ Resolved | Renamed `test_option_⤺` → `test_option_return` |
| `type_compat_test.sg` | `⤺` in identifier | ✅ Resolved | Renamed `test_optional_⤺` → `test_optional_return` |
| `test_ir_operations.sg` | `⎉` in variable name | ✅ Resolved (parse) | Renamed `let_⎉` → `let_else_op`. Now hits type errors (separate issue) |
| `test_morphemes.sg` | Multi-param closure in morphemes | ✅ Resolved | Implemented multi-param closures, σ{}, ρ{init,...}, δ distinct |
| `bootstrap_test.sg` | `expected LBrace, found Else` | ❌ Open | Requires `each...of` loop syntax (P2) |

---

## 2. Resolved Gaps

### 2.1 `else if` Chains (`⎉ ⎇`) — ✅ ALREADY WORKS

Already implemented at `parser.rs:7813-7815`. The `bootstrap_test.sg` failure is caused by `each ... of` loops, not else-if.

### 2.2 Sigil Symbols in Identifiers — ✅ RESOLVED (workaround)

**Fix:** Renamed identifiers containing Sigil symbols (`⤺`, `⎉`) in 3 test files.

### 2.3 Multi-param Morpheme Closures — ✅ RESOLVED

**Changes made:**

1. **`parse_morpheme_closure()`** (`parser.rs:7477`): Now supports `a, b => expr` syntax by parsing comma-separated idents after the first ident. Returns `Vec<ClosureParam>` instead of single `Pattern`.

2. **`parse_pipe_op()` σ case** (`parser.rs:6631`): Added `{` handling for `σ{a, b => b - a}` → `PipeOp::SortBy(closure)`.

3. **`parse_pipe_op()` ρ case** (`parser.rs:6701`): When `looks_like_morpheme_closure()` returns false, parses `init_expr, closure` → `PipeOp::ReduceWithInit(init, closure)`.

4. **`parse_pipe_op()` δ case** (`parser.rs:6792`): Added `Token::Delta` → `PipeOp::Unique` for distinct morpheme.

5. **AST** (`ast.rs:1444`): Added `ReduceWithInit(Box<Expr>, Box<Expr>)` and `SortBy(Box<Expr>)` to `PipeOp` enum.

6. **Interpreter** (`interpreter.rs:12834-12904`): Implemented `SortBy` (bubble sort with closure comparator) and `ReduceWithInit` (fold with explicit initial value and named closure params).

7. **Backends**: Added match arms for `SortBy` and `ReduceWithInit` in typeck.rs, codegen.rs, wasm/morphemes.rs.

### 2.4 Assignment in Unsupported Position — ✅ RESOLVED (was test file issue)

The `test_ir_operations.sg` error was caused by `⎉` in variable name `let_⎉`, not an assignment syntax gap. Renamed to `let_else_op`. File now progresses to type checking (separate type errors remain).

---

## 3. Open Gaps

### 3.1 `each ... of` Loop Syntax (bootstrap_test.sg)

**Priority:** P2
**Occurrences:** 6 in `bootstrap_test.sg`

```sigil
each source of COMPILER_SOURCES {
    // loop body
}
```

This is an alternative for-loop syntax. The parser doesn't recognize `each` as a loop keyword.

**Implementation approach:**
1. Add `each` as keyword/contextual keyword in lexer
2. In statement parser, handle `each <ident> of <expr> { ... }` as `Expr::ForIn`
3. Map to existing for-in semantics

**Other issues in bootstrap_test.sg:**
- `⊳` continue symbol (6 occurrences) — not handled as continue statement
- Various self-hosted compiler features that depend on runtime methods (`parse_file`, etc.)

---

## 4. Test Results

| Metric | Before (v0.1.0) | After (v0.3.0) |
|--------|---------|--------|
| Parse errors in uncounted tests | 5 files | 1 file |
| `test_morphemes.sg` | Parse error | ✅ All tests pass |
| `option_syntax_test.sg` | Parse error | ✅ Passes |
| `type_compat_test.sg` | Parse error | ✅ Passes |
| `test_ir_operations.sg` | Parse error | Promoted to type errors |
| `bootstrap_test.sg` | Parse error | Still parse error (each...of) |
| Uncounted pass rate | 10/36 → 14/37 | **17/37** |

---

## 5. Relationship to Other Specs

- **02-SYNTAX.md**: Control flow syntax definition (`⎇`/`⎉`/`⌥`)
- **01-LEXICAL.md**: Token definitions and identifier rules
- **ASSERT-API-SPEC.md**: `bootstrap_test.sg` also needs assert overloads to fully pass
- **TYPE-COERCION-GAPS-SPEC.md**: `test_ir_operations.sg` type errors

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-03 | Initial gap discovery. 5 parse failures in uncounted test files. |
| 0.2.0 | 2026-02-04 | else-if chains confirmed already working (misdiagnosis). Added `each...of` loop syntax as new gap. Updated priorities. |
| 0.3.0 | 2026-02-04 | Resolved 4 of 5 parse failures. Implemented multi-param morpheme closures, σ{comparator}, ρ{init,closure}, δ distinct. Only bootstrap_test.sg remains. |
