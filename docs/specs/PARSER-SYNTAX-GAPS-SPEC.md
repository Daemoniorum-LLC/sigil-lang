# Parser Syntax Gaps Spec

**Version:** 0.1.0
**Status:** ! Draft
**Date:** 2026-02-03
**Discovery:** Running 36 uncounted top-level test files revealed 5 parse failures on syntax constructs that should be valid Sigil.

---

## 1. Gap Discovery

### 1.1 Context

Five top-level test files fail with parse errors on constructs that represent valid Sigil syntax or reasonable language expectations. These were invisible because the test runner never scanned the top-level directory.

### 1.2 Affected Files

| File | Error | Root Cause |
|------|-------|------------|
| `bootstrap_test.sg` | `expected LBrace, found Else` | `else if` chains not supported |
| `option_syntax_test.sg` | `expected LParen, found Return` | `⤺` in identifier name |
| `type_compat_test.sg` | `expected LParen, found Return` | `⤺` in identifier name |
| `test_ir_operations.sg` | `expected LBrace, found Eq` | Assignment in unsupported position |
| `test_morphemes.sg` | `expected expression, found Comma` | Tuple/comma expression parsing gap |

---

## 2. Gap Analysis

### 2.1 `else if` Chains (`⎉ ⎇`)

**Severity:** High
**Impact:** Blocks any non-trivial control flow with multiple branches

**Current behavior:**
```sigil
// WORKS: simple if/else
⎇ condition {
    // ...
} ⎉ {
    // ...
}

// FAILS: else-if chain
⎇ arg == "--verbose" {
    // ...
} ⎉ ⎇ arg == "--base-dir" ∧ i + 1 < args.len() {  // Parse error here
    // ...
} ⎉ ⎇ arg == "--smoke" {
    // ...
} ⎉ {
    // ...
}
```

**Expected behavior:** `⎉ ⎇` (else if) should parse as a chained conditional, producing the same AST as nested `else { if ... }` blocks.

**Spec reference:** Section 02-SYNTAX.md should define `⎉ ⎇` as the `else if` construct. The parser currently expects `LBrace` after `⎉` but encounters `⎇` (the if token).

**Implementation:** In the parser's if-expression handler, after consuming `⎉` (else), check if the next token is `⎇` (if). If so, parse another if-expression as the else branch instead of requiring `{`.

### 2.2 Sigil Symbols in Identifiers

**Severity:** Medium
**Impact:** Cannot use Sigil vocabulary symbols in test/function names

**Current behavior:**
```sigil
// FAILS: ⤺ in function name triggers return-statement parsing
rite test_option_⤺() { ... }

// Parser sees: rite test_option_ [return statement] () ...
// Produces: "expected LParen, found Return"
```

**Expected behavior:** When a Sigil symbol (`⤺`, `⎇`, `⎉`, `⌥`, etc.) appears as part of an identifier (adjacent to alphanumeric/underscore characters with no whitespace), it should be treated as identifier text, not as its keyword meaning.

**Analysis:** This is a lexer-level issue. The lexer tokenizes `⤺` as `Token::Return` regardless of context. When it appears immediately after `test_option_`, the parser sees `Ident("test_option_")` followed by `Token::Return` instead of a single `Ident("test_option_⤺")`.

**Implementation options:**
1. **Lexer fix:** Extend the identifier regex to include Sigil symbols when preceded by identifier characters (complex, may break other things)
2. **Test file fix:** Rename functions to avoid symbols in names (pragmatic, addresses symptom)
3. **Allow symbols in backtick-quoted identifiers:** `` `test_option_⤺` `` (language feature, clean solution)

**Recommendation:** Option 3 (backtick identifiers) is the cleanest long-term solution and aligns with other languages (Kotlin, Scala). Option 2 is the immediate workaround.

### 2.3 Assignment in Unsupported Position

**Severity:** Low
**Impact:** Affects `test_ir_operations.sg` only

**Current behavior:**
```
expected LBrace, found Eq at 6330..6331
```

**Analysis:** The test file attempts an assignment (`=`) in a position where the parser expects a block expression (`{`). This is likely a construct like `⎇ x = foo { ... }` or similar pattern that conflates assignment with pattern binding.

**Recommendation:** Investigate the specific line and determine if this is a valid syntax that needs support or a test file error.

### 2.4 Tuple/Comma Expression Gaps

**Severity:** Low
**Impact:** Affects `test_morphemes.sg` only

**Current behavior:**
```
expected expression, found Comma at 1803..1804
```

**Analysis:** A comma appears where the parser expects an expression. This could be:
- Tuple literal syntax `(a, b, c)` in an unsupported context
- Trailing comma in a function call or array literal
- Morpheme argument list syntax

**Recommendation:** Investigate the specific line to determine the intended syntax.

---

## 3. Priority

| Gap | Priority | Rationale |
|-----|----------|-----------|
| `else if` chains | **P0** | Fundamental control flow; blocks real-world code |
| Symbols in identifiers | **P2** | Test naming convenience; workaround exists |
| Assignment position | **P2** | Single file; needs investigation |
| Comma expression | **P2** | Single file; needs investigation |

---

## 4. Implementation Strategy

### Phase 1: `else if` chains (P0)

| Component | Change |
|-----------|--------|
| `parser.rs` | In if-expression parsing, after consuming `⎉` (else token), check for `⎇` (if token). If found, recursively parse if-expression as the else branch. |
| Test | `bootstrap_test.sg` should pass after this fix |

**Expected AST:**
```
IfExpr {
    condition: ...,
    then_branch: ...,
    else_branch: Some(IfExpr {    // <-- recursive, not Block
        condition: ...,
        then_branch: ...,
        else_branch: Some(Block { ... })
    })
}
```

### Phase 2: Investigate remaining gaps (P2)

- Read `test_ir_operations.sg` line ~6330 to understand the assignment context
- Read `test_morphemes.sg` line ~1803 to understand the comma context
- Determine if these are valid syntax or test file errors
- Spec out fixes if valid

---

## 5. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| `bootstrap_test.sg` | Parse error | Passes (after else-if + assert fixes) |
| `option_syntax_test.sg` | Parse error | Passes (after rename or backtick-ident) |
| `type_compat_test.sg` | Parse error | Passes (after rename or backtick-ident) |
| `test_ir_operations.sg` | Parse error | Investigated, spec updated |
| `test_morphemes.sg` | Parse error | Investigated, spec updated |

---

## 6. Relationship to Other Specs

- **02-SYNTAX.md**: Control flow syntax definition (`⎇`/`⎉`/`⌥`)
- **01-LEXICAL.md**: Token definitions and identifier rules
- **ASSERT-API-SPEC.md**: `bootstrap_test.sg` also needs assert overloads to fully pass

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-03 | Initial gap discovery. 5 parse failures in uncounted test files. |
