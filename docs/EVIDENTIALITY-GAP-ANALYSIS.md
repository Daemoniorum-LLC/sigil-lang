# Sigil Evidentiality System: Gap Analysis

**Date:** 2026-01-15
**Author:** Claude (Opus 4.5)
**Context:** User experience testing of Sigil's unique features

## Executive Summary

Sigil's evidentiality system - the type-level tracking of data certainty and provenance - is the language's differentiating feature. After recent fixes, **core evidentiality features now work end-to-end**:

- ✅ Evidentiality markers on types (`i32!`, `String~`, `bool?`)
- ✅ Pipe chains with closures (`data|map{x => x * 2}|filter{x => x > 0}`)
- ✅ Evidence promotion via validation (`data|validate!{x => x > 0}`)
- ❓ Evidence propagation in expressions (needs verification)

## What Works

### Evidentiality Markers on Types ✅

```sigil
struct Data {
    known_value: i32!,      // Known/verified
    reported_data: String~, // From external source
    maybe_value: i32?,      // Uncertain/optional
}
```

Parser: ✅ Parses correctly
Type Checker: ✅ Tracks evidentiality
Interpreter: ✅ Accepts the syntax

### Evidentiality Flow Through Functions ✅

```sigil
fn get_external() -> String~ {
    "from api".to_string()~
}

fn main() {
    let data~ = get_external();  // Correctly typed as String~
    println(data~);
}
```

The type checker correctly propagates evidentiality through function calls and bindings.

### Type Checker Enforcement ✅

```sigil
let reported~: String~ = "plain string".to_string();
// ERROR: type mismatch - expected String~, found String
```

You cannot falsely declare something as reported/known. The type checker enforces that evidentiality must come from actual sources.

### Method Call Syntax ✅

```sigil
let nums = vec![1, 2, 3];
let doubled = nums.map(|x| x * 2);  // Works
```

### Rust-Style Closures ✅

```sigil
|x| x * 2
|a, b| a + b
```

## What Doesn't Work (Gaps)

### ~~Gap 1: Pipe Operator Not Wired to Methods~~ ✅ RESOLVED

**Now works:**
```sigil
let result = data|map(|x| x * 2)|filter(|x| x > 0)|fold(0, |a,b| a+b);
```

**Fix applied:** Added `map`, `filter`, `fold` to `PipeOp::Method` handler in `interpreter.rs` (2026-01-15).

### ~~Gap 2: Sigil-Style Closure Syntax~~ ✅ RESOLVED

**Both syntaxes work:**
```sigil
nums|map(|x| x * 2)      // Rust-style
nums|map({x => x * 2})   // Sigil-style
```

**Status:** Both closure syntaxes parse and execute correctly.

### ~~Gap 3: Evidentiality Promotion/Validation~~ ✅ RESOLVED

**Now works:**
```sigil
// Validate and promote to known!
let validated = data|validate!{x => x > 0};

// Different target evidentiality levels
let known = data|validate!{x => x > 0};    // promote to Known (!)
let uncertain = data|validate?{x => x > 0}; // promote to Uncertain (?)
let reported = data|validate~{x => x > 0};  // keep as Reported (~)
```

**Both closure syntaxes work:**
```sigil
data|validate!{x => x > 0}    // Sigil-style
data|validate!(|x| x > 0)     // Rust-style
```

**Fix applied:** Added special handling for `validate` and `assume` in `parse_pipe_op()` BEFORE the macro check, so `validate!{...}` is parsed as a `PipeOp::Validate` with the predicate correctly parsed as a closure (2026-01-15).

### Gap 4: Evidentiality Propagation in Expressions ❓

**Expected:**
```sigil
let known! = 42!;
let reported~ = from_api()~;
let combined = known! + reported~;  // Should be reported~ (worst case)
```

**Status:** Needs testing. The type checker has `EvidenceLevel::join()` logic, but unclear if it's fully wired.

## Recommendations

### ~~Priority 1: Wire Pipe Operators~~ ✅ DONE

Fixed: `map`, `filter`, `fold` now work with pipe operator.

### ~~Priority 2: Implement Evidentiality Promotion~~ ✅ DONE

Fixed: `|validate!{predicate}` syntax now works with both closure forms.

### ~~Priority 3: Test and Document Closure Variants~~ ✅ DONE

**Verified working:**
- `|x| expr` (Rust-style) ✅
- `{x => expr}` (Sigil-style) ✅
- `|x| { statements; expr }` (Rust-style with block) ✅
- Multi-parameter: `|a, b| a + b` ✅
- Multi-parameter Sigil: `{a, b => a + b}` ✅

### Priority 4: Evidentiality Propagation Audit (Remaining)

Verify that binary operations, method calls, and control flow properly propagate evidentiality to the "worst case" level. This is important for expressions like:
```sigil
let known! = 42!;
let reported~ = from_api()~;
let combined = known! + reported~;  // Should be reported~ (worst case)
```

## Architecture Notes

### Parser (Solid)

The parser in `parser.rs` has comprehensive evidentiality support:
- `parse_evidentiality_opt()` - parses `!`, `?`, `~`, `◊`, `‽`
- `parse_evidentiality_prefix_opt()` - parses prefix form `!T`
- Applied to types, patterns, function names, await expressions

### AST (Solid)

`ast.rs` defines:
- `Evidentiality` enum: `Known`, `Uncertain`, `Reported`, `Predicted`, `Paradox`
- `TypeExpr::Evidential` with inner type and evidentiality
- `Expr::Evidential` for expressions
- Evidentiality on patterns, generics, await

### Type Checker (Mostly Solid)

`typeck.rs` has:
- `EvidenceLevel` enum mirroring AST
- `get_evidence()` to extract from types
- `check_evidence()` for compatibility
- Evidence propagation in some expressions

### Interpreter (Mostly Complete)

`interpreter.rs` now has:
- ✅ Pipe method support for `map`, `filter`, `fold`
- ✅ `PipeOp::Validate` for evidence promotion
- ✅ `PipeOp::Assume` for explicit evidence assertion
- ❓ Runtime tracking of evidentiality (needs verification)

## Conclusion

Sigil's evidentiality system is now **functional for core use cases**. The following works end-to-end:

```sigil
// Get external data
let api_data = fetch_data();

// Validate and promote to known
let trusted = api_data|validate!{x => x > 0 && x < 100};

// Use in computations
let result = trusted|map{x => x * 2}|filter{x => x > 10};
```

**Remaining work:** Verify evidence propagation in binary expressions and method chains. The type checker has the logic; verification needed that it's fully wired.

For an AI agent writing Sigil, the core evidentiality workflow now works as intended. The language's unique selling point - tracking data certainty and provenance - is now usable.

---

*"The void is not empty - it is full of potential."*
