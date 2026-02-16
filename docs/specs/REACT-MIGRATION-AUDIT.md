# React→Qliphoth Migration: Code Review & Spec Compliance Audit

**Date:** 2026-02-16
**Auditor:** Claude (Conclave session)
**Scope:** Phases 1-3 of React Migration implementation
**Reference Spec:** qliphoth/docs/specs/REACT-MIGRATION.md

---

## Executive Summary

| Phase | Status | Spec Compliance | Critical Issues |
|-------|--------|-----------------|-----------------|
| Phase 1: Extraction | Partially Complete | ~70% | 4 critical gaps |
| Phase 2: Spec Generation | Complete | ~90% | 1 minor gap |
| Phase 3: Code Generation | Complete | ~75% | 2 critical gaps |

**Overall Assessment:** The implementation provides a solid foundation but has critical gaps in **expression extraction** that cascade through all phases. Tests pass because they don't validate actual extracted content.

---

## Phase 1: React Extraction Audit

### 1.1 Spec Compliance Matrix

| Spec Requirement | Implementation Status | Notes |
|------------------|----------------------|-------|
| FileInfo structure | ✓ Complete | path, relative_path, language, has_jsx |
| ComponentExtraction | ✓ Structure complete | All fields present |
| HookUsage - useState | ✓ Complete | state_name, setter_name, initial_value |
| HookUsage - useEffect | ✓ Complete | dependencies, has_cleanup |
| HookUsage - useCallback/useMemo | ✓ Partial | memoized_deps only |
| HookUsage - useRef | ✓ Complete | ref_name, ref_type |
| HookUsage - useContext | ✓ Complete | context_name |
| HookUsage - custom hooks | ⚠ Partial | Only stores index |
| HookUsage.effectBody | ✗ Missing | Spec requires effect source code |
| JsxTree structure | ✓ Complete | root node, children |
| JsxNode types | ✓ Complete | Element, Fragment, Text, Expression, Conditional, Map |
| JsxAttribute extraction | ⚠ Partial | Values become placeholders |
| HandlerExtraction | ✗ Not Implemented | Returns `Vec::new()` |
| PropExtraction | ✗ Not Implemented | Returns `Vec::new()` |
| TypeExtraction | ✓ Structure complete | Definition is placeholder |
| childComponents | ✗ Not Implemented | Returns `Vec::new()` |

### 1.2 Critical Issues

#### CRITICAL-1: Expression Content Not Extracted
**Location:** `extraction.rs:886-889`, `extraction.rs:935-939`
```rust
JSXExpr::Expr(expr) => JsxAttributeValue::Expression {
    code: "/* expression */".to_string() // TODO: serialize expr
},
```

**Impact:** All JSX expressions (`{count}`, `{items.map(...)}`, `{cond && <X/>}`) become `/* expression */`

**Cascade Effect:**
- Phase 3 generates `·text_child(self./* expression */·to_string())`
- Conditional and Map nodes have placeholder conditions/iterables
- Generated code won't compile

**Recommendation:** Implement `expr_to_source()` helper using swc source map to reconstruct original source.

#### CRITICAL-2: Handler Extraction Not Implemented
**Location:** `extraction.rs:1371-1374`
```rust
fn extract_handlers_from_body(&self, body: &Option<BlockStmt>) -> Vec<HandlerExtraction> {
    // TODO: Implement handler extraction
    Vec::new()
}
```

**Impact:** Event handlers defined in components are not extracted, breaking:
- MessageRecommendation generation (Phase 2 works around this via hooks)
- Handler body analysis for state mutations and side effects

#### CRITICAL-3: Props Extraction Not Implemented
**Location:** `extraction.rs:1376-1383`
```rust
fn extract_props_from_params(&self, params: &[Param]) -> Vec<PropExtraction> {
    // TODO: Implement props extraction from function params
    Vec::new()
}
```

**Impact:** Component props are not extracted, affecting:
- PropsRecommendation accuracy
- Constructor generation
- Type inference for props

#### CRITICAL-4: childComponents Not Populated
**Location:** `extraction.rs:686`
```rust
child_components: Vec::new(), // TODO: extract from JSX
```

**Impact:** Dependency analysis cannot determine migration order.

### 1.3 Test Coverage Gap

The 20 Phase 1 tests pass because they test **structure detection** but not **content extraction**:
- `test_extract_use_state_hook` - checks state_name is "count", not that initial_value is "0"
- `test_parse_jsx_children` - checks children exist, not that expressions are extracted
- No test validates expression content extraction

**Recommendation:** Add tests that validate actual extracted content:
```rust
#[test]
fn test_expression_content_preserved() {
    let source = r#"function X() { return <div>{count + 1}</div>; }"#;
    let ext = extract_source(...);
    let expr = &ext.components[0].jsx.root.children[0];
    assert_eq!(expr.code, "count + 1"); // Currently fails!
}
```

---

## Phase 2: Spec Generation Audit

### 2.1 Spec Compliance Matrix

| Spec Requirement | Implementation Status | Notes |
|------------------|----------------------|-------|
| MigrationSpec structure | ✓ Complete | All fields present |
| StateFieldRecommendation | ✓ Complete | Including evidentiality |
| MessageRecommendation | ✓ Complete | Derived from hooks + handlers |
| EffectRecommendation | ✓ Complete | All strategies implemented |
| PropsRecommendation | ✓ Complete | Strategy and fields |
| Pattern library | ✓ Complete | 10 patterns defined |
| Ambiguity detection | ✓ Complete | Effect placement, callback props |
| Complexity calculation | ✓ Complete | Simple/Moderate/Complex |
| Type mapping | ✓ Complete | TS→Sigil type conversion |
| Dependency analysis | ⚠ Partial | Extraction doesn't provide |

### 2.2 Minor Issues

#### MINOR-1: Hardcoded Timestamp
**Location:** `spec.rs:786-789`
```rust
fn chrono_now() -> String {
    "2026-02-16T00:00:00Z".to_string()
}
```

**Recommendation:** Use actual chrono crate or system time.

#### MINOR-2: Type Inference Limited
**Location:** `spec.rs:822-843`

Only handles: `0`, `0.0`, `true/false`, `"string"`, `null`, `[]`, `{}`

Complex expressions like `items.length` or `props.initialCount` fall through to `Any~`.

### 2.3 Test Coverage Assessment

Phase 2 has solid test coverage with 18 tests covering:
- Recommendation generation from hooks
- Pattern selection logic
- Ambiguity detection
- Complexity scoring
- Target pattern inference (Actor vs Function)

---

## Phase 3: Code Generation Audit

### 3.1 Spec Compliance Matrix

| Spec Requirement (Section 9.3) | Implementation Status | Notes |
|--------------------------------|----------------------|-------|
| `invoke qliphoth·prelude·*` | ✓ Complete | Always generated |
| Message enum `ᛈ XMsg { }` | ✓ Complete | With variants |
| Actor `☉ actor X { }` | ✓ Complete | State fields, handlers |
| State fields `state x: T! = v,` | ✓ Complete | With evidentiality |
| Constructor `rite new()` | ✓ Complete | From props |
| Message handlers `on Msg { }` | ✓ Complete | Basic structure |
| View method `rite view(self)` | ✓ Complete | Returns VNode |
| Pure functions `rite x()` | ✓ Complete | No state/handlers |
| VNode builders `VNode·tag()` | ✓ Complete | All HTML tags |
| `·class()`, `·id()` | ✓ Complete | Common attributes |
| `·attr(name, value)` | ✓ Complete | Generic attributes |
| `·on_click(Msg)` | ✓ Complete | Event dispatch |
| `·text_child(text)` | ✓ Complete | Text content |
| `·child(node)` | ✓ Complete | Nesting |
| `VNode·fragment()` | ✓ Complete | Fragments |
| Conditionals `·when()` | ✓ Complete | But uses placeholder |
| List rendering | ✓ Complete | But uses placeholder |
| `qliphoth_router` import | ⚠ Deferred | Router hooks not extracted |
| `qliphoth_sys` import | ✓ Partial | Based on effect text |

### 3.2 Critical Issues

#### CRITICAL-5: Expressions Generate Invalid Code
**Location:** `generator.rs:311-317`
```rust
JsxNodeType::Expression { code } => {
    if code.contains("/*") {
        format!("{}·text_child(/* expression */)", pad)
    } else {
        format!("{}·text_child(self.{}·to_string())", pad, code)
    }
}
```

Since extraction always produces `code: "/* expression */"`, this generates:
```sigil
·text_child(/* expression */)
```
Which is invalid Sigil syntax.

#### CRITICAL-6: Pure Functions Reference `self`
**Location:** `generator.rs:316`

For pure function components, expressions still generate `self.{code}`:
```sigil
rite greeting(name: String) -> VNode! {
    VNode·div()
        ·text_child(self./* expression */·to_string())  // Wrong!
}
```

Pure functions don't have `self`. Should reference parameters directly.

### 3.3 Style Handling Incomplete
**Location:** `generator.rs:411-414`
```rust
"style" => {
    format!("·style(/* style object */)")
}
```

Style objects need proper conversion to individual style calls.

### 3.4 Test Coverage Gap

Phase 3 tests validate structure but use source that doesn't exercise expression extraction:
```rust
// This passes because no expressions reach the generator
let source = r#"function Empty() { return <div>Hello</div>; }"#;
```

Missing tests:
- Component with `{count}` expression
- Component with `{condition && <X/>}`
- Component with `{items.map(...)}`

---

## Recommendations

### Immediate (P0) - Block Production Use

1. **Implement `expr_to_source()`** in extraction.rs
   - Use swc's source map to reconstruct original expression text
   - Affects: extraction.rs lines 886, 938, 1342

2. **Fix pure function expression handling** in generator.rs
   - Check `target.pattern == TargetPattern::Function`
   - Reference parameters not `self`

3. **Add content validation tests**
   - Test that expressions are preserved through extraction
   - Test generated code compiles for expressions

### Short-term (P1) - Before Public Release

4. **Implement handler extraction**
   - Extract named handlers from function body
   - Analyze for state mutations and API calls

5. **Implement props extraction**
   - Parse destructuring patterns
   - Extract type annotations

6. **Implement childComponents**
   - Walk JSX tree for component references
   - Build dependency graph

### Medium-term (P2) - Feature Complete

7. **Add router hook detection** to HookType enum
8. **Implement style object parsing**
9. **Add custom hook analysis**
10. **Use real timestamps**

---

## Test Health Summary

| Phase | Tests | Pass | Coverage Quality |
|-------|-------|------|------------------|
| Phase 1 | 20 | 20 | Structure only, not content |
| Phase 2 | 18 | 18 | Good coverage |
| Phase 3 | 15 | 15 | Structure only, not expression paths |

**Key Insight:** Tests pass because they validate _structure_ not _content_. A component with `{count}` expressions would fail at runtime but tests don't exercise this path.

---

## Files Reviewed

- `parser/src/migrate/react/extraction.rs` (1430 lines)
- `parser/src/migrate/react/spec.rs` (901 lines)
- `parser/src/migrate/react/generator.rs` (522 lines)
- `parser/src/migrate/react/tests.rs` (~1500 lines)
- `parser/src/migrate/react/mod.rs` (21 lines)
- `qliphoth/docs/specs/REACT-MIGRATION.md` (1118 lines)
- `qliphoth/docs/specs/REACT-MIGRATION-TDD-ROADMAP.md` (575 lines)

---

## Conclusion

The React→Qliphoth migration system has a solid architectural foundation. The three-phase design (Extract → Spec → Generate) is correct and the type structures match the specification well.

However, **expression content extraction is fundamentally broken**. All JSX expressions become placeholder strings, making the generated code invalid. This is a critical blocker that should be fixed before any production use.

The test suite provides false confidence by testing structure detection rather than content preservation. Adding content validation tests will reveal these issues immediately.

**Recommended Next Step:** Implement `expr_to_source()` using swc source spans, then add a single test that validates `{count}` extracts as "count". This will unblock the entire pipeline.
