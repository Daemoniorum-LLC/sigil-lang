# Jormungandr TDD Guide

> **Test-Driven Development for the Sigil Bootstrap Compiler**

This document establishes the TDD workflow for fixing codegen bugs and achieving fixed-point bootstrap.

---

## 1. The Problem

The self-hosted compiler's C code generation (`codegen.sg`) has bugs that require ugly post-processing hacks in `build.sh`. These include sed, awk, and Python scripts to fix malformed output.

**This is unacceptable.** The codegen should produce correct C directly.

---

## 2. Identified Codegen Bugs

| Bug ID | Description | Current Hack | Priority |
|--------|-------------|--------------|----------|
| CG-001 | Field access emits spurious space: `obj. field` | sed | Critical |
| CG-002 | Cast expression emits `/* unknown token */` | sed | Critical |
| CG-003 | Array index should use `sigil_index()` not `arr[N]` | sed | Critical |
| CG-004 | Variable redeclared in same scope | sed | Critical |
| CG-005 | `Vec::push` result not captured back to variable | Python | Critical |
| CG-006 | Duplicate function definitions emitted | awk | High |
| CG-007 | Closure missing self parameter when capturing | sed | High |
| CG-008 | Format string escaping issues | various | Medium |

---

## 3. TDD Workflow

### For Each Bug:

#### Step 1: Write Failing Test

Create or add to `tests/test_codegen.sg`:

```sigil
fn test_cg001_field_access_no_space() -> !bool {
    print("  [CG-001] Field access has no space... ");

    // Input that triggers the bug
    let source = "fn main() { let p = Point { x: 1 }; p.x }";

    // Compile to C
    let c_code = compile_to_c(source);

    // Assert correct output
    if c_code.contains(". ") {
        eprintln("FAIL: Found '.' followed by space");
        eprintln("Generated: {}", c_code);
        return false;
    }

    if !c_code.contains("sigil_struct_field(") {
        eprintln("FAIL: Should use sigil_struct_field()");
        return false;
    }

    println("PASS");
    true
}
```

#### Step 2: Run Test (Must Fail)

```bash
cd sigil-lang/parser
cargo run --release -- run-dir ../self-hosted/src -- test ../self-hosted/tests/test_codegen.sg
```

Expected output: `FAIL: Found '.' followed by space`

If the test passes, either:
- The bug is already fixed (verify by removing the hack from `build.sh`)
- The test doesn't actually trigger the bug (refine the test)

#### Step 3: Locate Bug in `codegen.sg`

Search for field access emission:

```bash
grep -n "\..*push_str\|emit.*field" src/codegen.sg
```

#### Step 4: Fix the Bug

Example fix:

```sigil
// BEFORE (buggy):
fn emit_field_access(&mut self, expr: !IrExpr, field: !String) {
    self.emit_expr(expr);
    self.output.push_str(". ");  // BUG HERE
    self.output.push_str(&field);
}

// AFTER (correct per spec):
fn emit_field_access(&mut self, expr: !IrExpr, field: !String) {
    self.output.push_str("sigil_struct_field(");
    self.emit_expr(expr);
    self.output.push_str(", \"");
    self.output.push_str(&field);
    self.output.push_str("\")");
}
```

#### Step 5: Run Test (Must Pass)

```bash
cargo run --release -- run-dir ../self-hosted/src -- test ../self-hosted/tests/test_codegen.sg
```

Expected output: `PASS`

#### Step 6: Remove Hack from `build.sh`

Delete the corresponding sed/awk/python workaround:

```bash
# DELETE THIS:
sed -i -E 's/([a-zA-Z_][a-zA-Z0-9_]*)\. ([a-zA-Z_][a-zA-Z0-9_]*)/sigil_struct_field(\1, "\2")/g' "$UNIFIED"
```

#### Step 7: Verify Build Still Works

```bash
cd self-hosted
./build.sh
```

#### Step 8: Run Full Test Suite

```bash
./run_tests.sh
```

---

## 4. Test Helper Functions

Add these to a shared test utilities module or each test file:

```sigil
/// Compile Sigil source to C code
fn compile_to_c(source: !&str) -> !String {
    let lexer = Lexer::new(source.to_string());
    let mut parser = Parser::new(lexer);
    let ast = parser.parse_file()?;
    let ir = lower(ast);
    CodeGen::new().generate(ir)
}

/// Assert C output contains expected pattern
fn assert_c_contains(c_code: !&str, expected: !&str, msg: !&str) {
    if !c_code.contains(expected) {
        panic(format!("FAIL {}: expected '{}' not found\nOutput:\n{}", msg, expected, c_code));
    }
}

/// Assert C output does NOT contain forbidden pattern
fn assert_c_not_contains(c_code: !&str, forbidden: !&str, msg: !&str) {
    if c_code.contains(forbidden) {
        panic(format!("FAIL {}: forbidden '{}' found\nOutput:\n{}", msg, forbidden, c_code));
    }
}

/// Assert C code compiles with GCC (integration test)
fn assert_c_compiles(c_code: !&str) -> !bool {
    // Write to temp file
    let path = "/tmp/sigil_test.c";
    write_file(path, c_code);

    // Try to compile
    let result = exec("gcc", ["-c", "-w", path, "-o", "/tmp/sigil_test.o"]);
    result.success
}
```

---

## 5. Expected C Output Reference

### Field Access

```sigil
// Input
p.x

// Expected C
sigil_struct_field(p, "x")
```

### Method Call

```sigil
// Input
vec.push(item)

// Expected C (mutable - captures result)
vec = sigil_Vec____push(vec, item)
```

### Index Access

```sigil
// Input
arr[i]

// Expected C
sigil_index(arr, i)
```

### Cast

```sigil
// Input
x as u32

// Expected C
sigil_cast(x, "u32")
// OR direct cast if we know the type
(uint32_t)x.v.i
```

### Variable Declaration

```sigil
// Input
let x = 1;
x = 2;

// Expected C
SigilValue x = sigil_int(1);
x = sigil_int(2);  // NO redeclaration
```

### Closure

```sigil
// Input (closure that captures self)
impl Foo {
    fn bar(&self) {
        let f = |x| self.process(x);
    }
}

// Expected C
static SigilValue sigil_closure_0(SigilValue self, SigilValue x) {
    return sigil_Foo____process(self, x);
}
```

---

## 6. Progress Checklist

### Phase 1: Critical Bugs

- [ ] CG-001: Field access (no space)
- [ ] CG-002: Cast expression
- [ ] CG-003: Array indexing
- [ ] CG-004: Variable scoping
- [ ] CG-005: Mutable method capture

### Phase 2: High Priority

- [ ] CG-006: Duplicate functions
- [ ] CG-007: Closure capture

### Phase 3: Medium Priority

- [ ] CG-008: Format strings

### Phase 4: Verification

- [ ] All post-processing removed from `build.sh`
- [ ] `./build.sh` produces valid C directly
- [ ] Fixed point test passes

---

## 7. Definition of Done

A bug is considered **fixed** when:

1. ✅ Test exists in `test_codegen.sg`
2. ✅ Test passes
3. ✅ Corresponding hack removed from `build.sh`
4. ✅ Full test suite still passes
5. ✅ Bootstrap build still succeeds

---

## 8. Reference

- [17-JORMUNGANDR-BOOTSTRAP.md](docs/specs/17-JORMUNGANDR-BOOTSTRAP.md) — Full codegen specification
- [tests/TEST_COVERAGE_GAPS.md](tests/TEST_COVERAGE_GAPS.md) — Existing gap analysis
- [build.sh](build.sh) — See hacks to remove
