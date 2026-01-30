# Jormungandr Bootstrap Specification

> *"The World Serpent bites its own tail"*

## 1. Overview

**Jormungandr** is the self-hosting bootstrap initiative for the Sigil compiler. This specification defines:

1. The **C code generation semantics** for the Rust bootstrap compiler
2. The **runtime value representation** (SigilValue)
3. The **fixed-point verification** criteria
4. The **TDD test requirements** for achieving bootstrap

### 1.1 Bootstrap Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOOTSTRAP PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Self-Hosted Compiler (.sg files)                               │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────┐                                        │
│  │ Rust Interpreter    │  (sigil-lang/parser)                   │
│  │ "compile" mode      │                                        │
│  └─────────────────────┘                                        │
│           │                                                      │
│           ▼                                                      │
│  Generated C Code (sigil_bootstrap.c)                           │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────┐                                        │
│  │ GCC/Clang           │                                        │
│  └─────────────────────┘                                        │
│           │                                                      │
│           ▼                                                      │
│  Native Binary (build/sigil)                                    │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────┐                                        │
│  │ Self-Compilation    │  sigil compile *.sg -o sigil2.c        │
│  └─────────────────────┘                                        │
│           │                                                      │
│           ▼                                                      │
│  FIXED POINT: sigil_bootstrap.c == sigil2.c                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **No post-processing** — Generated C must be valid without sed/awk/python fixes
2. **Deterministic output** — Same input always produces identical C code
3. **Readable C** — Generated code should be human-debuggable
4. **Minimal runtime** — Runtime support functions are small and auditable

---

## 2. Runtime Value Representation

All Sigil values are represented at runtime as a tagged union:

### 2.1 SigilValue Structure

```c
typedef struct SigilValue {
    uint8_t tag;       // Type discriminant (TAG_*)
    uint8_t evidence;  // Evidentiality level (SIGIL_*)
    union {
        bool b;                                              // TAG_BOOL
        int64_t i;                                           // TAG_INT
        double f;                                            // TAG_FLOAT
        char c;                                              // TAG_CHAR
        char* s;                                             // TAG_STRING
        struct { SigilValue* data; size_t len; size_t cap; } arr;  // TAG_ARRAY
        struct { SigilValue* fields; size_t count; } tup;    // TAG_TUPLE
        struct { char* name; SigilValue* fields; size_t count; } struc;  // TAG_STRUCT
        void* ptr;                                           // TAG_VARIANT, etc.
    } v;
} SigilValue;
```

### 2.2 Type Tags

| Tag | Value | Sigil Type | C Access |
|-----|-------|------------|----------|
| `TAG_UNIT` | 0 | `()` | N/A |
| `TAG_BOOL` | 1 | `bool` | `v.b` |
| `TAG_INT` | 2 | `i8`..`i128`, `u8`..`u128` | `v.i` |
| `TAG_FLOAT` | 3 | `f32`, `f64` | `v.f` |
| `TAG_CHAR` | 4 | `char` | `v.c` |
| `TAG_STRING` | 5 | `String`, `str` | `v.s` |
| `TAG_ARRAY` | 6 | `[T]`, `Vec<T>` | `v.arr` |
| `TAG_TUPLE` | 7 | `(T, U, ...)` | `v.tup` |
| `TAG_STRUCT` | 8 | `struct Name { ... }` | `v.struc` |
| `TAG_NULL` | 9 | `null` | N/A |
| `TAG_RESULT_OK` | 14 | `Result::Ok(T)` | `v.ptr` |
| `TAG_RESULT_ERR` | 15 | `Result::Err(E)` | `v.ptr` |
| `TAG_VARIANT` | 16 | Enum variant | `v.ptr` |

### 2.3 Evidentiality Levels

| Level | Value | Sigil Marker | Meaning |
|-------|-------|--------------|---------|
| `SIGIL_KNOWN` | 0 | `!` | Computed locally, definitely exists |
| `SIGIL_UNCERTAIN` | 1 | `?` | May or may not exist |
| `SIGIL_REPORTED` | 2 | `~` | From external source |
| `SIGIL_PARADOX` | 3 | `‽` | Trust boundary crossed |

---

## 3. C Code Generation Semantics

This section defines **exactly** what C code must be generated for each Sigil construct.

### 3.1 Literals

#### 3.1.1 Integer Literals

| Sigil | Generated C |
|-------|-------------|
| `42` | `sigil_int(42)` |
| `42i32` | `sigil_int(42)` |
| `42u64` | `sigil_int(42)` |
| `0xFF` | `sigil_int(255)` |
| `0b1010` | `sigil_int(10)` |
| `0o755` | `sigil_int(493)` |
| `1_000_000` | `sigil_int(1000000)` |

#### 3.1.2 Float Literals

| Sigil | Generated C |
|-------|-------------|
| `3.14` | `sigil_float(3.14)` |
| `1e10` | `sigil_float(1e10)` |
| `2.5E-3` | `sigil_float(2.5e-3)` |

#### 3.1.3 String Literals

| Sigil | Generated C |
|-------|-------------|
| `"hello"` | `sigil_string("hello")` |
| `"line1\nline2"` | `sigil_string("line1\nline2")` |
| `"quote\"here"` | `sigil_string("quote\"here")` |

#### 3.1.4 Other Literals

| Sigil | Generated C |
|-------|-------------|
| `true` | `sigil_bool(true)` |
| `false` | `sigil_bool(false)` |
| `'a'` | `sigil_char('a')` |
| `'\n'` | `sigil_char('\n')` |
| `null` | `sigil_null()` |
| `()` | `sigil_unit()` |

### 3.2 Variables and Bindings

#### 3.2.1 Let Bindings

```sigil
let x = 42;
let mut y = "hello";
```

**Generated C:**
```c
SigilValue x = sigil_int(42);
SigilValue y = sigil_string("hello");
```

**Rules:**
- All variables are `SigilValue` type
- Mutability is not reflected in C (Sigil enforces at compile time)
- Variable names are preserved (no mangling for locals)

#### 3.2.2 Variable Reassignment

```sigil
let mut x = 1;
x = 2;
```

**Generated C:**
```c
SigilValue x = sigil_int(1);
x = sigil_int(2);
```

**Rules:**
- Second assignment must NOT redeclare: `x = ...` not `SigilValue x = ...`
- Codegen must track declared variables per scope

### 3.3 Operators

#### 3.3.1 Binary Operators

| Sigil | Generated C |
|-------|-------------|
| `a + b` | `sigil_add(a, b)` |
| `a - b` | `sigil_sub(a, b)` |
| `a * b` | `sigil_mul(a, b)` |
| `a / b` | `sigil_div(a, b)` |
| `a % b` | `sigil_rem(a, b)` |
| `a == b` | `sigil_eq(a, b)` |
| `a != b` | `sigil_ne(a, b)` |
| `a < b` | `sigil_lt(a, b)` |
| `a <= b` | `sigil_le(a, b)` |
| `a > b` | `sigil_gt(a, b)` |
| `a >= b` | `sigil_ge(a, b)` |
| `a && b` | `sigil_and(a, b)` |
| `a \|\| b` | `sigil_or(a, b)` |
| `a & b` | `sigil_bit_and(a, b)` |
| `a \| b` | `sigil_bit_or(a, b)` |
| `a ^ b` | `sigil_bit_xor(a, b)` |
| `a << b` | `sigil_shl(a, b)` |
| `a >> b` | `sigil_shr(a, b)` |

#### 3.3.2 Unary Operators

| Sigil | Generated C |
|-------|-------------|
| `-x` | `sigil_neg(x)` |
| `!x` | `sigil_not(x)` |
| `*x` | `sigil_deref(x)` |
| `&x` | `sigil_ref(x)` |

### 3.4 Field Access

#### 3.4.1 Struct Field Access

```sigil
let x = point.x;
```

**Generated C:**
```c
SigilValue x = sigil_struct_field(point, "x");
```

**CRITICAL:** No space between variable and dot. The pattern `point. x` is INVALID.

#### 3.4.2 Tuple Field Access

```sigil
let first = tuple.0;
```

**Generated C:**
```c
SigilValue first = sigil_tuple_field(tuple, 0);
```

### 3.5 Array/Index Access

```sigil
let item = arr[i];
```

**Generated C:**
```c
SigilValue item = sigil_index(arr, i);
```

**NOT:** `arr[i]` — C arrays are not SigilValue arrays

### 3.6 Function Calls

#### 3.6.1 Free Functions

```sigil
let result = foo(a, b);
```

**Generated C:**
```c
SigilValue result = sigil_foo(a, b);
```

**Rules:**
- Function names prefixed with `sigil_`
- All arguments passed as `SigilValue`

#### 3.6.2 Method Calls

```sigil
let len = vec.len();
let result = vec.push(item);
```

**Generated C:**
```c
SigilValue len = sigil_Vec____len(vec);
SigilValue result = sigil_Vec____push(vec, item);
```

**Rules:**
- Methods become `sigil_TypeName____method_name(self, args...)`
- `::` becomes `____` (four underscores)
- Self is always first argument

#### 3.6.3 Mutable Method Semantics

**CRITICAL:** Methods that mutate must capture return value:

```sigil
vec.push(item);
```

**WRONG:**
```c
sigil_Vec____push(vec, item);  // Result lost!
```

**CORRECT:**
```c
vec = sigil_Vec____push(vec, item);  // Captures potentially reallocated pointer
```

### 3.7 Control Flow

#### 3.7.1 If Expression

```sigil
let x = if cond { 1 } else { 2 };
```

**Generated C:**
```c
SigilValue x;
if (sigil_truthy(cond)) {
    x = sigil_int(1);
} else {
    x = sigil_int(2);
}
```

#### 3.7.2 If Statement

```sigil
if cond {
    do_something();
}
```

**Generated C:**
```c
if (sigil_truthy(cond)) {
    sigil_do_something();
}
```

#### 3.7.3 Match Expression

```sigil
match value {
    Pattern::A(x) => expr1,
    Pattern::B => expr2,
    _ => expr3,
}
```

**Generated C:**
```c
SigilValue _match_result;
SigilValue _match_value = value;
if (sigil_variant_matches(_match_value, "Pattern", "A")) {
    SigilValue x = sigil_variant_field(_match_value, 0);
    _match_result = /* expr1 */;
} else if (sigil_variant_matches(_match_value, "Pattern", "B")) {
    _match_result = /* expr2 */;
} else {
    _match_result = /* expr3 */;
}
```

#### 3.7.4 Loop

```sigil
loop {
    if cond { break; }
}
```

**Generated C:**
```c
while (1) {
    if (sigil_truthy(cond)) { break; }
}
```

#### 3.7.5 While Loop

```sigil
while cond {
    body();
}
```

**Generated C:**
```c
while (sigil_truthy(cond)) {
    sigil_body();
}
```

#### 3.7.6 For Loop

```sigil
for item in iter {
    process(item);
}
```

**Generated C:**
```c
{
    SigilValue _iter = iter;
    for (size_t _i = 0; _i < sigil_len(_iter); _i++) {
        SigilValue item = sigil_index(_iter, sigil_int(_i));
        sigil_process(item);
    }
}
```

### 3.8 Functions

#### 3.8.1 Function Definition

```sigil
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

**Generated C:**
```c
/* Function: add */
SigilValue sigil_add(SigilValue a, SigilValue b) {
    return sigil_add(a, b);
}
```

**Rules:**
- Return type is always `SigilValue`
- All parameters are `SigilValue`
- Function name prefixed with `sigil_`
- Comment with original function name for debugging

#### 3.8.2 Methods (impl blocks)

```sigil
impl Point {
    fn distance(&self, other: &Point) -> f64 {
        // ...
    }
}
```

**Generated C:**
```c
/* Method: Point::distance */
SigilValue sigil_Point____distance(SigilValue self, SigilValue other) {
    // ...
}
```

### 3.9 Structs

#### 3.9.1 Struct Definition

```sigil
struct Point {
    x: f64,
    y: f64,
}
```

**Generated C:**
```c
/* Struct: Point */
/* Fields: x, y */
/* (No C struct generated - uses SigilValue with TAG_STRUCT) */
```

#### 3.9.2 Struct Construction

```sigil
let p = Point { x: 1.0, y: 2.0 };
```

**Generated C:**
```c
const char* _Point_fields[] = {"x", "y"};
SigilValue _Point_values[] = {sigil_float(1.0), sigil_float(2.0)};
SigilValue p = sigil_struct("Point", _Point_fields, _Point_values, 2);
```

### 3.10 Enums

#### 3.10.1 Enum Definition

```sigil
enum Option<T> {
    Some(T),
    None,
}
```

**Generated C:**
```c
/* Enum: Option */
/* Variants: Some(T), None */
```

#### 3.10.2 Variant Construction

```sigil
let x = Option::Some(42);
let y = Option::None;
```

**Generated C:**
```c
SigilValue x = sigil_variant("Option", "Some", sigil_int(42));
SigilValue y = sigil_variant("Option", "None", sigil_null());
```

### 3.11 Closures

```sigil
let f = |x| x + 1;
let result = f(5);
```

**Generated C:**
```c
/* Closure: closure_0 */
static SigilValue sigil_closure_0(SigilValue x) {
    return sigil_add(x, sigil_int(1));
}

SigilValue f = sigil_closure(sigil_closure_0);
SigilValue result = sigil_call_closure(f, sigil_int(5));
```

**Rules:**
- Closures become static functions with unique names
- Captured variables become additional parameters or a capture struct
- Closure value wraps function pointer

### 3.12 Result and Option Handling

#### 3.12.1 Try Operator (?)

```sigil
let value = fallible_fn()?;
```

**Generated C:**
```c
SigilValue _try_result = sigil_fallible_fn();
if (sigil_is_err(_try_result)) {
    return _try_result;  // Propagate error
}
SigilValue value = sigil_unwrap_result(_try_result);
```

#### 3.12.2 Null Coalescing (??)

```sigil
let value = maybe_null ?? default;
```

**Generated C:**
```c
SigilValue value = sigil_truthy(maybe_null) ? maybe_null : default;
```

---

## 4. Name Mangling

### 4.1 Rules

| Sigil Name | C Name |
|------------|--------|
| `foo` | `sigil_foo` |
| `Type::method` | `sigil_Type____method` |
| `mod::func` | `sigil_mod____func` |
| `Type<T>::method` | `sigil_Type____method` (generics erased) |

### 4.2 Special Cases

| Pattern | Handling |
|---------|----------|
| Operators (`+`, `-`, etc.) | Become function names: `sigil_add`, `sigil_sub` |
| Greek letters (`τ`, `φ`) | Transliterated: `sigil_tau`, `sigil_phi` |
| Unicode identifiers | Hex-encoded: `_XXXX` |

---

## 5. Scoping and Variable Tracking

### 5.1 Scope Blocks

Each `{ }` block introduces a new scope. Codegen must track:

1. **Declared variables** — Names that have been declared with `SigilValue name = ...`
2. **Assigned variables** — Names that exist but are being reassigned

### 5.2 Variable Redeclaration Bug

**WRONG:** Redeclaring in same scope:
```c
SigilValue x = sigil_int(1);
// ... later in same scope ...
SigilValue x = sigil_int(2);  // ERROR: redeclaration
```

**CORRECT:** Assignment after declaration:
```c
SigilValue x = sigil_int(1);
// ... later in same scope ...
x = sigil_int(2);  // OK: assignment
```

### 5.3 Scope Tracking Algorithm

```
for each statement:
    if is_let_binding(stmt):
        name = get_binding_name(stmt)
        if name in current_scope.declared:
            emit_assignment(name, value)  // Just assign
        else:
            emit_declaration(name, value)  // SigilValue name = ...
            current_scope.declared.add(name)
```

---

## 6. Fixed-Point Verification

### 6.1 Criteria

The bootstrap achieves **fixed point** when:

```bash
# Compile self-hosted with Rust interpreter
cargo run -- compile self-hosted/src/*.sg -o build/sigil_bootstrap.c
gcc -o build/sigil build/sigil_bootstrap.c -lm

# Compile self-hosted with native bootstrap
./build/sigil compile self-hosted/src/*.sg -o build/sigil2.c

# Verify identical output
diff build/sigil_bootstrap.c build/sigil2.c
# Must produce no output (files identical)
```

### 6.2 Determinism Requirements

For fixed-point to be achievable:

1. **No random identifiers** — Temp variable names must be deterministic
2. **Consistent ordering** — HashMap iteration order must be stable
3. **No timestamps** — No build time/date in output
4. **Canonical formatting** — Consistent whitespace/newlines

---

## 7. Test Requirements

### 7.1 Unit Tests for Codegen

Each section (3.1–3.12) requires corresponding tests:

```rust
// In parser/src/codegen_tests.rs

#[test]
fn codegen_integer_literal() {
    assert_codegen("42", "sigil_int(42)");
}

#[test]
fn codegen_field_access_no_space() {
    let code = compile_to_c("fn main() { let p = Point { x: 1 }; p.x }");
    assert!(!code.contains(". "), "Field access must not have space before field name");
    assert!(code.contains("sigil_struct_field(p, \"x\")"));
}

#[test]
fn codegen_vec_push_captures_result() {
    let code = compile_to_c("fn main() { let mut v = Vec::new(); v.push(1); }");
    assert!(code.contains("v = sigil_Vec____push(v,"),
            "Vec::push must capture result back to variable");
}

#[test]
fn codegen_no_variable_redeclaration() {
    let code = compile_to_c("fn main() { let mut x = 1; x = 2; }");
    let decl_count = code.matches("SigilValue x =").count();
    assert_eq!(decl_count, 1, "Variable should only be declared once");
}
```

### 7.2 Integration Tests

```rust
#[test]
fn bootstrap_produces_valid_c() {
    let c_code = compile_file("self-hosted/src/span.sg");

    // Write to temp file
    std::fs::write("/tmp/test.c", &c_code).unwrap();

    // Must compile with GCC without errors
    let status = Command::new("gcc")
        .args(["-c", "-w", "/tmp/test.c", "-o", "/tmp/test.o"])
        .status()
        .unwrap();

    assert!(status.success(), "Generated C must compile");
}
```

### 7.3 Fixed-Point Test

```rust
#[test]
#[ignore]  // Run manually: cargo test fixed_point -- --ignored
fn fixed_point_achieved() {
    // Build bootstrap
    let bootstrap_c = compile_all_modules("self-hosted/src/");
    std::fs::write("build/sigil_bootstrap.c", &bootstrap_c).unwrap();

    // Compile bootstrap
    assert!(Command::new("gcc")
        .args(["-O2", "-o", "build/sigil", "build/sigil_bootstrap.c", "-lm"])
        .status().unwrap().success());

    // Use bootstrap to compile itself
    let output = Command::new("./build/sigil")
        .args(["compile", "self-hosted/src/*.sg"])
        .output().unwrap();

    let sigil2_c = String::from_utf8(output.stdout).unwrap();

    // Compare
    assert_eq!(bootstrap_c, sigil2_c, "Fixed point not achieved");
}
```

---

## 8. Known Codegen Bugs (To Fix)

These bugs currently require post-processing hacks and must be fixed in `codegen.rs`:

| Bug ID | Description | Current Hack | Required Fix |
|--------|-------------|--------------|--------------|
| CG-001 | Field access emits space: `obj. field` | sed replacement | Fix field access emission in codegen |
| CG-002 | Cast emits `/* unknown token */` | sed replacement | Proper cast expression handling |
| CG-003 | Index emits `arr[ N]` instead of `sigil_index` | sed replacement | Use sigil_index for all indexing |
| CG-004 | Variable redeclaration in same scope | sed replacement | Track declared variables per scope |
| CG-005 | Vec::push result not captured | sed + python | Emit `var = method(var, ...)` for mutating methods |
| CG-006 | Duplicate function definitions | awk removal | Track emitted functions, emit once |
| CG-007 | Closure missing self parameter | sed fix | Proper closure capture codegen |
| CG-008 | Format string escaping issues | various | Proper string escape in format! expansion |

---

## 9. Appendix: Runtime Function Signatures

### 9.1 Constructors

```c
SigilValue sigil_unit(void);
SigilValue sigil_bool(bool b);
SigilValue sigil_int(int64_t i);
SigilValue sigil_float(double f);
SigilValue sigil_char(char c);
SigilValue sigil_string(const char* s);
SigilValue sigil_array(size_t cap);
SigilValue sigil_tuple(SigilValue* fields, size_t count);
SigilValue sigil_struct(const char* name, const char** field_names, SigilValue* values, size_t count);
SigilValue sigil_variant(const char* enum_name, const char* variant_name, SigilValue payload);
SigilValue sigil_null(void);
```

### 9.2 Accessors

```c
SigilValue sigil_struct_field(SigilValue s, const char* field);
SigilValue sigil_tuple_field(SigilValue t, size_t index);
SigilValue sigil_index(SigilValue arr, SigilValue idx);
size_t sigil_len(SigilValue v);
bool sigil_truthy(SigilValue v);
```

### 9.3 Operators

```c
SigilValue sigil_add(SigilValue a, SigilValue b);
SigilValue sigil_sub(SigilValue a, SigilValue b);
SigilValue sigil_mul(SigilValue a, SigilValue b);
SigilValue sigil_div(SigilValue a, SigilValue b);
SigilValue sigil_rem(SigilValue a, SigilValue b);
SigilValue sigil_neg(SigilValue a);
SigilValue sigil_eq(SigilValue a, SigilValue b);
SigilValue sigil_ne(SigilValue a, SigilValue b);
SigilValue sigil_lt(SigilValue a, SigilValue b);
SigilValue sigil_le(SigilValue a, SigilValue b);
SigilValue sigil_gt(SigilValue a, SigilValue b);
SigilValue sigil_ge(SigilValue a, SigilValue b);
SigilValue sigil_and(SigilValue a, SigilValue b);
SigilValue sigil_or(SigilValue a, SigilValue b);
SigilValue sigil_not(SigilValue a);
```

### 9.4 Result/Option

```c
SigilValue sigil_Ok(SigilValue v);
SigilValue sigil_Err(SigilValue e);
bool sigil_is_ok(SigilValue v);
bool sigil_is_err(SigilValue v);
SigilValue sigil_unwrap_result(SigilValue v);
```

### 9.5 Display/Debug

```c
SigilValue sigil_display(SigilValue v);
SigilValue sigil_debug(SigilValue v);
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial specification |

---

*This specification is part of the Jormungandr bootstrap initiative for the Sigil programming language.*
