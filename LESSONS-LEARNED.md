# Sigil Language - Lessons Learned

This file captures organizational memory across agent sessions. Read before starting work.
Document discoveries and mistakes when ending sessions.

---

## 2026-02-11 - LLVM Codegen Float Operations

### Context
Implementing DCT (Discrete Cosine Transform) benchmark to compare Sigil LLVM vs Rust performance.

### What Happened
Float operations produced garbage values (`3.0 + 2.0 = -1.11254e-308`) even though
code compiled without errors.

### Root Cause
Sigil stores ALL values as i64 (including floats as bit patterns). The LLVM codegen
was treating float operations as integer operations:
- Used `add` instead of `fadd`
- Used `mul` instead of `fmul`
- No bitcast from i64 to f64 before operations

### Lesson
When Sigil represents floats as i64 bit patterns, every float operation must:
1. Bitcast i64 → f64
2. Perform float operation (fadd, fsub, fmul, fdiv)
3. Bitcast f64 → i64 for storage

This requires tracking which variables are float-typed through the compilation scope.

### Prevention
- Added `float_vars: HashSet<String>` to `CompileScope`
- Added `is_float_expr_with_scope()` function to detect float expressions
- Added `compile_float_binary_op()` for proper float arithmetic

---

## 2026-02-11 - Vec Memory Layout

### Context
Vec indexing returned wrong values (struct fields instead of data elements).

### What Happened
`v[0]` returned values like 3, 5, 10 instead of expected data. These corresponded to
the `len`, `cap`, and first data element positions.

### Root Cause
Vec layout is `{len: i64, cap: i64, data: i64[]}` with data stored INLINE starting at
offset 2, NOT as a pointer. The codegen was treating `data` as a pointer field at offset 2,
when it's actually the start of inline data.

### Lesson
Vec data is inline, not pointed-to:
```
offset 0: len (i64)
offset 1: cap (i64)
offset 2: data[0] (i64)
offset 3: data[1] (i64)
...
```

To access `v[i]`, compute `base_ptr + (i + 2) * 8`.

### Prevention
- All Vec index operations now add 2 to the index before GEP
- This applies to both read and write operations

---

## 2026-02-11 - Math Functions (PI, sqrt, cos)

### Context
DCT calculation returned 0 for PI and all trig functions.

### What Happened
`PI()` returned 0. `sqrt()`, `cos()`, `sin()` all returned 0.

### Root Cause
1. `PI()` function was never implemented in LLVM codegen or C runtime
2. Method calls like `value.sqrt()` weren't mapped to runtime functions

### Lesson
Math operations in LLVM codegen require explicit mapping:
- `PI()` → custom `sigil_pi()` function (returns f64 bits as i64)
- `.sqrt()` → `sigil_sqrt()`
- `.cos()` → `sigil_cos()`
- etc.

### Prevention
- Added `sigil_pi()` to both JIT (Rust) and C runtime
- Added explicit MethodCall handling for math methods in LLVM codegen
- When adding new stdlib functions, ensure all backends implement them

---

## 2026-02-11 - Benchmark Optimization Away

### Context
Running DCT benchmark showed 0 microseconds for all operations.

### What Happened
Benchmark loop ran but timing showed 0:
```
Size     DCT (μs)
  16         0
  32         0
```

### Root Cause
Using `_ = dct_1d(data, n)` allowed LLVM to optimize away the entire function call
since the result was unused.

### Lesson
To prevent dead code elimination in benchmarks:
1. Accumulate results: `acc = acc + result[0]`
2. Print the accumulator at the end
3. This forces LLVM to actually execute the code

### Prevention
- Benchmark template now includes accumulator pattern
- Always use benchmark results in observable output

---

## 2026-02-21 - Compiler Audit: Gaps Relevant to sigil-codec and stdlib Authoring

### Context
Deep audit of the sigil-lang compiler source to understand what is actually
implemented vs. documented, before relying on it for sigil-codec and
ritualis-core.

### Findings

#### 1. Evidentiality Enforcement is Incomplete
`typeck.rs` tracks evidentiality markers (`!`, `~`, `?`, `‽`) and propagates
them through the type system, but enforcement is not complete:
- Markers are parsed and stored on types
- Some validation exists (e.g. can't assign `~` to `!` without local check)
- Propagation through function calls, closures, and trait method dispatch is
  incomplete — several `TODO` comments in typeck.rs confirm this
- **Impact:** Code using `~` return types may silently compile even if the
  evidentiality chain is incorrect. Do not rely on the compiler to catch
  evidentiality bugs yet; reason about them manually.

#### 2. Generic Resolution Has Known Gaps
- Generic structs: work
- Generic traits: partially work
- Turbofish syntax (`::<T>`) on method calls: **not fully tested**
- Nested generics in turbofish (`collect::<CodecResult[Vec[T>]>>()`) are
  particularly risky — the parser handles them but the type checker may
  not resolve them correctly in all positions
- **Impact:** The `collect::<Result[Vec[semver·Version], _>>()` patterns
  in registry.sigil and the codec may fail to type-check.

#### 3. `collect()` Turbofish + Nested Generics
Related to #2. The specific pattern:
```sigil
·map({...})·collect::<Result[Vec[T], _>>()
```
uses nested generic brackets inside turbofish. The Sigil parser uses `[`/`]`
for generics (not `<`/>`), so:
```sigil
collect::<codec·CodecResult[Vec[PatchOp>]>>()
```
is syntactically ambiguous — the `>]>>` closing is hard to parse correctly.
**Recommendation:** Prefer collecting into an explicit intermediate binding:
```sigil
≔ items!: Vec[PatchOp] = Vec·new();
∀ v ∈ arr { items.push(PatchOp·decode(v)?); }
```

#### 4. Float Handling in Interpreter vs. LLVM
Float operations work correctly in the interpreter (default backend). In the
LLVM backend, floats require the bitcast wrapping pattern (see 2026-02-11
entry). The codec `primitives.sg` uses `f64` and `f32` — these will work
correctly under the interpreter but may have issues under LLVM until all
float codepaths are audited.

#### 5. No Remote Dependency Resolution
`Sigil.toml` only supports `path = "..."` local dependencies. Version
specifiers (`"^1.0"`, etc.) are not resolved. The `sigil-codec` path dep
in ritualis-core's Sigil.toml (`path = "../../../sigil-lang/stdlib/codec"`)
is the correct approach — registry-based deps are not yet supported.

#### 6. Feature Flags in Sigil.toml Not Implemented
`//@ rune: cfg(feature = "...")` on scroll/module declarations is not yet
wired up — `ast::Module` has no `attrs` field, so feature-gated modules
do not compile conditionally. Avoid relying on feature flags for now.

#### 7. Concurrency Primitives Are Stubs in Interpreter
`weave`, `flow`, `voice`, `|await·all`, `|await·race` exist in the parser
grammar but are stubs in the interpreter. They return `RuntimeError::new("todo: ...")`
on execution. Async/await via `tokio` interop is the working path for now
(through the Rust transpiler backend).

#### 8. `//@ rune: cfg(test)` vs `scroll tests` Pattern
The codebase uses both `// cfg(test)` (comment) and `//@ rune: cfg(test)`
(annotation). The `scroll tests { ... }` pattern with `// cfg(test)` above
it is parsed as a regular module — the test-only gate is not enforced by the
compiler yet. Test scrolls compile in all modes. This is fine for now but
means test code is included in release builds.

#### 9. `#[...]` Attribute Syntax Still Accepted
Rust-compat `#[derive(...)]`, `#[cfg(...)]` etc. are accepted as a shim.
The canonical Sigil form is `//@ rune: ...`. New code should use the rune
form; the `#[...]` shim is planned for removal (see STASH-TODO.md).

#### 10. Agent Infrastructure (Aegis, Anima, etc.) Does Not Exist
Nine v0.4.0 agent modules listed in the README (`aegis`, `anima`, `commune`,
`covenant`, `daemon`, `gnosis`, `omen`, `oracle`, `engram`) exist only as
directories with README files. There is no implementation. Do not reference
them in any code.

#### 11. "Polycultural" stdlib Functions Do Not Exist
README lists functions like `gematria()`, `cast_iching()`, `sacred_freq()`,
`vigesimal_encode()`, etc. None of these are in `stdlib.rs` or any `.sg`
file. Do not use them.

### What Is Solid
- Lexer: complete, all Sigil tokens correctly lexed
- Parser: all declared syntax parses (11,000+ rule grammar)
- Interpreter: ~1,494 stdlib functions, tree-walking, fully functional
- Cranelift JIT: works (feature-gated)
- LLVM backend: works with known float/Vec caveats (see earlier entries)
- `sigil build`: reads Sigil.toml, resolves path deps, compiles
- `sigil check`: type checks, produces diagnostics
- `sigil test`: runs `//@ rune: test` annotated functions
- Pure Sigil libraries: codec (JSON+TOML) self-hosts correctly

### Recommendations for stdlib Authors
1. Test under the interpreter first (`sigil run`) before worrying about LLVM
2. Avoid complex turbofish generics — prefer explicit intermediate bindings
3. Do not use `weave`/`flow`/`voice` yet
4. Track evidentiality manually — the compiler won't catch all violations
5. Use `//@ rune:` annotations, not `#[...]`
6. Path-based deps only in Sigil.toml

---

## Template for Future Entries

```markdown
## [Date] - [Brief Title]

### Context
What were you trying to do?

### What Happened
What went wrong or unexpectedly right?

### Root Cause
Why did this happen?

### Lesson
What should future agents know?

### Prevention
How do we avoid this in future?
```
