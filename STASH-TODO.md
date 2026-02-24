# Stash Recovery TODO - sigil-lang

**Date**: 2026-02-08
**Backup Location**: `/home/crook/stash-backup/`

## Stashes to Review

### @{5} - model-mgmt (CRITICAL - Self-hosted compiler)

Patch: `stash-5-model-mgmt.patch`

Changes:
- `self-hosted/build/sigil_bootstrap.c`
- `self-hosted/src/codegen.sg`
- `self-hosted/src/driver.sg`
- `self-hosted/src/parser.sg`

### sigil-stash-0-gdpr (Website)

Patch: `sigil-stash-0-gdpr.patch`

Changes:
- `website/css/styles.css` - Hero section styling
- `website/index.html` - Hero section content

## Recovery Steps

### Self-hosted Compiler (@{5})
1. This is CRITICAL work on the self-hosted Sigil compiler
2. Review parser, codegen, and driver changes carefully
3. Test bootstrap compilation after applying
4. Apply with: `git apply --3way`

### Website (sigil-stash-0)
1. Hero section redesign:
   - Smaller logo (180px → 140px)
   - Condensed copy
   - New hero-subtitle class
   - Responsive adjustments
2. Apply with: `git apply sigil-stash-0-gdpr.patch`

## Notes

- Self-hosted compiler changes are significant
- Website changes are cosmetic but improve above-fold
- bootstrap.c is generated - may need regeneration

## Eliminate Rust-compat `#[...]` Attribute Syntax

**Context:** Sigil's native annotation syntax is `//@ rune: name`. The `#[...]`
bracket form is accepted as a Rust compat shim but erodes Sigil's identity.

**What needs to happen:**

1. **Define the rune vocabulary** — establish canonical `//@ rune:` forms for
   every `#[...]` attribute in use across the codebase:
   - `#[repr(C)]` → `//@ rune: repr(C)`
   - `#[cfg(...)]` → `//@ rune: cfg(...)`
   - `#[derive(...)]` → `//@ rune: derive(...)` (or a Sigil-native equivalent)
   - `#[allow(...)]`, `#[inline]`, etc.

2. **Wire `//@ rune: cfg(feature = "...")` on scroll/module declarations** —
   currently `Module` in the AST has no `attrs` field. Gating a `scroll` on a
   feature requires:
   - Add `attrs: Vec<Attribute>` to `ast::Module`
   - Parse rune annotations before `☉ scroll name;` in the parser
   - Check attrs in all five compilation passes (collect_uses, collect_types,
     prescan, collect_sigs, compile_bodies) in `statements.rs`
   - Pass active features from `sigil.toml` into the compiler context

3. **Sweep the codebase** — replace all `#[...]` with `//@ rune:` equivalents
   in one pass once the vocabulary is settled.

**Known occurrences today:**
- `qliphoth/src/platform/native.sigil:63` — `#[repr(C)]` on `NativeEventData`

**Driving motivation:** `platform/mod.sigil` has a pending `scroll native;` that
needs feature-gating (`native = ["gtk"]` already declared in `sigil.toml`) — the
correct form once implemented:
```
//@ rune: cfg(feature = "native")
☉ scroll native;
```

## LLVM Backend Optimization (DCT Benchmark)

**Current Status:** Sigil LLVM is ~1.2x slower than Rust on float-heavy workloads (DCT benchmark: 36ms vs 30ms)

**Benchmark:** `/tmp/bench_dct_sigil.sg` - 1000 iterations of 64x64 DCT transform

**Potential Optimizations to Investigate:**
1. **Eliminate alloca for float reinterpretation** - Currently using store/load through stack for i64↔f64 conversion. Could use LLVM bitcast intrinsics if available in inkwell.
2. **SIMD vectorization** - DCT inner loop is highly parallelizable. Could use F32x16 SIMD primitives.
3. **Loop unrolling** - LLVM should do this automatically, but may need hints.
4. **Bounds check elimination** - Vec indexing may have redundant bounds checks.
5. **cos/sin intrinsics** - Verify we're using LLVM intrinsics vs libm calls.

**Reference commits:**
- `ebc4574` - Float type tracking
- `c6bc8fe` - Vec element type tracking
- `9b792a0` - Mixed int/float operations

## Interpreter Limitations Discovered via Ritualis Integration (2026-02-22)

The following bugs were found when driving the interpreter from real application
code (`ritualis-core`).  Each is worked around by a dedicated native binding for
now; they should be fixed in the interpreter so the native-binding workarounds can
be removed.

### INT-001: Generic `json·from_str<T>` Does Not Dispatch T::decode

**Symptom:**
```sigil
≔ manifest~: PackageManifest = json·from_str(&body)?;
// manifest has no fields — accessing manifest·package errors
```
`json·from_str` is a generic Sigil function `from_str<T: Decode>(s) -> CodecResult<T>`.
The interpreter ignores the type annotation on the binding (`PackageManifest`), so
`T::decode` is never called and the raw parsed value (a `Value::Map` or empty struct)
is returned instead of the decoded struct.

**Expected behaviour:** The interpreter should propagate the declared type of the
binding into the function call as the concrete `T`, then dispatch `T::decode`.

**Workaround:** Dedicated native binding `registry·parse_manifest(s)` in `stdlib.rs`
that uses `serde_json` and builds the Sigil struct tree directly.

**Affects:** Any call to a generic Sigil function where the return type is constrained
by a trait (`Decode`, `From`, etc.) and the caller relies on the type annotation to
select the concrete impl.

---

### INT-002: `Vec[T]` Type Annotation Breaks `push()` When T Contains `·`

**Symptom:**
```sigil
≔ Δ versions~: Vec[semver·Version] = Vec·new();
versions.push(ver);  // ver: semver·Version
// Runtime error: type mismatch: expected Vec<semver>.push(semver), found semver·Version
```
The interpreter parses `Vec[semver·Version]` and strips the `·Version` suffix,
creating a Vec typed as `Vec<semver>`. When `push(ver)` is called with a full
`semver·Version` struct, the type check rejects it.

**Expected behaviour:** `Vec[semver·Version]` should create a Vec whose element type
is the struct `semver·Version`, not the module `semver`.

**Workaround:** Omit the type annotation — `≔ Δ versions~ = Vec·new()` — and let
the interpreter infer the element type from the first `push`.

**Affects:** Any `Vec[X·Y]` annotation where the element type is a qualified path
with two or more components.

---

## Resolved Interpreter Issues

### DONE: INT-003 — Scroll sub-module functions not callable via dot notation

**Status:** Fixed — commit `808ca70` in sigil-lang

`json·value_from_str(args)` (and any `scroll·fn(args)` form where `scroll` is a
loaded sub-module) was not dispatching to the registered function.  Sub-module
functions are stored in `self.globals` as `module·fn_name`, but the 2-segment
compound-name check in `eval_call` only searched `self.environment`.

**Fix:** In `eval_call`, extended the early-exit compound-name check to also probe
`self.globals`.  `call_function_by_name` already searches both, so the call path
is correct once the guard passes.

**Spec tests added** (3 tests in `interpreter.rs` `#[cfg(test)] mod tests`):
- `test_scroll_submodule_fn_callable_via_compound_name`
- `test_scroll_submodule_fn_overrides_variable_lookup`
- `test_scroll_submodule_fn_returns_correct_value`

---

### DONE: `rsplit` missing from String method dispatch

**Status:** Fixed — commit `5a5c1fe` in sigil-lang

`String·rsplit(sep)` was not handled in the interpreter's method dispatch table.
Calling `url·rsplit('/')·first()` (needed in `download.sigil` to extract the
filename from a URL) raised a runtime error.

**Fix:** Added `(Value::String(s), "rsplit")` branch alongside `split` in
`interpreter.rs`, supporting both `String` and `char` separators. Returns parts
in right-to-left order, matching Rust's `str::rsplit` semantics.

**Spec tests added** (5 tests in `interpreter.rs` `#[cfg(test)] mod tests`):
- `test_rsplit_with_string_sep_returns_parts_right_to_left`
- `test_rsplit_with_char_sep_matches_string_sep_order`
- `test_rsplit_no_separator_found_returns_whole_string`
- `test_rsplit_first_gives_last_path_component`
- `test_rsplit_symmetric_with_split_on_non_overlapping_sep`
