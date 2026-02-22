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

