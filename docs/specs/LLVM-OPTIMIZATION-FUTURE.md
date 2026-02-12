# LLVM Optimization - Future Work

**Status**: Backlog
**Last Updated**: 2026-02-11

## Completed Optimizations

- [x] Phase 1: Native float compilation with LLVM intrinsics
- [x] Phase 2: Vec base pointer caching
- [x] Phase 3: Extended math intrinsics (sin, cos, pow, etc.)
- [x] Phase 4: Compile-time constant folding
- [x] Phase 5: AVX-512 auto-vectorization

**Current Performance**: Rust parity on vectorizable workloads

## Future Optimization Candidates

### 1. Vectorization Hints
Add LLVM loop metadata to encourage vectorization of complex loops:
- `llvm.loop.vectorize.enable`
- `llvm.loop.vectorize.width`
- `llvm.loop.unroll.count`

**Benefit**: More loops vectorized automatically

### 2. SIMD Reductions
Enable vectorized sum/product reductions:
- Currently scalar due to loop-carried dependency
- LLVM can use horizontal adds with proper hints
- Consider `-ffast-math` equivalent flags

**Benefit**: 4-8x speedup on reduction operations

### 3. Alias Analysis Annotations
Add `noalias`/`restrict` to function parameters:
- Help LLVM prove non-overlapping memory access
- Enables more aggressive load/store reordering

**Benefit**: Better optimization of pointer-heavy code

### 4. Aggressive Inlining
Tune inlining thresholds for hot paths:
- Mark small functions with `always_inline`
- Increase inline threshold for numerical code

**Benefit**: Reduced function call overhead

### 5. Native f64 Storage
Change Vec<f64> to store f64 directly:
- Currently stores as i64 bits (requires bitcasts)
- Would eliminate load/store conversions
- Larger refactor affecting runtime

**Benefit**: Cleaner IR, potentially better optimization

### 6. LTO Enhancements
Improve link-time optimization:
- Cross-module inlining
- Dead code elimination
- Whole-program devirtualization

**Benefit**: Better optimization across module boundaries

## Priority Assessment

| Optimization | Effort | Impact | Priority |
|--------------|--------|--------|----------|
| Vectorization hints | Low | Medium | P2 |
| SIMD reductions | Medium | High | P1 |
| Alias annotations | Low | Medium | P2 |
| Aggressive inlining | Low | Low | P3 |
| Native f64 storage | High | Medium | P3 |
| LTO enhancements | Medium | Medium | P2 |

## Notes

These optimizations are documented for a future session. Current focus is on lucifer infrastructure work.
